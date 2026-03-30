"""
主执行循环模块

从 Orchestrator 拆分，负责 Agent 主循环的执行逻辑：
- 轮次循环控制
- 工具调用执行与去重
- 监控检查与升级
- 反思检查点注入
- 最终摘要生成
"""

import logging
import sys
from collections.abc import Callable

from mem_deep_research_core.core.hooks import HookContext, hooks
from mem_deep_research_core.core.monitoring import (
    EscalationAction,
    TurnCounter,
)

logger = logging.getLogger("mem_deep_research")


class MainLoopRunner:
    """主执行循环运行器

    封装 Agent 主循环的完整执行逻辑，从 Orchestrator 拆分出来以降低复杂度。
    通过依赖注入获取所需组件，不直接引用 Orchestrator。
    """

    def __init__(
        self,
        *,
        cfg,
        monitor,
        context_manager,
        stream_handler,
        tool_executor,
        sub_agent_runner,
        llm_handler,
        summary_handler,
        task_planner,
        inline_skill_selector,
        llm_client,
        output_formatter,
        task_log,
        context: dict,
        chinese_context: bool,
        monitoring_schema_dict: dict,
        # callbacks
        handle_llm_call: Callable,
        handle_summary: Callable,
        intercept_key_message: Callable,
        streaming_final_message: Callable,
        stream_tool_reasoning: Callable,
        extract_recent_tool_names: Callable,
        deduplicate_trailing_messages: Callable,
        build_user_identity_context: Callable,
        detect_language: Callable,
    ):
        self.cfg = cfg
        self.monitor = monitor
        self.context_manager = context_manager
        self.stream_handler = stream_handler
        self.tool_executor = tool_executor
        self.sub_agent_runner = sub_agent_runner
        self.llm_handler = llm_handler
        self.summary_handler = summary_handler
        self.task_planner = task_planner
        self.inline_skill_selector = inline_skill_selector
        self.llm_client = llm_client
        self.output_formatter = output_formatter
        self.task_log = task_log
        self.context = context
        self.chinese_context = chinese_context
        self._monitoring_schema_dict = monitoring_schema_dict

        # Callbacks (injected from Orchestrator)
        self._handle_llm_call = handle_llm_call
        self._handle_summary = handle_summary
        self._intercept_key_message = intercept_key_message
        self._streaming_final_message = streaming_final_message
        self._stream_tool_reasoning = stream_tool_reasoning
        self._extract_recent_tool_names = extract_recent_tool_names
        self._deduplicate_trailing_messages = deduplicate_trailing_messages
        self._build_user_identity_context = build_user_identity_context
        self._detect_language = detect_language

        # 当前 Agent ID
        self.current_agent_id: str | None = None

    async def run(
        self,
        system_prompt: str,
        message_history: list,
        tool_definitions: list,
        main_agent_prompt_instance,
        deep_research_cfg: dict | None,
        task_description: str,
        task_guidence: str,
        keep_tool_result: int,
    ) -> str:
        """运行主执行循环

        Returns:
            最终答案文本
        """
        max_turns = self.cfg.main_agent.max_turns
        if max_turns < 0:
            max_turns = sys.maxsize
        max_tool_calls = self.cfg.main_agent.max_tool_calls_per_turn

        # 初始化监控和计数器
        self.monitor.reset()
        self.context_manager.reset()
        turn_counter = TurnCounter(
            max_turns=max_turns,
            reflection_enabled=deep_research_cfg and deep_research_cfg.get("enabled", False),
            reflection_interval=deep_research_cfg.get("reflection_interval", 5)
            if deep_research_cfg
            else 5,
        )

        task_failed = False
        self.current_agent_id = await self.stream_handler.stream_start_agent("main")
        await self.stream_handler.stream_start_llm("main")

        # Hook: on_agent_start
        hooks.call(
            "on_agent_start",
            HookContext(
                hook_name="on_agent_start",
                query=task_description,
                context=self.context,
                extra={"agent_type": "main"},
            ),
        )

        # 自动任务分解（仅深度研究模式 + auto_planning 启用时）
        if self.task_planner.enabled:
            plan = await self.task_planner.create_plan(
                task_description=task_description,
                llm_client=self.llm_client,
            )
            if plan:
                message_history.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"[RESEARCH PLAN]\n{plan.to_context_string()}"}
                        ],
                    }
                )
                self.task_log.log_step(
                    "auto_planning",
                    f"Generated research plan with {len(plan.sub_questions)} sub-questions",
                )
                logger.info("[TaskPlanner] Plan injected into message history")

        while not turn_counter.is_max_reached():
            turn_count = turn_counter.increment()
            self.context_manager.set_turn(turn_count)
            logger.debug(f"\n--- Main Agent Turn {turn_count} ---")

            # Hook: on_turn_start
            hooks.call(
                "on_turn_start",
                HookContext(
                    hook_name="on_turn_start",
                    turn_number=turn_count,
                    query=task_description,
                    context=self.context,
                ),
            )
            self.task_log.save()

            # 监控前检查
            terminate_reason = await self.monitor.pre_turn_check()
            if terminate_reason:
                break

            # LLM 调用
            assistant_response_text, should_break, tool_calls = await self._handle_llm_call(
                system_prompt,
                message_history,
                tool_definitions,
                turn_count,
                f"Main agent turn {turn_count}",
                keep_tool_result=keep_tool_result,
                agent_type="main",
                stream_message_callback=self._intercept_key_message,
            )

            # 清除温度覆盖（无论 LLM 调用成功与否）
            self.llm_client.clear_temperature_override()

            # 监控后检查
            terminate_reason = await self.monitor.post_turn_check(
                response_text=assistant_response_text,
                llm_call_failed=tool_calls == "context_limit" or assistant_response_text is None,
            )

            # 立即检查终止原因
            if terminate_reason:
                task_failed = True
                break

            # 响应循环升级到 INJECT_HINT 时注入强制策略指令 + 温度提升
            if self.monitor.last_loop_action == EscalationAction.INJECT_HINT:
                recent_tools = self._extract_recent_tool_names(message_history)
                hint_text = self.monitor.get_loop_break_hint(recent_tool_names=recent_tools)
                message_history.append(
                    {"role": "user", "content": [{"type": "text", "text": hint_text}]}
                )
                temp_boost = self._monitoring_schema_dict.get("temperature_boost", 0.3)
                temp_cap = self._monitoring_schema_dict.get("temperature_boost_cap", 1.0)
                self.llm_client.set_temperature_boost(boost=temp_boost, cap=temp_cap)

            # Inline Skill: 从 LLM 回复中解析 <next_skills>，下一轮动态注入
            if self.inline_skill_selector and assistant_response_text:
                next_skills = self.inline_skill_selector.update_pending_skills(
                    assistant_response_text
                )
                if next_skills:
                    system_prompt = self.inline_skill_selector.inject_pending_skills(system_prompt)
                    logger.info(f"[InlineSkill] Injected skills for next turn: {next_skills}")

            # 处理 LLM 响应
            if assistant_response_text:
                if should_break:
                    break
            else:
                if tool_calls == "context_limit":
                    # Level 3: 紧急裁剪
                    emergency_count = self.context_manager.apply_emergency(
                        message_history,
                        turn_count,
                        system_prompt,
                        self.llm_client.max_context_length,
                    )
                    if emergency_count > 0:
                        self.task_log.log_step(
                            "context_emergency",
                            f"[CONTEXT L3] Emergency: processed {emergency_count} messages, "
                            f"history now {len(message_history)} messages",
                            "info",
                        )
                        continue  # retry LLM call
                    self.task_log.log_step(
                        "main_agent_context_limit_reached", "Context limit reached", "warning"
                    )
                else:
                    self.task_log.log_step("main_agent", "LLM call failed", "failed")
                task_failed = True
                continue

            # 检查是否有工具调用
            if (
                not tool_calls
                or len(tool_calls) < 2
                or (len(tool_calls[0]) == 0 and len(tool_calls[1]) == 0)
            ):
                logger.debug("LLM did not request tool use, process ends.")
                break

            # 跨轮次去重过滤
            calls_to_execute = tool_calls[0][:max_tool_calls]
            to_execute, cached_results = self.context_manager.filter_duplicate_calls(
                calls_to_execute
            )

            # 执行工具调用（仅非重复的）
            executed_tool_calls = (
                [tool_calls[0][:0], tool_calls[1]] if len(tool_calls) > 1 else [[], []]
            )
            executed_tool_calls[0] = to_execute
            all_tool_results_with_id = []

            if to_execute:
                modified_tool_calls = [to_execute, tool_calls[1] if len(tool_calls) > 1 else []]
                all_tool_results_with_id = await self._execute_tools(
                    modified_tool_calls, max_tool_calls, keep_tool_result
                )

            # 添加缓存结果（dedup 命中的）
            for call_id, cached_content in cached_results:
                all_tool_results_with_id.append((call_id, cached_content))

            # 注册 tool results
            if to_execute and all_tool_results_with_id:
                executed_results = all_tool_results_with_id[: len(to_execute)]
                self.context_manager.register_tool_results(to_execute, executed_results, turn_count)

            # 记录策略摘要
            for call in to_execute:
                self.monitor.record_strategy_summary(
                    f"{call.get('tool_name', '?')}({str(call.get('arguments', ''))[:100]})"
                )

            # 更新消息历史
            tool_calls_exceeded = len(tool_calls[0]) > max_tool_calls
            message_history = self.llm_client.update_message_history(
                message_history, all_tool_results_with_id, tool_calls_exceeded
            )

            # 三级 Context 管理
            action = self.context_manager.manage_context(
                message_history,
                turn_count,
                system_prompt,
                self.llm_client.max_context_length,
            )
            if action == "need_summarize":
                await self.context_manager.apply_summarize(
                    message_history,
                    turn_count,
                    system_prompt,
                    self.llm_client.max_context_length,
                    llm_call_fn=self._context_summarize_call,
                )

            # Hook: on_turn_end
            tool_calls_count = len(tool_calls[0]) if tool_calls and len(tool_calls) > 0 else 0
            hooks.call(
                "on_turn_end",
                HookContext(
                    hook_name="on_turn_end",
                    turn_number=turn_count,
                    tool_calls_count=tool_calls_count,
                    query=task_description,
                    context=self.context,
                ),
            )

            # 反思检查点
            if turn_counter.should_inject_reflection():
                from mem_deep_research_core.core.llm_call_handler import generate_reflection_prompt

                reflection_prompt = generate_reflection_prompt(
                    turn_count, task_description, self.chinese_context
                )
                message_history.append(
                    {"role": "user", "content": [{"type": "text", "text": reflection_prompt}]}
                )
                self.task_log.log_step(
                    "reflection_checkpoint", f"Injected at turn {turn_count}", "info"
                )
                logger.debug(f"[Deep Research] Reflection checkpoint injected at turn {turn_count}")

        # 清理
        self.context_manager.cancel()

        # Hook: on_agent_end
        hooks.call(
            "on_agent_end",
            HookContext(
                hook_name="on_agent_end",
                query=task_description,
                turn_number=turn_counter.current_turn,
                result=task_failed,
                context=self.context,
                extra={
                    "agent_type": "main",
                    "task_failed": task_failed,
                    "turns_used": turn_counter.current_turn,
                },
            ),
        )

        # 退出主 LLM/Agent
        await self.stream_handler.stream_end_llm("main")
        await self.stream_handler.stream_end_agent("main", self.current_agent_id)

        # 记录循环结束
        if turn_counter.is_max_reached():
            if not task_failed:
                task_failed = True
            self.task_log.log_step(
                "max_turns_reached", f"Reached maximum turns ({max_turns})", "warning"
            )
        else:
            self.task_log.log_step(
                "main_loop_completed", f"Completed after {turn_counter.current_turn} turns"
            )

        # 循环终止时清理重复的 assistant 响应
        if task_failed:
            self._deduplicate_trailing_messages(message_history)

        # 注入引用信息到消息历史
        citation_summary = self.context_manager.source_registry.get_citation_summary()
        if citation_summary:
            message_history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"[COLLECTED SOURCES]\n{citation_summary}\n\nPlease include these sources in your final summary where relevant.",
                        }
                    ],
                }
            )
            self.task_log.log_step(
                "citation_injection",
                f"Injected {len(self.context_manager.source_registry.get_all_sources())} sources into message history",
            )

        # 生成最终摘要
        self.task_log.log_step("final_summary", "Generating final summary")

        self.current_agent_id = await self.stream_handler.stream_start_agent("reporter")
        await self.stream_handler.stream_start_llm("reporter")

        final_answer_text = await self._handle_summary(
            system_prompt,
            main_agent_prompt_instance,
            message_history,
            tool_definitions,
            "Final summary generation",
            task_description,
            task_failed,
            agent_type="main",
            task_guidence=task_guidence,
            stream_message_callback=self._streaming_final_message,
            deep_research_cfg=deep_research_cfg,
        )

        return final_answer_text

    async def _execute_tools(
        self,
        tool_calls: list,
        max_tool_calls: int,
        keep_tool_result: int,
    ) -> list:
        """执行工具调用"""
        all_tool_results_with_id = []

        tool_calls_exceeded = len(tool_calls[0]) > max_tool_calls
        if tool_calls_exceeded:
            logger.warning(
                f"[ERROR] Tool call count too high ({len(tool_calls[0])}), "
                f"only processing first {max_tool_calls}"
            )

        for call in tool_calls[0][:max_tool_calls]:
            server_name = call["server_name"]
            tool_name = call["tool_name"]
            arguments = call["arguments"]
            call_id = call["id"]

            if server_name.startswith("agent-"):
                # 子 Agent 调用
                await self.stream_handler.stream_end_llm("main")
                await self.stream_handler.stream_end_agent("main", self.current_agent_id)

                sub_agent_result = await self.sub_agent_runner.run(
                    server_name, str(arguments), keep_tool_result
                )

                tool_result = {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "result": sub_agent_result,
                }

                self.current_agent_id = await self.stream_handler.stream_start_agent(
                    "main", display_name="Summarizing"
                )
                await self.stream_handler.stream_start_llm("main", display_name="Summarizing")

                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(tool_result)
                all_tool_results_with_id.append((call_id, tool_result_for_llm))
            else:
                # 普通工具调用
                tool_result, _ = await self.tool_executor.execute_single_tool(
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments=arguments,
                    call_id=call_id,
                    agent_name="main",
                )

                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(tool_result)
                all_tool_results_with_id.append((call_id, tool_result_for_llm))

        # 处理失败的工具调用
        if len(tool_calls) > 1 and len(tool_calls[1]) > 0:
            _, failed_results = self.tool_executor.handle_failed_tool_calls(tool_calls[1])
            all_tool_results_with_id.extend(failed_results)

        return all_tool_results_with_id

    async def _context_summarize_call(
        self,
        summarize_system_prompt: str,
        summarize_messages: list,
        purpose: str,
    ) -> str:
        """Level 2 context 压缩的 LLM 调用"""
        response_text, _, _ = await self.llm_handler.handle_llm_call(
            summarize_system_prompt,
            summarize_messages,
            [],
            999,
            purpose,
            agent_type="main",
        )
        return response_text or ""
