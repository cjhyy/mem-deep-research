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
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mem_deep_research_core.core.constants import (
    DEFAULT_TEMPERATURE_BOOST,
    DEFAULT_TEMPERATURE_BOOST_CAP,
    MAX_CONTEXT_LIMIT_RETRIES,
    SUB_AGENT_PREFIX,
    TAG_COLLECTED_SOURCES,
    TAG_RESEARCH_PLAN,
    generate_message_id,
)
from mem_deep_research_core.core.hooks import HookContext, hooks
from mem_deep_research_core.core.llm_call_handler import generate_reflection_prompt
from mem_deep_research_core.core.monitoring import (
    EscalationAction,
    TurnCounter,
)

logger = logging.getLogger("mem_deep_research")


@dataclass
class MainLoopContext:
    """Bundles all dependencies for MainLoopRunner to avoid 25-param constructor."""

    cfg: Any
    monitor: Any
    context_manager: Any
    stream_handler: Any
    tool_executor: Any
    sub_agent_runner: Any
    llm_handler: Any
    summary_handler: Any
    task_planner: Any
    inline_skill_selector: Any
    llm_client: Any
    output_formatter: Any
    task_log: Any
    context: dict
    chinese_context: bool

    # Callbacks
    handle_llm_call: Callable
    handle_summary: Callable
    intercept_key_message: Callable
    streaming_final_message: Callable
    stream_tool_reasoning: Callable
    extract_recent_tool_names: Callable
    deduplicate_trailing_messages: Callable

    # Language (with default)
    response_language: str = "auto"

    # Agent name for stream events and hooks (default "main")
    agent_name: str = "main"


class MainLoopRunner:
    """主执行循环运行器

    封装 Agent 主循环的完整执行逻辑，从 Orchestrator 拆分出来以降低复杂度。
    通过依赖注入获取所需组件，不直接引用 Orchestrator。
    """

    def __init__(self, ctx: MainLoopContext):
        self.cfg = ctx.cfg
        self.monitor = ctx.monitor
        self.context_manager = ctx.context_manager
        self.stream_handler = ctx.stream_handler
        self.tool_executor = ctx.tool_executor
        self.sub_agent_runner = ctx.sub_agent_runner
        self.llm_handler = ctx.llm_handler
        self.summary_handler = ctx.summary_handler
        self.task_planner = ctx.task_planner
        self.inline_skill_selector = ctx.inline_skill_selector
        self.llm_client = ctx.llm_client
        self.output_formatter = ctx.output_formatter
        self.task_log = ctx.task_log
        self.context = ctx.context
        self.chinese_context = ctx.chinese_context
        self.response_language = ctx.response_language
        self.agent_name = ctx.agent_name

        # Callbacks (injected from Orchestrator)
        self._handle_llm_call = ctx.handle_llm_call
        self._handle_summary = ctx.handle_summary
        self._intercept_key_message = ctx.intercept_key_message
        self._streaming_final_message = ctx.streaming_final_message
        self._stream_tool_reasoning = ctx.stream_tool_reasoning
        self._extract_recent_tool_names = ctx.extract_recent_tool_names
        self._deduplicate_trailing_messages = ctx.deduplicate_trailing_messages

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
        task_guidance: str,
        keep_tool_result: int,
    ) -> tuple[str, bool]:
        """运行主执行循环

        Returns:
            (最终答案文本, is_simple_response)
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
        self.current_agent_id = await self.stream_handler.stream_start_agent(self.agent_name)
        await self.stream_handler.stream_start_llm(self.agent_name)

        # Hook: on_agent_start
        hooks.call(
            "on_agent_start",
            HookContext(
                hook_name="on_agent_start",
                query=task_description,
                context=self.context,
                extra={"agent_type": self.agent_name},
            ),
        )

        # Auto-detect response language from query (hook can override via on_agent_start)
        if self.response_language == "auto":
            from mem_deep_research_core.core.user_context import detect_language_by_chars
            self.response_language = detect_language_by_chars(task_description)
            self.chinese_context = self.response_language == "Chinese"
            logger.info(f"[Language] Auto-detected: {self.response_language}")

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
                            {"type": "text", "text": f"{TAG_RESEARCH_PLAN}\n{plan.to_context_string()}"}
                        ],
                    }
                )
                self.task_log.log_step(
                    "auto_planning",
                    f"Generated research plan with {len(plan.sub_questions)} sub-questions",
                )
                logger.info("[TaskPlanner] Plan injected into message history")

        total_tool_calls_executed = 0
        last_assistant_text = ""
        _context_limit_retries = 0
        _perf_main_loop_start = time.perf_counter()
        _perf_total_llm_time = 0.0
        _perf_total_tool_time = 0.0

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
            _perf_llm_start = time.perf_counter()
            assistant_response_text, should_break, tool_calls = await self._handle_llm_call(
                system_prompt,
                message_history,
                tool_definitions,
                turn_count,
                f"{self.agent_name} agent turn {turn_count}",
                keep_tool_result=keep_tool_result,
                agent_type=self.agent_name,
                stream_message_callback=self._intercept_key_message,
            )
            _perf_llm_elapsed = time.perf_counter() - _perf_llm_start
            _perf_total_llm_time += _perf_llm_elapsed
            self.task_log.append_perf("llm_call_durations", _perf_llm_elapsed)

            last_assistant_text = assistant_response_text or ""
            if assistant_response_text is not None:
                _context_limit_retries = 0

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
                hint_text = self.monitor.get_loop_break_hint(
                    recent_tool_names=recent_tools,
                    chinese=self.chinese_context,
                )
                message_history.append(
                    {"role": "user", "content": [{"type": "text", "text": hint_text}]}
                )
                temp_boost = getattr(self.monitor.config, 'temperature_boost', DEFAULT_TEMPERATURE_BOOST)
                temp_cap = getattr(self.monitor.config, 'temperature_boost_cap', DEFAULT_TEMPERATURE_BOOST_CAP)
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
            if assistant_response_text is not None:
                if should_break:
                    # 保存最终 checkpoint（即使 1 轮就结束）
                    self.task_log.save_checkpoint(
                        turn=turn_count,
                        message_count=len(message_history),
                        tool_calls_executed=total_tool_calls_executed,
                        last_assistant_text=last_assistant_text,
                        task_failed=task_failed,
                    )
                    break
            else:
                if tool_calls == "context_limit":
                    _context_limit_retries += 1
                    if _context_limit_retries > MAX_CONTEXT_LIMIT_RETRIES:
                        self.task_log.log_step(
                            "context_limit_exhausted",
                            f"Context limit retry exhausted after {MAX_CONTEXT_LIMIT_RETRIES} attempts",
                            "failed",
                        )
                        task_failed = True
                        break
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
                        hooks.call(
                            "on_context_compact",
                            HookContext(
                                hook_name="on_context_compact",
                                turn_number=turn_count,
                                compact_action="emergency",
                                query=task_description,
                                context=self.context,
                                extra={"message_count": len(message_history)},
                            ),
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

            # Hook: on_tool_filter — 去重后、执行前，可修改/重排/拦截工具调用列表
            if to_execute and hooks.has_hooks("on_tool_filter"):
                filtered = hooks.call(
                    "on_tool_filter",
                    HookContext(
                        hook_name="on_tool_filter",
                        turn_number=turn_count,
                        tool_calls_batch=to_execute,
                        query=task_description,
                        context=self.context,
                    ),
                )
                if filtered is not None and isinstance(filtered, list):
                    to_execute = filtered

            # 执行工具调用（仅非重复的）
            executed_tool_calls = (
                [tool_calls[0][:0], tool_calls[1]] if len(tool_calls) > 1 else [[], []]
            )
            executed_tool_calls[0] = to_execute
            all_tool_results_with_id = []

            if to_execute:
                _perf_tool_start = time.perf_counter()
                modified_tool_calls = [to_execute, tool_calls[1] if len(tool_calls) > 1 else []]
                all_tool_results_with_id = await self._execute_tools(
                    modified_tool_calls, max_tool_calls, keep_tool_result
                )
                _perf_tool_elapsed = time.perf_counter() - _perf_tool_start
                _perf_total_tool_time += _perf_tool_elapsed
                self.task_log.append_perf("tool_batch_durations", _perf_tool_elapsed)
                total_tool_calls_executed += len(to_execute)

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

            # Hook: on_context_compact — 压缩发生后通知业务层
            if action is not None:
                hooks.call(
                    "on_context_compact",
                    HookContext(
                        hook_name="on_context_compact",
                        turn_number=turn_count,
                        compact_action="summarize" if action == "need_summarize" else "masking",
                        query=task_description,
                        context=self.context,
                        extra={"message_count": len(message_history)},
                    ),
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
                    extra={
                        "assistant_text": last_assistant_text,
                        "message_count": len(message_history),
                        "total_tool_calls": total_tool_calls_executed,
                    },
                ),
            )

            # Turn checkpoint — save progress for debugging and potential resume
            self.task_log.save_checkpoint(
                turn=turn_count,
                message_count=len(message_history),
                tool_calls_executed=total_tool_calls_executed,
                last_assistant_text=last_assistant_text,
                task_failed=task_failed,
            )

            # 反思检查点
            if turn_counter.should_inject_reflection():

                reflection_prompt = generate_reflection_prompt(
                    turn_count, task_description, self.chinese_context
                )

                # Hook: on_reflection_build — 可修改反思 prompt
                if hooks.has_hooks("on_reflection_build"):
                    modified_prompt = hooks.call(
                        "on_reflection_build",
                        HookContext(
                            hook_name="on_reflection_build",
                            turn_number=turn_count,
                            result=reflection_prompt,
                            query=task_description,
                            context=self.context,
                        ),
                    )
                    if modified_prompt is not None and isinstance(modified_prompt, str):
                        reflection_prompt = modified_prompt

                message_history.append(
                    {"role": "user", "content": [{"type": "text", "text": reflection_prompt}]}
                )
                self.task_log.log_step(
                    "reflection_checkpoint", f"Injected at turn {turn_count}", "info"
                )
                logger.debug(f"[Deep Research] Reflection checkpoint injected at turn {turn_count}")

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
                    "agent_type": self.agent_name,
                    "task_failed": task_failed,
                    "turns_used": turn_counter.current_turn,
                    "total_tool_calls": total_tool_calls_executed,
                    "duration_seconds": time.perf_counter() - _perf_main_loop_start,
                    "message_count": len(message_history),
                },
            ),
        )

        # 退出 LLM/Agent
        await self.stream_handler.stream_end_llm(self.agent_name)
        await self.stream_handler.stream_end_agent(self.agent_name, self.current_agent_id)

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
                            "text": f"{TAG_COLLECTED_SOURCES}\n{citation_summary}\n\nPlease include these sources in your final summary where relevant.",
                        }
                    ],
                }
            )
            self.task_log.log_step(
                "citation_injection",
                f"Injected {len(self.context_manager.source_registry.get_all_sources())} sources into message history",
            )

        # 检测简单响应：无工具调用且有有效文本
        is_simple_response = (
            not task_failed
            and total_tool_calls_executed == 0
            and last_assistant_text
            and last_assistant_text.strip()
        )

        # Record main loop timing
        _perf_main_loop_elapsed = time.perf_counter() - _perf_main_loop_start
        self.task_log.record_perf("main_loop_duration", _perf_main_loop_elapsed)
        self.task_log.record_perf("main_loop_total_llm_time", _perf_total_llm_time)
        self.task_log.record_perf("main_loop_total_tool_time", _perf_total_tool_time)
        self.task_log.record_perf("main_loop_turns", turn_counter.current_turn, unit="")
        self.task_log.record_perf("main_loop_tool_calls", total_tool_calls_executed, unit="")

        if is_simple_response:
            # 简单响应：跳过 summary LLM 调用，直接使用最后的 assistant 文本
            self.task_log.log_step("final_summary", "Simple response detected, skipping summary LLM call")
            self.task_log.record_perf("summary_skipped", 1, unit="bool")
            self.task_log.record_perf("summary_duration", 0.0)

            self.current_agent_id = await self.stream_handler.stream_start_agent("reporter")
            await self.stream_handler.stream_start_llm("reporter")

            await self._streaming_final_message(generate_message_id(), last_assistant_text, True)
            final_answer_text = last_assistant_text
        else:
            # 生成最终摘要
            self.task_log.log_step("final_summary", "Generating final summary")
            self.task_log.record_perf("summary_skipped", 0, unit="bool")

            self.current_agent_id = await self.stream_handler.stream_start_agent("reporter")
            await self.stream_handler.stream_start_llm("reporter")

            _perf_summary_start = time.perf_counter()
            final_answer_text = await self._handle_summary(
                system_prompt,
                main_agent_prompt_instance,
                message_history,
                tool_definitions,
                "Final summary generation",
                task_description,
                task_failed,
                agent_type=self.agent_name,
                task_guidance=task_guidance,
                stream_message_callback=self._streaming_final_message,
                deep_research_cfg=deep_research_cfg,
            )
            self.task_log.record_perf("summary_duration", time.perf_counter() - _perf_summary_start)

        return final_answer_text, is_simple_response

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

            if server_name.startswith(SUB_AGENT_PREFIX):
                if self.sub_agent_runner is None:
                    # Sub-agents cannot spawn sub-agents
                    tool_result_for_llm = {"type": "text", "text": "[Error] Sub-agent spawning is not available in this context."}
                    all_tool_results_with_id.append((call_id, tool_result_for_llm))
                    continue

                # 子 Agent 调用
                await self.stream_handler.stream_end_llm(self.agent_name)
                await self.stream_handler.stream_end_agent(self.agent_name, self.current_agent_id)

                sub_agent_result = await self.sub_agent_runner.run(
                    server_name, arguments, keep_tool_result
                )

                tool_result = {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "result": sub_agent_result,
                }

                self.current_agent_id = await self.stream_handler.stream_start_agent(
                    self.agent_name, display_name="Summarizing"
                )
                await self.stream_handler.stream_start_llm(self.agent_name, display_name="Summarizing")

                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(tool_result)
                all_tool_results_with_id.append((call_id, tool_result_for_llm))
            else:
                # 普通工具调用
                tool_result, _ = await self.tool_executor.execute_single_tool(
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments=arguments,
                    call_id=call_id,
                    agent_name=self.agent_name,
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
            agent_type=self.agent_name,
        )
        return response_text or ""
