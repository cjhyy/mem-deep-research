"""
子 Agent 运行模块

处理子 Agent 的完整生命周期，包括初始化、执行循环和结果汇总。
"""

import logging
import sys
import uuid
from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig

from mem_deep_research_core.core.hooks import HookContext, hooks
from mem_deep_research_core.core.tool_executor import ToolExecutor
from mem_deep_research_core.core.tool_result_formatter import ToolResultFormatter
from mem_deep_research_core.llm.provider_client_base import LLMProviderClientBase
from mem_deep_research_core.mem_deep_research_logging.logger import truncate_for_log
from mem_deep_research_core.mem_deep_research_logging.task_tracer import TaskTracer
from mem_deep_research_core.tool.manager import ToolManager
from mem_deep_research_core.utils.external_loader import external_loader
from mem_deep_research_core.utils.io_utils import OutputFormatter
from mem_deep_research_core.utils.tool_utils import _load_agent_prompt

logger = logging.getLogger("mem_deep_research")


class SubAgentRunner:
    """子 Agent 运行器"""

    def __init__(
        self,
        sub_agent_tool_managers: dict[str, ToolManager],
        sub_agent_llm_client: LLMProviderClientBase,
        output_formatter: OutputFormatter,
        cfg: DictConfig,
        task_log: TaskTracer,
        context: dict[str, Any] | None = None,
        chinese_context: bool = False,
        # 流式回调
        stream_start_agent: Callable | None = None,
        stream_end_agent: Callable | None = None,
        stream_start_llm: Callable | None = None,
        stream_end_llm: Callable | None = None,
        stream_tool_call: Callable | None = None,
        stream_tool_reasoning: Callable | None = None,
        stream_usage_info: Callable | None = None,
        # LLM 调用回调
        handle_llm_call: Callable | None = None,
        handle_summary: Callable | None = None,
        # 消息拦截回调
        intercept_key_message: Callable | None = None,
    ):
        """
        初始化子 Agent 运行器

        Args:
            sub_agent_tool_managers: 子 Agent 工具管理器字典
            sub_agent_llm_client: 子 Agent 的 LLM 客户端
            output_formatter: 输出格式化器
            cfg: 配置
            task_log: 任务日志
            context: 用户上下文
            chinese_context: 是否使用中文上下文
            stream_*: 各种流式输出回调
            handle_llm_call: LLM 调用处理回调
            handle_summary: 摘要生成回调
            intercept_key_message: 消息拦截回调
        """
        self.sub_agent_tool_managers = sub_agent_tool_managers
        self.sub_agent_llm_client = sub_agent_llm_client
        self.output_formatter = output_formatter
        self.cfg = cfg
        self.task_log = task_log
        self.context = context or {}
        self.chinese_context = chinese_context

        # 流式回调
        self.stream_start_agent = stream_start_agent
        self.stream_end_agent = stream_end_agent
        self.stream_start_llm = stream_start_llm
        self.stream_end_llm = stream_end_llm
        self.stream_tool_call = stream_tool_call
        self.stream_tool_reasoning = stream_tool_reasoning
        self.stream_usage_info = stream_usage_info

        # LLM 调用回调
        self.handle_llm_call = handle_llm_call
        self.handle_summary = handle_summary

        # 消息拦截回调
        self.intercept_key_message = intercept_key_message

        # 工具结果格式化器
        self.tool_result_formatter = ToolResultFormatter(context)

        # 预缓存的工具定义
        self._cached_tool_definitions: dict[str, list[dict]] | None = None

    def set_cached_tool_definitions(self, definitions: dict[str, list[dict]]):
        """设置预缓存的子 Agent 工具定义"""
        self._cached_tool_definitions = definitions

    async def _get_tool_definitions(self, sub_agent_name: str) -> list[dict]:
        """获取子 Agent 的工具定义"""
        if self._cached_tool_definitions:
            return self._cached_tool_definitions.get(sub_agent_name, [])

        # 动态获取
        tool_manager = self.sub_agent_tool_managers.get(sub_agent_name)
        if tool_manager:
            return await tool_manager.get_all_tool_definitions()
        return []

    def _create_tool_executor(self, sub_agent_name: str) -> ToolExecutor:
        """为子 Agent 创建工具执行器"""
        tool_manager = self.sub_agent_tool_managers.get(sub_agent_name)
        if not tool_manager:
            raise ValueError(f"Tool manager not found for sub-agent: {sub_agent_name}")

        return ToolExecutor(
            tool_manager=tool_manager,
            output_formatter=self.output_formatter,
            tool_result_formatter=self.tool_result_formatter,
            context=self.context,
            stream_tool_call=self.stream_tool_call,
            stream_tool_reasoning=self.stream_tool_reasoning,
            stream_usage_info=self.stream_usage_info,
        )

    def _load_sub_agent_prompt(self, sub_agent_name: str):
        """加载子 Agent 的提示词实例"""
        if not self.cfg.sub_agents or sub_agent_name not in self.cfg.sub_agents:
            raise ValueError(f"Sub-agent {sub_agent_name} not found in configuration")

        sub_agent_cfg = self.cfg.sub_agents[sub_agent_name]

        # 获取 prompt 配置
        prompt_cfg = {}
        if hasattr(sub_agent_cfg, "prompt") and sub_agent_cfg.prompt:
            prompt_cfg = dict(sub_agent_cfg.prompt)
        # 子 Agent 默认为 worker 类型
        if "agent_type" not in prompt_cfg:
            prompt_cfg["agent_type"] = "worker"

        return _load_agent_prompt(prompt_cfg)

    def _generate_system_prompt(
        self,
        sub_agent_name: str,
        tool_definitions: list[dict],
        task_description: str,
    ) -> str:
        """生成子 Agent 的系统提示词"""
        prompt_instance = self._load_sub_agent_prompt(sub_agent_name)

        system_prompt = prompt_instance.generate_system_prompt_with_mcp_tools(
            mcp_servers=tool_definitions,
            chinese_context=self.chinese_context,
        )

        # 注入 Skills
        skill_injector = external_loader.get_skill_injector()
        if skill_injector and task_description:
            tools_to_use = [t.get("name", "") for t in tool_definitions if isinstance(t, dict)]
            system_prompt = skill_injector.inject_skills(
                base_prompt=system_prompt,
                query=task_description,
                context=self.context,
                tools_to_use=tools_to_use,
            )

        return system_prompt

    async def run(
        self,
        sub_agent_name: str,
        task_description: str,
        keep_tool_result: int = -1,
    ) -> str:
        """
        运行子 Agent

        Args:
            sub_agent_name: 子 Agent 名称
            task_description: 任务描述
            keep_tool_result: 保留工具结果数量

        Returns:
            str: 子 Agent 的最终回答
        """
        logger.debug(f"\n=== Starting Sub Agent {sub_agent_name} ===")
        task_description += "\n\nPlease provide the answer and detailed supporting information of the subtask given to you."
        logger.debug(f"Subtask: {task_description}")

        # 发送子 Agent 开始事件
        display_name = sub_agent_name.replace("agent-", "")
        sub_agent_id = None
        if self.stream_start_agent:
            sub_agent_id = await self.stream_start_agent(display_name)
        if self.stream_start_llm:
            await self.stream_start_llm(display_name)

        # 开始新的子 Agent 会话
        self.task_log.start_sub_agent_session(sub_agent_name, task_description)

        # 初始化消息历史
        initial_user_content = [{"type": "text", "text": task_description}]
        message_history = [{"role": "user", "content": initial_user_content}]

        # 获取工具定义
        tool_definitions = await self._get_tool_definitions(sub_agent_name)
        self.task_log.log_step(f"get_sub_{sub_agent_name}_tool_definitions", f"{tool_definitions}")

        if not tool_definitions:
            logger.debug(
                "Warning: Failed to get any tool definitions. LLM may not be able to use tools."
            )
            self.task_log.log_step(
                f"{sub_agent_name}_no_tools",
                f"No tool definitions available for {sub_agent_name}",
                "warning",
            )

        # 生成系统提示词
        prompt_instance = self._load_sub_agent_prompt(sub_agent_name)
        system_prompt = self._generate_system_prompt(
            sub_agent_name, tool_definitions, task_description
        )

        # 创建工具执行器
        tool_executor = self._create_tool_executor(sub_agent_name)

        # 获取配置
        sub_agent_cfg = self.cfg.sub_agents[sub_agent_name]
        max_turns = sub_agent_cfg.max_turns
        if max_turns < 0:
            max_turns = sys.maxsize
        max_tool_calls = sub_agent_cfg.max_tool_calls_per_turn

        # Hook: on_agent_start
        hooks.call(
            "on_agent_start",
            HookContext(
                hook_name="on_agent_start",
                query=task_description,
                context=self.context,
                extra={"agent_type": sub_agent_name},
            ),
        )

        # 执行循环
        turn_count = 0
        task_failed = False

        while turn_count < max_turns:
            turn_count += 1
            logger.debug(f"\n--- Sub Agent {sub_agent_name} Turn {turn_count} ---")

            # Hook: on_turn_start
            hooks.call(
                "on_turn_start",
                HookContext(
                    hook_name="on_turn_start",
                    turn_number=turn_count,
                    query=task_description,
                    context=self.context,
                    extra={"agent_type": sub_agent_name},
                ),
            )
            self.task_log.save()

            # 调用 LLM
            if not self.handle_llm_call:
                logger.error("handle_llm_call callback not set")
                break

            assistant_response_text, should_break, tool_calls = await self.handle_llm_call(
                system_prompt,
                message_history,
                tool_definitions,
                turn_count,
                f"Sub agent {sub_agent_name} turn {turn_count}",
                keep_tool_result=keep_tool_result,
                agent_type=sub_agent_name,
                stream_message_callback=self.intercept_key_message,
            )

            # 处理 LLM 响应
            if assistant_response_text:
                if should_break:
                    self.task_log.log_step(
                        "sub_agent_early_termination",
                        f"Sub agent {sub_agent_name} terminated early on turn {turn_count}",
                    )
                    break
            else:
                if tool_calls == "context_limit":
                    self.task_log.log_step(
                        "sub_agent_context_limit_reached",
                        f"Sub agent {sub_agent_name} context limit reached, jumping to summary",
                        "warning",
                    )
                else:
                    self.task_log.log_step(
                        "sub_agent_llm_call_failed",
                        "LLM call failed",
                        "failed",
                    )
                task_failed = True
                break

            # 检查是否有工具调用
            if (
                tool_calls is None
                or len(tool_calls) < 2
                or (len(tool_calls[0]) == 0 and len(tool_calls[1]) == 0)
            ):
                logger.debug(f"Sub Agent {sub_agent_name} did not request tool use, ending task.")
                self.task_log.log_step(
                    "sub_agent_no_tool_calls",
                    f"No tool calls found in sub agent {sub_agent_name}, ending on turn {turn_count}",
                )
                break

            # 执行工具调用
            (
                tool_calls_data,
                tool_results_with_id,
                exceeded,
            ) = await tool_executor.execute_tool_calls(
                tool_calls[0],
                max_tool_calls,
                agent_name=sub_agent_name,
            )

            # 处理失败的工具调用
            if len(tool_calls) > 1 and len(tool_calls[1]) > 0:
                failed_data, failed_results = tool_executor.handle_failed_tool_calls(tool_calls[1])
                tool_calls_data.extend(failed_data)
                tool_results_with_id.extend(failed_results)

            # 更新消息历史
            message_history = self.sub_agent_llm_client.update_message_history(
                message_history, tool_results_with_id, exceeded
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
                    extra={"agent_type": sub_agent_name},
                ),
            )

        # Hook: on_agent_end
        hooks.call(
            "on_agent_end",
            HookContext(
                hook_name="on_agent_end",
                query=task_description,
                turn_number=turn_count,
                result=task_failed,
                context=self.context,
                extra={
                    "agent_type": sub_agent_name,
                    "task_failed": task_failed,
                    "turns_used": turn_count,
                },
            ),
        )

        # 记录循环结束
        logger.debug(f"\n=== Sub Agent {sub_agent_name} Completed ({turn_count} turns) ===")

        if turn_count >= max_turns:
            if not task_failed:
                task_failed = True
            self.task_log.log_step(
                "sub_agent_max_turns_reached",
                f"Sub agent {sub_agent_name} reached maximum turns ({max_turns})",
                "warning",
            )
        else:
            self.task_log.log_step(
                "sub_agent_loop_completed",
                f"Sub agent {sub_agent_name} loop completed after {turn_count} turns",
            )

        # 生成最终摘要
        self.task_log.log_step(
            "sub_agent_final_summary",
            f"Generating sub agent {sub_agent_name} final summary",
        )

        if self.stream_tool_call:
            await self.stream_tool_call("Partial Summary", {}, tool_call_id=str(uuid.uuid4()))

        # 使用 context limit 重试逻辑生成最终摘要
        final_answer_text = ""
        if self.handle_summary:
            final_answer_text = await self.handle_summary(
                system_prompt,
                prompt_instance,
                message_history,
                tool_definitions,
                f"Sub agent {sub_agent_name} final summary",
                task_description,
                task_failed,
                agent_type=sub_agent_name,
                stream_message_callback=self.intercept_key_message,
            )

        if final_answer_text:
            self.task_log.log_step(
                "sub_agent_final_answer",
                f"Sub agent {sub_agent_name} final answer generated successfully",
            )
        else:
            final_answer_text = f"No final answer generated by sub agent {sub_agent_name}."
            self.task_log.log_step(
                "sub_agent_final_answer",
                f"Failed to generate sub agent {sub_agent_name} final answer",
                "failed",
            )

        logger.debug(
            f"Sub Agent {sub_agent_name} Final Answer: {truncate_for_log(final_answer_text)}"
        )

        # 保存消息历史
        self.task_log.sub_agent_message_history_sessions[
            self.task_log.current_sub_agent_session_id
        ] = {"system_prompt": system_prompt, "message_history": message_history}
        self.task_log.save()

        # 结束子 Agent 会话
        self.task_log.end_sub_agent_session(sub_agent_name)
        self.task_log.log_step(
            "sub_agent_completed", f"Sub agent {sub_agent_name} completed", "info"
        )

        # 发送子 Agent 结束事件
        if self.stream_end_llm:
            await self.stream_end_llm(display_name)
        if self.stream_end_agent and sub_agent_id:
            await self.stream_end_agent(display_name, sub_agent_id)

        return final_answer_text
