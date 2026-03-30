"""
工具执行模块

处理工具调用的执行、结果收集和错误处理。
"""

import datetime
import logging
import time
from collections.abc import Callable
from typing import Any

from mem_deep_research_core.core.hooks import HookContext, hooks
from mem_deep_research_core.core.secure_context import resolve_placeholders_in_args
from mem_deep_research_core.core.tool_result_formatter import ToolResultFormatter
from mem_deep_research_core.tool.manager import ToolManager
from mem_deep_research_core.utils.io_utils import OutputFormatter

logger = logging.getLogger("mem_deep_research")


# ========== 默认钩子实现 ==========


def _default_on_tool_start(ctx: HookContext):
    """工具调用开始 - 默认实现，返回原始 arguments"""
    return ctx.arguments


def _default_on_tool_end(ctx: HookContext):
    """工具调用完成 - 默认实现，返回原始 tool_result"""
    return ctx.tool_result


hooks.set_default("on_tool_start", _default_on_tool_start)
hooks.set_default("on_tool_end", _default_on_tool_end)


class ToolExecutor:
    """工具调用执行器"""

    def __init__(
        self,
        tool_manager: ToolManager,
        output_formatter: OutputFormatter,
        tool_result_formatter: ToolResultFormatter,
        context: dict[str, Any] | None = None,
        stream_tool_call: Callable | None = None,
        stream_tool_reasoning: Callable | None = None,
        stream_usage_info: Callable | None = None,
    ):
        """
        初始化工具执行器

        Args:
            tool_manager: 工具管理器
            output_formatter: 输出格式化器
            tool_result_formatter: 工具结果格式化器
            context: 用户上下文
            stream_tool_call: 流式工具调用回调
            stream_tool_reasoning: 流式推理输出回调
            stream_usage_info: 流式使用信息回调
        """
        self.tool_manager = tool_manager
        self.output_formatter = output_formatter
        self.tool_result_formatter = tool_result_formatter
        self.context = context or {}
        self.stream_tool_call = stream_tool_call
        self.stream_tool_reasoning = stream_tool_reasoning
        self.stream_usage_info = stream_usage_info

        # Scrape 结果最大长度
        self.scrape_max_length = 20000

    def set_scrape_max_length(self, length: int):
        """设置 scrape 结果最大长度"""
        self.scrape_max_length = length

    def _get_scrape_result(self, result: str) -> str:
        """处理 scrape 结果，限制长度"""
        import json

        try:
            scrape_result_dict = json.loads(result)
            text = scrape_result_dict.get("text")
            if text and len(text) > self.scrape_max_length:
                text = text[: self.scrape_max_length]
            return json.dumps({"text": text}, ensure_ascii=False)
        except json.JSONDecodeError:
            if isinstance(result, str) and len(result) > self.scrape_max_length:
                result = result[: self.scrape_max_length]
            return result

    def _post_process_tool_result(self, tool_name: str, tool_result: dict) -> dict:
        """后处理工具调用结果"""
        if "result" in tool_result and tool_name == "scrape":
            tool_result["result"] = self._get_scrape_result(tool_result["result"])
        return tool_result

    async def execute_single_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict,
        call_id: str,
        agent_name: str = "main",
    ) -> tuple[dict[str, Any], int]:
        """
        执行单个工具调用

        Args:
            server_name: 服务器名称
            tool_name: 工具名称
            arguments: 工具参数
            call_id: 调用 ID
            agent_name: Agent 名称（用于日志）

        Returns:
            Tuple[Dict, int]: (工具结果, 执行耗时毫秒)
        """
        call_start_time = time.time()

        try:
            # 工具调用开始时输出 REASONING
            if self.stream_tool_reasoning:
                try:
                    start_description = self.tool_result_formatter.get_tool_thinking_description(
                        tool_name, arguments
                    )
                    await self.stream_tool_reasoning(tool_name, "START", start_description)
                except Exception as e:
                    logger.debug(f"Failed to stream start reasoning: {e}")

            # 发送工具调用开始事件
            tool_call_id = None
            if self.stream_tool_call:
                tool_call_id = await self.stream_tool_call(tool_name, arguments)

            # Hook: on_tool_start — 可修改 arguments
            modified = hooks.call(
                "on_tool_start",
                HookContext(
                    hook_name="on_tool_start",
                    tool_name=tool_name,
                    server_name=server_name,
                    arguments=arguments,
                    context=self.context,
                ),
            )
            if modified is not None and isinstance(modified, dict):
                arguments = modified

            # SecureContext: 将 LLM 生成的 [SECURE:xxx] 占位符替换回真实值
            arguments = resolve_placeholders_in_args(arguments, self.context)

            # 执行工具调用
            tool_result = await self.tool_manager.execute_tool_call(
                server_name=server_name,
                tool_name=tool_name,
                arguments=arguments,
                context=self.context,
            )

            # 后处理结果
            tool_result = self._post_process_tool_result(tool_name, tool_result)

            # Hook: on_tool_end — 可修改 tool_result
            modified_result = hooks.call(
                "on_tool_end",
                HookContext(
                    hook_name="on_tool_end",
                    tool_name=tool_name,
                    server_name=server_name,
                    arguments=arguments,
                    tool_result=tool_result,
                    duration_ms=int((time.time() - call_start_time) * 1000),
                    context=self.context,
                ),
            )
            if modified_result is not None and isinstance(modified_result, dict):
                tool_result = modified_result

            # 发送工具调用完成事件
            result = (
                tool_result.get("result") if tool_result.get("result") else tool_result.get("error")
            )
            if self.stream_tool_call and tool_call_id:
                await self.stream_tool_call(
                    tool_name, {"result": result}, tool_call_id=tool_call_id
                )

            # 发送使用信息
            if self.stream_usage_info:
                await self.stream_usage_info(agent_name, {"tool_name": tool_name}, "tool_call")

            call_end_time = time.time()
            call_duration_ms = int((call_end_time - call_start_time) * 1000)

            # 工具调用后输出 REASONING - 显示结果摘要
            if self.stream_tool_reasoning:
                try:
                    result_summary = self.tool_result_formatter.summarize_tool_result(
                        tool_name, tool_result, call_duration_ms, arguments
                    )
                    if result_summary:
                        await self.stream_tool_reasoning(
                            tool_name, "RESULT_SUMMARY", result_summary
                        )
                except Exception as e:
                    logger.debug(f"Failed to stream result reasoning: {e}")

            return tool_result, call_duration_ms

        except Exception as e:
            call_end_time = time.time()
            call_duration_ms = int((call_end_time - call_start_time) * 1000)

            # 处理空错误消息
            error_msg = str(e) or (
                "[ERROR]: Tool execution timeout"
                if isinstance(e, TimeoutError)
                else f"Tool execution failed: {type(e).__name__}"
            )

            # 工具错误时输出 REASONING
            if self.stream_tool_reasoning:
                try:
                    await self.stream_tool_reasoning(
                        tool_name, "ERROR", f"{error_msg} (耗时: {call_duration_ms}ms)"
                    )
                except Exception as reason_err:
                    logger.debug(f"Failed to stream error reasoning: {reason_err}")

            tool_result = {
                "server_name": server_name,
                "tool_name": tool_name,
                "error": error_msg,
            }
            return tool_result, call_duration_ms

    async def execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        max_tool_calls: int,
        agent_name: str = "main",
    ) -> tuple[list[dict[str, Any]], list[tuple[str, Any]], bool]:
        """
        执行一组工具调用

        Args:
            tool_calls: 工具调用列表，每个元素包含 server_name, tool_name, arguments, id
            max_tool_calls: 单轮最大工具调用数
            agent_name: Agent 名称

        Returns:
            Tuple[List[Dict], List[Tuple[str, Any]], bool]:
                - tool_calls_data: 工具调用数据记录
                - tool_results_with_id: (call_id, formatted_result) 列表
                - exceeded: 是否超出最大调用数
        """
        tool_calls_data = []
        tool_results_with_id = []
        exceeded = len(tool_calls) > max_tool_calls

        if exceeded:
            logger.warning(
                f"[ERROR] Single turn tool call count too high ({len(tool_calls)} calls), "
                f"only processing first {max_tool_calls}"
            )

        for call in tool_calls[:max_tool_calls]:
            try:
                server_name = call["server_name"]
                tool_name = call["tool_name"]
                arguments = call["arguments"]
                call_id = call["id"]
            except (KeyError, TypeError) as e:
                logger.error(f"[ToolExecutor] Malformed tool call, missing key: {e}. Call: {call}")
                continue

            tool_result, call_duration_ms = await self.execute_single_tool(
                server_name=server_name,
                tool_name=tool_name,
                arguments=arguments,
                call_id=call_id,
                agent_name=agent_name,
            )

            # 记录工具调用数据
            if "error" in tool_result:
                tool_calls_data.append(
                    {
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "error": tool_result.get("error"),
                        "duration_ms": call_duration_ms,
                        "call_time": datetime.datetime.now(),
                    }
                )
            else:
                tool_calls_data.append(
                    {
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result": tool_result,
                        "duration_ms": call_duration_ms,
                        "call_time": datetime.datetime.now(),
                    }
                )

            # 格式化结果供 LLM 使用
            tool_result_for_llm = self.output_formatter.format_tool_result_for_user(tool_result)
            tool_results_with_id.append((call_id, tool_result_for_llm))

        return tool_calls_data, tool_results_with_id, exceeded

    def handle_failed_tool_calls(
        self,
        failed_calls: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[tuple[str, Any]]]:
        """
        处理失败的工具调用（格式错误等）

        Args:
            failed_calls: 失败的调用列表，每个元素包含 error 字段

        Returns:
            Tuple[List[Dict], List[Tuple[str, Any]]]:
                - tool_calls_data: 工具调用数据记录
                - tool_results_with_id: (call_id, formatted_result) 列表
        """
        tool_calls_data = []
        tool_results_with_id = []

        for failed_call in failed_calls:
            error_msg = failed_call.get("error", "Unknown error")
            tool_result = {
                "result": f"Your tool call format was incorrect, and the tool invocation failed, "
                f"error_message: {error_msg}; please review it carefully and try calling again.",
                "server_name": "re-think",
                "tool_name": "re-think",
            }

            tool_calls_data.append(
                {
                    "server_name": "",
                    "tool_name": "",
                    "arguments": "",
                    "result": tool_result,
                    "duration_ms": 0,
                    "call_time": datetime.datetime.now(),
                }
            )

            tool_result_for_llm = self.output_formatter.format_tool_result_for_user(tool_result)
            tool_results_with_id.append(("FAILED", tool_result_for_llm))

        return tool_calls_data, tool_results_with_id
