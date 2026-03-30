"""
LLM 调用处理模块

处理 LLM 调用、日志记录、错误处理和重试逻辑。
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from mem_deep_research_core.core.constants import (
    EMERGENCY_SUMMARY_MIN_CHARS,
    FALLBACK_EMERGENCY_SUMMARY,
    FALLBACK_SUMMARY_ERROR,
    MAX_SUMMARY_CONTEXT_RETRIES,
    SUMMARY_GENERATION_TIMEOUT,
    SUMMARY_NETWORK_MAX_RETRIES,
    TAG_CONTENT_REMOVED,
    TASK_PREVIEW_LENGTH,
    generate_message_id,
)
from mem_deep_research_core.core.hooks import HookContext, hooks
from mem_deep_research_core.exceptions import ContextLimitError, GuardrailError
from mem_deep_research_core.llm.provider_client_base import LLMProviderClientBase
from mem_deep_research_core.mem_deep_research_logging.task_tracer import TaskTracer
from mem_deep_research_core.prompts.template_loader import PromptTemplateLoader

logger = logging.getLogger("mem_deep_research")


class LLMCallHandler:
    """LLM 调用处理器"""

    def __init__(
        self,
        main_llm_client: LLMProviderClientBase,
        sub_agent_llm_client: LLMProviderClientBase | None = None,
        task_log: TaskTracer | None = None,
        add_message_id: bool = False,
        keep_tool_result: int = -1,
        stream_error_callback: Callable | None = None,
    ):
        """
        初始化 LLM 调用处理器

        Args:
            main_llm_client: 主 Agent 的 LLM 客户端
            sub_agent_llm_client: 子 Agent 的 LLM 客户端（可选）
            task_log: 任务日志记录器
            add_message_id: 是否为用户消息添加 ID
            keep_tool_result: 保留工具结果数量
            stream_error_callback: 流式错误回调函数
        """
        self.main_llm_client = main_llm_client
        self.sub_agent_llm_client = sub_agent_llm_client or main_llm_client
        self.task_log = task_log
        self.add_message_id = add_message_id
        self.keep_tool_result = keep_tool_result
        self.stream_error_callback = stream_error_callback

    def get_client(self, agent_type: str = "main") -> LLMProviderClientBase:
        """获取指定类型的 LLM 客户端"""
        return self.main_llm_client if agent_type == "main" else self.sub_agent_llm_client

    def _add_message_ids(self, message_history: list) -> None:
        """为用户消息添加 ID（就地修改）"""
        if not self.add_message_id:
            return

        for message in message_history:
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, list):
                    for content_item in content:
                        if content_item.get("type") == "text":
                            text = content_item["text"]
                            if not text.startswith("[msg_"):
                                message_id = generate_message_id()
                                content_item["text"] = f"[{message_id}] {text}"
                elif isinstance(content, str) and not content.startswith("[msg_"):
                    message_id = generate_message_id()
                    message["content"] = f"[{message_id}] {content}"

    def _save_message_history(
        self, system_prompt: str, message_history: list, agent_type: str
    ) -> None:
        """保存消息历史到任务日志"""
        if not self.task_log:
            return

        history_data = {
            "system_prompt": system_prompt,
            "message_history": message_history,
        }

        if agent_type == "main":
            self.task_log.main_agent_message_history = history_data
        elif self.task_log.current_sub_agent_session_id:
            self.task_log.sub_agent_message_history_sessions[
                self.task_log.current_sub_agent_session_id
            ] = history_data

        self.task_log.save()

    def _log_step(self, purpose: str, status: str, message: str) -> None:
        """记录步骤日志"""
        if self.task_log:
            step_name = f"{purpose.lower().replace(' ', '_')}_{status}"
            self.task_log.log_step(
                step_name, message, "failed" if status in ["failed", "error", "timeout"] else "info"
            )

    async def handle_llm_call(
        self,
        system_prompt: str,
        message_history: list,
        tool_definitions: list,
        step_id: int,
        purpose: str = "LLM call",
        agent_type: str = "main",
        stream_message_callback: Callable | None = None,
    ) -> tuple[str | None, bool, Any]:
        """
        统一的 LLM 调用处理

        Args:
            system_prompt: 系统提示词
            message_history: 消息历史
            tool_definitions: 工具定义列表
            step_id: 步骤 ID
            purpose: 调用目的描述
            agent_type: Agent 类型 ("main" 或子 agent 名称)
            stream_message_callback: 流式消息回调

        Returns:
            tuple: (response_text, should_break, tool_calls_info)
                - response_text: LLM 响应文本，失败时为 None
                - should_break: 是否应该终止循环
                - tool_calls_info: 工具调用信息，context_limit 时为 "context_limit"
        """
        current_llm_client = self.get_client(agent_type)

        # 添加消息 ID
        self._add_message_ids(message_history)

        # 保存调用前的消息历史
        self._save_message_history(system_prompt, message_history, agent_type)

        try:
            # Guardrail: pre-LLM validation
            if hooks.has_hooks("on_before_llm_call"):
                try:
                    hooks.call(
                        "on_before_llm_call",
                        HookContext(
                            hook_name="on_before_llm_call",
                            query=system_prompt[:200],
                            context={"message_count": len(message_history), "turn": step_id},
                            extra={"agent_type": agent_type},
                        ),
                    )
                except GuardrailError as e:
                    logger.warning(f"[Guardrail] Pre-LLM blocked: {e}")
                    self._log_step(purpose, "guardrail_blocked", str(e))
                    return None, True, None

            response = await current_llm_client.create_message(
                system_prompt=system_prompt,
                message_history=message_history,
                tool_definitions=tool_definitions,
                keep_tool_result=self.keep_tool_result,
                step_id=step_id,
                task_log=self.task_log,
                agent_type=agent_type,
                stream_message_callback=stream_message_callback,
            )

            if response:
                # 处理 LLM 响应
                assistant_response_text, should_break = current_llm_client.process_llm_response(
                    response, message_history, agent_type
                )

                # 保存响应后的消息历史
                self._save_message_history(system_prompt, message_history, agent_type)

                # 提取工具调用信息
                tool_calls_info = current_llm_client.extract_tool_calls_info(
                    response, assistant_response_text
                )

                if assistant_response_text:
                    # Guardrail: post-LLM validation
                    if hooks.has_hooks("on_after_llm_call"):
                        try:
                            hooks.call(
                                "on_after_llm_call",
                                HookContext(
                                    hook_name="on_after_llm_call",
                                    result=assistant_response_text,
                                    context={"turn": step_id},
                                    extra={"agent_type": agent_type},
                                ),
                            )
                        except GuardrailError as e:
                            logger.warning(f"[Guardrail] Post-LLM rejected: {e}")
                            self._log_step(purpose, "guardrail_rejected", str(e))
                            return None, True, None

                    self._log_step(purpose, "success", f"{purpose} completed successfully")
                    return assistant_response_text, should_break, tool_calls_info
                else:
                    self._log_step(purpose, "failed", f"{purpose} returned no valid response")
                    return None, True, None
            else:
                self._log_step(purpose, "failed", f"{purpose} returned no valid response")
                return None, True, None

        except TimeoutError:
            logger.debug(f"⚠️ {purpose} timed out")
            if self.stream_error_callback:
                await self.stream_error_callback(
                    "show_error", {"error": f"LLM Response Error: {purpose} timed out"}, True
                )
            self._log_step(purpose, "timeout", f"{purpose} timed out")
            return None, True, None

        except ContextLimitError as e:
            logger.debug(f"⚠️ {purpose} context limit exceeded: {e}")
            self._log_step(purpose, "context_limit", f"{purpose} context limit exceeded: {str(e)}")
            return None, True, "context_limit"

        except Exception as e:
            # Check if this is a RetryError wrapping a ContextLimitError
            # tenacity.RetryError wraps the last exception in its chain
            if hasattr(e, "last_attempt"):
                try:
                    inner = e.last_attempt.exception()
                    if isinstance(inner, ContextLimitError):
                        logger.debug(f"⚠️ {purpose} context limit exceeded (from RetryError): {e}")
                        self._log_step(
                            purpose, "context_limit",
                            f"{purpose} context limit exceeded (RetryError): {inner}",
                        )
                        return None, True, "context_limit"
                except Exception:
                    pass

            logger.debug(f"⚠️ {purpose} call failed: {e}")
            if self.stream_error_callback:
                await self.stream_error_callback(
                    "show_error", {"error": f"LLM Response Error: {purpose} {str(e)}"}, True
                )
            self._log_step(purpose, "error", f"{purpose} failed: {str(e)}")
            return None, True, None


class SummaryHandler:
    """摘要生成处理器，处理 context limit 重试逻辑"""

    def __init__(
        self,
        llm_call_handler: LLMCallHandler,
        chinese_context: bool = False,
        response_language: str = "English",
    ):
        """
        初始化摘要处理器

        Args:
            llm_call_handler: LLM 调用处理器
            chinese_context: 是否使用中文上下文
            response_language: 响应语言
        """
        self.llm_call_handler = llm_call_handler
        self.chinese_context = chinese_context
        self.response_language = response_language
        self.context: dict[str, Any] | None = None

    async def handle_summary_with_retry(
        self,
        system_prompt: str,
        agent_prompt_instance: Any,
        message_history: list,
        tool_definitions: list,
        purpose: str,
        task_description: str,
        task_failed: bool,
        agent_type: str = "main",
        task_guidance: str = "",
        stream_message_callback: Callable | None = None,
    ) -> str:
        """
        处理摘要生成，包含 context limit 重试逻辑

        Args:
            system_prompt: 系统提示词
            agent_prompt_instance: Agent 提示词实例
            message_history: 消息历史
            tool_definitions: 工具定义
            purpose: 调用目的
            task_description: 任务描述
            task_failed: 任务是否失败
            agent_type: Agent 类型
            task_guidance: 任务指导
            stream_message_callback: 流式消息回调

        Returns:
            str: 摘要文本，失败时返回错误消息
        """
        import time as _time
        _summary_deadline = _time.perf_counter() + SUMMARY_GENERATION_TIMEOUT

        retry_count = 0
        max_context_retries = MAX_SUMMARY_CONTEXT_RETRIES

        while retry_count < max_context_retries:
            if _time.perf_counter() > _summary_deadline:
                logger.warning(f"[SummaryHandler] Summary generation deadline exceeded ({SUMMARY_GENERATION_TIMEOUT}s)")
                break
            target_language = self.response_language

            # 生成摘要提示词
            summary_prompt = agent_prompt_instance.generate_summarize_prompt(
                task_description + task_guidance,
                task_failed=task_failed,
                chinese_context=self.chinese_context,
                target_language=target_language,
            )

            # Hook: on_summarize_prompt_build
            hook_result = hooks.call("on_summarize_prompt_build", HookContext(
                hook_name="on_summarize_prompt_build",
                result=summary_prompt,
                context=self.context,
            ))
            if isinstance(hook_result, str):
                summary_prompt = hook_result

            # 处理消息历史合并
            current_llm_client = self.llm_call_handler.get_client(agent_type)
            summary_prompt = current_llm_client.handle_max_turns_reached_summary_prompt(
                message_history, summary_prompt
            )

            # 添加摘要提示到消息历史
            message_history.append(
                {"role": "user", "content": [{"type": "text", "text": summary_prompt}]}
            )

            # 网络重试循环
            network_max_retries = SUMMARY_NETWORK_MAX_RETRIES
            for network_retry_count in range(network_max_retries):
                history_len_before = len(message_history)

                response_text, _, tool_calls_info = await self.llm_call_handler.handle_llm_call(
                    system_prompt,
                    message_history,
                    tool_definitions,
                    999,
                    purpose,
                    agent_type=agent_type,
                    stream_message_callback=stream_message_callback,
                )

                # Context limit: break immediately and go to history reduction
                if tool_calls_info == "context_limit":
                    response_text = None
                    break

                # 检查有效响应
                if response_text is not None and response_text.strip():
                    break
                else:
                    # 检查消息历史中是否有有效的 assistant 响应
                    if len(message_history) > history_len_before:
                        last_msg = message_history[-1]
                        if last_msg.get("role") == "assistant":
                            content = last_msg.get("content", "")
                            if content and (isinstance(content, str) and content.strip()):
                                logger.info(
                                    f"Found valid assistant response in message_history (len={len(content)})"
                                )
                                response_text = content
                                break
                            else:
                                message_history.pop()
                                logger.debug("Removed empty assistant response before retry")

                    # 递减重试等待
                    retry_wait = max(1, 5 - network_retry_count)
                    logger.warning(
                        f"LLM summary returned empty response, attempt {network_retry_count + 1}/{network_max_retries}, retrying after {retry_wait}s..."
                    )
                    await asyncio.sleep(retry_wait)

            if response_text:
                return response_text

            # All retries failed (context_limit or persistent errors): reduce history and retry
            retry_count += 1
            fail_reason = (
                "context_limit" if tool_calls_info == "context_limit" else "persistent_error"
            )
            logger.warning(
                f"Summary failed ({fail_reason}), attempt {retry_count}, "
                f"removing recent dialogue (history_len={len(message_history)})"
            )

            if self.llm_call_handler.task_log:
                self.llm_call_handler.task_log.log_step(
                    "final_summary_generation_error",
                    f"Summary retry {retry_count} ({fail_reason}), reducing history from {len(message_history)} messages",
                    "warning",
                )

            # 移除刚添加的摘要提示
            if message_history and message_history[-1]["role"] == "user":
                message_history.pop()
            # 移除最近的 assistant 消息（可能是空响应）
            if message_history and message_history[-1]["role"] == "assistant":
                message_history.pop()

            task_failed = True

            # 检查是否只剩初始消息
            if len(message_history) <= 2:
                logger.warning(
                    "Removed all removable dialogues, but still unable to generate summary"
                )
                break

            # --- Binary reduction: remove half of middle messages ---
            # Keep: first message (index 0) + last 2 messages
            keep_tail = 2
            if len(message_history) <= keep_tail + 1:
                # Too few messages, replace content of middle messages
                for i in range(1, len(message_history)):
                    message_history[i]["content"] = TAG_CONTENT_REMOVED
            else:
                # Token-aware reduction: if client supports it, calculate target
                target_count = None
                current_llm_client = self.llm_call_handler.get_client(agent_type)
                if (
                    hasattr(current_llm_client, "_estimate_tokens")
                    and hasattr(current_llm_client, "max_context_length")
                    and current_llm_client.max_context_length > 0
                ):
                    # Estimate current tokens
                    total_text = system_prompt + " ".join(
                        str(m.get("content", "")) for m in message_history
                    )
                    current_tokens = current_llm_client._estimate_tokens(total_text)
                    target_tokens = int(current_llm_client.max_context_length * 0.6)
                    if current_tokens > target_tokens:
                        # Calculate how many middle messages to remove to hit target
                        middle_msgs = message_history[1:-keep_tail]
                        middle_text = " ".join(str(m.get("content", "")) for m in middle_msgs)
                        middle_tokens = current_llm_client._estimate_tokens(middle_text)
                        tokens_to_remove = current_tokens - target_tokens
                        if middle_tokens > 0:
                            remove_ratio = min(tokens_to_remove / middle_tokens, 0.9)
                            target_count = max(1, int(len(middle_msgs) * (1 - remove_ratio)))
                        logger.info(
                            f"[CONTEXT] Token-aware reduction: {current_tokens} tokens -> target {target_tokens}, "
                            f"removing {len(middle_msgs) - (target_count or len(middle_msgs) // 2)} middle messages"
                        )

                # Binary reduction: keep half of middle messages (or token-calculated count)
                middle_msgs = message_history[1:-keep_tail]
                if target_count is not None:
                    keep_count = target_count
                else:
                    keep_count = max(1, len(middle_msgs) // 2)

                # Keep the most recent middle messages (they have more relevant context)
                kept_middle = (
                    middle_msgs[-keep_count:] if keep_count < len(middle_msgs) else middle_msgs
                )
                removed_count = len(middle_msgs) - len(kept_middle)
                message_history[1:-keep_tail] = kept_middle

                logger.info(
                    f"[CONTEXT] Binary reduction: removed {removed_count} messages, "
                    f"history now {len(message_history)} messages"
                )

        # --- Emergency fallback: extract last assistant response ---
        emergency_summary = self._extract_emergency_summary(message_history)
        if emergency_summary:
            fallback_text = f"{FALLBACK_EMERGENCY_SUMMARY}\n\n{emergency_summary}"
            logger.warning(
                f"[CONTEXT] Using emergency summary from last assistant response (len={len(emergency_summary)})"
            )
        else:
            fallback_text = FALLBACK_SUMMARY_ERROR
            logger.error(f"Summary failed after {retry_count} context reduction attempts")

        # 将 fallback 文本通过流式回调发送给客户端，防止客户端收到 content_length=0
        if stream_message_callback and fallback_text:
            try:
                fallback_msg_id = f"fallback_{id(fallback_text)}"
                await stream_message_callback(fallback_msg_id, fallback_text, True)
            except Exception as e:
                logger.debug(f"Failed to stream fallback text: {e}")

        return fallback_text

    @staticmethod
    def _extract_emergency_summary(message_history: list) -> str | None:
        """Scan message history for the last substantial assistant response as emergency fallback."""
        for msg in reversed(message_history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content.strip()) > EMERGENCY_SUMMARY_MIN_CHARS:
                    return content.strip()
                elif isinstance(content, list):
                    texts = [
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    combined = "\n".join(t for t in texts if t.strip())
                    if len(combined) > EMERGENCY_SUMMARY_MIN_CHARS:
                        return combined
        return None


_reflection_loader = PromptTemplateLoader()


def generate_reflection_prompt(
    turn_count: int, task_description: str, chinese_context: bool = False
) -> str:
    """
    生成深度研究的反思检查点提示词

    Args:
        turn_count: 当前轮次
        task_description: 任务描述
        chinese_context: 是否使用中文

    Returns:
        反思提示词
    """
    task_preview = (
        task_description[:TASK_PREVIEW_LENGTH] + "..." if len(task_description) > TASK_PREVIEW_LENGTH else task_description
    )

    template_name = "reflection/reflection_chinese" if chinese_context else "reflection/reflection"
    return _reflection_loader.load_and_render(
        template_name,
        turn_count=turn_count,
        task_preview=task_preview,
    )
