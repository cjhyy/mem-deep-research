import asyncio
import contextvars
import dataclasses
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import (
    Any,
    Optional,
)

from omegaconf import DictConfig
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

from mem_deep_research_core.mem_deep_research_logging.logger import bootstrap_logger
from mem_deep_research_core.mem_deep_research_logging.task_tracer import TaskTracer

_temperature_override_var: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "_temperature_override", default=None
)

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)


@dataclasses.dataclass
class LLMProviderClientBase(ABC):
    # Required arguments (no default value)
    task_id: str
    cfg: DictConfig

    # Optional arguments (with default value)
    task_log: Optional["TaskTracer"] = None

    # post_init
    client: Any = dataclasses.field(init=False)

    def __post_init__(self):
        # Explicitly assign from cfg object
        self.provider_class: str = self.cfg.llm.provider_class
        self.model_name: str = self.cfg.llm.model_name
        self.temperature: float = self.cfg.llm.temperature
        self.top_p: float = self.cfg.llm.top_p
        self.min_p: float = self.cfg.llm.min_p
        self.top_k: int = self.cfg.llm.top_k
        self.reasoning_effort: str = self.cfg.llm.get("reasoning_effort", "medium")
        self.thinking_mode: str = self.cfg.llm.get("thinking_mode", "auto")
        self.repetition_penalty: float = self.cfg.llm.get("repetition_penalty", 1.0)
        self.max_tokens: int = self.cfg.llm.max_tokens
        self.max_context_length: int = self.cfg.llm.get("max_context_length", -1)
        self.oai_tool_thinking: bool = self.cfg.llm.oai_tool_thinking
        self.async_client: bool = self.cfg.llm.async_client
        self.enable_streaming: bool = self.cfg.llm.get("enable_streaming", True)

        self.use_tool_calls: bool | None = self.cfg.llm.get("use_tool_calls")
        self.openrouter_provider: str | None = self.cfg.llm.get("openrouter_provider")

        # Timeout and retry configuration (with defaults)
        self.timeout: int = self.cfg.llm.get("timeout", 1800)  # default 30 min
        self.retry_max_attempts: int = self.cfg.llm.get("retry_max_attempts", 5)
        self.retry_wait_seconds: int = self.cfg.llm.get("retry_wait_seconds", 10)
        self.retry_multiplier: int = self.cfg.llm.get("retry_multiplier", 5)
        self.retry_strategy: str = self.cfg.llm.get(
            "retry_strategy", "exponential"
        )  # "exponential" or "fixed"
        # Safely handle string to bool conversion
        disable_cache_control_val = self.cfg.llm.get("disable_cache_control", False)
        if isinstance(disable_cache_control_val, str):
            self.disable_cache_control: bool = disable_cache_control_val.lower().strip() == "true"
        else:
            self.disable_cache_control: bool = bool(disable_cache_control_val)

        logger.info(
            f"openrouter_provider config value: {self.openrouter_provider} (type: {type(self.openrouter_provider)})"
        )

        logger.info(
            f"disable_cache_control config value: {disable_cache_control_val} (type: {type(disable_cache_control_val)}) -> parsed as: {self.disable_cache_control}"
        )

        # Usage tracking: accumulated across all API calls
        self._usage_records: list[dict] = []

        # ObserverRegistry（可选，由 Orchestrator 注入）；None 时默认空 registry no-op
        self._observers: Any = None

        self.client = self._create_client(self.cfg)

        logger.info(
            f"LLMClient (class={self.__class__.__name__},provider={self.provider_class},model_name={self.model_name},timeout={self.timeout}) initialized"
        )

    def supports_adaptive_thinking(self) -> bool:
        """是否支持 adaptive thinking（模型自决定推理深度）。

        子类覆盖：Claude → True，其他 → False。
        """
        return False

    def get_thinking_params(self) -> dict[str, Any]:
        """获取当前 thinking 配置参数，用于注入 API 调用。

        子类覆盖以返回 provider 特定的参数格式。
        默认返回空 dict（不注入 thinking 参数）。

        Returns:
            provider 特定的 thinking 参数 dict，如：
            - Claude: {"thinking": {"type": "adaptive"}}
            - GPT-5: {"reasoning_effort": "medium"}
        """
        mode = self.thinking_mode
        if mode == "none":
            return {}
        if mode == "auto":
            # auto 模式下由子类决定最优策略
            return self._auto_thinking_params()
        if mode == "adaptive":
            if self.supports_adaptive_thinking():
                return self._adaptive_thinking_params()
            # 不支持 adaptive，fallback 到 fixed
            return self._fixed_thinking_params()
        if mode == "fixed":
            return self._fixed_thinking_params()
        return {}

    def _auto_thinking_params(self) -> dict[str, Any]:
        """auto 模式的默认实现：不注入参数。子类覆盖。"""
        return {}

    def _adaptive_thinking_params(self) -> dict[str, Any]:
        """adaptive thinking 参数。子类覆盖。"""
        return {}

    def _fixed_thinking_params(self) -> dict[str, Any]:
        """fixed thinking 参数。子类覆盖。"""
        return {}

    def get_retry_decorator(self, exception_to_skip=None):
        """
        Create a retry decorator based on configuration.

        Args:
            exception_to_skip: Exception type that should NOT be retried (e.g., ContextLimitError)

        Returns:
            A tenacity retry decorator configured with instance settings
        """
        if self.retry_strategy == "fixed":
            wait_strategy = wait_fixed(self.retry_wait_seconds)
        else:  # exponential
            wait_strategy = wait_exponential(multiplier=self.retry_multiplier)

        retry_kwargs = {
            "wait": wait_strategy,
            "stop": stop_after_attempt(self.retry_max_attempts),
        }

        if exception_to_skip:
            retry_kwargs["retry"] = retry_if_not_exception_type(exception_to_skip)

        return retry(**retry_kwargs)

    def get_effective_temperature(self) -> float:
        """Return temperature_override if set, otherwise the configured temperature."""
        override = _temperature_override_var.get(None)
        if override is not None:
            return override
        return self.temperature

    def set_temperature_boost(self, boost: float = 0.3, cap: float = 1.0):
        """Temporarily boost temperature by `boost`, capped at `cap`."""
        _temperature_override_var.set(min(self.temperature + boost, cap))

    def clear_temperature_override(self):
        """Remove the temporary temperature override."""
        _temperature_override_var.set(None)

    def reset_for_new_task(self, task_id: str) -> None:
        """Reset mutable state for reuse across tasks.

        Preserves the HTTP client/connection pool while clearing
        task-specific state like temperature overrides.
        """
        self.task_id = task_id
        _temperature_override_var.set(None)

    # ------------------------------------------------------------------
    # ContextVar snapshot contract (HITL / durable execution)
    # ------------------------------------------------------------------

    def save_contextvar_state(self) -> dict:
        """Capture ContextVar state owned by this provider for snapshot.

        Subclasses that introduce additional ContextVars must override,
        extending the dict returned by super().save_contextvar_state().
        """
        return {"temperature_override": _temperature_override_var.get(None)}

    def restore_contextvar_state(self, state: dict) -> None:
        """Restore ContextVar state captured by save_contextvar_state().

        Unknown keys are ignored to keep forward-compat with future provider
        subclasses that snapshot additional vars.
        """
        if "temperature_override" in state:
            _temperature_override_var.set(state["temperature_override"])

    @abstractmethod
    def _create_client(self, config: DictConfig) -> Any:
        """Create specific LLM client"""
        raise NotImplementedError("must override in subclass")

    @abstractmethod
    async def _create_message(
        self,
        system_prompt: str,
        messages: list[dict],
        tools_definitions: list[dict],
        keep_tool_result: int = -1,
        stream_message_callback: Callable | None = None,
    ) -> Any:
        """Create provider-specific message - implemented by subclass"""
        raise NotImplementedError("subclass must implement this")

    @abstractmethod
    def process_llm_response(
        self, llm_response, message_history, agent_type="main"
    ) -> tuple[str, bool]:
        """Process LLM response - implemented by subclass"""
        pass

    @abstractmethod
    def extract_tool_calls_info(self, llm_response, assistant_response_text) -> tuple[list, list]:
        """Extract tool call information - implemented by subclass"""
        pass

    def _remove_tool_result_from_messages(self, messages, keep_tool_result):
        """Remove tool results from messages.

        Only touches messages with _type=MT.TOOL_RESULT.
        Messages without _type (legacy) or with other types are never modified.
        """
        from mem_deep_research_core.core.constants import MT

        messages_copy = [m.copy() for m in messages]
        if keep_tool_result >= 0:
            tool_result_indices = [
                i for i, msg in enumerate(messages_copy)
                if msg.get("_type") == MT.TOOL_RESULT
            ]

            if tool_result_indices:
                num_to_keep = (
                    0 if keep_tool_result == 0 else min(keep_tool_result, len(tool_result_indices))
                )
                indices_to_keep = set(tool_result_indices[-num_to_keep:]) if num_to_keep > 0 else set()

                logger.debug(
                    f"[keep_tool_result] {len(tool_result_indices)} tool results found, "
                    f"keeping {num_to_keep}, omitting {len(tool_result_indices) - num_to_keep}"
                )

                for i in tool_result_indices:
                    if i not in indices_to_keep:
                        messages_copy[i]["content"] = "Tool result is omitted to save tokens."

        return messages_copy

    async def create_message(
        self,
        system_prompt: str,
        message_history: list[dict],
        tool_definitions: list[dict],
        keep_tool_result: int = -1,
        step_id: int = 1,
        task_log: Optional["TaskTracer"] = None,
        agent_type: str = "main",
        stream_message_callback: Callable | None = None,
    ):
        """
        Call LLM to generate response, supports tool calls - unified implementation
        """
        # Observer context：包裹整个 LLM 调用
        import time as _time
        from mem_deep_research_core.observability import LLMCallContext, ObserverRegistry

        _observers = self._observers or ObserverRegistry()
        _obs_ctx = LLMCallContext(
            agent_name=agent_type,
            turn_number=step_id,
            provider=self.provider_class,
            model=self.model_name,
            messages_count=len(message_history),
        )
        _start = _time.time()

        async with _observers.around_llm_call(_obs_ctx):
            try:
                # Unified LLM call handling
                # Note: _remove_tool_result_from_messages() is called inside each provider's _create_message()
                response = await self._create_message(
                    system_prompt,
                    message_history,
                    tool_definitions,
                    keep_tool_result=keep_tool_result,
                    stream_message_callback=stream_message_callback,
                )
            except Exception as e:
                _obs_ctx.error = str(e)
                _obs_ctx.duration_ms = int((_time.time() - _start) * 1000)
                raise

            _obs_ctx.duration_ms = int((_time.time() - _start) * 1000)
            # 尝试填充 observer 可用的响应元数据（best-effort，不抛错）
            try:
                _obs_ctx.stop_reason = getattr(response, "stop_reason", None) or (
                    response.choices[0].finish_reason
                    if getattr(response, "choices", None)
                    else None
                )
            except Exception:
                pass
            try:
                usage = self.get_usage() if hasattr(self, "get_usage") else {}
                if usage:
                    _obs_ctx.token_usage = {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }
            except Exception:
                pass
            return response

    def set_observer_registry(self, observers: Any) -> None:
        """注入 ObserverRegistry（由 Orchestrator 在初始化时调用）。"""
        self._observers = observers

    @staticmethod
    async def convert_tool_definition_to_tool_call(tools_definitions):
        tool_list = []
        name_map = {}  # flat_name -> (server_name, tool_name)
        for server in tools_definitions:
            if "tools" in server and len(server["tools"]) > 0:
                for tool in server["tools"]:
                    flat_name = f"{server['name']}--{tool['name']}"
                    name_map[flat_name] = (server["name"], tool["name"])
                    tool_def = {
                        "type": "function",
                        "function": {
                            "name": flat_name,
                            "description": tool["description"],
                            "parameters": tool["schema"],
                        },
                    }
                    tool_list.append(tool_def)
        return tool_list, name_map

    def close(self):
        """Close client connection (sync version)"""
        try:
            if hasattr(self.client, "close"):
                if asyncio.iscoroutinefunction(self.client.close):
                    # For async clients, try to run in event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # Schedule async close and ensure it completes
                        loop.create_task(self._async_close())
                    except RuntimeError:
                        # No running event loop, create one temporarily
                        asyncio.run(self._async_close())
                else:
                    self.client.close()
            elif hasattr(self.client, "_client") and hasattr(self.client._client, "close"):
                # Some clients may have an internal _client attribute
                self.client._client.close()
            logger.debug(f"LLMClient closed (sync) for task_id={self.task_id}")
        except Exception as e:
            logger.warning(f"Error closing LLM client (sync): {e}")

    async def close_async(self):
        """Close client connection (async version) - preferred for async contexts"""
        await self._async_close()

    async def _async_close(self):
        """Internal async close implementation"""
        try:
            if hasattr(self.client, "close"):
                if asyncio.iscoroutinefunction(self.client.close):
                    await self.client.close()
                else:
                    self.client.close()
            elif hasattr(self.client, "_client") and hasattr(self.client._client, "close"):
                if asyncio.iscoroutinefunction(self.client._client.close):
                    await self.client._client.close()
                else:
                    self.client._client.close()
            logger.debug(f"LLMClient closed (async) for task_id={self.task_id}")
        except Exception as e:
            logger.warning(f"Error closing LLM client (async): {e}")

    def _format_response_for_log(self, response) -> dict:
        """Format response for logging"""
        if not response:
            return {}

        # Basic response information
        formatted: dict[str, Any] = {
            "response_type": type(response).__name__,
        }

        # Anthropic response
        if hasattr(response, "content"):
            formatted["content"] = []
            for block in response.content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        formatted["content"].append(
                            {
                                "type": "text",
                                "text": block.text[:500] + "..."
                                if len(block.text) > 500
                                else block.text,
                            }
                        )
                    elif block.type == "tool_use":
                        formatted["content"].append(
                            {
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": str(block.input)[:200] + "..."
                                if len(str(block.input)) > 200
                                else str(block.input),
                            }
                        )

        # OpenAI response
        if hasattr(response, "choices"):
            formatted["choices"] = []
            for choice in response.choices:
                choice_data = {"finish_reason": choice.finish_reason}
                if hasattr(choice, "message"):
                    message = choice.message
                    choice_data["message"] = {
                        "role": message.role,
                        "content": message.content[:500] + "..."
                        if message.content and len(message.content) > 500
                        else message.content,
                    }
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        choice_data["message"]["tool_calls_count"] = len(message.tool_calls)
                formatted["choices"].append(choice_data)

        return formatted

    @abstractmethod
    def update_message_history(
        self,
        message_history: list[dict[str, Any]],
        tool_call_info: list[Any],
        tool_calls_exceeded: bool = False,
    ):
        raise NotImplementedError("must implement in subclass")

    def handle_max_turns_reached_summary_prompt(
        self, message_history: list[dict[str, Any]], summary_prompt: str
    ):
        """Default: merge summary with last user message if present.

        Subclasses can override for provider-specific behavior.
        """
        if message_history and message_history[-1].get("role") == "user":
            last_user_message = message_history[-1]  # 只读取不 pop，pop 由调用方负责
            content = last_user_message.get("content", "")
            if isinstance(content, list) and content:
                text = (
                    content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
                )
            else:
                text = str(content)
            return text + "\n\n-----------------\n\n" + summary_prompt
        return summary_prompt

    def _apply_cache_control(self, messages, include_system: bool = False):
        """Apply ephemeral cache control to the last user message.

        Args:
            messages: Message history list.
            include_system: If True, also apply cache control to system messages.
        """
        cached_messages = []
        user_turns_processed = 0
        for turn in reversed(messages):
            should_process = (turn["role"] == "user" and user_turns_processed < 1) or (
                include_system and turn["role"] == "system"
            )
            if should_process:
                new_content = []
                processed_text = False
                if isinstance(turn.get("content"), list):
                    for item in turn["content"]:
                        if (
                            item.get("type") == "text"
                            and len(item.get("text", "")) > 0
                            and not processed_text
                        ):
                            text_item = item.copy()
                            text_item["cache_control"] = {"type": "ephemeral"}
                            new_content.append(text_item)
                            processed_text = True
                        else:
                            new_content.append(item.copy())
                    cached_messages.append({"role": turn["role"], "content": new_content})
                else:
                    logger.debug(
                        "Warning: Message content is not in expected list format, cache control not applied."
                    )
                    cached_messages.append(turn)
                if turn["role"] == "user":
                    user_turns_processed += 1
            else:
                cached_messages.append(turn)
        return list(reversed(cached_messages))

    def _record_usage(self, response) -> None:
        """Extract and record usage from an API response.

        Works with both OpenAI and Anthropic response formats.
        Called automatically from _post_response_hook in subclasses.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        record = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
        # Anthropic uses input_tokens / output_tokens
        if record["prompt_tokens"] == 0:
            record["prompt_tokens"] = getattr(usage, "input_tokens", 0) or 0
        if record["completion_tokens"] == 0:
            record["completion_tokens"] = getattr(usage, "output_tokens", 0) or 0
        if record["total_tokens"] == 0:
            record["total_tokens"] = record["prompt_tokens"] + record["completion_tokens"]

        self._usage_records.append(record)

    def get_usage(self) -> dict[str, Any]:
        """
        Get accumulated usage statistics for this LLM client.

        :return: Dictionary containing usage statistics with keys:
            - records: List of all usage records (each with prompt_tokens, completion_tokens, total_tokens)
            - total_prompt_tokens: Sum of all prompt tokens
            - total_completion_tokens: Sum of all completion tokens
            - total_tokens: Sum of all total tokens
            - request_count: Number of API requests made
        """
        total_prompt = sum(r.get("prompt_tokens", 0) for r in self._usage_records)
        total_completion = sum(r.get("completion_tokens", 0) for r in self._usage_records)
        return {
            "records": list(self._usage_records),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "request_count": len(self._usage_records),
        }

    def get_output_truncated(self) -> bool:
        """Check if the last LLM response was truncated due to output length limits.

        Providers that detect truncation should override this method.
        Default returns False (no truncation detected).
        """
        return getattr(self, "_output_truncated_flag", False) or False

    def clear_output_truncated(self) -> None:
        """Reset the output truncated flag after handling recovery."""
        self._output_truncated_flag = False
