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

_temperature_override_var: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    '_temperature_override', default=None
)
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

from mem_deep_research_core.mem_deep_research_logging.logger import bootstrap_logger
from mem_deep_research_core.mem_deep_research_logging.task_tracer import TaskTracer

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

        self.client = self._create_client(self.cfg)

        logger.info(
            f"LLMClient (class={self.__class__.__name__},provider={self.provider_class},model_name={self.model_name},timeout={self.timeout}) initialized"
        )

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
        messages_copy = [m.copy() for m in messages]
        """Remove tool results from messages"""
        if keep_tool_result >= 0:
            # Find indices of all user messages
            user_indices = [
                i
                for i, msg in enumerate(messages_copy)
                if msg.get("role") == "user" or msg.get("role") == "tool"
            ]

            if len(user_indices) > 1:  # Only proceed if there are more than one user message
                first_user_idx = user_indices[0]  # Always keep the first user message

                # Calculate how many messages to keep from the end
                # If keep_tool_result is 0, we only keep the first message
                num_to_keep = (
                    0 if keep_tool_result == 0 else min(keep_tool_result, len(user_indices) - 1)
                )

                # Get indices of messages to keep from the end
                last_indices_to_keep = user_indices[-num_to_keep:] if num_to_keep > 0 else []

                # Combine first message and last k messages
                indices_to_keep = [first_user_idx] + last_indices_to_keep

                logger.debug("\n=======>>>>>> Message retention summary:")
                logger.debug(f"Total user messages: {len(user_indices)}")
                logger.debug(f"Keeping first message at index: {first_user_idx}")
                logger.debug(
                    f"Keeping last {num_to_keep} messages at indices: {last_indices_to_keep}"
                )
                logger.debug(f"Total messages to keep: {len(indices_to_keep)}")

                for i, msg in enumerate(messages_copy):
                    if (
                        msg.get("role") == "user" or msg.get("role") == "tool"
                    ) and i not in indices_to_keep:
                        logger.debug(f"Omitting content for user message at index {i}")
                        msg["content"] = "Tool result is omitted to save tokens."
            elif user_indices:  # This means only 1 user message exists
                logger.debug("\n=======>>>>>> Only 1 user message found. Keeping it as is.")
            else:  # No user messages at all
                logger.debug("\n=======>>>>>> No user messages found in the history.")

            logger.debug(
                f"\n\n=======>>>>>> Messages after potential content omission: {json.dumps(messages_copy, indent=4, ensure_ascii=False)}\n\n"
            )
        elif keep_tool_result == -1:
            # No processing
            pass

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
        # Filter message history
        filtered_history = self._filter_message_history(message_history, keep_tool_result)

        response = None

        # Unified LLM call handling
        response = await self._create_message(
            system_prompt,
            filtered_history,
            tool_definitions,
            keep_tool_result=keep_tool_result,
            stream_message_callback=stream_message_callback,
        )
        return response

    @staticmethod
    async def convert_tool_definition_to_tool_call(tools_definitions):
        tool_list = []
        for server in tools_definitions:
            if "tools" in server and len(server["tools"]) > 0:
                for tool in server["tools"]:
                    tool_def = {
                        "type": "function",
                        "function": {
                            "name": f"{server['name']}-{tool['name']}",
                            "description": tool["description"],
                            "parameters": tool["schema"],
                        },
                    }
                    tool_list.append(tool_def)
        return tool_list

    def close(self):
        """Close client connection (sync version)"""
        try:
            if hasattr(self.client, "close"):
                if asyncio.iscoroutinefunction(self.client.close):
                    # For async clients, try to run in event loop
                    try:
                        asyncio.get_running_loop()
                        # Schedule async close but don't wait
                        asyncio.create_task(self._async_close())
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

    def _filter_message_history(
        self, message_history: list[dict], keep_tool_result: int
    ) -> list[dict]:
        """Filter message history, keep specified number of tool results"""
        if keep_tool_result == -1:
            return message_history

        # Complex filtering logic can be implemented here
        # For now, simply return the last keep_tool_result messages
        if keep_tool_result > 0 and len(message_history) > keep_tool_result:
            return message_history[-keep_tool_result:]
        return message_history

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

    @abstractmethod
    def handle_max_turns_reached_summary_prompt(
        self, message_history: list[dict[str, Any]], summary_prompt: str
    ):
        raise NotImplementedError("must implement in subclass")

    def get_usage(self) -> dict[str, Any]:
        """
        Get usage statistics for this LLM client.

        Default implementation returns an empty result indicating usage tracking is not supported.
        Subclasses should override this method to provide actual usage statistics.

        :return: Dictionary containing usage statistics with keys:
            - records: List of all usage records (each with prompt_tokens, completion_tokens, total_tokens)
            - total_prompt_tokens: Sum of all prompt tokens
            - total_completion_tokens: Sum of all completion tokens
            - total_tokens: Sum of all total tokens
            - request_count: Number of API requests made
        """
        return {
            "records": [],
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "request_count": 0,
        }
