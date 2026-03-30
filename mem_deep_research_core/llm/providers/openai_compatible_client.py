"""
OpenAI-compatible LLM provider base class.

Extracts shared logic from ClaudeOpenRouter, GPT5OpenRouter, GPT5OpenAI,
and DeepSeekOpenRouter providers.
"""

import asyncio
import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Callable
from typing import Any

import tiktoken
from omegaconf import DictConfig
from openai import AsyncOpenAI, OpenAI

from mem_deep_research_core.exceptions import ContextLimitError
from mem_deep_research_core.llm.provider_client_base import LLMProviderClientBase
from mem_deep_research_core.llm.util import collect_openai_stream
from mem_deep_research_core.mem_deep_research_logging.logger import (
    bootstrap_logger,
    truncate_for_log,
)

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)


@dataclasses.dataclass
class OpenAICompatibleClient(LLMProviderClientBase):
    """Base class for all providers that use the OpenAI SDK."""

    # ========== Context limit error patterns ==========

    _CONTEXT_LIMIT_PATTERNS: list[str] = dataclasses.field(
        default_factory=lambda: [
            "Input is too long for requested model",
            "input length and `max_tokens` exceed context limit",
            "maximum context length",
            "prompt is too long",
            "exceeds the maximum length",
            "exceeds the maximum allowed length",
            "Input tokens exceed the configured limit",
            "Requested token count exceeds the model's maximum context length",
            "Stream was empty",  # Some providers return empty stream when context is too long
        ]
    )

    # ========== Hook methods for subclass customization ==========

    def _get_api_credentials(self) -> tuple[str, str]:
        """Return (api_key, base_url). Must be overridden by subclasses."""
        raise NotImplementedError("Subclass must implement _get_api_credentials")

    def _validate_model(self) -> None:
        """Validate model name. Override if model validation is needed."""
        pass

    def _build_extra_body(self) -> dict:
        """Build extra_body for the API request (OpenRouter provider routing)."""
        provider_config = (self.openrouter_provider or "").strip().lower()
        logger.info(f"provider_config: {provider_config}")
        if provider_config == "google":
            extra_body = {
                "provider": {
                    "only": [
                        "google-vertex/us",
                        "google-vertex/europe",
                        "google-vertex/global",
                    ]
                }
            }
        elif provider_config == "anthropic":
            extra_body = {"provider": {"only": ["anthropic"]}}
        elif provider_config == "amazon":
            extra_body = {"provider": {"only": ["amazon-bedrock"]}}
        elif provider_config != "":
            extra_body = {"provider": {"only": [provider_config]}}
        else:
            extra_body = {}

        # Add top_k and min_p through extra_body for OpenRouter
        if self.top_k != -1:
            extra_body["top_k"] = self.top_k
        if self.min_p != 0.0:
            extra_body["min_p"] = self.min_p
        if self.repetition_penalty != 1.0:
            extra_body["repetition_penalty"] = self.repetition_penalty

        return extra_body

    def _customize_params(self, params: dict) -> dict:
        """Customize API parameters before sending. Override to add extra params."""
        return params

    def _get_context_limit_patterns(self) -> list[str]:
        """Return context limit error matching patterns."""
        return self._CONTEXT_LIMIT_PATTERNS

    def _post_response_hook(self, response, messages_copy: list = None) -> None:
        """Post-response processing hook. Override for logging, usage tracking, etc."""
        pass

    def _use_cache_control(self) -> bool:
        """Whether to use cache control. Override to disable."""
        return True

    # ========== Core implementation ==========

    def _create_client(self, config: DictConfig):
        """Create configured OpenAI client using credentials from _get_api_credentials."""
        api_key, base_url = self._get_api_credentials()

        if self.async_client:
            return AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.timeout,
            )
        else:
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.timeout,
            )

    async def _create_message(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools_definitions,
        keep_tool_result: int = -1,
        stream_message_callback: Callable | None = None,
    ):
        """Send message to OpenAI-compatible API."""
        logger.debug(f" Calling LLM ({'async' if self.async_client else 'sync'})")

        # Inject system prompt
        if system_prompt:
            target_role = "system"
            if messages and messages[0]["role"] in ["system", "developer"]:
                messages[0] = {
                    "role": target_role,
                    "content": [{"type": "text", "text": system_prompt}],
                }
            else:
                messages.insert(
                    0,
                    {
                        "role": target_role,
                        "content": [{"type": "text", "text": system_prompt}],
                    },
                )

        messages_copy = self._remove_tool_result_from_messages(messages, keep_tool_result)

        # Pre-flight context check: trim if approaching context limit
        if self.max_context_length > 0:
            messages_copy = self._preflight_context_check(system_prompt, messages_copy)

        # Apply cache control
        if self._use_cache_control() and not self.disable_cache_control:
            processed_messages = self._apply_cache_control(messages_copy)
        else:
            processed_messages = messages_copy

        # Allow subclass model validation
        self._validate_model()

        params = None
        try:
            temperature = self.get_effective_temperature()
            extra_body = self._build_extra_body()

            params = {
                "model": self.model_name,
                "temperature": temperature,
                "max_tokens": self.max_tokens,
                "messages": processed_messages,
                "stream": self.enable_streaming,
            }

            # Only add extra_body if non-empty
            if extra_body:
                params["extra_body"] = extra_body

            # Native tool calling: convert tool definitions to OpenAI tools format
            # and pass via API `tools` parameter (required for models that don't
            # follow XML tool format instructions, e.g. Sonnet 4.6 via OpenRouter)
            if tools_definitions:
                tool_list = await self.convert_tool_definition_to_tool_call(tools_definitions)
                if tool_list:
                    params["tools"] = tool_list

            # Add optional parameters only if they have non-default values
            if self.top_p != 1.0:
                params["top_p"] = self.top_p

            # Allow subclass to customize params
            params = self._customize_params(params)

            # Apply configurable retry decorator to the API call
            retry_decorator = self.get_retry_decorator(exception_to_skip=ContextLimitError)
            create_completion_with_retry = retry_decorator(self._create_completion)
            response = await create_completion_with_retry(
                params, self.async_client, stream_message_callback
            )

            if response is None or response.choices is None or len(response.choices) == 0:
                logger.debug(f"LLM call failed: response = {response}")
                raise Exception(f"LLM call failed [rare case]: response = {response}")

            if response.choices and response.choices[0].finish_reason == "length":
                logger.debug("LLM finish_reason is 'length', triggering ContextLimitError")
                raise ContextLimitError(
                    "(finish_reason=length) Response truncated due to maximum context length"
                )

            # Check for empty content on 'stop'
            if response.choices and response.choices[0].finish_reason == "stop":
                content = response.choices[0].message.content
                if content is None or (isinstance(content, str) and content.strip() == ""):
                    logger.warning(
                        "LLM finish_reason is 'stop', but content is empty/None - API may be unstable"
                    )
                    raise Exception("LLM returned empty response (content is None or empty)")

            logger.debug(
                f"LLM call finish_reason: {getattr(response.choices[0], 'finish_reason', 'N/A')}"
            )

            # Post-response hook for subclass customization
            self._post_response_hook(response, messages_copy)

            return response
        except asyncio.CancelledError:
            logger.debug("[WARNING] LLM API call was cancelled during execution")
            raise Exception("LLM API call was cancelled during execution")
        except Exception as e:
            error_str = str(e)
            if self._check_context_limit_error(error_str):
                logger.debug(f"LLM Context limit exceeded: {error_str}")
                raise ContextLimitError(f"Context limit exceeded: {error_str}")

            # Redact message history from logged params to avoid leaking sensitive data
            safe_params = {k: v for k, v in params.items() if k != "messages"}
            safe_params["messages"] = f"[{len(params.get('messages', []))} messages redacted]"
            logger.error(
                f"LLM call failed: {str(e)}, params = {json.dumps(safe_params, default=str)}",
                exc_info=True,
            )
            raise e

    def _check_context_limit_error(self, error_str: str) -> bool:
        """Check if an error string matches context limit patterns."""
        patterns = self._get_context_limit_patterns()
        for pattern in patterns:
            if pattern in error_str:
                return True
        # Special case: "BadRequestError" AND "context length"
        return bool("BadRequestError" in error_str and "context length" in error_str)

    async def _create_completion(
        self,
        params: dict[str, Any],
        is_async: bool,
        stream_message_callback: Callable | None = None,
    ):
        """Helper to create a completion, handling async and sync calls."""
        if is_async:
            response = await self.client.chat.completions.create(**params)
        else:
            # Run sync client in executor to avoid blocking the event loop
            import functools

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, functools.partial(self.client.chat.completions.create, **params)
            )

        # If streaming is enabled, collect all chunks into a complete response
        if self.enable_streaming:
            return await collect_openai_stream(response, stream_message_callback)

        return response

    def _clean_user_content_from_response(self, text: str) -> str:
        """Remove content between \\n\\nUser: and <use_mcp_tool> in assistant response."""
        pattern = r"\n\nUser:.*?(?=<use_mcp_tool>|$)"
        cleaned_text = re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL)
        return cleaned_text

    def process_llm_response(
        self, llm_response, message_history, agent_type="main"
    ) -> tuple[str, bool]:
        """Process OpenAI LLM response."""
        if not llm_response or not llm_response.choices:
            error_msg = "LLM did not return a valid response."
            logger.error(f"Should never happen: {error_msg}")
            return "", True

        # Extract LLM response text
        finish_reason = llm_response.choices[0].finish_reason
        if finish_reason == "stop":
            assistant_response_text = llm_response.choices[0].message.content or ""
            assistant_response_text = self._clean_user_content_from_response(
                assistant_response_text
            )
            message_history.append({"role": "assistant", "content": assistant_response_text})
        elif finish_reason == "tool_calls":
            # Native tool calling: model returned tool_calls via API
            assistant_response_text = llm_response.choices[0].message.content or ""
            message_history.append({"role": "assistant", "content": assistant_response_text})
        elif finish_reason == "length":
            assistant_response_text = llm_response.choices[0].message.content or ""
            if assistant_response_text == "":
                assistant_response_text = "LLM response is empty. This is likely due to thinking block used up all tokens."
            else:
                assistant_response_text = self._clean_user_content_from_response(
                    assistant_response_text
                )
            message_history.append({"role": "assistant", "content": assistant_response_text})
        else:
            logger.error(f"Unsupported finish reason: {finish_reason}")
            assistant_response_text = "Successful response, but unsupported finish reason: " + str(
                finish_reason
            )
            message_history.append({"role": "assistant", "content": assistant_response_text})
        logger.debug(f"LLM Response: {truncate_for_log(assistant_response_text)}")

        return assistant_response_text, False

    def extract_tool_calls_info(self, llm_response, assistant_response_text):
        """Extract tool call information from OpenAI LLM response.

        Checks native tool_calls first (from API response), then falls back
        to XML text parsing for models using xml tool_format.
        """
        from mem_deep_research_core.utils.parsing_utils import parse_llm_response_for_tool_calls

        # Check for native tool_calls in the API response (e.g. Sonnet via OpenRouter)
        if llm_response and llm_response.choices:
            message = llm_response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                return parse_llm_response_for_tool_calls(message.tool_calls)

        # Fallback: parse XML <use_mcp_tool> tags from text
        return parse_llm_response_for_tool_calls(assistant_response_text)

    def _deduplicate_tool_results(self, tool_call_info: list) -> list:
        """Deduplicate identical tool results by hashing their text content.

        Collapses N identical results into 1 unique result + a note about removed duplicates.
        """
        if len(tool_call_info) <= 1:
            return tool_call_info

        seen_hashes: dict[str, int] = {}  # hash -> index in deduped list
        deduped: list = []
        duplicate_counts: dict[int, int] = {}  # index in deduped -> count of duplicates

        for tool_id, content in tool_call_info:
            text = content.get("text", "")
            text_hash = hashlib.md5(text.encode()).hexdigest()

            if text_hash in seen_hashes:
                idx = seen_hashes[text_hash]
                duplicate_counts[idx] = duplicate_counts.get(idx, 1) + 1
            else:
                seen_hashes[text_hash] = len(deduped)
                deduped.append((tool_id, content))

        # Add duplicate count notes
        total_removed = 0
        for idx, count in duplicate_counts.items():
            removed = count - 1
            total_removed += removed
            tool_id, content = deduped[idx]
            content = dict(content)  # copy to avoid mutating original
            content["text"] = (
                content["text"] + f"\n\n[Note: {removed} duplicate tool result(s) removed]"
            )
            deduped[idx] = (tool_id, content)

        if total_removed > 0:
            logger.info(
                f"[CONTEXT] Deduplicated tool results: removed {total_removed} duplicates from {len(tool_call_info)} results"
            )

        return deduped

    def update_message_history(self, message_history, tool_call_info, tool_calls_exceeded=False):
        """Update message history with tool calls data."""
        # Filter tool call results with type "text"
        tool_call_info = [item for item in tool_call_info if item[1]["type"] == "text"]

        # Deduplicate identical tool results before processing
        tool_call_info = self._deduplicate_tool_results(tool_call_info)

        # Separate valid tool calls and bad tool calls
        valid_tool_calls = [
            (tool_id, content) for tool_id, content in tool_call_info if tool_id != "FAILED"
        ]
        bad_tool_calls = [
            (tool_id, content) for tool_id, content in tool_call_info if tool_id == "FAILED"
        ]

        total_calls = len(valid_tool_calls) + len(bad_tool_calls)

        # Build output text
        output_parts = []

        if total_calls > 1:
            if tool_calls_exceeded:
                output_parts.append(
                    f"You made too many tool calls. I can only afford to process {len(valid_tool_calls)} valid tool calls in this turn."
                )
            else:
                output_parts.append(
                    f"I have processed {len(valid_tool_calls)} valid tool calls in this turn."
                )

            for i, (_tool_id, content) in enumerate(valid_tool_calls, 1):
                output_parts.append(f"Valid tool call {i} result:\n{content['text']}")

            for i, (_tool_id, content) in enumerate(bad_tool_calls, 1):
                output_parts.append(f"Failed tool call {i} result:\n{content['text']}")
        else:
            for _tool_id, content in valid_tool_calls:
                output_parts.append(content["text"])
            for _tool_id, content in bad_tool_calls:
                output_parts.append(content["text"])

        merged_text = "\n\n".join(output_parts)

        message_history.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": merged_text}],
            }
        )
        return message_history

    def parse_llm_response(self, llm_response) -> str:
        """Parse OpenAI LLM response to get text content."""
        if not llm_response or not llm_response.choices:
            raise ValueError("LLM did not return a valid response.")
        return llm_response.choices[0].message.content

    def _preflight_context_check(self, system_prompt: str, messages: list) -> list:
        """Check estimated token count before sending to API and trim if needed.

        If total tokens exceed 85% of max_context_length, progressively trim:
        1. First: replace old tool result content with placeholders
        2. Then: remove old message pairs from the middle
        """
        threshold = 0.85
        target_ratio = 0.70

        # Estimate total tokens
        all_text_parts = [system_prompt]
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                all_text_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        all_text_parts.append(item.get("text", ""))
        total_text = " ".join(all_text_parts)
        estimated_tokens = self._estimate_tokens(total_text)
        token_limit = self.max_context_length

        if token_limit <= 0 or estimated_tokens <= token_limit * threshold:
            return messages

        logger.warning(
            f"[CONTEXT] Pre-flight check: {estimated_tokens} tokens estimated, "
            f"limit={token_limit} ({estimated_tokens / token_limit * 100:.0f}% used). Trimming..."
        )

        target_tokens = int(token_limit * target_ratio)
        messages_copy = [m.copy() for m in messages]

        # Phase 1: Replace old user/tool message content with placeholders (skip first and last 5)
        user_indices = [
            i
            for i, msg in enumerate(messages_copy)
            if msg.get("role") in ("user", "tool") and i > 0
        ]
        if len(user_indices) > 5:
            for idx in user_indices[:-5]:
                content = messages_copy[idx].get("content", "")
                if isinstance(content, str) and len(content) > 200:
                    messages_copy[idx]["content"] = "[Tool result trimmed to save context]"
                elif isinstance(content, list):
                    for item in content:
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "text"
                            and len(item.get("text", "")) > 200
                        ):
                            item["text"] = "[Tool result trimmed to save context]"

            # Re-estimate
            all_text_parts = [system_prompt]
            for msg in messages_copy:
                content = msg.get("content", "")
                if isinstance(content, str):
                    all_text_parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            all_text_parts.append(item.get("text", ""))
            estimated_tokens = self._estimate_tokens(" ".join(all_text_parts))

            if estimated_tokens <= target_tokens:
                logger.info(
                    f"[CONTEXT] Pre-flight trimmed tool results: {estimated_tokens} tokens "
                    f"({estimated_tokens / token_limit * 100:.0f}% of limit)"
                )
                return messages_copy

        # Phase 2: Remove old message pairs from middle
        keep_head = 1
        keep_tail = 4
        if len(messages_copy) > keep_head + keep_tail:
            middle = messages_copy[keep_head:-keep_tail]
            # Remove half of middle messages (keep more recent ones)
            keep_count = max(1, len(middle) // 2)
            kept_middle = middle[-keep_count:]
            removed = len(middle) - len(kept_middle)
            messages_copy = messages_copy[:keep_head] + kept_middle + messages_copy[-keep_tail:]
            logger.info(
                f"[CONTEXT] Pre-flight removed {removed} old messages, "
                f"history now {len(messages_copy)} messages"
            )

        return messages_copy

    def _estimate_tokens(self, text: str) -> int:
        """Use tiktoken to estimate token count of text."""
        if not hasattr(self, "_encoding"):
            try:
                self._encoding = tiktoken.get_encoding("o200k_base")
            except Exception:
                self._encoding = tiktoken.get_encoding("cl100k_base")

        try:
            return len(self._encoding.encode(text))
        except Exception:
            return len(text) // 4

    # handle_max_turns_reached_summary_prompt: uses base class default

    # _apply_cache_control: override to include system messages
    def _apply_cache_control(self, messages, include_system: bool = True):
        return super()._apply_cache_control(messages, include_system=include_system)
