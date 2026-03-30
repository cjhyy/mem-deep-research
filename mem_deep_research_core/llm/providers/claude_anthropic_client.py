import asyncio
import dataclasses
import os
from collections.abc import Callable

from anthropic import (
    NOT_GIVEN,
    Anthropic,
    AsyncAnthropic,
)
from omegaconf import DictConfig

from mem_deep_research_core.exceptions import ContextLimitError
from mem_deep_research_core.llm.provider_client_base import LLMProviderClientBase
from mem_deep_research_core.llm.util import collect_anthropic_stream
from mem_deep_research_core.mem_deep_research_logging.logger import (
    bootstrap_logger,
    truncate_for_log,
)

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)


@dataclasses.dataclass
class ClaudeAnthropicClient(LLMProviderClientBase):
    def __post_init__(self):
        super().__post_init__()

    def _create_client(self, config: DictConfig):
        """Create Anthropic client"""
        api_key = self.cfg.llm.anthropic_api_key

        if self.async_client:
            return AsyncAnthropic(
                api_key=api_key,
                base_url=self.cfg.llm.anthropic_base_url,
                timeout=self.timeout,
            )
        else:
            return Anthropic(
                api_key=api_key,
                base_url=self.cfg.llm.anthropic_base_url,
                timeout=self.timeout,
            )

    async def _create_message(
        self,
        system_prompt,
        messages,
        tools_definitions,
        keep_tool_result: int = -1,
        stream_message_callback: Callable | None = None,
    ):
        """
        Send message to Anthropic API.
        :param system_prompt: System prompt string.
        :param messages: Message history list.
        :return: Anthropic API response object or None (if error).
        """
        logger.debug(f" Calling LLM ({'async' if self.async_client else 'sync'})")

        messages_copy = self._remove_tool_result_from_messages(messages, keep_tool_result)

        processed_messages = self._apply_cache_control(messages_copy)

        try:
            # Apply configurable retry decorator to the API call
            retry_decorator = self.get_retry_decorator(exception_to_skip=ContextLimitError)
            create_completion_with_retry = retry_decorator(self._create_completion)
            response = await create_completion_with_retry(system_prompt, processed_messages)

            # If streaming is enabled, collect all chunks into a complete response
            if self.enable_streaming:
                response = await collect_anthropic_stream(response)

            logger.debug(f"LLM call status: {getattr(response, 'stop_reason', 'N/A')}")
            return response
        except asyncio.CancelledError:
            logger.exception("[WARNING] LLM API call was cancelled during execution")
            raise
        except ContextLimitError:
            raise
        except Exception as e:
            error_str = str(e)
            # Anthropic raises BadRequestError with specific messages for context limit
            if any(
                pattern in error_str
                for pattern in [
                    "prompt is too long",
                    "maximum context length",
                    "exceeds the maximum",
                    "context_length_exceeded",
                    "Input is too long",
                ]
            ):
                logger.debug(f"Anthropic context limit exceeded: {error_str}")
                raise ContextLimitError(f"Context limit exceeded: {error_str}") from e
            logger.exception("Anthropic LLM endpoint failed")
            raise

    async def _create_completion(self, system_prompt, processed_messages):
        """Helper to create a completion, handling async and sync calls."""
        if self.async_client:
            return await self.client.messages.create(
                model=self.model_name,
                temperature=self.get_effective_temperature(),
                top_p=self.top_p if self.top_p != 1.0 else NOT_GIVEN,
                top_k=self.top_k if self.top_k != -1 else NOT_GIVEN,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=processed_messages,
                stream=self.enable_streaming,
            )
        else:
            return self.client.messages.create(
                model=self.model_name,
                temperature=self.get_effective_temperature(),
                top_p=self.top_p if self.top_p != 1.0 else NOT_GIVEN,
                top_k=self.top_k if self.top_k != -1 else NOT_GIVEN,
                max_tokens=self.max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=processed_messages,
                stream=self.enable_streaming,
            )

    def process_llm_response(
        self, llm_response, message_history, agent_type="main"
    ) -> tuple[str, bool]:
        """Process Anthropic LLM response"""
        if not llm_response:
            logger.debug("[ERROR] LLM call failed, skipping this response.")
            return "", True

        if not hasattr(llm_response, "content") or not llm_response.content:
            logger.debug("[ERROR] LLM response is empty or doesn't contain content.")
            return "", True

        # Extract response content
        assistant_response_text = ""
        assistant_response_content = []

        for block in llm_response.content:
            if block.type == "text":
                assistant_response_text += block.text + "\n"
                assistant_response_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_response_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        message_history.append({"role": "assistant", "content": assistant_response_content})

        logger.debug(f"LLM Response: {truncate_for_log(assistant_response_text)}")

        return assistant_response_text, False

    def extract_tool_calls_info(self, llm_response, assistant_response_text):
        """Extract tool call information from Anthropic LLM response"""
        from mem_deep_research_core.utils.parsing_utils import parse_llm_response_for_tool_calls

        # For Anthropic, parse tool calls from the response text
        return parse_llm_response_for_tool_calls(assistant_response_text)

    def update_message_history(
        self, message_history, tool_call_info, tool_calls_exceeded: bool = False
    ):
        """Update message history with tool calls data (llm client specific)"""

        merged_text = "\n".join(
            [item[1]["text"] for item in tool_call_info if item[1]["type"] == "text"]
        )

        message_history.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": merged_text}],
            }
        )

        return message_history

    def handle_max_turns_reached_summary_prompt(self, message_history, summary_prompt):
        """Handle max turns reached summary prompt"""
        # if message_history[-1]["role"] == "user":
        #     last_user_message = message_history.pop()
        #     return (
        #         last_user_message["content"][0]["text"]
        #         + "\n*************\n"
        #         + summary_prompt
        #     )
        # else:
        return summary_prompt

    def _apply_cache_control(self, messages):
        """Apply cache control to the last user message and system message (if applicable)"""
        cached_messages = []
        user_turns_processed = 0
        for turn in reversed(messages):
            if turn["role"] == "user" and user_turns_processed < 1:
                # Add ephemeral cache control to the text part of the last user message
                new_content = []
                processed_text = False
                # Check if content is a list
                if isinstance(turn.get("content"), list):
                    # see example here
                    # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
                    for item in turn["content"]:
                        if (
                            item.get("type") == "text"
                            and len(item.get("text")) > 0
                            and not processed_text
                        ):
                            # Copy and add cache control
                            text_item = item.copy()
                            text_item["cache_control"] = {"type": "ephemeral"}
                            new_content.append(text_item)
                            processed_text = True
                        else:
                            # Other types of content (like image) copied directly
                            new_content.append(item.copy())
                    cached_messages.append({"role": "user", "content": new_content})
                else:
                    # If content is not a list (e.g., plain text), add as is without cache control
                    # Or adjust logic as needed
                    logger.debug(
                        "Warning: User message content is not in expected list format, cache control not applied."
                    )
                    cached_messages.append(turn)

                user_turns_processed += 1
            else:
                # Add other messages directly
                cached_messages.append(turn)
        return list(reversed(cached_messages))
