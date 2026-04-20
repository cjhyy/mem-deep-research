import asyncio
import dataclasses
import os
from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig
from openai import AsyncOpenAI, OpenAI

# tenacity imports removed - using configurable retry from base class
from mem_deep_research_core.llm.provider_client_base import LLMProviderClientBase
from mem_deep_research_core.llm.util import collect_openai_stream
from mem_deep_research_core.mem_deep_research_logging.logger import (
    bootstrap_logger,
    truncate_for_log,
)

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
# OPENAI reasoning models only support temperature=1
OPENAI_REASONING_MODEL_SET = {"o1", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-5-2025-08-07"}

logger = bootstrap_logger(level=LOGGER_LEVEL)


@dataclasses.dataclass
class GPTOpenAIClient(LLMProviderClientBase):
    def _create_client(self, config: DictConfig):
        """Create configured OpenAI client"""
        if self.async_client:
            return AsyncOpenAI(
                api_key=self.cfg.llm.openai_api_key,
                base_url=self.cfg.llm.openai_base_url,
                timeout=self.timeout,
            )
        else:
            return OpenAI(
                api_key=self.cfg.llm.openai_api_key,
                base_url=self.cfg.llm.openai_base_url,
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
        """
        Send message to OpenAI API.
        :param system_prompt: System prompt string.
        :param messages: Message history list.
        :return: OpenAI API response object or None (if error occurs).
        """
        is_oai_new_model = (
            self.model_name.startswith("o1")
            or self.model_name.startswith("o3")
            or self.model_name.startswith("o4")
            or self.model_name.startswith("gpt-4.1")
            or self.model_name.startswith("gpt-4o")
            or self.model_name.startswith("gpt-5")
        )
        logger.debug(f" Calling LLM ({'async' if self.async_client else 'sync'})")
        # Avoid mutating the caller's message list (deep copy dicts to prevent side effects)
        import copy

        messages = [copy.copy(m) for m in messages]
        # put the system prompt in the first message since OpenAI API does not support system prompt in
        if system_prompt:
            target_role = "developer" if is_oai_new_model else "system"

            # Check if there's already a system or developer message
            if messages and messages[0]["role"] in ["system", "developer"]:
                # Replace existing message with correct role
                messages[0] = {
                    "role": target_role,
                    "content": [{"type": "text", "text": system_prompt}],
                }
            else:
                # Insert new message
                messages.insert(
                    0,
                    {
                        "role": target_role,
                        "content": [{"type": "text", "text": system_prompt}],
                    },
                )

        messages_copy = self._remove_tool_result_from_messages(messages, keep_tool_result)

        tool_list, name_map = await self.convert_tool_definition_to_tool_call(tools_definitions)
        self._native_tool_name_map = name_map

        try:
            # Set temperature=1 for reasoning models
            temperature = (
                1.0
                if self.model_name in OPENAI_REASONING_MODEL_SET
                else self.get_effective_temperature()
            )

            params = {
                "model": self.model_name,
                "temperature": temperature,
                "max_completion_tokens": self.max_tokens,
                "messages": messages_copy,
                "tools": tool_list,
                "stream": self.enable_streaming,
            }

            if self.top_p != 1.0:
                params["top_p"] = self.top_p
            # NOTE: min_p and top_k are not supported by OpenAI chat completion API, but SGLANG and VLLM support them
            if self.min_p != 0.0:
                params["min_p"] = self.min_p
            if self.top_k != -1:
                params["top_k"] = self.top_k

            # Apply configurable retry decorator to the API call
            retry_decorator = self.get_retry_decorator()

            if self.oai_tool_thinking:
                handle_oai_tool_thinking_with_retry = retry_decorator(
                    self._handle_oai_tool_thinking
                )
                response = await handle_oai_tool_thinking_with_retry(
                    params, messages, self.async_client, stream_message_callback
                )
            else:
                create_completion_with_retry = retry_decorator(self._create_completion)
                response = await create_completion_with_retry(
                    params, self.async_client, stream_message_callback
                )

            logger.debug(f"LLM call status: {getattr(response.choices[0], 'finish_reason', 'N/A')}")
            return response
        except asyncio.CancelledError:
            logger.exception("[WARNING] LLM API call was cancelled during execution")
            raise
        except Exception as e:
            logger.exception(f"OpenAI LLM call failed: {str(e)}")
            raise e

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
            response = self.client.chat.completions.create(**params)

        # If streaming is enabled, collect all chunks into a complete response
        if self.enable_streaming:
            return await collect_openai_stream(response, stream_message_callback)

        return response

    async def _handle_oai_tool_thinking(
        self,
        params: dict[str, Any],
        messages: list[dict[str, Any]],
        is_async: bool,
        stream_message_callback: Callable | None = None,
    ):
        """Handles the logic for oai_tool_thinking.

        Uses a copy of the messages list to avoid mutating the caller's state.
        """
        # Work on a copy so the caller's message list is not mutated
        messages_copy = list(messages)

        # ---- Step 1: Let AI output text first, without calling tools ----
        params["tool_choice"] = "none"
        response = await self._create_completion(params, is_async, stream_message_callback)

        text_reply = response.choices[0].message.content
        messages_copy.append({"role": "assistant", "content": text_reply})
        params["messages"] = messages_copy

        # ---- Step 2: Allow tool_call ----
        del params["tool_choice"]
        response_tool = await self._create_completion(params, is_async, stream_message_callback)

        if response_tool.choices[0].finish_reason == "tool_calls":
            response_tool.choices[0].message.content = text_reply
            response = response_tool

        return response

    def process_llm_response(
        self, llm_response, message_history, agent_type="main"
    ) -> tuple[str, bool]:
        """Process OpenAI LLM response"""

        if not llm_response or not llm_response.choices:
            error_msg = "LLM did not return a valid response."
            logger.debug(f"Error: {error_msg}")
            return "", True  # Exit loop

        # Extract LLM response text
        if llm_response.choices[0].finish_reason == "stop":
            assistant_response_text = llm_response.choices[0].message.content or ""
            message_history.append({"role": "assistant", "content": assistant_response_text})
        elif llm_response.choices[0].finish_reason == "tool_calls":
            # For tool_calls, we need to extract tool call information as text
            tool_calls = llm_response.choices[0].message.tool_calls
            assistant_response_text = llm_response.choices[0].message.content or ""

            # If there's no text content, we generate a text describing the tool call
            if not assistant_response_text:
                tool_call_descriptions = []
                for tool_call in tool_calls:
                    tool_call_descriptions.append(
                        f"Using tool {tool_call.function.name} with arguments: {tool_call.function.arguments}"
                    )
                assistant_response_text = "\n".join(tool_call_descriptions)

            message_history.append(
                {
                    "role": "assistant",
                    "content": assistant_response_text,
                    "tool_calls": [
                        {
                            "id": _.id,
                            "type": "function",
                            "function": {
                                "name": _.function.name,
                                "arguments": _.function.arguments,
                            },
                        }
                        for _ in tool_calls
                    ],
                }
            )
        elif llm_response.choices[0].finish_reason == "length":
            assistant_response_text = llm_response.choices[0].message.content or ""
            if assistant_response_text == "":
                assistant_response_text = "LLM response is empty. This is likely due to thinking block used up all tokens."
            message_history.append({"role": "assistant", "content": assistant_response_text})
        else:
            raise ValueError(f"Unsupported finish reason: {llm_response.choices[0].finish_reason}")
        logger.debug(f"LLM Response: {truncate_for_log(assistant_response_text)}")

        # Claude Code-style loop exit: trust finish_reason.
        finish_reason = llm_response.choices[0].finish_reason
        should_break = finish_reason != "tool_calls"
        if should_break:
            logger.debug(f"[GPT] finish_reason={finish_reason!r} → should_break=True")
        return assistant_response_text, should_break

    def extract_tool_calls_info(self, llm_response, assistant_response_text):
        """Extract tool call information from OpenAI LLM response"""
        from mem_deep_research_core.utils.parsing_utils import parse_llm_response_for_tool_calls

        name_map = getattr(self, "_native_tool_name_map", None)

        # For OpenAI, get tool calls directly from response object
        if llm_response.choices[0].finish_reason == "tool_calls":
            return parse_llm_response_for_tool_calls(
                llm_response.choices[0].message.tool_calls, name_map=name_map
            )
        else:
            return [], []

    def update_message_history(
        self, message_history, tool_call_info, tool_calls_exceeded: bool = False
    ):
        """Update message history with tool calls data (llm client specific)"""

        from mem_deep_research_core.core.constants import make_tool_result_msg_native

        for cur_call_id, tool_result in tool_call_info:
            message_history.append(
                make_tool_result_msg_native(cur_call_id, tool_result["text"])
            )

        return message_history

    def handle_max_turns_reached_summary_prompt(self, message_history, summary_prompt):
        """Handle max turns reached summary prompt"""
        return summary_prompt
