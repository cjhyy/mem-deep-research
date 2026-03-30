import contextvars
import dataclasses
import os

from mem_deep_research_core.llm.providers.openai_compatible_client import OpenAICompatibleClient
from mem_deep_research_core.mem_deep_research_logging.logger import (
    bootstrap_logger,
    truncate_for_log,
)

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)

# 使用 contextvars 替代实例变量，消除并发竞态
_pending_tool_list_var: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "_pending_tool_list", default=None
)
_native_tool_name_map_var: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "_native_tool_name_map", default=None
)


@dataclasses.dataclass
class DeepSeekOpenRouterClient(OpenAICompatibleClient):
    def _get_api_credentials(self) -> tuple[str, str]:
        return self.cfg.llm.openrouter_api_key, self.cfg.llm.openrouter_base_url

    async def _create_message(
        self,
        system_prompt,
        messages,
        tools_definitions,
        keep_tool_result=-1,
        stream_message_callback=None,
    ):
        """Override to inject native tool_calls list into params via _customize_params."""
        tool_list, name_map = await self.convert_tool_definition_to_tool_call(tools_definitions)
        token1 = _pending_tool_list_var.set(tool_list)
        token2 = _native_tool_name_map_var.set(name_map)
        # 同步设置实例变量供 process_llm_response 中的 name_map 查找使用
        self._native_tool_name_map = name_map
        try:
            return await super()._create_message(
                system_prompt,
                messages,
                tools_definitions,
                keep_tool_result,
                stream_message_callback,
            )
        finally:
            _pending_tool_list_var.reset(token1)
            _native_tool_name_map_var.reset(token2)

    def _customize_params(self, params: dict) -> dict:
        params["stream"] = False
        pending = _pending_tool_list_var.get(None)
        if pending:
            params["tools"] = pending
        return params

    def process_llm_response(
        self, llm_response, message_history, agent_type="main"
    ) -> tuple[str, bool]:
        """Process OpenAI LLM response - extended with tool_calls finish_reason support."""
        if not llm_response or not llm_response.choices:
            error_msg = "LLM did not return a valid response."
            logger.error(f"Should never happen: {error_msg}")
            return "", True

        finish_reason = llm_response.choices[0].finish_reason

        if finish_reason == "tool_calls":
            tool_calls = llm_response.choices[0].message.tool_calls
            assistant_response_text = llm_response.choices[0].message.content or ""

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
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            logger.debug(f"LLM Response: {truncate_for_log(assistant_response_text)}")
            return assistant_response_text, False

        # Delegate to base class for stop/length/other
        return super().process_llm_response(llm_response, message_history, agent_type)

    def extract_tool_calls_info(self, llm_response, assistant_response_text):
        """Extract tool call information - from response object for tool_calls finish_reason."""
        from mem_deep_research_core.utils.parsing_utils import parse_llm_response_for_tool_calls

        name_map = _native_tool_name_map_var.get(None) or getattr(
            self, "_native_tool_name_map", None
        )

        if llm_response.choices[0].finish_reason == "tool_calls":
            return parse_llm_response_for_tool_calls(
                llm_response.choices[0].message.tool_calls, name_map=name_map
            )
        else:
            return [], []

    # handle_max_turns_reached_summary_prompt: uses base class default
