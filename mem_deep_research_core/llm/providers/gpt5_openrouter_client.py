import dataclasses
import os

from mem_deep_research_core.llm.providers.openai_compatible_client import OpenAICompatibleClient
from mem_deep_research_core.mem_deep_research_logging.logger import bootstrap_logger

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)


@dataclasses.dataclass
class GPT5OpenRouterClient(OpenAICompatibleClient):
    def _get_api_credentials(self) -> tuple[str, str]:
        return self.cfg.llm.openrouter_api_key, self.cfg.llm.openrouter_base_url

    def _validate_model(self) -> None:
        valid_models = ["gpt-5-2025-08-07", "gpt-5"]
        if self.model_name not in valid_models:
            raise ValueError(
                f"Invalid model_name '{self.model_name}'. Must be one of: {valid_models}"
            )

    def _customize_params(self, params: dict) -> dict:
        # GPT-5 uses max_completion_tokens instead of max_tokens
        params["max_completion_tokens"] = params.pop("max_tokens")
        params["reasoning_effort"] = self.reasoning_effort
        params["stream"] = False
        return params

    # handle_max_turns_reached_summary_prompt: uses base class default
