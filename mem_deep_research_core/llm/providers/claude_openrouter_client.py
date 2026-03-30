import dataclasses
import os

from omegaconf import DictConfig

from mem_deep_research_core.llm.providers.openai_compatible_client import OpenAICompatibleClient
from mem_deep_research_core.mem_deep_research_logging.logger import bootstrap_logger

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)


@dataclasses.dataclass
class ClaudeOpenRouterClient(OpenAICompatibleClient):
    def _get_api_credentials(self) -> tuple[str, str]:
        return self.cfg.llm.openrouter_api_key, self.cfg.llm.openrouter_base_url

    def _create_client(self, config: DictConfig):
        """Create configured OpenAI client with API key logging."""
        api_key, base_url = self._get_api_credentials()
        if api_key:
            masked_key = f"{api_key[:15]}...{api_key[-8:]}" if len(api_key) > 23 else "***"
            logger.info(
                f"[OpenRouter] Creating client with API key: {masked_key}, base_url: {base_url}"
            )
        else:
            logger.warning("[OpenRouter] API key is empty or None!")
        return super()._create_client(config)
