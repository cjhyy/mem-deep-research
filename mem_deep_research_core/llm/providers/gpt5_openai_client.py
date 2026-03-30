import dataclasses

from mem_deep_research_core.llm.providers.gpt5_openrouter_client import GPT5OpenRouterClient


@dataclasses.dataclass
class GPT5OpenAIClient(GPT5OpenRouterClient):
    """GPT-5 via OpenAI direct API. Same behavior as GPT5OpenRouterClient but with OpenAI credentials."""

    def _get_api_credentials(self) -> tuple[str, str]:
        return self.cfg.llm.openai_api_key, self.cfg.llm.openai_base_url
