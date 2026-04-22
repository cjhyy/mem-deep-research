"""FactExtractionStrategy — 工具结果回来后用轻量 LLM 抽 facts 存 session_memory

适合主 LLM 不可控（弱模型 / 用户 prompt 自由度大）的场景，
以轻量 LLM 的成本换 extraction 质量的确定性。

默认不启用。用户按需传入：
    FactExtractionStrategy(extractor_llm_client=haiku_client)
"""

import hashlib
import logging
from typing import Any

from mem_deep_research_core.core.memory import EvidenceItem
from mem_deep_research_core.memory_extraction.base import (
    ExtractionContext,
    MemoryExtractionStrategy,
)

logger = logging.getLogger("mem_deep_research")


DEFAULT_FACT_EXTRACTION_PROMPT = """你是信息提炼助手。从下面工具返回的内容中提炼最多 {max_facts} 条关键事实。

要求：
- 每条事实一行，以 "- " 开头
- 只提炼文本里明确存在的信息，不推测、不总结
- 事实要具体（含数字、实体、时间等），避免泛泛描述
- 如果没有值得提炼的事实，返回空

工具名: {tool}

内容:
{result}

提炼的事实（每行一条，以 "- " 开头）:"""


class FactExtractionStrategy(MemoryExtractionStrategy):
    """工具结果回来后用轻量 LLM 抽 facts 存 session_memory。

    Snapshot 维护 "已抽取过的 tool_result hash" 集合，避免 resume 后重抽。
    """

    name = "fact_extraction"

    def __init__(
        self,
        extractor_llm_client: Any = None,
        prompt_template: str = DEFAULT_FACT_EXTRACTION_PROMPT,
        max_facts_per_result: int = 5,
        min_result_size: int = 500,
        max_result_chars: int = 10000,
    ):
        super().__init__()
        self.extractor = extractor_llm_client
        self.prompt_template = prompt_template
        self.max_facts = max_facts_per_result
        self.min_result_size = min_result_size
        self.max_result_chars = max_result_chars
        # 已抽取过的 (tool_name, hash(content))，resume-safe
        self._processed: set[tuple[str, str]] = set()

    async def on_tool_result(
        self,
        tool_name: str,
        tool_result: Any,
        ctx: ExtractionContext,
    ) -> None:
        if self.extractor is None:
            return
        text = self._coerce_text(tool_result)
        if len(text) < self.min_result_size:
            return

        # 去重：相同工具 + 相同内容指纹只抽一次（跨 resume 也有效）
        fingerprint = hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:16]
        key = (tool_name, fingerprint)
        if key in self._processed:
            return
        self._processed.add(key)

        prompt = self.prompt_template.format(
            tool=tool_name,
            result=text[: self.max_result_chars],
            max_facts=self.max_facts,
        )
        try:
            response = await self._call_extractor(prompt)
        except Exception as e:
            logger.warning(f"[FactExtraction] Extractor call failed for {tool_name}: {e}")
            return

        facts = self._parse_facts(response)[: self.max_facts]
        for fact in facts:
            ctx.session_memory.add_evidence(
                EvidenceItem(
                    tool_name=tool_name,
                    turn=ctx.turn_number,
                    summary=fact,
                )
            )
        if facts:
            logger.info(
                f"[FactExtraction] Extracted {len(facts)} facts from {tool_name} at turn {ctx.turn_number}"
            )

    @staticmethod
    def _coerce_text(tool_result: Any) -> str:
        if isinstance(tool_result, str):
            return tool_result
        if isinstance(tool_result, dict):
            return tool_result.get("text", "") or str(tool_result)
        return str(tool_result)

    async def _call_extractor(self, prompt: str) -> str:
        """调用轻量 LLM。兼容多种 client 接口。"""
        # 支持 client.extract(prompt) 形式（测试 / 自定义 client）
        if hasattr(self.extractor, "extract"):
            result = self.extractor.extract(prompt)
            if hasattr(result, "__await__"):
                result = await result
            return str(result) if not isinstance(result, str) else result
        # 支持标准 create_message 接口
        if hasattr(self.extractor, "create_message"):
            response = await self.extractor.create_message(
                system_prompt="",
                message_history=[{"role": "user", "content": prompt}],
                tool_definitions=[],
            )
            # 尝试多种返回形态
            if hasattr(response, "content"):
                blocks = response.content
                if isinstance(blocks, list):
                    texts = [b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "") for b in blocks]
                    return "\n".join(t for t in texts if t)
            return str(response)
        raise TypeError(
            f"extractor_llm_client {type(self.extractor)} has no 'extract' or 'create_message' method"
        )

    @staticmethod
    def _parse_facts(response: str) -> list[str]:
        """从响应里解析 "- xxx" 开头的行。"""
        facts: list[str] = []
        for line in response.splitlines():
            s = line.strip()
            if s.startswith("- "):
                fact = s[2:].strip()
                if fact:
                    facts.append(fact)
        return facts

    # ========== Snapshot ==========

    def snapshot(self) -> dict:
        # set of tuples → list of lists（JSON-serializable）
        return {"processed": [list(t) for t in self._processed]}

    def restore(self, state: dict) -> None:
        processed = state.get("processed", [])
        self._processed = {tuple(t) for t in processed if isinstance(t, (list, tuple)) and len(t) == 2}
