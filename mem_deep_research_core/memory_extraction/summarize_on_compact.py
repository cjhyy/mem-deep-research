"""SummarizeOnCompactStrategy — Context 压缩时把整段 summary 存为 memory anchor

和 SummaryEvidenceStrategy 互补：SummaryEvidence 抽 summary 里的 ## Evidence 段，
本 strategy 把整段 summary 本身作为 "记忆锚点" 保存到 session_memory.evidence_items。

适合 LangGraph / Mastra 风格的 "memory is summary" 思路：
当原消息被 compact 掉，整段 summary 作为 EvidenceItem 留存，后续可追溯。

默认不启用。用户按需追加：
    StandardProfile(config={"extraction_strategies_extra": [SummarizeOnCompactStrategy()]})
"""

import logging

from mem_deep_research_core.core.memory import EvidenceItem
from mem_deep_research_core.memory_extraction.base import (
    ExtractionContext,
    MemoryExtractionStrategy,
)

logger = logging.getLogger("mem_deep_research")


class SummarizeOnCompactStrategy(MemoryExtractionStrategy):
    """把 LLMSummarize 产出的整段 summary 作为 memory anchor 存 session_memory。

    EvidenceItem:
    - tool_name = "compact_anchor"
    - summary = summary 全文（按 max_anchor_chars 截断）
    - turn = up_to_turn
    """

    name = "summarize_on_compact"

    def __init__(self, max_anchor_chars: int = 4000):
        super().__init__()
        self.max_anchor_chars = max_anchor_chars

    async def on_compact(
        self,
        summary: str,
        up_to_turn: int,
        ctx: ExtractionContext,
    ) -> None:
        if not summary or ctx.session_memory is None:
            return
        text = summary.strip()
        if not text:
            return
        if len(text) > self.max_anchor_chars:
            text = text[: self.max_anchor_chars] + "..."
        ctx.session_memory.add_evidence(
            EvidenceItem(
                tool_name="compact_anchor",
                turn=up_to_turn,
                summary=text,
            )
        )
        logger.info(
            f"[SummarizeOnCompact] Anchored summary (turns 1-{up_to_turn}, {len(text)} chars)"
        )
