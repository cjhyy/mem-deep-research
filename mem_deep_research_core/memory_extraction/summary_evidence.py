"""SummaryEvidenceStrategy — 从 LLM 压缩 summary 的 ## Evidence 段抽取细节

封装原 window_strategy.LLMSummarizeStrategy._extract_evidence_from_summary 的逻辑。
Context 压缩时 LLM 产出的 summary 如果包含 ## Evidence 段，抽取存 session_memory。

这是 "细节保鲜" 的第二道防线：即使原消息被 compact 掉，关键事实仍以
EvidenceItem 形式留在 session_memory 里，供后续 summary / 引用。

所有 profile 默认启用。
"""

import logging

from mem_deep_research_core.core.memory import EvidenceItem
from mem_deep_research_core.memory_extraction.base import (
    ExtractionContext,
    MemoryExtractionStrategy,
)

logger = logging.getLogger("mem_deep_research")


class SummaryEvidenceStrategy(MemoryExtractionStrategy):
    """Context 压缩后从 summary 的 ## Evidence 段抽取细节存 session_memory。

    解析逻辑：
    - 查找 summary 中 "## Evidence" 标题
    - 截取到下一个 "## " 或末尾
    - 整段作为一个 EvidenceItem（tool_name="llm_summarize"）存入 session_memory
    """

    name = "summary_evidence"

    async def on_compact(
        self,
        summary: str,
        up_to_turn: int,
        ctx: ExtractionContext,
    ) -> None:
        if not summary or ctx.session_memory is None:
            return

        evidence_start = summary.find("## Evidence")
        if evidence_start == -1:
            return

        evidence_text = summary[evidence_start + len("## Evidence") :]
        # 截取到下一个 ## 或末尾
        next_section = evidence_text.find("\n## ")
        if next_section != -1:
            evidence_text = evidence_text[:next_section]
        evidence_text = evidence_text.strip()

        if not evidence_text:
            return

        ctx.session_memory.add_evidence(
            EvidenceItem(
                tool_name="llm_summarize",
                turn=up_to_turn,
                summary=evidence_text[:1000],
            )
        )
        logger.info(
            f"[SummaryEvidence] Extracted from summary "
            f"(turns 1-{up_to_turn}, {len(evidence_text)} chars)"
        )
