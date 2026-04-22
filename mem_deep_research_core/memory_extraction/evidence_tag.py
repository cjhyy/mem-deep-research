"""EvidenceTagStrategy — 抽取 LLM 回复中的 <evidence> tag 存 session_memory

封装原 main_loop._extract_evidence_tags 的逻辑。
研究场景 prompt 引导 LLM 在回复中产 <evidence>...</evidence> tag。

原则：
- Strategy 只读取 tag 内容写入 session_memory
- 不清理 tag（tag 清理由 Runtime 在 strategy 链完成后统一做）
"""

import logging
import re

from mem_deep_research_core.core.constants import EVIDENCE_MAX_CHARS
from mem_deep_research_core.core.memory import EvidenceItem
from mem_deep_research_core.memory_extraction.base import (
    ExtractionContext,
    MemoryExtractionStrategy,
)

logger = logging.getLogger("mem_deep_research")

_RE_EVIDENCE = re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL)
_RE_SOURCE = re.compile(r"\(source:\s*(https?://[^\s)]+)\)", re.IGNORECASE)
_RE_CONFIDENCE = re.compile(r"\(confidence:\s*(high|medium|low)\)", re.IGNORECASE)


def _parse_evidence_line(line: str) -> tuple[str, str, str]:
    """从单行证据中提取 source_url 和 confidence。

    Returns:
        (clean_text, source_url, confidence)
    """
    source_url = ""
    confidence = ""
    m = _RE_SOURCE.search(line)
    if m:
        source_url = m.group(1)
        line = line[: m.start()] + line[m.end() :]
    m = _RE_CONFIDENCE.search(line)
    if m:
        confidence = m.group(1).lower()
        line = line[: m.start()] + line[m.end() :]
    return line.strip().strip("-").strip(), source_url, confidence


class EvidenceTagStrategy(MemoryExtractionStrategy):
    """抽取 <evidence>...</evidence> tag 存 session_memory。

    依赖 prompt 引导 LLM 产出 tag 格式。DeepResearchProfile 的 prompt 模板
    默认包含引导指令；其他 profile 除非自行引导，LLM 不会产此 tag，
    strategy 执行但抽不到（无副作用）。

    支持两种格式：
    1. 逐行结构化（推荐）：每行以 "-" 开头，可带 (source: URL) (confidence: high/medium/low)
    2. 整块文本（兼容旧格式）
    """

    name = "evidence_tag"

    async def on_llm_response(
        self,
        assistant_text: str,
        ctx: ExtractionContext,
    ) -> str:
        if not assistant_text:
            return assistant_text

        matches = _RE_EVIDENCE.findall(assistant_text)
        if not matches:
            return assistant_text

        count = 0
        for block in matches:
            block = block.strip()
            if not block:
                continue

            # 尝试逐行解析（每行以 - 开头）
            lines = [l.strip() for l in block.split("\n") if l.strip().startswith("-")]
            if lines:
                for line in lines:
                    text, source_url, confidence = _parse_evidence_line(line)
                    if not text:
                        continue
                    if len(text) > EVIDENCE_MAX_CHARS:
                        text = text[:EVIDENCE_MAX_CHARS] + "..."
                    ctx.session_memory.add_evidence(
                        EvidenceItem(
                            tool_name="llm_extraction",
                            turn=ctx.turn_number,
                            summary=text,
                            source_url=source_url,
                            confidence=confidence,
                        )
                    )
                    count += 1
            else:
                # 整块作为一个 EvidenceItem（兼容旧格式）
                if len(block) > EVIDENCE_MAX_CHARS:
                    block = block[:EVIDENCE_MAX_CHARS] + "..."
                ctx.session_memory.add_evidence(
                    EvidenceItem(
                        tool_name="llm_extraction",
                        turn=ctx.turn_number,
                        summary=block,
                    )
                )
                count += 1

        if count:
            logger.debug(
                f"[EvidenceTag] Extracted {count} evidence items from turn {ctx.turn_number}"
            )
        return assistant_text
