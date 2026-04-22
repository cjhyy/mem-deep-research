"""OffloadEvidenceStrategy — 绑定 LLM 产出的 <offload_evidence> 到 offload registry

封装原 main_loop._extract_offload_evidence 的逻辑。
工具结果 offload 时，主循环注入 [OFFLOAD PREP] sidecar 引导 LLM 产 <offload_evidence ref="...">；
本 strategy 抽取并绑定到 context_manager._offload_registry，保证大工具结果细节保鲜。

所有 profile 默认启用（StandardProfile 默认含此 strategy）。
"""

import logging
import re

from mem_deep_research_core.memory_extraction.base import (
    ExtractionContext,
    MemoryExtractionStrategy,
)

logger = logging.getLogger("mem_deep_research")

_RE_OFFLOAD_EVIDENCE = re.compile(
    r'<offload_evidence\s+ref="([^"]+)">(.*?)</offload_evidence>', re.DOTALL
)


class OffloadEvidenceStrategy(MemoryExtractionStrategy):
    """抽取 <offload_evidence ref="..."> 并绑定到 offload registry。

    工作流：
    1. 工具结果超阈值 → context_manager 把它 offload 到文件，注入 [OFFLOAD PREP] sidecar
    2. LLM 下轮产 <offload_evidence ref="abc.txt">- key fact 1\\n- key fact 2</offload_evidence>
    3. 本 strategy 抽取 → context_manager.update_offload_evidence(ref, lines)

    这样即使原始 offload 文件后续被清理，关键事实已绑定到 offload record，
    在 context 管理 / 最终 summary 里可恢复引用。

    所有 profile 默认启用 —— "细节保鲜" 是通用能力，不是研究专属。
    """

    name = "offload_evidence"

    async def on_llm_response(
        self,
        assistant_text: str,
        ctx: ExtractionContext,
    ) -> str:
        if not assistant_text:
            return assistant_text

        matches = _RE_OFFLOAD_EVIDENCE.findall(assistant_text)
        if not matches:
            return assistant_text

        for ref, block in matches:
            lines = [
                l.strip().lstrip("- ").strip()
                for l in block.strip().split("\n")
                if l.strip() and l.strip() != "-"
            ]
            if lines:
                ctx.context_manager.update_offload_evidence(ref, lines)

        return assistant_text
