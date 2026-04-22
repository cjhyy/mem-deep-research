"""StandardProfile — 默认执行策略

全空实现，所有 lifecycle 钩子继承基类默认行为。

默认 extraction_strategies：
- OffloadEvidenceStrategy: 绑定 LLM 产出的 <offload_evidence> 到 offload registry
- SummaryEvidenceStrategy: 从 LLM 压缩 summary 的 ## Evidence 段抽细节

这两个是 "大工具结果细节保鲜" 的通用基础设施，不依赖 prompt 引导
（或 LLM 主动产出），所有 profile 都应具备。

用户如需完全自定义 / 禁用，通过 profile_config:
    profile_config={"extraction_strategies": []}  # 完全覆盖
"""

from mem_deep_research_core.core.profiles.base import Profile, ProfileContext
from mem_deep_research_core.memory_extraction import (
    OffloadEvidenceStrategy,
    SummaryEvidenceStrategy,
)


class StandardProfile(Profile):
    """通用 agent profile。

    Lifecycle 钩子默认返回 "不做任何研究专属动作"：
    - 不注入 reflection prompt
    - 不跑 verify checkpoint
    - 不自动 task plan
    - 不处理 inline skills
    - 不强制 summary（由用户配置 generate_summary 决定）

    默认 extraction strategies：
    - OffloadEvidenceStrategy：绑定 <offload_evidence> 到 offload registry
    - SummaryEvidenceStrategy：从 LLM 压缩 summary 的 ## Evidence 段抽细节
    （这两个是通用 "细节保鲜" 基础设施，所有 profile 都应具备）
    """

    name = "standard"
    default_extraction_strategies = [
        OffloadEvidenceStrategy(),
        SummaryEvidenceStrategy(),
    ]

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        cfg = config or {}
        # Standard profile 也支持用户显式启用 summary（少数场景）
        self.generate_summary_default: bool = cfg.get("generate_summary", False)

    async def needs_final_summary(
        self,
        tool_calls_executed: int,
        last_assistant_text: str,
        ctx: ProfileContext,
    ) -> bool:
        """Standard profile 默认不生成 summary。"""
        return self.generate_summary_default
