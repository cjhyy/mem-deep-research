"""DeepResearchProfile — 研究场景执行策略

聚合研究专属能力：
- Extraction strategies: Offload + Summary evidence + free-form EvidenceTag
- Lifecycle 钩子决策：reflection / verify / task plan / inline skills / final summary

实现原则：Profile 负责 "要不要做"（policy decision），主循环负责 "怎么做"（execution）。
Profile 不直接持有 runtime 对象（task_planner、summary_handler 等），只通过 ctx 读配置和状态。
"""

from typing import Any

from mem_deep_research_core.core.profiles.base import Profile, ProfileContext
from mem_deep_research_core.memory_extraction import (
    EvidenceTagStrategy,
    OffloadEvidenceStrategy,
    SummaryEvidenceStrategy,
)


class DeepResearchProfile(Profile):
    """研究场景 profile。

    Lifecycle 钩子实现 "mode=deep 时启用 reflection/verify/summary" 等研究专属
    决策，原本散落在 main_loop.py 的 29 处 is_deep_mode 分支被替换为钩子调用。

    Config 字段：
    - reflection_enabled: bool = True（mode=quick 时自动禁用）
    - enable_verify: bool = True（verify checkpoint 开关）
    - generate_summary: bool = True（mode=deep + tool_calls>0 时强制 summary）
    - auto_task_plan: bool = None（None = 跟随 task_planner 实例的 enabled；True/False 显式覆盖）
    """

    name = "deep_research"
    default_extraction_strategies = [
        OffloadEvidenceStrategy(),
        SummaryEvidenceStrategy(),
        EvidenceTagStrategy(),
    ]

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        cfg = config or {}
        self.reflection_enabled: bool = cfg.get("reflection_enabled", True)
        self.enable_verify: bool = cfg.get("enable_verify", True)
        self.generate_summary_default: bool = cfg.get("generate_summary", True)
        self.auto_task_plan: bool | None = cfg.get("auto_task_plan", None)

    # ========== Reflection ==========

    async def should_inject_reflection(self, ctx: ProfileContext) -> bool:
        """Deep / standard 模式 + reflection interval 达成时注入反思。

        对应原 main_loop:1650 `if not is_quick_mode and turn_counter.should_inject_reflection()`。
        Quick 模式不反思。Turn counter 状态由 runtime 管理，这里只给"是否开启反思能力"的决策。
        """
        return self.reflection_enabled and ctx.mode != "quick"

    # ========== Verify ==========

    async def should_run_verify(self, ctx: ProfileContext) -> bool:
        """Deep + 有工具调用 + enable_verify + session_memory 非空 → 跑 verify。

        对应原 main_loop:1752-1763。
        """
        if not self.enable_verify:
            return False
        if ctx.mode != "deep":
            return False
        if getattr(ctx, "tool_calls_executed", 0) <= 0:
            return False
        sm = getattr(ctx, "session_memory", None)
        if sm is None or sm.is_empty():
            return False
        return True

    # ========== Task Plan ==========

    async def create_task_plan(self, ctx: ProfileContext) -> str | None:
        """不直接产 plan 文本，主循环通过 ctx.task_planner 调用。
        Profile 只决策 "是否应该产 plan"（Phase 2c 以 policy 形式表达）。

        返回 None 表示无 plan；Profile 在这里不调用 TaskPlanner（TaskPlanner 是 runtime 对象）。
        实际 plan 生成由主循环在看到 should_create_task_plan=True 时触发。
        """
        return None  # 实际生成路径在 main_loop 里，见 should_create_task_plan

    async def should_create_task_plan(self, ctx: ProfileContext) -> bool:
        """决定是否生成 task plan。对应原 main_loop:807
        `if self.task_planner.enabled and not is_quick_mode`。

        Quick 模式不生成。实际启用由 runtime 的 task_planner.enabled 决定。
        """
        if ctx.mode == "quick":
            return False
        if self.auto_task_plan is False:
            return False
        return True  # 跟随 task_planner.enabled；True 表示 "不反对"

    # ========== Skills ==========

    async def should_process_inline_skills(self, ctx: ProfileContext) -> bool:
        """Quick 模式不处理 inline skills。对应原 main_loop:1201
        `if not is_quick_mode and self.inline_skill_selector and assistant_response_text`。
        """
        return ctx.mode != "quick"

    # ========== Final Summary ==========

    async def needs_final_summary(
        self,
        tool_calls_executed: int,
        last_assistant_text: str,
        ctx: ProfileContext,
    ) -> bool:
        """Deep 模式 + 有工具调用 → 强制 summary。对应原 main_loop:1771-1772
        `if effective_mode == EXECUTION_MODE_DEEP and total_tool_calls_executed > 0`。

        其他模式遵循用户配置的 generate_summary（默认 False / True，此处按 profile 默认）。
        """
        if ctx.mode == "deep" and tool_calls_executed > 0:
            return True
        return self.generate_summary_default
