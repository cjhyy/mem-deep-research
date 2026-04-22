"""Profile 抽象基类

Profile 是执行策略的抽象：决定 "这一类任务要怎么跑"（研究 / 自动化 / 编码 / workflow 节点 / ...）。
与 Mode 正交 —— Profile 决定 "做什么类型的任务"，Mode 决定 "投入多少资源"（quick / standard / deep）。

设计依据：docs/22-profile-boundary.md、docs/25-profile-contract.md

Phase 1 状态：接口落地，StandardProfile 全空实现，主循环钩子调用点加上但 runtime 行为零变化。
Phase 2 将把当前主循环里 29 处研究专属分支迁移到 DeepResearchProfile。
"""

from abc import ABC
from typing import Any, Protocol, runtime_checkable

from mem_deep_research_core.memory_extraction import (
    ExtractionContext,
    MemoryExtractionStrategy,
)


@runtime_checkable
class ProfileContext(Protocol):
    """Profile 方法可访问的 runtime 状态视图。

    大多数字段是只读 —— Profile 不能直接改 message_history / session_memory，
    通过返回值（如 on_llm_response 返回修改后的 text）或显式 API 写入。

    实际对象由 MainLoopRunner 构造；这里只是 Protocol，便于类型检查和测试 mock。
    """

    # 任务元数据
    turn_number: int
    task_description: str
    mode: str  # "quick" / "standard" / "deep" / "auto"

    # 执行状态
    tool_calls_executed: int
    assistant_response_text: str
    last_assistant_text: str

    # 可访问的运行时对象（profile 可读，不应写）
    message_history: list[dict]
    session_memory: Any
    todo_tracker: Any
    context_manager: Any
    llm_client: Any
    hooks: Any


class Profile(ABC):
    """执行策略抽象基类。

    所有钩子方法都提供默认实现（= StandardProfile 的行为），子类按需覆盖。
    主循环在固定生命周期点调用这些钩子。

    钩子调用时序（每轮）：
        on_agent_start()                     # 仅首轮
        build_initial_system_prompt()        # 仅首轮
        ┌── on_turn_start()
        │   should_inject_reflection() / build_reflection_prompt()
        │   ...LLM call...
        │   on_llm_response()
        │   on_before_tools() / tool execute / on_after_tools()
        │   should_run_verify() / run_verify()
        └── (loop)
        build_final_answer()                 # 主循环退出后
    """

    name: str = "base"

    # Profile 的默认 memory extraction strategies。子类覆盖。
    # 运行时通过 config 可以完全覆盖（extraction_strategies）或追加（extraction_strategies_extra）。
    default_extraction_strategies: list[MemoryExtractionStrategy] = []

    def __init__(self, config: dict | None = None):
        """基类构造：解析 extraction_strategies 配置。

        config 支持两个 key：
        - extraction_strategies: 完全覆盖 default_extraction_strategies
        - extraction_strategies_extra: 追加在 default 后面
        """
        config = config or {}
        if "extraction_strategies" in config:
            self.extraction_strategies: list[MemoryExtractionStrategy] = list(
                config["extraction_strategies"]
            )
        else:
            # 复制 class-level 默认（避免多实例共享同一 list）
            self.extraction_strategies = list(self.__class__.default_extraction_strategies)
            extra = config.get("extraction_strategies_extra", [])
            if extra:
                self.extraction_strategies.extend(extra)

    # ========== Memory Extraction Strategy 链调用 ==========
    # Runtime 在 4 个触发点调这些方法，不需要遍历 strategies list。

    async def run_strategies_on_llm_response(
        self, assistant_text: str, ctx: "ExtractionContext",
    ) -> str:
        """按 list 顺序调 strategy.on_llm_response，串联返回值。"""
        for strat in self.extraction_strategies:
            assistant_text = await strat.on_llm_response(assistant_text, ctx)
        return assistant_text

    async def run_strategies_on_tool_result(
        self, tool_name: str, tool_result: Any, ctx: "ExtractionContext",
    ) -> None:
        for strat in self.extraction_strategies:
            await strat.on_tool_result(tool_name, tool_result, ctx)

    async def run_strategies_on_compact(
        self, summary: str, up_to_turn: int, ctx: "ExtractionContext",
    ) -> None:
        for strat in self.extraction_strategies:
            await strat.on_compact(summary, up_to_turn, ctx)

    async def run_strategies_on_offload(
        self, ref: str, tool_name: str, original_content: str, ctx: "ExtractionContext",
    ) -> None:
        for strat in self.extraction_strategies:
            await strat.on_offload(ref, tool_name, original_content, ctx)

    # ========== 启动阶段 ==========

    async def on_agent_start(self, ctx: ProfileContext) -> None:
        """Agent 启动时调一次。

        典型用途：DeepResearchProfile 在此创建并注入 task plan。
        """
        return None

    async def build_initial_system_prompt(
        self,
        base_prompt: str,
        ctx: ProfileContext,
    ) -> str:
        """可选修改 initial system prompt。

        默认返回原 prompt 不做修改。
        """
        return base_prompt

    # ========== 每轮生命周期 ==========

    async def on_turn_start(self, ctx: ProfileContext) -> None:
        """每轮开始时调。"""
        return None

    async def should_inject_reflection(self, ctx: ProfileContext) -> bool:
        """决定本轮是否注入反思 prompt。

        StandardProfile: False（不反思）。
        DeepResearchProfile: 按 reflection_interval 决定。
        """
        return False

    async def build_reflection_prompt(self, ctx: ProfileContext) -> str | None:
        """返回反思 prompt 文本，None 表示不注入。

        仅在 should_inject_reflection 返回 True 时调用。
        """
        return None

    # ========== LLM 响应后 ==========

    async def on_llm_response(
        self,
        assistant_text: str,
        ctx: ProfileContext,
    ) -> str:
        """LLM 响应返回后调，返回可能被修改的 assistant_text。

        典型用途：DeepResearchProfile 抽取 <evidence> tag、清理研究专属标记。
        默认直接返回原 text。
        """
        return assistant_text

    # ========== 工具执行前后 ==========

    async def on_before_tools(
        self,
        tool_calls: list[dict],
        ctx: ProfileContext,
    ) -> list[dict]:
        """工具批次执行前调，返回可能被修改/重排的 tool_calls。

        注意：这是 profile 专属决策，和通用的 on_tool_filter hook 不同。
        hook 先于 profile 方法调用。默认直接返回原 list。
        """
        return tool_calls

    async def on_after_tools(
        self,
        results: list[tuple[str, dict]],
        ctx: ProfileContext,
    ) -> None:
        """工具结果收齐后调。

        典型用途：DeepResearchProfile 更新 source registry / evidence binding。
        """
        return None

    # ========== 验证检查点 ==========

    async def should_run_verify(self, ctx: ProfileContext) -> bool:
        """决定是否执行 verify checkpoint。

        StandardProfile: False。
        DeepResearchProfile: 按 mode + enable_verify 决定。
        """
        return False

    async def run_verify(self, ctx: ProfileContext) -> dict | None:
        """执行 verify checkpoint。

        仅在 should_run_verify 返回 True 时调用。
        返回 dict（verify 结果，供后续决策）或 None（无需处理）。
        """
        return None

    # ========== 任务规划 ==========

    async def create_task_plan(self, ctx: "ProfileContext") -> str | None:
        """创建任务分解 plan，返回注入 message_history 的文本；None 表示不注入。

        StandardProfile: None。
        DeepResearchProfile: 按配置使用 TaskPlanner。
        """
        return None

    # ========== Skill 注入 ==========

    async def should_process_inline_skills(self, ctx: "ProfileContext") -> bool:
        """是否处理 <next_skills> inline skill 选择。

        StandardProfile: False。
        DeepResearchProfile: True（quick 模式由 ctx.mode 判断）。
        """
        return False

    # ========== 最终答案 ==========

    async def needs_final_summary(
        self,
        tool_calls_executed: int,
        last_assistant_text: str,
        ctx: "ProfileContext",
    ) -> bool:
        """是否需要跑 SummaryHandler 生成最终 summary。

        StandardProfile: False（直接用 last_assistant_text）。
        DeepResearchProfile: mode=deep + tool_calls>0 时 True。
        """
        return False

    async def build_final_answer(
        self,
        last_assistant_text: str,
        message_history: list[dict],
        ctx: ProfileContext,
    ) -> str:
        """决定如何生成最终答案。

        StandardProfile: 直接返回 last_assistant_text。
        DeepResearchProfile: 有 tool 调用时跑 SummaryHandler 生成结构化 summary。
        """
        return last_assistant_text

    # ========== 配置 ==========

    @classmethod
    def default_config(cls) -> dict:
        """返回 profile 默认配置。"""
        return {}

    def validate_config(self, config: dict) -> dict:
        """验证并返回规范化的配置。无效时抛 ConfigValidationError。"""
        return config

    # ========== Snapshot (Phase 2 HITL 衔接) ==========

    def snapshot(self) -> dict:
        """返回 profile 内部状态（含 strategies），用于 checkpoint。"""
        return {
            "name": self.name,
            "strategies": {
                strat.name: strat.snapshot() for strat in self.extraction_strategies
            },
        }

    def restore(self, state: dict) -> None:
        """从 snapshot 恢复 profile 内部状态（含 strategies）。"""
        strategy_states = state.get("strategies", {})
        for strat in self.extraction_strategies:
            if strat.name in strategy_states:
                strat.restore(strategy_states[strat.name])
