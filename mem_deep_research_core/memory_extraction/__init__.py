"""Memory Extraction Strategy 系统

Strategy 是 "长任务细节保鲜" 的可插拔扩展机制。

使用：
    from mem_deep_research_core.memory_extraction import (
        MemoryExtractionStrategy,
        EvidenceTagStrategy,
        OffloadEvidenceStrategy,
        SummaryEvidenceStrategy,
    )

    # 内置 strategy 已通过 Profile 默认启用，无需手动配置
    # 自定义 strategy：
    class MyStrategy(MemoryExtractionStrategy):
        name = "my_strategy"
        async def on_llm_response(self, text, ctx):
            ...
            return text

    dr = DeepResearch(
        profile="standard",
        profile_config={"extraction_strategies_extra": [MyStrategy()]},
    )
"""

from mem_deep_research_core.memory_extraction.base import (
    ExtractionContext,
    MemoryExtractionStrategy,
)
from mem_deep_research_core.memory_extraction.evidence_tag import EvidenceTagStrategy
from mem_deep_research_core.memory_extraction.fact_extraction import FactExtractionStrategy
from mem_deep_research_core.memory_extraction.offload_evidence import OffloadEvidenceStrategy
from mem_deep_research_core.memory_extraction.summarize_on_compact import (
    SummarizeOnCompactStrategy,
)
from mem_deep_research_core.memory_extraction.summary_evidence import SummaryEvidenceStrategy

# Registry: name → Strategy class (用于 YAML 配置解析)
_STRATEGY_REGISTRY: dict[str, type[MemoryExtractionStrategy]] = {
    "evidence_tag": EvidenceTagStrategy,
    "offload_evidence": OffloadEvidenceStrategy,
    "summary_evidence": SummaryEvidenceStrategy,
    "fact_extraction": FactExtractionStrategy,
    "summarize_on_compact": SummarizeOnCompactStrategy,
}


def register_strategy(cls: type[MemoryExtractionStrategy]) -> None:
    """注册自定义 strategy 类到全局 registry。

    Strategy 类必须设置 name 属性。重名会覆盖。
    """
    if not issubclass(cls, MemoryExtractionStrategy):
        raise TypeError(f"{cls!r} must be a MemoryExtractionStrategy subclass")
    name = getattr(cls, "name", None)
    if not name or name == "base":
        raise ValueError(
            f"Strategy class {cls!r} must define a non-empty 'name' attribute"
        )
    _STRATEGY_REGISTRY[name] = cls


def resolve_strategy(
    strategy: str | MemoryExtractionStrategy | type[MemoryExtractionStrategy],
    config: dict | None = None,
) -> MemoryExtractionStrategy:
    """将 strategy 参数解析为实例。

    接受：
    - 字符串 → 从 registry 查找对应 class 并实例化
    - Strategy class → 实例化
    - Strategy instance → 直接返回
    """
    if isinstance(strategy, MemoryExtractionStrategy):
        return strategy

    if isinstance(strategy, type) and issubclass(strategy, MemoryExtractionStrategy):
        return _build_from_class(strategy, config)

    if isinstance(strategy, str):
        if strategy not in _STRATEGY_REGISTRY:
            available = sorted(_STRATEGY_REGISTRY.keys())
            raise ValueError(
                f"Unknown strategy name {strategy!r}. Available: {available}"
            )
        return _build_from_class(_STRATEGY_REGISTRY[strategy], config)

    raise TypeError(
        f"strategy must be str, instance, or subclass; got {type(strategy)!r}"
    )


def _build_from_class(
    cls: type[MemoryExtractionStrategy],
    config: dict | None,
) -> MemoryExtractionStrategy:
    """用 config 构造 strategy，失败时无参构造。"""
    if config:
        try:
            return cls(**config)  # type: ignore[call-arg]
        except TypeError:
            return cls()
    return cls()


def list_strategies() -> list[str]:
    """返回所有已注册的 strategy 名称（按名称排序）。"""
    return sorted(_STRATEGY_REGISTRY.keys())


__all__ = [
    "ExtractionContext",
    "MemoryExtractionStrategy",
    "EvidenceTagStrategy",
    "OffloadEvidenceStrategy",
    "SummaryEvidenceStrategy",
    "FactExtractionStrategy",
    "SummarizeOnCompactStrategy",
    "register_strategy",
    "resolve_strategy",
    "list_strategies",
]
