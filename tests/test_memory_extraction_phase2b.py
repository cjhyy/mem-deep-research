"""Phase 2b 测试：on_tool_result / on_offload 触发点 + 2 个可选 strategy"""

from unittest.mock import MagicMock

import pytest

from mem_deep_research_core.core.memory import SessionMemory
from mem_deep_research_core.memory_extraction import (
    ExtractionContext,
    FactExtractionStrategy,
    SummarizeOnCompactStrategy,
    list_strategies,
    resolve_strategy,
)


def _ext_ctx(*, session_memory=None, turn=1, task="test"):
    return ExtractionContext(
        turn_number=turn,
        task_description=task,
        mode="standard",
        session_memory=session_memory or SessionMemory(),
        context_manager=MagicMock(),
        llm_client=None,
    )


# =========================================================
# Registry
# =========================================================


class TestRegistry:
    def test_all_optional_strategies_registered(self):
        names = list_strategies()
        assert "fact_extraction" in names
        assert "summarize_on_compact" in names

    def test_resolve_each(self):
        assert isinstance(resolve_strategy("fact_extraction"), FactExtractionStrategy)
        assert isinstance(resolve_strategy("summarize_on_compact"), SummarizeOnCompactStrategy)


# =========================================================
# FactExtractionStrategy
# =========================================================


class _FakeExtractor:
    """Mock 轻量 LLM extractor。按预设 response 返回。"""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[str] = []

    async def extract(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.response


class TestFactExtractionStrategy:
    @pytest.mark.asyncio
    async def test_no_extractor_no_op(self):
        """未提供 extractor 时不抽取。"""
        sm = SessionMemory()
        strat = FactExtractionStrategy()
        big_result = "x" * 5000
        await strat.on_tool_result("search", big_result, _ext_ctx(session_memory=sm))
        assert len(sm.evidence_items) == 0

    @pytest.mark.asyncio
    async def test_below_min_size_skipped(self):
        """小于 min_result_size 的结果不抽取。"""
        sm = SessionMemory()
        strat = FactExtractionStrategy(
            extractor_llm_client=_FakeExtractor("- fact 1\n- fact 2"),
            min_result_size=1000,
        )
        await strat.on_tool_result("search", "short", _ext_ctx(session_memory=sm))
        assert len(sm.evidence_items) == 0

    @pytest.mark.asyncio
    async def test_extract_facts(self):
        sm = SessionMemory()
        extractor = _FakeExtractor("Here are facts:\n- alice revenue 5M\n- bob revenue 3M\n- charlie 2M")
        strat = FactExtractionStrategy(
            extractor_llm_client=extractor,
            min_result_size=100,
            max_facts_per_result=3,
        )
        big_result = "x" * 500
        await strat.on_tool_result("search", big_result, _ext_ctx(session_memory=sm, turn=5))
        assert len(sm.evidence_items) == 3
        assert sm.evidence_items[0].summary == "alice revenue 5M"
        assert sm.evidence_items[0].tool_name == "search"
        assert sm.evidence_items[0].turn == 5

    @pytest.mark.asyncio
    async def test_dedup_same_content(self):
        """相同 (tool, content) 只抽一次。"""
        sm = SessionMemory()
        extractor = _FakeExtractor("- fact 1")
        strat = FactExtractionStrategy(
            extractor_llm_client=extractor,
            min_result_size=10,
        )
        content = "x" * 500
        await strat.on_tool_result("search", content, _ext_ctx(session_memory=sm))
        await strat.on_tool_result("search", content, _ext_ctx(session_memory=sm))
        assert len(extractor.calls) == 1  # 第二次被去重

    @pytest.mark.asyncio
    async def test_snapshot_restore(self):
        strat1 = FactExtractionStrategy(
            extractor_llm_client=_FakeExtractor("- fact"),
            min_result_size=10,
        )
        sm = SessionMemory()
        await strat1.on_tool_result("t", "x" * 100, _ext_ctx(session_memory=sm))
        snap = strat1.snapshot()
        assert len(snap["processed"]) == 1

        strat2 = FactExtractionStrategy(
            extractor_llm_client=_FakeExtractor("- another"),
            min_result_size=10,
        )
        strat2.restore(snap)
        # 新实例看到相同 content 不应再抽
        await strat2.on_tool_result("t", "x" * 100, _ext_ctx(session_memory=sm))
        assert strat2.extractor.calls == []

    @pytest.mark.asyncio
    async def test_extractor_error_logged(self):
        """extractor 抛错时不传播，只 warning。"""
        class FailingExtractor:
            async def extract(self, prompt):
                raise RuntimeError("extractor down")

        sm = SessionMemory()
        strat = FactExtractionStrategy(
            extractor_llm_client=FailingExtractor(),
            min_result_size=10,
        )
        # 不应该抛异常
        await strat.on_tool_result("t", "x" * 100, _ext_ctx(session_memory=sm))
        assert len(sm.evidence_items) == 0


# =========================================================
# SummarizeOnCompactStrategy
# =========================================================


class TestSummarizeOnCompactStrategy:
    @pytest.mark.asyncio
    async def test_anchor_full_summary(self):
        sm = SessionMemory()
        strat = SummarizeOnCompactStrategy()
        summary = "Multi-turn summary with ## Evidence section and more content"
        await strat.on_compact(summary, up_to_turn=7, ctx=_ext_ctx(session_memory=sm))
        assert len(sm.evidence_items) == 1
        assert sm.evidence_items[0].tool_name == "compact_anchor"
        assert sm.evidence_items[0].turn == 7
        assert sm.evidence_items[0].summary == summary.strip()

    @pytest.mark.asyncio
    async def test_truncate_long_summary(self):
        sm = SessionMemory()
        strat = SummarizeOnCompactStrategy(max_anchor_chars=20)
        await strat.on_compact("a" * 100, up_to_turn=1, ctx=_ext_ctx(session_memory=sm))
        assert len(sm.evidence_items) == 1
        assert sm.evidence_items[0].summary.endswith("...")
        assert len(sm.evidence_items[0].summary) <= 25  # 20 + "..."

    @pytest.mark.asyncio
    async def test_empty_summary_noop(self):
        sm = SessionMemory()
        strat = SummarizeOnCompactStrategy()
        await strat.on_compact("", up_to_turn=1, ctx=_ext_ctx(session_memory=sm))
        assert len(sm.evidence_items) == 0
