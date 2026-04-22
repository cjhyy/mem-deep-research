"""Memory Extraction Strategy Phase 2a 测试

验证：
1. Strategy ABC 默认行为
2. 三个内置 strategy 的抽取逻辑（evidence_tag / offload_evidence / summary_evidence）
3. Profile 的 run_strategies_on_* 方法串联调用
4. StandardProfile / DeepResearchProfile 的默认 strategies
5. extraction_strategies vs extraction_strategies_extra 配置
6. Strategy snapshot / restore（HITL 衔接）
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mem_deep_research_core.core.memory import EvidenceItem, SessionMemory
from mem_deep_research_core.core.profiles import (
    DeepResearchProfile,
    Profile,
    StandardProfile,
)
from mem_deep_research_core.memory_extraction import (
    EvidenceTagStrategy,
    ExtractionContext,
    MemoryExtractionStrategy,
    OffloadEvidenceStrategy,
    SummaryEvidenceStrategy,
    list_strategies,
    register_strategy,
    resolve_strategy,
)


def _make_ext_ctx(*, session_memory=None, context_manager=None, turn=1, mode="standard"):
    return ExtractionContext(
        turn_number=turn,
        task_description="test",
        mode=mode,
        session_memory=session_memory or SessionMemory(),
        context_manager=context_manager or MagicMock(),
        llm_client=None,
    )


# =========================================================
# 1. Strategy ABC 默认行为
# =========================================================


class TestStrategyDefaults:
    @pytest.mark.asyncio
    async def test_on_llm_response_passthrough(self):
        s = MemoryExtractionStrategy()
        assert await s.on_llm_response("hello", _make_ext_ctx()) == "hello"

    @pytest.mark.asyncio
    async def test_on_tool_result_noop(self):
        s = MemoryExtractionStrategy()
        assert await s.on_tool_result("t", "r", _make_ext_ctx()) is None

    @pytest.mark.asyncio
    async def test_on_compact_noop(self):
        s = MemoryExtractionStrategy()
        assert await s.on_compact("summary", 5, _make_ext_ctx()) is None

    @pytest.mark.asyncio
    async def test_on_offload_noop(self):
        s = MemoryExtractionStrategy()
        assert await s.on_offload("ref.txt", "tool", "content", _make_ext_ctx()) is None

    def test_snapshot_empty(self):
        assert MemoryExtractionStrategy().snapshot() == {}

    def test_restore_noop(self):
        MemoryExtractionStrategy().restore({"any": "state"})


# =========================================================
# 2. EvidenceTagStrategy
# =========================================================


class TestEvidenceTagStrategy:
    @pytest.mark.asyncio
    async def test_extract_line_format_with_source_confidence(self):
        sm = SessionMemory()
        strat = EvidenceTagStrategy()
        text = (
            "Analysis follows.\n"
            "<evidence>\n"
            "- alice's revenue is $5M (source: https://example.com/report) (confidence: high)\n"
            "- bob's revenue is $3M (source: https://example.com/db)\n"
            "</evidence>\n"
            "End of response."
        )
        result = await strat.on_llm_response(text, _make_ext_ctx(session_memory=sm, turn=3))
        # Tag 不由 strategy 清理
        assert "<evidence>" in result
        assert len(sm.evidence_items) == 2
        assert sm.evidence_items[0].summary.startswith("alice")
        assert sm.evidence_items[0].source_url == "https://example.com/report"
        assert sm.evidence_items[0].confidence == "high"
        assert sm.evidence_items[0].turn == 3

    @pytest.mark.asyncio
    async def test_extract_legacy_block_format(self):
        sm = SessionMemory()
        strat = EvidenceTagStrategy()
        text = "<evidence>This is a free-form evidence block without list</evidence>"
        await strat.on_llm_response(text, _make_ext_ctx(session_memory=sm, turn=2))
        assert len(sm.evidence_items) == 1
        assert sm.evidence_items[0].summary.startswith("This is a free-form")

    @pytest.mark.asyncio
    async def test_no_tag_no_extraction(self):
        sm = SessionMemory()
        strat = EvidenceTagStrategy()
        await strat.on_llm_response("plain response", _make_ext_ctx(session_memory=sm))
        assert len(sm.evidence_items) == 0

    @pytest.mark.asyncio
    async def test_empty_text_safe(self):
        sm = SessionMemory()
        strat = EvidenceTagStrategy()
        assert await strat.on_llm_response("", _make_ext_ctx(session_memory=sm)) == ""


# =========================================================
# 3. OffloadEvidenceStrategy
# =========================================================


class TestOffloadEvidenceStrategy:
    @pytest.mark.asyncio
    async def test_bind_to_offload_registry(self):
        cm = MagicMock()
        strat = OffloadEvidenceStrategy()
        text = (
            'Analysis...\n'
            '<offload_evidence ref="toolmsg_abc.txt">\n'
            '- alice revenue $5M\n'
            '- bob revenue $3M\n'
            '</offload_evidence>\n'
            'Done.'
        )
        result = await strat.on_llm_response(text, _make_ext_ctx(context_manager=cm))
        # Tag 保留（runtime 清理）
        assert "<offload_evidence" in result
        cm.update_offload_evidence.assert_called_once_with(
            "toolmsg_abc.txt", ["alice revenue $5M", "bob revenue $3M"]
        )

    @pytest.mark.asyncio
    async def test_no_tag_no_call(self):
        cm = MagicMock()
        strat = OffloadEvidenceStrategy()
        await strat.on_llm_response("plain text", _make_ext_ctx(context_manager=cm))
        cm.update_offload_evidence.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_offload_blocks(self):
        cm = MagicMock()
        strat = OffloadEvidenceStrategy()
        text = (
            '<offload_evidence ref="a.txt">- line a1\n- line a2</offload_evidence>\n'
            '<offload_evidence ref="b.txt">- line b1</offload_evidence>'
        )
        await strat.on_llm_response(text, _make_ext_ctx(context_manager=cm))
        assert cm.update_offload_evidence.call_count == 2


# =========================================================
# 4. SummaryEvidenceStrategy
# =========================================================


class TestSummaryEvidenceStrategy:
    @pytest.mark.asyncio
    async def test_extract_evidence_section(self):
        sm = SessionMemory()
        strat = SummaryEvidenceStrategy()
        summary = (
            "# Task Summary\n\n"
            "## Context\n\n"
            "Some context here.\n\n"
            "## Evidence\n\n"
            "- alice data\n- bob data\n\n"
            "## Next Steps\n\n"
            "..."
        )
        await strat.on_compact(summary, up_to_turn=5, ctx=_make_ext_ctx(session_memory=sm))
        assert len(sm.evidence_items) == 1
        assert "alice data" in sm.evidence_items[0].summary
        assert "## Next Steps" not in sm.evidence_items[0].summary
        assert sm.evidence_items[0].turn == 5
        assert sm.evidence_items[0].tool_name == "llm_summarize"

    @pytest.mark.asyncio
    async def test_no_evidence_section_no_extraction(self):
        sm = SessionMemory()
        strat = SummaryEvidenceStrategy()
        summary = "## Context\n\nSome text\n"
        await strat.on_compact(summary, up_to_turn=3, ctx=_make_ext_ctx(session_memory=sm))
        assert len(sm.evidence_items) == 0

    @pytest.mark.asyncio
    async def test_empty_summary_safe(self):
        sm = SessionMemory()
        strat = SummaryEvidenceStrategy()
        await strat.on_compact("", up_to_turn=1, ctx=_make_ext_ctx(session_memory=sm))
        assert len(sm.evidence_items) == 0


# =========================================================
# 5. Profile.run_strategies_on_* 串联调用
# =========================================================


class _CountingStrategy(MemoryExtractionStrategy):
    """记录所有调用的 strategy，用于验证链式。"""

    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.calls = []

    async def on_llm_response(self, text, ctx):
        self.calls.append(("llm", text))
        return text + f"[{self.tag}]"  # 显式修改 text 验证串联

    async def on_compact(self, summary, up_to_turn, ctx):
        self.calls.append(("compact", summary, up_to_turn))


class TestProfileStrategyChain:
    @pytest.mark.asyncio
    async def test_run_on_llm_response_in_list_order(self):
        """按 list 顺序串联 on_llm_response。"""
        s1 = _CountingStrategy("s1")
        s2 = _CountingStrategy("s2")

        class P(Profile):
            name = "p"
            default_extraction_strategies = [s1, s2]

        p = P()
        result = await p.run_strategies_on_llm_response("text", _make_ext_ctx())
        assert result == "text[s1][s2]"
        assert s1.calls == [("llm", "text")]
        assert s2.calls == [("llm", "text[s1]")]

    @pytest.mark.asyncio
    async def test_run_on_compact_fan_out(self):
        """on_compact 调所有 strategy，不串联返回值。"""
        s1 = _CountingStrategy("s1")
        s2 = _CountingStrategy("s2")

        class P(Profile):
            name = "p"
            default_extraction_strategies = [s1, s2]

        await P().run_strategies_on_compact("summary", 3, _make_ext_ctx())
        assert ("compact", "summary", 3) in s1.calls
        assert ("compact", "summary", 3) in s2.calls


# =========================================================
# 6. StandardProfile / DeepResearchProfile 默认 strategies
# =========================================================


class TestProfileDefaults:
    def test_standard_profile_has_base_strategies(self):
        sp = StandardProfile()
        names = [s.name for s in sp.extraction_strategies]
        assert "offload_evidence" in names
        assert "summary_evidence" in names
        assert "evidence_tag" not in names

    def test_deep_research_profile_has_base_plus_evidence_tag(self):
        dp = DeepResearchProfile()
        names = [s.name for s in dp.extraction_strategies]
        assert "offload_evidence" in names
        assert "summary_evidence" in names
        assert "evidence_tag" in names

    def test_deep_research_profile_is_superset_of_standard(self):
        sp_names = {s.name for s in StandardProfile().extraction_strategies}
        dp_names = {s.name for s in DeepResearchProfile().extraction_strategies}
        assert sp_names.issubset(dp_names)


# =========================================================
# 7. Config: extraction_strategies vs extraction_strategies_extra
# =========================================================


class TestProfileConfig:
    def test_extraction_strategies_full_override(self):
        custom = _CountingStrategy("custom")
        sp = StandardProfile(config={"extraction_strategies": [custom]})
        assert sp.extraction_strategies == [custom]
        # 默认 strategies 被完全覆盖
        assert not any(s.name == "offload_evidence" for s in sp.extraction_strategies)

    def test_extraction_strategies_extra_appends(self):
        custom = _CountingStrategy("custom")
        sp = StandardProfile(config={"extraction_strategies_extra": [custom]})
        names = [s.name for s in sp.extraction_strategies]
        # 保留默认 + 追加
        assert "offload_evidence" in names
        assert "summary_evidence" in names
        assert custom in sp.extraction_strategies
        assert sp.extraction_strategies[-1] is custom

    def test_empty_override_disables_all(self):
        sp = StandardProfile(config={"extraction_strategies": []})
        assert sp.extraction_strategies == []


# =========================================================
# 8. Snapshot / Restore (HITL 衔接)
# =========================================================


class TestProfileSnapshot:
    def test_snapshot_contains_strategies_dict(self):
        dp = DeepResearchProfile()
        snap = dp.snapshot()
        assert snap["name"] == "deep_research"
        assert set(snap["strategies"].keys()) == {
            "offload_evidence", "summary_evidence", "evidence_tag",
        }

    def test_restore_delegates_to_strategies(self):
        class StatefulStrategy(MemoryExtractionStrategy):
            name = "stateful"
            def __init__(self):
                super().__init__()
                self.counter = 0
            def snapshot(self):
                return {"counter": self.counter}
            def restore(self, state):
                self.counter = state.get("counter", 0)

        strat = StatefulStrategy()
        strat.counter = 42

        class P(Profile):
            name = "p"
            default_extraction_strategies = [strat]

        p = P()
        snap = p.snapshot()
        assert snap["strategies"]["stateful"] == {"counter": 42}

        # Restore 到新实例
        strat2 = StatefulStrategy()
        class P2(Profile):
            name = "p"
            default_extraction_strategies = [strat2]
        p2 = P2()
        p2.restore(snap)
        assert strat2.counter == 42


# =========================================================
# 9. Registry
# =========================================================


class TestStrategyRegistry:
    def test_builtin_strategies_registered(self):
        names = list_strategies()
        assert "evidence_tag" in names
        assert "offload_evidence" in names
        assert "summary_evidence" in names

    def test_resolve_by_name(self):
        s = resolve_strategy("evidence_tag")
        assert isinstance(s, EvidenceTagStrategy)

    def test_resolve_instance(self):
        inst = EvidenceTagStrategy()
        assert resolve_strategy(inst) is inst

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            resolve_strategy("nonexistent")

    def test_register_custom(self):
        class MyS(MemoryExtractionStrategy):
            name = "test_my_s"

        register_strategy(MyS)
        try:
            s = resolve_strategy("test_my_s")
            assert isinstance(s, MyS)
        finally:
            from mem_deep_research_core.memory_extraction import _STRATEGY_REGISTRY
            _STRATEGY_REGISTRY.pop("test_my_s", None)


# =========================================================
# 10. 行为等价：原 evidence 抽取逻辑对应 strategy 的等价产出
# =========================================================


class TestBehaviorEquivalence:
    @pytest.mark.asyncio
    async def test_evidence_tag_equivalent_to_old_extract(self):
        """EvidenceTagStrategy 产生的 EvidenceItem 列表等价于原 _extract_evidence_tags。"""
        from mem_deep_research_core.core.main_loop import _extract_evidence_tags

        sm_old = SessionMemory()
        sm_new = SessionMemory()
        text = (
            "<evidence>\n"
            "- fact 1 (source: https://a.com) (confidence: high)\n"
            "- fact 2\n"
            "</evidence>"
        )
        # 旧路径
        _extract_evidence_tags(text, 5, sm_old)
        # 新路径
        strat = EvidenceTagStrategy()
        await strat.on_llm_response(text, _make_ext_ctx(session_memory=sm_new, turn=5))

        assert len(sm_old.evidence_items) == len(sm_new.evidence_items)
        for a, b in zip(sm_old.evidence_items, sm_new.evidence_items):
            assert a.summary == b.summary
            assert a.source_url == b.source_url
            assert a.confidence == b.confidence
            assert a.turn == b.turn

    @pytest.mark.asyncio
    async def test_offload_evidence_equivalent(self):
        """OffloadEvidenceStrategy 的 update_offload_evidence 调用等价于原 _extract_offload_evidence。"""
        from mem_deep_research_core.core.main_loop import _extract_offload_evidence

        cm_old = MagicMock()
        cm_new = MagicMock()
        text = '<offload_evidence ref="x.txt">- a\n- b</offload_evidence>'

        _extract_offload_evidence(text, cm_old)
        strat = OffloadEvidenceStrategy()
        await strat.on_llm_response(text, _make_ext_ctx(context_manager=cm_new))

        assert cm_old.update_offload_evidence.call_args_list == cm_new.update_offload_evidence.call_args_list
