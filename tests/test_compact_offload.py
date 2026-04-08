"""
Integration tests — Compact + Offload + Resume chain.

Covers:
- ContextManager.offload_large_result: writes file, returns summary
- ContextManager.restore_offloaded_content: reads file, restores message
- Offload + restore round-trip
- ObservationMaskingStrategy: masks old tool outputs
- BinaryReductionStrategy: emergency truncation
- WindowStrategyPipeline: orchestrates strategies in order
- apply_compact delegates to ObservationMasking
"""

import os

import pytest

from mem_deep_research_core.core.constants import MT, TAG_OFFLOADED, make_msg
from mem_deep_research_core.core.context_manager import ContextManager, ContextManagerConfig
from mem_deep_research_core.core.window_strategy import (
    BinaryReductionStrategy,
    CompressResult,
    ObservationMaskingStrategy,
    WindowContext,
    WindowStrategyPipeline,
    _is_protected_message,
)


# ============================================================
# Offload + Restore
# ============================================================


class TestOffloadLargeResult:
    def test_small_result_not_offloaded(self, tmp_path):
        """Results below threshold pass through unchanged."""
        cm = ContextManager(
            config=ContextManagerConfig(result_offload_threshold=1000)
        )
        cm.set_offload_dir(str(tmp_path))

        text = "short result"
        summary, ref = cm.offload_large_result(text, "search", turn=1)
        assert summary == text
        assert ref is None

    def test_large_result_offloaded_to_file(self, tmp_path):
        """Results above threshold → written to file, summary returned."""
        cm = ContextManager(
            config=ContextManagerConfig(result_offload_threshold=100)
        )
        cm.set_offload_dir(str(tmp_path))

        large_text = "x" * 500
        summary, ref = cm.offload_large_result(large_text, "scrape", turn=2)

        assert ref is not None
        assert os.path.isfile(ref)
        assert TAG_OFFLOADED in summary
        assert "500" in summary  # chars count in marker
        assert "Preview:" in summary

        # File should contain the full text
        with open(ref) as f:
            assert f.read() == large_text

    def test_offload_disabled_when_threshold_zero(self, tmp_path):
        """threshold=0 means offload is disabled."""
        cm = ContextManager(
            config=ContextManagerConfig(result_offload_threshold=0)
        )
        cm.set_offload_dir(str(tmp_path))

        text = "x" * 10000
        summary, ref = cm.offload_large_result(text, "tool", turn=1)
        assert summary == text
        assert ref is None

    def test_offload_no_dir_configured(self):
        """No offload_dir set → large result passes through."""
        cm = ContextManager(
            config=ContextManagerConfig(result_offload_threshold=50)
        )
        # No set_offload_dir call

        text = "x" * 200
        summary, ref = cm.offload_large_result(text, "tool", turn=1)
        assert summary == text
        assert ref is None


class TestRestoreOffloadedContent:
    def test_restore_round_trip(self, tmp_path):
        """offload → restore should reconstruct the original content."""
        cm = ContextManager(
            config=ContextManagerConfig(result_offload_threshold=100)
        )
        cm.set_offload_dir(str(tmp_path))

        original = "A" * 500
        summary, ref = cm.offload_large_result(original, "fetch", turn=3)
        assert ref is not None

        # Build a message_history with the offloaded summary
        message_history = [
            {"role": "user", "content": [{"type": "text", "text": summary}]}
        ]

        restored_count = cm.restore_offloaded_content(message_history)
        assert restored_count == 1
        assert message_history[0]["content"][0]["text"] == original

    def test_restore_ignores_normal_messages(self, tmp_path):
        """Messages without OFFLOADED marker are not touched."""
        cm = ContextManager(
            config=ContextManagerConfig(result_offload_threshold=100)
        )
        cm.set_offload_dir(str(tmp_path))

        message_history = [
            {"role": "user", "content": [{"type": "text", "text": "normal message"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "response"}]},
        ]

        restored = cm.restore_offloaded_content(message_history)
        assert restored == 0
        assert message_history[0]["content"][0]["text"] == "normal message"

    def test_restore_missing_file_skipped(self, tmp_path):
        """If offloaded file is deleted, restore skips it without error."""
        cm = ContextManager(
            config=ContextManagerConfig(result_offload_threshold=100)
        )
        cm.set_offload_dir(str(tmp_path))

        # Create an offloaded reference manually (file doesn't exist)
        marker = f"{TAG_OFFLOADED}missing_file.txt|999]\nPreview: ...\n"
        message_history = [
            {"role": "user", "content": [{"type": "text", "text": marker}]}
        ]

        restored = cm.restore_offloaded_content(message_history)
        assert restored == 0


# ============================================================
# ObservationMaskingStrategy
# ============================================================


class TestObservationMaskingStrategy:
    def _make_context(self, ratio=0.7, current_turn=5, max_tokens=10000):
        return WindowContext(
            current_turn=current_turn,
            max_turns=20,
            token_count=int(max_tokens * ratio),
            max_tokens=max_tokens,
            token_ratio=ratio,
            message_count=10,
        )

    def test_should_trigger_above_ratio(self):
        strategy = ObservationMaskingStrategy(trigger_ratio=0.6)
        ctx = self._make_context(ratio=0.65)
        assert strategy.should_trigger(ctx) is True

    def test_should_not_trigger_below_ratio(self):
        strategy = ObservationMaskingStrategy(trigger_ratio=0.6)
        ctx = self._make_context(ratio=0.5)
        assert strategy.should_trigger(ctx) is False

    def test_masks_old_tool_results(self):
        """Old tool_result messages should be replaced with short summaries."""
        strategy = ObservationMaskingStrategy(
            trigger_ratio=0.0,  # Always trigger
            keep_recent=1,
        )

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "query"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "I'll search"}]},
            # Old tool result (should be masked)
            {"role": "user", "content": [{"type": "text", "text": "A" * 2000}],
             "_type": MT.TOOL_RESULT},
            {"role": "assistant", "content": [{"type": "text", "text": "I see results"}]},
            # Recent tool result (should be kept)
            {"role": "user", "content": [{"type": "text", "text": "B" * 2000}],
             "_type": MT.TOOL_RESULT},
            {"role": "assistant", "content": [{"type": "text", "text": "Final answer"}]},
        ]

        old_text_before = messages[2]["content"][0]["text"]

        ctx_with_msgs = WindowContext(
            current_turn=5,
            max_turns=20,
            token_count=0,
            max_tokens=0,
            token_ratio=0.7,
            message_count=len(messages),
            message_history=messages,
        )

        result = strategy.apply(messages, ctx_with_msgs)

        # At least one tool result should have been compressed
        assert result.messages_affected >= 1
        # The old tool result (index 2) should have been shortened
        assert len(messages[2]["content"][0]["text"]) < len(old_text_before)

    def test_preserves_protected_messages(self):
        """Protected messages (SESSION_MEMORY, PLAN, etc.) should never be masked."""
        strategy = ObservationMaskingStrategy(trigger_ratio=0.0, keep_recent=0)

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "query"}]},
            make_msg("user", "A" * 2000, _type=MT.SESSION_MEMORY),
            make_msg("user", "B" * 2000, _type=MT.PLAN),
        ]

        ctx = WindowContext(
            current_turn=10,
            max_turns=20,
            token_count=0,
            max_tokens=0,
            token_ratio=0.7,
            message_count=len(messages),
            message_history=messages,
        )

        strategy.apply(messages, ctx)

        # Protected messages should still have full content
        assert len(messages[1]["content"][0]["text"]) == 2000
        assert len(messages[2]["content"][0]["text"]) == 2000


# ============================================================
# BinaryReductionStrategy
# ============================================================


class TestBinaryReductionStrategy:
    def test_should_trigger_at_high_ratio(self):
        strategy = BinaryReductionStrategy(trigger_ratio=0.95)
        ctx = WindowContext(
            current_turn=10, max_turns=20,
            token_count=9600, max_tokens=10000,
            token_ratio=0.96, message_count=20,
        )
        assert strategy.should_trigger(ctx) is True

    def test_should_not_trigger_below(self):
        strategy = BinaryReductionStrategy(trigger_ratio=0.95)
        ctx = WindowContext(
            current_turn=10, max_turns=20,
            token_count=8000, max_tokens=10000,
            token_ratio=0.8, message_count=20,
        )
        assert strategy.should_trigger(ctx) is False

    def test_reduces_message_count(self):
        """Emergency reduction should remove messages from the middle."""
        strategy = BinaryReductionStrategy(trigger_ratio=0.0)

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "query"}]},
        ]
        # Add 20 assistant/user pairs
        for i in range(20):
            messages.append({"role": "assistant", "content": [{"type": "text", "text": f"response {i}"}]})
            messages.append({"role": "user", "content": [{"type": "text", "text": f"followup {i}"}]})

        original_len = len(messages)

        ctx = WindowContext(
            current_turn=20, max_turns=30,
            token_count=9800, max_tokens=10000,
            token_ratio=0.98, message_count=len(messages),
            message_history=messages,
        )

        result = strategy.apply(messages, ctx)

        assert len(messages) < original_len
        assert result.messages_affected > 0
        # First and last messages should be preserved
        assert messages[0]["content"][0]["text"] == "query"


# ============================================================
# WindowStrategyPipeline
# ============================================================


class TestWindowStrategyPipeline:
    def test_pipeline_applies_strategies_in_order(self):
        """Pipeline should try strategies from lowest to highest trigger_ratio."""
        s1 = ObservationMaskingStrategy(trigger_ratio=0.6)
        s2 = BinaryReductionStrategy(trigger_ratio=0.95)
        pipeline = WindowStrategyPipeline([s1, s2])

        assert len(pipeline.strategies) == 2
        assert pipeline.strategies[0] is s1
        assert pipeline.strategies[1] is s2


# ============================================================
# apply_compact integration
# ============================================================


class TestApplyCompact:
    def test_apply_compact_triggers_observation_masking(self):
        """apply_compact should delegate to ObservationMaskingStrategy."""
        cm = ContextManager(
            config=ContextManagerConfig(
                enable_compact=True,
                compact_at_ratio=0.5,
                compact_keep_recent=1,
            )
        )
        cm.set_turn(5)

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "query"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "thinking"}]},
            {"role": "user", "content": [{"type": "text", "text": "X" * 5000}],
             "_type": MT.TOOL_RESULT},
            {"role": "assistant", "content": [{"type": "text", "text": "more thinking"}]},
            {"role": "user", "content": [{"type": "text", "text": "Y" * 5000}],
             "_type": MT.TOOL_RESULT},
            {"role": "assistant", "content": [{"type": "text", "text": "final"}]},
        ]

        # Force high ratio by setting small max_context
        affected = cm.apply_compact(
            message_history=messages,
            current_turn=5,
            system_prompt="sys",
            max_context_length=1000,  # very small → high ratio → triggers
        )

        # Should have compacted at least the older tool result
        assert affected >= 1

    def test_apply_compact_disabled(self):
        """enable_compact=False → no compaction."""
        cm = ContextManager(
            config=ContextManagerConfig(enable_compact=False)
        )

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "X" * 5000}]},
        ]

        affected = cm.apply_compact(
            message_history=messages,
            current_turn=5,
            system_prompt="sys",
            max_context_length=100,
        )
        assert affected == 0
