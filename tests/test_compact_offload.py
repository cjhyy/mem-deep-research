"""
Integration tests — Compact + Offload + Resume chain.

Covers:
- ContextManager.backup_large_result: writes file, returns logical ref
- Sliding window offload: finalize_offload_candidates replaces old results
- Evidence inlined in OFFLOADED markers
- restore_single_file: reads backed-up content by ref
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


class TestBackupLargeResult:
    def test_small_result_not_backed_up(self, tmp_path):
        """Results below threshold → no backup, returns None."""
        cm = ContextManager(
            config=ContextManagerConfig(result_offload_threshold=1000)
        )
        cm.set_offload_dir(str(tmp_path))

        ref = cm.backup_large_result("short result", "search", turn=1)
        assert ref is None

    def test_large_result_backed_up_to_file(self, tmp_path):
        """Results above threshold → written to file, returns ref."""
        cm = ContextManager(
            config=ContextManagerConfig(result_offload_threshold=100)
        )
        cm.set_offload_dir(str(tmp_path))

        large_text = "x" * 500
        ref = cm.backup_large_result(large_text, "scrape", turn=2)

        assert ref is not None
        assert ref.startswith("toolmsg_")
        assert ref.endswith(".txt")

        # File should contain the full text
        file_path = os.path.join(str(tmp_path), ref)
        assert os.path.isfile(file_path)
        with open(file_path) as f:
            assert f.read() == large_text

        # Registry should have a record
        assert ref in cm._offload_registry
        assert cm._offload_registry[ref].state == "backed_up"
        assert cm._offload_registry[ref].char_count == 500

    def test_backup_disabled_when_threshold_zero(self, tmp_path):
        """threshold=0 means backup is disabled."""
        cm = ContextManager(
            config=ContextManagerConfig(result_offload_threshold=0)
        )
        cm.set_offload_dir(str(tmp_path))

        ref = cm.backup_large_result("x" * 10000, "tool", turn=1)
        assert ref is None

    def test_backup_no_dir_configured(self):
        """No offload_dir set → no backup."""
        cm = ContextManager(
            config=ContextManagerConfig(result_offload_threshold=50)
        )
        ref = cm.backup_large_result("x" * 200, "tool", turn=1)
        assert ref is None


class TestSlidingWindowOffload:
    def test_recent_turns_kept_intact(self, tmp_path):
        """Messages within keep_recent window should NOT be offloaded."""
        cm = ContextManager(
            config=ContextManagerConfig(
                result_offload_threshold=100,
                compact_keep_recent=2,
            )
        )
        cm.set_offload_dir(str(tmp_path))

        ref = cm.backup_large_result("A" * 500, "search", turn=1)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "query"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "thinking"}]},
            {"role": "user", "content": [{"type": "text", "text": "A" * 500}],
             "_type": MT.TOOL_RESULT, "_offload_refs": [ref]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]

        # current_turn=2, keep_recent=2: turn 1 is within window
        replaced = cm.finalize_offload_candidates(messages, current_turn=2, keep_recent=2)
        assert replaced == 0
        assert "A" * 500 == messages[2]["content"][0]["text"]

    def test_old_turns_offloaded_with_marker(self, tmp_path):
        """Messages outside keep_recent window should be replaced with OFFLOADED marker."""
        cm = ContextManager(
            config=ContextManagerConfig(
                result_offload_threshold=100,
                compact_keep_recent=1,
            )
        )
        cm.set_offload_dir(str(tmp_path))

        ref = cm.backup_large_result("B" * 500, "fetch", turn=1)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "query"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "turn 1"}]},
            {"role": "user", "content": [{"type": "text", "text": "B" * 500}],
             "_type": MT.TOOL_RESULT, "_offload_refs": [ref]},
            {"role": "assistant", "content": [{"type": "text", "text": "turn 2"}]},
            {"role": "user", "content": [{"type": "text", "text": "recent result"}],
             "_type": MT.TOOL_RESULT},
        ]

        # current_turn=3, keep_recent=1: only turn 3 is kept, turn 1 is old
        replaced = cm.finalize_offload_candidates(messages, current_turn=3, keep_recent=1)
        assert replaced == 1
        assert TAG_OFFLOADED in messages[2]["content"][0]["text"]
        assert ref in messages[2]["content"][0]["text"]
        assert messages[2]["_type"] == MT.OFFLOADED
        assert cm._offload_registry[ref].state == "offloaded"

    def test_evidence_inlined_in_marker(self, tmp_path):
        """OFFLOADED marker should inline evidence if available."""
        cm = ContextManager(
            config=ContextManagerConfig(
                result_offload_threshold=100,
                compact_keep_recent=1,
            )
        )
        cm.set_offload_dir(str(tmp_path))

        ref = cm.backup_large_result("C" * 500, "search", turn=1)
        cm.update_offload_evidence(ref, ["key fact 1", "key fact 2"])

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "query"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "turn 1"}]},
            {"role": "user", "content": [{"type": "text", "text": "C" * 500}],
             "_type": MT.TOOL_RESULT, "_offload_refs": [ref]},
            {"role": "assistant", "content": [{"type": "text", "text": "turn 2"}]},
            {"role": "user", "content": [{"type": "text", "text": "recent"}],
             "_type": MT.TOOL_RESULT},
        ]

        cm.finalize_offload_candidates(messages, current_turn=3, keep_recent=1)
        marker_text = messages[2]["content"][0]["text"]
        assert "Evidence:" in marker_text
        assert "key fact 1" in marker_text
        assert "key fact 2" in marker_text
        assert f'read_result("{ref}")' in marker_text

    def test_read_result_restores_backed_up_content(self, tmp_path):
        """read_result(ref) should be able to restore backed-up content."""
        cm = ContextManager(
            config=ContextManagerConfig(result_offload_threshold=100)
        )
        cm.set_offload_dir(str(tmp_path))

        original = "D" * 500
        ref = cm.backup_large_result(original, "fetch", turn=1)
        assert ref is not None

        content = cm.restore_single_file(ref)
        assert content == original

    def test_no_backup_means_no_finalize(self, tmp_path):
        """Messages without _offload_refs should not be touched by finalize."""
        cm = ContextManager(
            config=ContextManagerConfig(compact_keep_recent=1)
        )

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "query"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "turn 1"}]},
            {"role": "user", "content": [{"type": "text", "text": "small result"}],
             "_type": MT.TOOL_RESULT},
            {"role": "assistant", "content": [{"type": "text", "text": "turn 2"}]},
        ]

        replaced = cm.finalize_offload_candidates(messages, current_turn=5, keep_recent=1)
        assert replaced == 0
        assert messages[2]["content"][0]["text"] == "small result"


class TestRestoreOffloadedContent:
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

        marker = f"{TAG_OFFLOADED}missing_file.txt|999]\nPreview: ...\n"
        message_history = [
            {"role": "user", "content": [{"type": "text", "text": marker}]}
        ]

        restored = cm.restore_offloaded_content(message_history)
        assert restored == 0

    def test_registry_rebuilt_after_restore(self, tmp_path):
        """After restore, _offload_registry should be rebuilt from message_history."""
        cm = ContextManager(
            config=ContextManagerConfig(result_offload_threshold=100)
        )
        cm.set_offload_dir(str(tmp_path))

        # Backup a large result (populates registry + writes file)
        original = "E" * 500
        ref = cm.backup_large_result(original, "search", turn=1)
        assert ref is not None

        # Build message_history with an offloaded marker and a restored message
        marker = f"{TAG_OFFLOADED}{ref}|500]\nFull content: read_result(\"{ref}\")"
        message_history = [
            {"role": "user", "content": [{"type": "text", "text": "query"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "turn 1"}]},
            # This message still has _offload_refs and will be restored
            {"role": "user", "content": [{"type": "text", "text": marker}],
             "_offload_refs": [ref]},
        ]

        # Clear registry to simulate resume (fresh ContextManager)
        cm._offload_registry.clear()
        assert len(cm._offload_registry) == 0

        cm.restore_offloaded_content(message_history)

        # Registry should have been rebuilt
        assert ref in cm._offload_registry
        assert cm._offload_registry[ref].state == "backed_up"

    def test_registry_rebuilt_for_unrestorable_markers(self, tmp_path):
        """OFFLOADED markers that can't be restored still get registry entries."""
        cm = ContextManager(
            config=ContextManagerConfig(result_offload_threshold=100)
        )
        cm.set_offload_dir(str(tmp_path))

        # Marker for a file that doesn't exist on disk
        ref = "toolmsg_deadbeef.txt"
        marker = f"{TAG_OFFLOADED}{ref}|1234]\nEvidence:\n- key fact"
        message_history = [
            {"role": "user", "content": [{"type": "text", "text": marker}]},
        ]

        cm.restore_offloaded_content(message_history)

        # Registry should still be rebuilt from the marker metadata
        assert ref in cm._offload_registry
        assert cm._offload_registry[ref].state == "offloaded"
        assert cm._offload_registry[ref].char_count == 1234


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
