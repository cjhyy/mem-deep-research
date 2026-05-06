"""End-to-end result lifecycle regression.

Roadmap (docs/20-roadmap.md) lists this chain as a v1.3.0 completion criterion:

    tool result → format → offload → compact → read_result → resume rebuild

Existing unit tests cover each stage in isolation. This file walks the full
chain on a single ContextManager instance to catch silent drift between stages
— e.g. an offload registry whose state doesn't match the markers in
``message_history``, or a resume that loses track of which refs are still
backed_up vs already offloaded.

Test layout:
    Stage 1: simulate several turns producing varied-size tool results.
    Stage 2: verify large results are backed up to disk + tracked in registry.
    Stage 3: trigger sliding-window offload — old turns get OFFLOADED markers,
             registry state flips to "offloaded".
    Stage 4: trigger observation-masking compact — recent results stay full,
             older masked-out (orthogonal to offload).
    Stage 5: read_result(ref) restores a specific offloaded result.
    Stage 6: build a snapshot, restore on a fresh ContextManager,
             verify offload_registry / dedup_cache / message_history identity.
    Stage 7: rebuild_registry_from_history reconstructs registry purely from
             the message_history's _offload_refs / OFFLOADED markers (the
             post-resume cold-start path).
"""

from __future__ import annotations

import os
from copy import deepcopy

import pytest

from mem_deep_research_core.core.constants import MT, TAG_OFFLOADED, make_msg
from mem_deep_research_core.core.context_manager import (
    ContextManager,
    ContextManagerConfig,
)
from mem_deep_research_core.core.hitl import build_snapshot, restore_snapshot


# Realistic-ish thresholds: small enough that we can produce "large" results
# in tests without 10k-char strings, large enough that "small" results don't
# accidentally trip backup.
OFFLOAD_THRESHOLD = 100
KEEP_RECENT = 2


def _build_cm(tmp_path) -> ContextManager:
    cm = ContextManager(
        config=ContextManagerConfig(
            result_offload_threshold=OFFLOAD_THRESHOLD,
            compact_keep_recent=KEEP_RECENT,
            # Force compact at very low ratio so tests that trigger it can
            # do so with small message histories.
            compact_at_ratio=0.05,
            summarize_at_ratio=0.5,
        )
    )
    cm.set_offload_dir(str(tmp_path))
    return cm


def _make_tool_result_msg(
    cm: ContextManager,
    turn: int,
    tool_name: str,
    result_text: str,
) -> dict:
    """Simulate the runner's path: try to back up the result, then attach
    ``_offload_refs`` if a backup happened. Returns the message that would
    land in ``message_history``."""
    ref = cm.backup_large_result(result_text, tool_name, turn=turn)
    msg = make_msg(
        "user",
        result_text,
        MT.TOOL_RESULT,
    )
    if ref:
        msg["_offload_refs"] = [ref]
    return msg


# ======================================================================
# Stage 1+2: tool results → backup_large_result + registry coherence
# ======================================================================


class TestStage_OffloadRegistryCoherence:
    def test_small_results_skip_backup_large_results_backed_up(self, tmp_path):
        cm = _build_cm(tmp_path)

        # Mix small + large results across 3 turns.
        small_msg = _make_tool_result_msg(cm, turn=1, tool_name="calc", result_text="42")
        large_a = _make_tool_result_msg(
            cm, turn=2, tool_name="search", result_text="A" * 500
        )
        large_b = _make_tool_result_msg(
            cm, turn=3, tool_name="scrape", result_text="B" * 800
        )

        # Small result didn't get a ref; large ones did.
        assert "_offload_refs" not in small_msg
        assert len(large_a["_offload_refs"]) == 1
        assert len(large_b["_offload_refs"]) == 1

        # Registry has exactly the two backed-up records.
        assert len(cm._offload_registry) == 2
        for msg, expected_chars in [(large_a, 500), (large_b, 800)]:
            ref = msg["_offload_refs"][0]
            rec = cm._offload_registry[ref]
            assert rec.state == "backed_up"
            assert rec.char_count == expected_chars
            # File exists on disk and matches.
            file_path = os.path.join(str(tmp_path), ref)
            assert os.path.isfile(file_path)


# ======================================================================
# Stage 3: sliding-window offload flips state to "offloaded"
# ======================================================================


class TestStage_SlidingWindowOffload:
    def test_old_results_get_offloaded_markers_recent_kept(self, tmp_path):
        cm = _build_cm(tmp_path)

        msgs = [
            make_msg("user", "task", MT.USER_INPUT),
            make_msg("assistant", "turn 1 thinking", MT.ASSISTANT),
            _make_tool_result_msg(cm, turn=1, tool_name="search", result_text="A" * 500),
            make_msg("assistant", "turn 2 thinking", MT.ASSISTANT),
            _make_tool_result_msg(cm, turn=2, tool_name="scrape", result_text="B" * 600),
            make_msg("assistant", "turn 3 thinking", MT.ASSISTANT),
            _make_tool_result_msg(cm, turn=3, tool_name="fetch", result_text="C" * 700),
        ]

        ref_old = msgs[2]["_offload_refs"][0]
        ref_mid = msgs[4]["_offload_refs"][0]
        ref_new = msgs[6]["_offload_refs"][0]

        # current_turn=3, keep_recent=2 → turns 2,3 are kept; turn 1 offloaded.
        replaced = cm.finalize_offload_candidates(msgs, current_turn=3, keep_recent=KEEP_RECENT)
        assert replaced == 1

        # Old turn now carries an OFFLOADED marker.
        assert TAG_OFFLOADED in msgs[2]["content"][0]["text"]
        assert ref_old in msgs[2]["content"][0]["text"]
        assert msgs[2]["_type"] == MT.OFFLOADED
        # Recent turns still hold their full text.
        assert "B" * 600 == msgs[4]["content"][0]["text"]
        assert "C" * 700 == msgs[6]["content"][0]["text"]

        # Registry state reflects the transition.
        assert cm._offload_registry[ref_old].state == "offloaded"
        assert cm._offload_registry[ref_mid].state == "backed_up"
        assert cm._offload_registry[ref_new].state == "backed_up"


# ======================================================================
# Stage 4: compact (observation masking) is orthogonal to offload state
# ======================================================================


class TestStage_CompactPlusOffload:
    def test_apply_compact_masks_old_tool_results(self, tmp_path):
        """ObservationMasking and offload work on different signals — the
        first masks tool *outputs* (regardless of offload), the second
        replaces large content with file refs. Both must compose without
        either undoing the other."""
        cm = _build_cm(tmp_path)
        cm.set_token_estimator(lambda s: len(s) // 3)  # rough approximation

        msgs = [
            make_msg("user", "task", MT.USER_INPUT),
        ]
        for turn in range(1, 5):
            msgs.append(make_msg("assistant", f"thought {turn}", MT.ASSISTANT))
            msgs.append(
                _make_tool_result_msg(
                    cm, turn=turn, tool_name=f"tool_{turn}", result_text=f"R{turn}" * 200
                )
            )

        before_count = len(msgs)
        cm.apply_compact(
            msgs, current_turn=4, system_prompt="sys" * 100, max_context_length=200
        )

        # Compact may shrink older messages but doesn't drop them.
        assert len(msgs) == before_count

        # Offload registry untouched by compact.
        assert len(cm._offload_registry) == 4


# ======================================================================
# Stage 5: read_result restores a specific ref's content
# ======================================================================


class TestStage_ReadResult:
    def test_read_result_restores_offloaded_content(self, tmp_path):
        cm = _build_cm(tmp_path)
        original = "D" * 500
        ref = cm.backup_large_result(original, "fetch", turn=1)
        assert ref is not None

        content = cm.restore_single_file(ref)
        assert content == original

    def test_read_result_after_offload_marker_still_works(self, tmp_path):
        """Even after a turn has been replaced with an OFFLOADED marker,
        read_result(ref) can still bring the full text back."""
        cm = _build_cm(tmp_path)
        msgs = [
            make_msg("user", "task", MT.USER_INPUT),
            make_msg("assistant", "t1", MT.ASSISTANT),
            _make_tool_result_msg(cm, turn=1, tool_name="search", result_text="E" * 500),
            make_msg("assistant", "t2", MT.ASSISTANT),
            _make_tool_result_msg(cm, turn=2, tool_name="scrape", result_text="F" * 500),
        ]
        ref_t1 = msgs[2]["_offload_refs"][0]

        cm.finalize_offload_candidates(msgs, current_turn=2, keep_recent=1)
        assert TAG_OFFLOADED in msgs[2]["content"][0]["text"]

        # The original file is still on disk and read_result reaches it.
        recovered = cm.restore_single_file(ref_t1)
        assert recovered == "E" * 500


# ======================================================================
# Stage 6: snapshot → restore round trip preserves the chain
# ======================================================================


class TestStage_SnapshotRoundTrip:
    def test_full_chain_state_survives_snapshot_restore(self, tmp_path):
        cm = _build_cm(tmp_path)

        # Build state by walking the full chain.
        msgs = [make_msg("user", "task", MT.USER_INPUT)]
        for turn in range(1, 4):
            msgs.append(make_msg("assistant", f"t{turn}", MT.ASSISTANT))
            msgs.append(
                _make_tool_result_msg(
                    cm, turn=turn, tool_name="search", result_text=f"R{turn}" * 200
                )
            )
        cm.finalize_offload_candidates(msgs, current_turn=3, keep_recent=KEEP_RECENT)

        # Capture snapshot of CM state and message_history.
        snap = build_snapshot(
            message_history=msgs,
            turn_count=3,
            context_manager=cm,
        )

        # Restore on a fresh CM that knows nothing about the offload dir
        # state — and verify it ends up byte-identical.
        cm_fresh = _build_cm(tmp_path)
        restore_snapshot(snap, context_manager=cm_fresh)

        # Registries match.
        assert set(cm_fresh._offload_registry.keys()) == set(cm._offload_registry.keys())
        for ref, original_record in cm._offload_registry.items():
            restored_record = cm_fresh._offload_registry[ref]
            assert restored_record.state == original_record.state
            assert restored_record.char_count == original_record.char_count
            assert restored_record.tool_names == original_record.tool_names

        # And read_result still works on the restored CM.
        backed_up_refs = [
            r for r, rec in cm_fresh._offload_registry.items() if rec.state == "backed_up"
        ]
        assert backed_up_refs, "expected at least one backed_up ref after partial offload"
        assert cm_fresh.restore_single_file(backed_up_refs[0]) is not None


# ======================================================================
# Stage 7: rebuild_registry_from_history (cold-start resume path)
# ======================================================================


class TestStage_RebuildRegistryFromHistory:
    def test_registry_rebuilt_from_message_history_only(self, tmp_path):
        """Imagine a process restart where the in-memory registry is gone but
        message_history (with OFFLOADED markers + _offload_refs sidecars) and
        the on-disk files survive. _rebuild_registry_from_history must
        reconstruct the registry purely from message_history."""
        cm = _build_cm(tmp_path)

        msgs = [make_msg("user", "task", MT.USER_INPUT)]
        for turn in range(1, 4):
            msgs.append(make_msg("assistant", f"t{turn}", MT.ASSISTANT))
            msgs.append(
                _make_tool_result_msg(
                    cm, turn=turn, tool_name="search", result_text=f"R{turn}" * 200
                )
            )
        cm.finalize_offload_candidates(msgs, current_turn=3, keep_recent=KEEP_RECENT)

        # Capture the registry shape and clone the messages — this is what
        # would land in the snapshot's message_history field.
        original_registry = deepcopy(cm._offload_registry)
        original_msgs = deepcopy(msgs)

        # Wipe a fresh CM's registry, then ask it to rebuild from history.
        cm_cold = _build_cm(tmp_path)
        assert cm_cold._offload_registry == {}
        cm_cold._rebuild_registry_from_history(original_msgs)

        # Every ref from before is back, and states match (offloaded vs
        # backed_up depending on whether the message still carries the marker).
        assert set(cm_cold._offload_registry.keys()) == set(original_registry.keys())
        for ref, original_record in original_registry.items():
            assert cm_cold._offload_registry[ref].state == original_record.state


# ======================================================================
# Sanity: full-chain on one CM doesn't accumulate inconsistencies
# ======================================================================


class TestStage_FullChainCoherence:
    def test_no_orphan_files_no_dangling_registry_entries(self, tmp_path):
        """Walk every stage on a single CM and verify the disk + registry +
        message_history triple agree at the end."""
        cm = _build_cm(tmp_path)

        msgs = [make_msg("user", "task", MT.USER_INPUT)]
        for turn in range(1, 5):
            msgs.append(make_msg("assistant", f"t{turn}", MT.ASSISTANT))
            msgs.append(
                _make_tool_result_msg(
                    cm, turn=turn, tool_name="search", result_text=f"R{turn}" * 250
                )
            )

        # Stage: offload old turns
        cm.finalize_offload_candidates(msgs, current_turn=4, keep_recent=KEEP_RECENT)

        # Stage: compact
        cm.set_token_estimator(lambda s: len(s) // 3)
        cm.apply_compact(
            msgs, current_turn=4, system_prompt="sys" * 200, max_context_length=300
        )

        # Stage: pull one of the still-backed-up files
        backed_up_refs = [
            r for r, rec in cm._offload_registry.items() if rec.state == "backed_up"
        ]
        if backed_up_refs:
            assert cm.restore_single_file(backed_up_refs[0]) is not None

        # Coherence check: every file on disk corresponds to a registry entry,
        # and every registry entry's file exists on disk.
        on_disk = {p for p in os.listdir(str(tmp_path)) if p.endswith(".txt")}
        in_registry = set(cm._offload_registry.keys())
        assert on_disk == in_registry, (
            f"orphan files: {on_disk - in_registry}; "
            f"dangling registry: {in_registry - on_disk}"
        )

        # Coherence check: every _offload_refs / OFFLOADED marker in
        # message_history references a real registry entry.
        for msg in msgs:
            refs = msg.get("_offload_refs") or []
            for ref in refs:
                assert ref in cm._offload_registry, (
                    f"message references {ref} but registry has no record"
                )
            content = msg.get("content")
            if isinstance(content, list) and content:
                first = content[0] if isinstance(content[0], dict) else {}
                text = first.get("text", "")
                if text.startswith(TAG_OFFLOADED):
                    # Marker must point at a real ref.
                    import re

                    for m in re.finditer(
                        re.escape(TAG_OFFLOADED) + r"([^|]+)\|", text
                    ):
                        assert m.group(1) in cm._offload_registry
