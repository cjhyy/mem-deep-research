"""Sub-agent / main-agent context manager isolation regression.

Roadmap (docs/20-roadmap.md) v1.3.0 completion criterion: "子 Agent 和主
Agent 的上下文管理行为不再静默漂移". The design contract is:

- Each sub-agent gets its own ``ContextManager`` instance (isolated dedup
  cache, isolated message tracking, isolated turn counter).
- The parent's ``offload_dir`` is shared so all offload files land in one
  place — single ``cleanup_offload_files()`` reaches them.
- ``merge_offload_registry()`` lifts sub-agent offload records back into
  the parent so cleanup sees them after the sub-agent terminates.
- Parent's ``ContextManagerConfig`` propagates to sub-agents as the base
  layer; sub-agent-specific overrides win.

Existing ``test_sub_agent.py`` covers spawn/run flows but not these
specific isolation invariants. This file checks them directly so future
refactors of ``_create_context_manager`` / ``merge_offload_registry`` get
caught.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from mem_deep_research_core.core.context_manager import (
    ContextManager,
    ContextManagerConfig,
    OffloadRecord,
    ToolCallRecord,
)


# ======================================================================
# 1. Offload directory is shared; offload files land in one place
# ======================================================================


class TestSharedOffloadDir:
    def test_sub_offload_files_land_in_parent_dir(self, tmp_path):
        """Sub-agent's ``backup_large_result`` writes to the parent's
        offload dir, so a single cleanup pass reaches them."""
        parent_dir = tmp_path / "shared_offload"
        parent_dir.mkdir()

        parent = ContextManager(config=ContextManagerConfig(result_offload_threshold=50))
        parent.set_offload_dir(str(parent_dir))

        # Simulate _create_context_manager: child inherits parent's offload_dir.
        child = ContextManager(config=ContextManagerConfig(result_offload_threshold=50))
        child.set_offload_dir(parent._offload_dir)

        parent_ref = parent.backup_large_result("PARENT" * 50, "search", turn=1)
        child_ref = child.backup_large_result("CHILD" * 50, "scrape", turn=1)
        assert parent_ref and child_ref

        # Both files live under the parent's directory.
        on_disk = {p for p in os.listdir(str(parent_dir)) if p.endswith(".txt")}
        assert parent_ref in on_disk
        assert child_ref in on_disk


# ======================================================================
# 2. Dedup cache is per-instance — sub-agent re-running the same
#    (tool, args) doesn't see the parent's earlier dedup hit
# ======================================================================


class TestDedupCacheIsolation:
    def test_dedup_cache_does_not_leak_between_instances(self):
        parent = ContextManager()
        child = ContextManager()

        record = ToolCallRecord(
            tool_name="search",
            arguments_hash="h_shared",
            arguments={"q": "x"},
            turn=1,
            result_hash="rh",
            result_brief="brief",
        )
        parent._dedup_cache["h_shared"] = record

        assert "h_shared" in parent._dedup_cache
        assert "h_shared" not in child._dedup_cache, (
            "sub-agent dedup cache must NOT see parent's entries — "
            "the design splits dedup by ContextManager instance"
        )


# ======================================================================
# 3. ConfigManager config propagation: parent → child
# ======================================================================


class TestConfigPropagation:
    def test_compact_ratios_inherited_unless_overridden(self):
        """The sub-agent factory layers parent config under sub-agent
        override. Verify the field-level merge keeps non-overridden parent
        ratios visible to the child."""
        parent_cfg = ContextManagerConfig(
            compact_at_ratio=0.55,
            summarize_at_ratio=0.85,
            compact_keep_recent=4,
        )

        # Simulate _create_context_manager's merge: parent dict + sub-agent
        # override. Here the sub-agent only overrides keep_recent.
        merged = {
            **dataclasses.asdict(parent_cfg),
            "compact_keep_recent": 1,  # sub-agent override wins
        }
        valid_fields = {f.name for f in dataclasses.fields(ContextManagerConfig)}
        child_cfg = ContextManagerConfig(
            **{k: v for k, v in merged.items() if k in valid_fields}
        )

        # Inherited:
        assert child_cfg.compact_at_ratio == 0.55
        assert child_cfg.summarize_at_ratio == 0.85
        # Overridden:
        assert child_cfg.compact_keep_recent == 1

    def test_invalid_parent_config_propagation_still_validated(self):
        """If parent config carries an invariant violation (compact >=
        summarize), the child instantiation must raise — not silently
        accept it."""
        merged = {
            "compact_at_ratio": 0.9,
            "summarize_at_ratio": 0.5,
        }
        with pytest.raises(ValueError, match="compact_at_ratio"):
            ContextManagerConfig(**merged)


# ======================================================================
# 4. merge_offload_registry: sub-agent records get lifted to parent
#    cleanly without trampling unrelated parent entries
# ======================================================================


class TestMergeOffloadRegistry:
    def test_merge_lifts_records_into_parent(self, tmp_path):
        parent = ContextManager()
        parent._offload_registry["parent_ref_a"] = OffloadRecord(
            ref="parent_ref_a", turn=1, char_count=200, tool_names=["main_search"]
        )

        child = ContextManager()
        child._offload_registry["sub_ref_x"] = OffloadRecord(
            ref="sub_ref_x", turn=2, char_count=400, tool_names=["sub_scrape"]
        )
        child._offload_registry["sub_ref_y"] = OffloadRecord(
            ref="sub_ref_y", turn=3, char_count=500, tool_names=["sub_fetch"]
        )

        parent.merge_offload_registry(child)

        # Parent now sees parent + child entries.
        assert set(parent._offload_registry.keys()) == {
            "parent_ref_a",
            "sub_ref_x",
            "sub_ref_y",
        }
        # Parent's existing entry is untouched (no overwrite by accident).
        assert parent._offload_registry["parent_ref_a"].tool_names == ["main_search"]
        # Sub-agent's records carried over verbatim.
        assert parent._offload_registry["sub_ref_x"].char_count == 400

    def test_merge_handles_collision_by_overwriting_with_child(self, caplog):
        """UUID collisions are unlikely but defined: child wins, so the
        sub-agent's file isn't orphaned. The collision is logged."""
        import logging

        parent = ContextManager()
        parent._offload_registry["clash"] = OffloadRecord(
            ref="clash", turn=1, char_count=100, tool_names=["parent_owner"]
        )
        child = ContextManager()
        child._offload_registry["clash"] = OffloadRecord(
            ref="clash", turn=9, char_count=999, tool_names=["child_owner"]
        )

        with caplog.at_level(logging.WARNING, logger="mem_deep_research"):
            parent.merge_offload_registry(child)

        # Child's record wins.
        assert parent._offload_registry["clash"].tool_names == ["child_owner"]
        assert parent._offload_registry["clash"].char_count == 999
        # Collision was warned.
        assert any("collision" in rec.getMessage().lower() for rec in caplog.records)

    def test_merge_empty_child_is_noop(self):
        parent = ContextManager()
        parent._offload_registry["a"] = OffloadRecord(
            ref="a", turn=1, char_count=10
        )
        child = ContextManager()  # empty registry

        before = dict(parent._offload_registry)
        parent.merge_offload_registry(child)
        assert parent._offload_registry == before


# ======================================================================
# 5. End-to-end: sub-agent backs up, registers, parent merges, cleanup
#    sees both
# ======================================================================


class TestE2ESubAgentLifecycle:
    def test_full_cycle_offload_merge_cleanup(self, tmp_path):
        """Verify the contract end to end:
            parent backup → child backup → child terminates →
            parent.merge_offload_registry(child) →
            parent.cleanup_offload_files() reaches both files.
        """
        parent_dir = tmp_path / "shared"
        parent_dir.mkdir()

        parent = ContextManager(
            config=ContextManagerConfig(
                result_offload_threshold=50,
                cleanup_offload_on_finish=True,
            )
        )
        parent.set_offload_dir(str(parent_dir))

        child = ContextManager(
            config=ContextManagerConfig(
                result_offload_threshold=50,
                cleanup_offload_on_finish=True,
            )
        )
        child.set_offload_dir(parent._offload_dir)

        parent_ref = parent.backup_large_result("P" * 200, "search", turn=1)
        child_ref = child.backup_large_result("C" * 300, "scrape", turn=1)

        # Lift child's records into parent (what sub_agent_runner does on exit).
        parent.merge_offload_registry(child)

        on_disk_before = {
            p for p in os.listdir(str(parent_dir)) if p.endswith(".txt")
        }
        assert {parent_ref, child_ref}.issubset(on_disk_before)

        # Cleanup reaches both files (and may remove the now-empty dir too).
        cleaned = parent.cleanup_offload_files()
        assert cleaned == 2

        # Either the dir is gone, or it still exists but is empty of .txt.
        if parent_dir.exists():
            on_disk_after = {
                p for p in os.listdir(str(parent_dir)) if p.endswith(".txt")
            }
            assert on_disk_after == set()
