"""Phase 2 HITL acceptance tests.

Covers:
- FilesystemCheckpointStore roundtrip + sweep_expired.
- Schema-version mismatch detected on restore.
- RuntimeSnapshot populated via MainLoopRunner._build_runtime_snapshot
  captures every module's state coherently.
- ``wait_for_human`` main-agent timeout raises ``PendingHumanException``
  carrying the request, and leaves the pending store entry open.

End-to-end pause → resume of a running task is exercised indirectly via
the pipeline-level resume entry in integration-style tests; the main-loop
heavy lifting is covered by the hook-level tests here.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mem_deep_research_core.core.context_manager import (
    ContextManager,
    ContextManagerConfig,
    OffloadRecord,
    ToolCallRecord,
)
from mem_deep_research_core.core.hitl import (
    PendingHumanException,
    PendingHumanRequest,
    RUNTIME_SNAPSHOT_SCHEMA_VERSION,
    RuntimeSnapshot,
    build_snapshot,
)
from mem_deep_research_core.core.hitl.checkpoint_store import (
    FilesystemCheckpointStore,
)
from mem_deep_research_core.core.hitl.pending_store import InMemoryPendingStore
from mem_deep_research_core.core.hitl.runtime_facade import RuntimeFacade
from mem_deep_research_core.core.hooks import HookRegistry
from mem_deep_research_core.core.monitoring import (
    EscalationAction,
    ExecutionMonitor,
    MonitoringConfig,
)
from mem_deep_research_core.skills.inline_selector import InlineSkillSelector
from mem_deep_research_core.skills.matcher import SkillMatcher


def _make_matcher() -> SkillMatcher:
    return SkillMatcher(skills_dir=Path(tempfile.mkdtemp(prefix="hitl2_skills_")))


# ======================================================================
# FilesystemCheckpointStore roundtrip
# ======================================================================


class TestFilesystemCheckpointStore:
    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_path):
        store = FilesystemCheckpointStore(tmp_path)
        snap = build_snapshot(turn_count=5, effective_mode="deep")
        snap.pending_human_request = PendingHumanRequest(
            prompt="approve?", tool_call_id="tc_1", sync_timeout=30.0
        )
        cid = await store.save(snap)
        loaded = await store.load(cid)

        assert loaded.schema_version == RUNTIME_SNAPSHOT_SCHEMA_VERSION
        assert loaded.turn_count == 5
        assert loaded.effective_mode == "deep"
        assert loaded.pending_human_request.tool_call_id == "tc_1"
        assert loaded.pending_human_request.checkpoint_id == cid

    @pytest.mark.asyncio
    async def test_task_description_round_trips(self, tmp_path):
        """Snapshot field added in v1.3.0 — resume needs no task_description arg."""
        store = FilesystemCheckpointStore(tmp_path)
        snap = build_snapshot(task_description="What is the capital of France?")
        snap.pending_human_request = PendingHumanRequest(prompt="approve?", tool_call_id="tc")
        cid = await store.save(snap)
        loaded = await store.load(cid)
        assert loaded.task_description == "What is the capital of France?"

    @pytest.mark.asyncio
    async def test_save_survives_restart(self, tmp_path):
        """Fresh store instance (different object) can load what the old one saved."""
        snap = build_snapshot(turn_count=7)
        snap.pending_human_request = PendingHumanRequest(prompt="re-open", tool_call_id="tc")

        store_a = FilesystemCheckpointStore(tmp_path)
        cid = await store_a.save(snap)

        store_b = FilesystemCheckpointStore(tmp_path)
        loaded = await store_b.load(cid)
        assert loaded.turn_count == 7

    @pytest.mark.asyncio
    async def test_delete_is_idempotent(self, tmp_path):
        store = FilesystemCheckpointStore(tmp_path)
        snap = build_snapshot()
        snap.pending_human_request = PendingHumanRequest(prompt="x", tool_call_id="tc")
        cid = await store.save(snap)

        await store.delete(cid)
        await store.delete(cid)  # idempotent

        assert await store.list_checkpoints() == []

    @pytest.mark.asyncio
    async def test_illegal_checkpoint_id_rejected(self, tmp_path):
        store = FilesystemCheckpointStore(tmp_path)
        with pytest.raises(ValueError):
            await store.load("../escape")

    @pytest.mark.asyncio
    async def test_sweep_expired(self, tmp_path):
        store = FilesystemCheckpointStore(tmp_path)

        # Fresh checkpoint — future expiry.
        snap_fresh = build_snapshot()
        snap_fresh.pending_human_request = PendingHumanRequest(
            prompt="fresh",
            tool_call_id="tc_fresh",
            async_timeout=3600.0,
        )
        cid_fresh = await store.save(snap_fresh)

        # Stale checkpoint — expires_at back-dated.
        snap_stale = build_snapshot()
        req_stale = PendingHumanRequest(prompt="stale", tool_call_id="tc_stale")
        req_stale.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        snap_stale.pending_human_request = req_stale
        cid_stale = await store.save(snap_stale)

        swept = await store.sweep_expired()
        assert cid_stale in swept
        assert cid_fresh not in swept
        assert await store.list_checkpoints() == [cid_fresh]


# ======================================================================
# Schema mismatch
# ======================================================================


class TestSchemaMismatch:
    @pytest.mark.asyncio
    async def test_save_rejects_mismatched_schema(self, tmp_path):
        store = FilesystemCheckpointStore(tmp_path)
        snap = build_snapshot()
        snap.pending_human_request = PendingHumanRequest(prompt="x", tool_call_id="tc")
        snap.schema_version = RUNTIME_SNAPSHOT_SCHEMA_VERSION + 1
        with pytest.raises(ValueError, match="schema"):
            await store.save(snap)


# ======================================================================
# Main-agent wait_for_human → PendingHumanException (Phase 2 behaviour)
# ======================================================================


class TestWaitForHumanMainAgentPath:
    @pytest.mark.asyncio
    async def test_timeout_raises_pending_human_exception(self):
        facade = RuntimeFacade(
            hooks=HookRegistry(), pending_store=InMemoryPendingStore()
        )
        facade.bind_tool_context(tool_call_id="tc_42", turn_number=3)

        with pytest.raises(PendingHumanException) as excinfo:
            await facade.wait_for_human(
                "approve?",
                payload={"tool": "send_email", "args": {"to": "a@b"}},
                sync_timeout=0.05,
                async_timeout=600.0,
                tags=["risky"],
            )

        request = excinfo.value.request
        assert request.tool_call_id == "tc_42"
        assert request.turn_number == 3
        assert request.prompt == "approve?"
        assert request.tags == ["risky"]
        # Pending store retains the request so resume can deliver later.
        assert facade.pending_store.has(request.request_id)


# ======================================================================
# RuntimeSnapshot coverage: build_snapshot captures every module
# ======================================================================


class TestBuildSnapshotCoverage:
    def test_every_module_state_lands_in_snapshot(self):
        cm = ContextManager(config=ContextManagerConfig())
        cm._current_turn = 4
        cm._offload_registry["r1"] = OffloadRecord(ref="r1", turn=1, char_count=100)
        cm._dedup_cache["h1"] = ToolCallRecord(
            tool_name="search",
            arguments_hash="h1",
            arguments={"q": "x"},
            turn=1,
            result_hash="rh",
            result_brief="brief",
        )

        mon = ExecutionMonitor(config=MonitoringConfig())
        mon.state.consecutive_empty_turns = 2
        mon._last_loop_action = EscalationAction.INJECT_HINT

        sel = InlineSkillSelector(matcher=_make_matcher())
        sel._pending_skills = ["alpha"]
        sel._first_turn = False

        pending_req = PendingHumanRequest(
            prompt="approve?", tool_call_id="tc_1", turn_number=4
        )

        snap = build_snapshot(
            message_history=[{"role": "user", "content": "hi"}],
            turn_count=4,
            session_memory={"key_findings": ["fact-1"]},
            todo_state={"items": []},
            last_assistant_text="partial",
            task_failed=False,
            tool_calls_executed=12,
            effective_mode="deep",
            reasoning_effort="high",
            reflection_pending=True,
            adaptive_pending=False,
            context_manager=cm,
            monitor=mon,
            inline_skill_selector=sel,
            assistant_response_text="asst",
            current_tool_calls=[{"id": "tc_1", "tool_name": "search"}],
            current_tool_index=0,
            completed_tool_results=[("tc_done", "ref_a")],
            effective_arguments={"q": "x"},
            pending_human_request=pending_req,
        )

        assert snap.turn_count == 4
        assert snap.effective_mode == "deep"
        assert snap.reasoning_effort == "high"
        assert snap.reflection_pending is True
        assert snap.context_manager_state["current_turn"] == 4
        assert "r1" in snap.context_manager_state["offload_registry"]
        assert snap.monitor_state["consecutive_empty_turns"] == 2
        assert snap.monitor_state["last_loop_action"] == "inject_hint"
        assert snap.inline_skill_state["pending_skills"] == ["alpha"]
        assert snap.current_tool_calls == [{"id": "tc_1", "tool_name": "search"}]
        assert snap.effective_arguments == {"q": "x"}
        assert snap.pending_human_request is pending_req
        assert snap.completed_tool_results == [("tc_done", "ref_a")]

    def test_roundtrip_through_filesystem_preserves_module_state(self, tmp_path):
        """End-to-end: build → save → load → restore gives equivalent module state."""
        import asyncio as _asyncio

        cm = ContextManager(config=ContextManagerConfig())
        cm._current_turn = 9
        cm._offload_registry["r1"] = OffloadRecord(ref="r1", turn=0, char_count=42)

        mon = ExecutionMonitor(config=MonitoringConfig())
        mon.state.consecutive_empty_turns = 1
        mon._last_loop_action = EscalationAction.WARN

        sel = InlineSkillSelector(matcher=_make_matcher())
        sel._pending_skills = ["beta"]

        snap = build_snapshot(
            turn_count=9,
            context_manager=cm,
            monitor=mon,
            inline_skill_selector=sel,
        )
        snap.pending_human_request = PendingHumanRequest(
            prompt="x", tool_call_id="tc_x"
        )

        async def _roundtrip():
            store = FilesystemCheckpointStore(tmp_path)
            cid = await store.save(snap)
            return await store.load(cid)

        restored = _asyncio.run(_roundtrip())

        assert restored.turn_count == 9
        assert restored.context_manager_state["current_turn"] == 9
        # OffloadRecord survives dataclass registry round-trip.
        offload_r1 = restored.context_manager_state["offload_registry"]["r1"]
        assert isinstance(offload_r1, OffloadRecord)
        assert offload_r1.char_count == 42
        assert restored.monitor_state["consecutive_empty_turns"] == 1
        assert restored.monitor_state["last_loop_action"] == "warn"
        assert restored.inline_skill_state["pending_skills"] == ["beta"]


# ======================================================================
# PendingHumanException carries snapshot when raised from MainLoopRunner
# ======================================================================


class TestHitlConfigWiring:
    """HitlConfig flags are honoured at the RuntimeFacade level."""

    @pytest.mark.asyncio
    async def test_enabled_false_auto_approves(self):
        """``enabled=False`` short-circuits wait_for_human with approved=True."""
        facade = RuntimeFacade(
            hooks=HookRegistry(),
            pending_store=InMemoryPendingStore(),
            enabled=False,
        )
        decision = await facade.wait_for_human(
            "Should-never-wait",
            sync_timeout=0.0,  # would time out immediately if honoured
        )
        assert decision.approved is True
        assert decision.reason == "hitl_disabled"

    @pytest.mark.asyncio
    async def test_enabled_false_does_not_register_pending_request(self):
        """Disabled runtime must not leave an entry in the pending store."""
        store = InMemoryPendingStore()
        facade = RuntimeFacade(
            hooks=HookRegistry(),
            pending_store=store,
            enabled=False,
        )
        await facade.wait_for_human("skip")
        # No futures registered — store stayed empty.
        assert store._futures == {}  # type: ignore[attr-defined]


class TestPendingHumanExceptionAttributes:
    def test_exception_accepts_optional_snapshot(self):
        request = PendingHumanRequest(prompt="x", tool_call_id="tc")
        snap = build_snapshot()
        snap.pending_human_request = request
        exc = PendingHumanException(request, snapshot=snap)
        assert exc.request is request
        assert exc.snapshot is snap

    def test_exception_snapshot_defaults_to_none(self):
        exc = PendingHumanException(PendingHumanRequest(prompt="x", tool_call_id="tc"))
        assert exc.snapshot is None
