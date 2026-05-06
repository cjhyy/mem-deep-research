"""HITL audit trail tests (v1.3.0+).

Covers:
- ``on_human_request_created`` (canonical) and ``on_await_human`` (deprecated
  alias) both fire on the same event.
- ``on_human_request_decided`` fires when a decision is delivered in time.
- Transcript records ``hitl_request_created`` / ``hitl_request_decided`` /
  ``hitl_request_resumed`` for compliance / replay.
"""

from __future__ import annotations

import asyncio

import pytest

from mem_deep_research_core.core.hitl import (
    HumanDecision,
    PendingHumanException,
    PendingHumanRequest,
)
from mem_deep_research_core.core.hitl.pending_store import InMemoryPendingStore
from mem_deep_research_core.core.hitl.runtime_facade import RuntimeFacade
from mem_deep_research_core.core.hooks import HookRegistry
from mem_deep_research_core.core.transcript import EventType, Transcript


# ======================================================================
# Hook fan-out: created (canonical) + on_await_human (deprecated)
# ======================================================================


class TestRequestCreatedHookFanout:
    @pytest.mark.asyncio
    async def test_canonical_hook_fires(self):
        hooks = HookRegistry()
        seen: list[str] = []

        async def notifier(ctx, next_fn):
            seen.append(ctx.extra["request"].prompt)
            return await next_fn(ctx)

        hooks.register_fn("on_human_request_created", notifier)
        facade = RuntimeFacade(hooks=hooks, pending_store=InMemoryPendingStore())

        with pytest.raises(PendingHumanException):
            await facade.wait_for_human("approve me?", sync_timeout=0.05)
        assert seen == ["approve me?"]

    @pytest.mark.asyncio
    async def test_deprecated_alias_still_fires(self):
        """Existing on_await_human users keep working through v1.3.x."""
        hooks = HookRegistry()
        canonical_seen: list[str] = []
        legacy_seen: list[str] = []

        hooks.register_fn(
            "on_human_request_created",
            lambda ctx, nxt: canonical_seen.append(ctx.extra["request"].prompt) or nxt(ctx),
        )
        hooks.register_fn(
            "on_await_human",
            lambda ctx, nxt: legacy_seen.append(ctx.extra["request"].prompt) or nxt(ctx),
        )
        facade = RuntimeFacade(hooks=hooks, pending_store=InMemoryPendingStore())

        with pytest.raises(PendingHumanException):
            await facade.wait_for_human("dual fire", sync_timeout=0.05)

        assert canonical_seen == ["dual fire"]
        assert legacy_seen == ["dual fire"]


class TestRequestDecidedHook:
    @pytest.mark.asyncio
    async def test_decided_fires_on_in_time_decision(self):
        hooks = HookRegistry()
        seen: list[bool] = []

        async def on_decided(ctx, next_fn):
            seen.append(ctx.extra["decision"].approved)
            return await next_fn(ctx)

        hooks.register_fn("on_human_request_decided", on_decided)
        facade = RuntimeFacade(hooks=hooks, pending_store=InMemoryPendingStore())

        async def approver():
            for _ in range(50):
                ids = list(facade.pending_store._futures.keys())  # type: ignore[attr-defined]
                if ids:
                    await facade.pending_store.put_decision(
                        ids[0], HumanDecision(approved=True, reason="ok")
                    )
                    return
                await asyncio.sleep(0.01)

        approver_task = asyncio.create_task(approver())
        decision = await facade.wait_for_human("decide", sync_timeout=2.0)
        await approver_task

        assert decision.approved is True
        assert seen == [True]

    @pytest.mark.asyncio
    async def test_decided_does_not_fire_on_timeout(self):
        """Timeout is a suspend, not a decision — decided hook stays silent."""
        hooks = HookRegistry()
        seen: list = []
        hooks.register_fn(
            "on_human_request_decided",
            lambda ctx, nxt: seen.append(1) or nxt(ctx),
        )
        facade = RuntimeFacade(hooks=hooks, pending_store=InMemoryPendingStore())

        with pytest.raises(PendingHumanException):
            await facade.wait_for_human("times out", sync_timeout=0.05)
        assert seen == []


# ======================================================================
# Transcript audit events
# ======================================================================


class TestTranscriptAudit:
    @pytest.mark.asyncio
    async def test_request_created_recorded_to_transcript(self):
        transcript = Transcript()
        facade = RuntimeFacade(
            hooks=HookRegistry(),
            pending_store=InMemoryPendingStore(),
            transcript=transcript,
            agent_name="main",
        )
        facade.bind_tool_context(tool_call_id="tc_audit", turn_number=2)

        with pytest.raises(PendingHumanException):
            await facade.wait_for_human(
                "Approve risky_thing?",
                payload={"tool": "send_email"},
                tags=["approval"],
                sync_timeout=0.05,
            )

        events = [e for e in transcript._events if e.event_type == EventType.HITL_REQUEST_CREATED.value]
        assert len(events) == 1, f"Expected 1 created event, got {len(events)}"
        evt = events[0]
        assert evt.data["tool_call_id"] == "tc_audit"
        assert evt.data["tags"] == ["approval"]
        assert evt.data["sync_timeout"] == 0.05
        assert evt.turn == 2

    @pytest.mark.asyncio
    async def test_request_decided_recorded_to_transcript(self):
        transcript = Transcript()
        facade = RuntimeFacade(
            hooks=HookRegistry(),
            pending_store=InMemoryPendingStore(),
            transcript=transcript,
        )

        async def approver():
            for _ in range(50):
                ids = list(facade.pending_store._futures.keys())  # type: ignore[attr-defined]
                if ids:
                    await facade.pending_store.put_decision(
                        ids[0],
                        HumanDecision(
                            approved=False, reason="not safe", decided_by="alice@example.com"
                        ),
                    )
                    return
                await asyncio.sleep(0.01)

        approver_task = asyncio.create_task(approver())
        decision = await facade.wait_for_human("audit me", sync_timeout=2.0)
        await approver_task

        decided_events = [
            e for e in transcript._events if e.event_type == EventType.HITL_REQUEST_DECIDED.value
        ]
        assert len(decided_events) == 1
        evt = decided_events[0]
        assert evt.data["approved"] is False
        assert evt.data["reason"] == "not safe"
        assert evt.data["decided_by"] == "alice@example.com"
        # And the matching created event is also there.
        created = [
            e for e in transcript._events if e.event_type == EventType.HITL_REQUEST_CREATED.value
        ]
        assert len(created) == 1
        # Same request_id ties created and decided together for replay.
        assert created[0].data["request_id"] == evt.data["request_id"]

    @pytest.mark.asyncio
    async def test_no_transcript_means_no_audit_recording_no_crash(self):
        """transcript=None is a supported config — must not raise."""
        facade = RuntimeFacade(
            hooks=HookRegistry(),
            pending_store=InMemoryPendingStore(),
            transcript=None,
        )
        with pytest.raises(PendingHumanException):
            await facade.wait_for_human("no audit", sync_timeout=0.05)
