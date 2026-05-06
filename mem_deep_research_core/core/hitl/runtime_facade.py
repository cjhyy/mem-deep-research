"""HITL runtime facade.

Exposes the ``wait_for_human`` API that hook authors consume. Phase 1 keeps
the implementation fully synchronous — timeout in a sync-HITL context raises
``asyncio.TimeoutError``, which the hook translates to ``GuardrailError`` /
tool rejection. Phase 2 extends the facade to raise
:class:`PendingHumanException` so the main loop can checkpoint.

The facade is discovered by hook code through a ContextVar
(:data:`_runtime_var`) so the :class:`HookContext` dataclass stays
invariant across the HITL rollout. Callers that need the facade inside a
hook do ``ctx.runtime.wait_for_human(...)``; the ``ctx.runtime`` property
resolves the ContextVar lazily.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from mem_deep_research_core.core.hitl.exceptions import PendingHumanException
from mem_deep_research_core.core.hitl.pending_store import (
    InMemoryPendingStore,
    PendingStore,
)
from mem_deep_research_core.core.hitl.types import (
    HumanDecision,
    PendingHumanRequest,
    _generate_request_id,
)
from mem_deep_research_core.core.sub_agent_runner import _is_sub_agent_var

if TYPE_CHECKING:
    from mem_deep_research_core.core.hooks import HookRegistry

logger = logging.getLogger("mem_deep_research")


# ContextVar that hook code resolves via ``HookContext.runtime``. Populated
# by the main loop around the hook chain for each turn; sub-agents inherit
# the parent runtime's facade (sub-agent restriction is enforced inside
# ``wait_for_human`` via ``_is_sub_agent_var``).
_runtime_var: contextvars.ContextVar["RuntimeFacade | None"] = contextvars.ContextVar(
    "_hitl_runtime", default=None
)


def get_current_runtime() -> "RuntimeFacade | None":
    """Return the runtime facade attached to the current execution context."""
    return _runtime_var.get(None)


def set_current_runtime(runtime: "RuntimeFacade | None"):
    """Attach ``runtime`` to the current context; returns a ``Token`` for reset()."""
    return _runtime_var.set(runtime)


@dataclass
class RuntimeFacade:
    """Narrow surface exposed to hooks — only HITL in Phase 1, more later.

    Attributes
    ----------
    hooks:
        The same :class:`HookRegistry` the main loop uses. Kept as a weak
        backreference so hook authors can fire ``on_await_human`` notifications
        without reaching into framework internals.
    pending_store:
        :class:`PendingStore` implementation; defaults to :class:`InMemoryPendingStore`.
    enabled:
        When False, ``wait_for_human`` short-circuits with an auto-approved
        :class:`HumanDecision` (``reason="hitl_disabled"``). Useful for tests
        and for environments without an approver wired up.
    """

    hooks: "HookRegistry | None" = None
    pending_store: PendingStore = field(default_factory=InMemoryPendingStore)
    enabled: bool = True
    # Optional transcript handle — when set, request lifecycle events
    # (created / decided / resumed) are recorded for audit / replay.
    transcript: Any = None
    agent_name: str = "main"
    # Tool-call id context for the current hook invocation. Populated by
    # ``ToolExecutor`` before it enters ``on_tool_start``; ``wait_for_human``
    # copies it into the :class:`PendingHumanRequest`.
    _current_tool_call_id: str = ""
    _current_turn_number: int = 0

    def bind_tool_context(self, tool_call_id: str, turn_number: int) -> None:
        """Called by the tool executor before each ``on_tool_start`` chain."""
        self._current_tool_call_id = tool_call_id
        self._current_turn_number = turn_number

    async def wait_for_human(
        self,
        prompt: str,
        *,
        payload: dict | None = None,
        sync_timeout: float = 30.0,
        async_timeout: float = 3600.0,
        tags: list[str] | None = None,
        hook_point: str = "on_tool_start",
    ) -> HumanDecision:
        """Block until a human decision arrives, then return it.

        Phase 2 behaviour:

        - **Disabled runtime** (``enabled=False``): short-circuits with an
          auto-approved :class:`HumanDecision`, no pending store side effect.
        - **Main agent path**: ``sync_timeout`` elapses without a decision →
          raises :class:`PendingHumanException`. The main loop catches at
          its outer try/except, builds a RuntimeSnapshot, persists a
          checkpoint, and returns ``awaiting_human``.
        - **Sub-agent path** (``_is_sub_agent_var=True``): durable suspend is
          disallowed (design doc, Phase 2 section). Timeout raises
          :class:`asyncio.TimeoutError`; the approval hook converts that to
          :class:`GuardrailError` so the sub-agent rejects the tool call
          cleanly.

        In both cases a received decision is returned immediately and the
        request is closed. Only the main-agent timeout path keeps the
        request open in the store so the resume entrypoint can deliver the
        decision post-restart.
        """
        if not self.enabled:
            logger.debug(
                "[HITL] wait_for_human short-circuited (enabled=False) for prompt=%r",
                prompt[:60],
            )
            return HumanDecision(
                approved=True,
                reason="hitl_disabled",
                decided_at=datetime.now(timezone.utc),
            )

        request = PendingHumanRequest(
            prompt=prompt,
            payload=payload or {},
            hook_point=hook_point,
            turn_number=self._current_turn_number,
            tool_call_id=self._current_tool_call_id,
            sync_timeout=sync_timeout,
            async_timeout=async_timeout,
            tags=list(tags or []),
        )

        await self.pending_store.put(request)
        await self._fire_request_created(request)

        is_sub_agent = _is_sub_agent_var.get(False)
        close_on_exit = True
        try:
            decision = await self.pending_store.wait_for_decision(
                request.request_id, timeout=sync_timeout
            )
            decision.decided_at = decision.decided_at or datetime.now(timezone.utc)
            await self._fire_request_decided(request, decision)
            return decision
        except asyncio.TimeoutError:
            logger.info(
                "[HITL] wait_for_human timeout for %s (sub_agent=%s)",
                request.request_id,
                is_sub_agent,
            )
            if is_sub_agent:
                # Sub-agents stay on the strictly-synchronous path; the
                # approval hook will translate TimeoutError into a
                # GuardrailError and the sub-agent rejects the tool.
                raise
            # Main agent: durable-suspend signal. Leave the request open so
            # ``resume_with_human_decision`` can deliver the human's answer
            # after the process restarts / checkpoint loads.
            close_on_exit = False
            raise PendingHumanException(request) from None
        finally:
            if close_on_exit:
                await self.pending_store.close(request.request_id)

    async def _fire_hook(
        self, hook_name: str, request: PendingHumanRequest, *, extra: dict | None = None
    ) -> None:
        """Best-effort fire-and-log for HITL lifecycle hooks.

        Failures inside notification hooks must NOT abort the wait — they're
        observability, not control flow. Each hook fires independently so a
        broken Slack notifier doesn't cascade into a broken transcript.
        """
        if self.hooks is None or not self.hooks.has_hooks(hook_name):
            return
        from mem_deep_research_core.core.hooks import HookContext

        ctx_extra = {"request": request}
        if extra:
            ctx_extra.update(extra)
        ctx = HookContext(hook_name=hook_name, extra=ctx_extra)
        try:
            await self.hooks.call(hook_name, ctx)
        except Exception as exc:  # pragma: no cover — best-effort
            logger.warning(
                "[HITL] %s hook failed for %s: %s", hook_name, request.request_id, exc
            )

    def _record_transcript(
        self, event_type: str, request: PendingHumanRequest, **payload: Any
    ) -> None:
        """Record an audit event onto the transcript (no-op if not configured).

        HITL audit trail goes to the same transcript as agent_start / tool_use
        / etc., so a single ``transcript.export()`` captures the full chain
        — including who decided what and when.
        """
        if self.transcript is None:
            return
        from mem_deep_research_core.core.transcript import EventType as ET

        try:
            self.transcript.record(
                event_type=ET(event_type),
                data={
                    "request_id": request.request_id,
                    "tool_call_id": request.tool_call_id,
                    "hook_point": request.hook_point,
                    "tags": list(request.tags),
                    **payload,
                },
                turn=request.turn_number,
                agent_name=self.agent_name,
            )
        except Exception as exc:  # pragma: no cover — observability is best-effort
            logger.debug("[HITL] transcript record %s failed: %s", event_type, exc)

    async def _fire_request_created(self, request: PendingHumanRequest) -> None:
        """Fire on_human_request_created (canonical) + on_await_human (deprecated).

        Two-name window so existing on_await_human users keep working through
        v1.3.x; v1.4.0 drops on_await_human entirely.
        """
        self._record_transcript(
            "hitl_request_created",
            request,
            prompt=request.prompt[:200],
            sync_timeout=request.sync_timeout,
            async_timeout=request.async_timeout,
        )
        await self._fire_hook("on_human_request_created", request)
        await self._fire_hook("on_await_human", request)

    async def _fire_request_decided(
        self, request: PendingHumanRequest, decision: HumanDecision
    ) -> None:
        self._record_transcript(
            "hitl_request_decided",
            request,
            approved=decision.approved,
            reason=(decision.reason or "")[:200],
            decided_by=decision.decided_by,
        )
        await self._fire_hook(
            "on_human_request_decided", request, extra={"decision": decision}
        )
