"""Pending HITL request store.

Responsibility: hold the open ``PendingHumanRequest`` records and act as a
rendezvous point between ``Runtime.wait_for_human()`` (which awaits a
``HumanDecision``) and external resumers (``inject_human_decision`` in
Phase 2).

Phase 1 ships an in-memory implementation — a single process holds the
future, the approver calls ``put_decision`` on the same store. Phase 2
adds a filesystem implementation so async HITL survives a process
restart.

The protocol uses async methods everywhere so Phase 3 can swap in
Redis / Postgres stores without forcing callers to change.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

from mem_deep_research_core.core.hitl.types import (
    HumanDecision,
    PendingHumanRequest,
)

logger = logging.getLogger("mem_deep_research")


@runtime_checkable
class PendingStore(Protocol):
    """Rendezvous store for open human-decision requests."""

    async def put(self, request: PendingHumanRequest) -> None:
        """Register a newly-opened request."""

    async def wait_for_decision(
        self, request_id: str, *, timeout: float
    ) -> HumanDecision:
        """Block until a decision is available, raising ``asyncio.TimeoutError`` on timeout.

        Implementations must guarantee that ``put_decision`` delivered between
        ``put`` and ``wait_for_decision`` is *not* lost — i.e. the future the
        waiter awaits is registered during ``put``, not during the wait call.
        """

    async def put_decision(self, request_id: str, decision: HumanDecision) -> None:
        """Deliver a decision to any waiter; must be idempotent if the request is already closed."""

    async def close(self, request_id: str) -> None:
        """Drop the request; any future awaiters get ``KeyError``."""

    def has(self, request_id: str) -> bool:
        """Check whether a request is still open."""


class InMemoryPendingStore:
    """Single-process PendingStore backed by ``asyncio.Future``.

    Thread-safe across a single event loop. Not suitable for multi-process
    async HITL — Phase 2 adds a filesystem-backed store for that.
    """

    def __init__(self) -> None:
        self._futures: dict[str, asyncio.Future[HumanDecision]] = {}
        self._requests: dict[str, PendingHumanRequest] = {}
        self._lock = asyncio.Lock()

    async def put(self, request: PendingHumanRequest) -> None:
        async with self._lock:
            if request.request_id in self._futures:
                raise ValueError(
                    f"PendingStore already tracking request {request.request_id}"
                )
            loop = asyncio.get_running_loop()
            self._futures[request.request_id] = loop.create_future()
            self._requests[request.request_id] = request
            logger.debug(
                "[PendingStore] Registered request %s (hook_point=%s, tool_call_id=%s)",
                request.request_id,
                request.hook_point,
                request.tool_call_id,
            )

    async def wait_for_decision(
        self, request_id: str, *, timeout: float
    ) -> HumanDecision:
        future = self._futures.get(request_id)
        if future is None:
            raise KeyError(f"Unknown or already-closed request {request_id}")
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        finally:
            # Do not remove the request here — timeout may want to transition
            # the same request into the async-HITL path (Phase 2). Callers
            # invoke ``close`` explicitly when they're done.
            pass

    async def put_decision(
        self, request_id: str, decision: HumanDecision
    ) -> None:
        async with self._lock:
            future = self._futures.get(request_id)
            if future is None or future.done():
                logger.debug(
                    "[PendingStore] put_decision for %s ignored (unknown or already closed)",
                    request_id,
                )
                return
            future.set_result(decision)
            logger.debug(
                "[PendingStore] Decision delivered for %s (approved=%s)",
                request_id,
                decision.approved,
            )

    async def close(self, request_id: str) -> None:
        async with self._lock:
            future = self._futures.pop(request_id, None)
            self._requests.pop(request_id, None)
            if future is not None and not future.done():
                # Cancel so any still-waiting coroutine wakes with CancelledError
                # (which is then translated to KeyError by callers that resubscribe).
                future.cancel()

    def has(self, request_id: str) -> bool:
        return request_id in self._futures

    def get_request(self, request_id: str) -> PendingHumanRequest | None:
        """Inspect the stored request (primarily for tests / debug / transcript)."""
        return self._requests.get(request_id)
