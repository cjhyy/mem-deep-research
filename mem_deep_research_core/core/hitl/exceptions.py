"""HITL runtime control-flow exceptions.

Kept separate from ``hitl/types.py`` so hook / tool-executor modules can
catch :class:`PendingHumanException` without importing the full type
surface (which transitively pulls in ``datetime`` / ``Path`` / etc.).

Phase 1 defines the exception; Phase 2 wires it into the main-loop
catch-path and checkpointing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mem_deep_research_core.core.hitl.types import PendingHumanRequest

if TYPE_CHECKING:  # avoid import cycle with runtime_snapshot at module load
    from mem_deep_research_core.core.hitl.types import RuntimeSnapshot


class HitlRejectedError(Exception):
    """Raised when ``cfg.hitl.rejection_strategy="abort_task"`` and a human
    rejected the pending tool call.

    The pipeline catches this in :func:`execute_hitl_resume_pipeline` and
    translates it to ``status="failed"`` with ``error_type="hitl_rejected"``,
    skipping the LLM-react path that ``rejection_strategy="tool_error"`` uses.

    Carries the request + decision so downstream tooling (transcript,
    error reporters) can attribute the failure to a specific approver.
    """

    def __init__(self, request: "PendingHumanRequest", decision):
        super().__init__(
            f"HITL rejection (abort_task) for request {request.request_id}: "
            f"{decision.reason or 'user rejected'}"
        )
        self.request = request
        self.decision = decision


class PendingHumanException(Exception):
    """Runtime control-flow exception — MUST NOT be swallowed.

    Raised by ``Runtime.wait_for_human()`` when ``sync_timeout`` elapses
    without a decision AND the context allows async degradation (i.e. we
    are not inside a sub-agent). The main loop catches it at the outer
    layer to build a checkpoint and return ``RunResult(status="awaiting_human")``.

    Must be re-raised through:
    - ``HookRegistry.call`` / ``call_sync``
    - ``ToolExecutor.execute_single_tool``
    - ``asyncio.gather`` fan-in paths
    - ``SubAgentRunner`` (lets it bubble to the parent main loop)

    Only the outermost ``MainLoopRunner`` is allowed to catch.

    Attributes
    ----------
    request:
        The :class:`PendingHumanRequest` that timed out waiting for a decision.
    snapshot:
        :class:`RuntimeSnapshot` captured when ``MainLoopRunner._run_inner`` intercepts
        this exception before re-raising. Upstream layers (pipeline) persist it
        via :class:`CheckpointStore.save`. ``None`` at the raise site; populated
        by the main loop on the way up.
    """

    def __init__(
        self,
        request: PendingHumanRequest,
        snapshot: "RuntimeSnapshot | None" = None,
    ):
        super().__init__(
            f"Pending human decision for request {request.request_id} "
            f"(hook_point={request.hook_point}, tool_call_id={request.tool_call_id})"
        )
        self.request = request
        self.snapshot: "RuntimeSnapshot | None" = snapshot
