"""HITL / RuntimeSnapshot data types.

Phase 0 shipped :class:`RuntimeSnapshot`. Phase 1 adds the HITL control-flow
surface: :class:`HumanDecision`, :class:`PendingHumanRequest`, and
:class:`RunResult`. :class:`PendingHumanException` lives in
:mod:`mem_deep_research_core.core.hitl.exceptions` to avoid circular imports
between exception-only modules and the types module that already depends on
the rest of the runtime.

See ``docs/23-hitl-design.md`` for the full design.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

# Bump whenever the snapshot dataclass gains/loses fields or changes meaning.
# Restoring an older snapshot with a newer code version must fail loudly
# (see ``restore_snapshot`` for the check).
RUNTIME_SNAPSHOT_SCHEMA_VERSION = 1


@dataclass
class RuntimeSnapshot:
    """Durable snapshot of MainLoopRunner state for HITL / durable execution.

    Phase 0 captures:
    - schema metadata (version / framework version / timestamp)
    - conversation state (message_history / turn_count / session_memory / todos)
    - module state (offload registry, dedup cache, monitor state, inline-skill state)
    - ContextVar state for each runtime-owned context var
    - main-loop transient flags (reflection_pending / adaptive routing result)

    Phase 2 (not yet populated by Phase 0) extends with:
    - ``current_tool_calls`` / ``current_tool_index`` / ``completed_tool_results``
    - ``effective_arguments`` (pending tool's HITL-approved args)
    - ``pending_human_request``

    Fields default to safe empties so callers can build a partial snapshot
    when suspending only a subset of state (e.g. first HITL rollout).
    """

    # -- Metadata ------------------------------------------------------
    schema_version: int = RUNTIME_SNAPSHOT_SCHEMA_VERSION
    framework_version: str = ""
    checkpoint_created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # -- Conversation / task state -------------------------------------
    # task_description: the original user query — captured so resume can be
    # invoked with just (checkpoint_id, decision) and not require the caller
    # to re-supply the prompt. Empty string for older snapshots that predate
    # this field (forward-compatible).
    task_description: str = ""
    message_history: list[dict] = field(default_factory=list)
    turn_count: int = 0
    session_memory: dict = field(default_factory=dict)
    todo_state: dict | None = None
    last_assistant_text: str = ""
    task_failed: bool = False
    tool_calls_executed: int = 0

    # -- Tool-boundary cursor (Phase 2 — placeholders so schema is stable) ---
    assistant_response_text: str = ""
    current_tool_calls: list[dict] = field(default_factory=list)
    current_tool_index: int = 0
    # (tool_call_id, offload_ref) pairs — full result lives in offload registry.
    completed_tool_results: list[tuple[str, str]] = field(default_factory=list)
    # Phase 2 single-pending shape; Phase 3 evolves to dict[tool_call_id, dict].
    effective_arguments: dict | None = None

    # -- Main-loop transient flags -------------------------------------
    reflection_pending: bool = False
    adaptive_pending: bool = False
    effective_mode: str = ""
    reasoning_effort: str | None = None

    # -- Module state (opaque dicts produced by each module's snapshot()) ---
    context_manager_state: dict = field(default_factory=dict)
    monitor_state: dict = field(default_factory=dict)
    inline_skill_state: dict = field(default_factory=dict)

    # -- ContextVar state ---------------------------------------------
    # Keyed by owner ("llm_provider" / "sub_agent_runner"), value = owner-supplied dict.
    contextvar_state: dict[str, dict[str, Any]] = field(default_factory=dict)

    # -- HITL request (Phase 2 — None in Phase 0) ----------------------
    pending_human_request: "PendingHumanRequest | None" = None


# ======================================================================
# HITL control-flow types (Phase 1)
# ======================================================================


# Grace period added when computing ``PendingHumanRequest.expires_at`` so clock
# drift between the approver and the framework host doesn't prematurely expire
# a legitimately-in-flight decision.
_DEFAULT_EXPIRES_GRACE_SECONDS = 60.0


def _generate_request_id() -> str:
    """Short, unique request identifier used across transcript / store / checkpoint."""
    return f"hitl_{uuid.uuid4().hex[:16]}"


@dataclass
class HumanDecision:
    """Decision emitted by the human approver.

    ``payload`` may carry modified tool arguments under the ``"args"`` key
    (the same convention the design doc uses). Additional fields are
    allowed but ignored by the framework in Phase 1.
    """

    approved: bool
    reason: str | None = None
    payload: dict | None = None
    decided_by: str | None = None
    decided_at: datetime | None = None


@dataclass
class PendingHumanRequest:
    """Snapshot of an open ``wait_for_human`` request.

    Phase 1 populates this mainly for tests and audit trails — the main
    loop does not yet persist it. Phase 2 checkpoints it alongside the
    ``RuntimeSnapshot``.
    """

    prompt: str
    payload: dict = field(default_factory=dict)
    hook_point: str = "on_tool_start"
    turn_number: int = 0
    tool_call_id: str = ""
    sync_timeout: float = 30.0
    async_timeout: float = 3600.0
    tags: list[str] = field(default_factory=list)
    request_id: str = field(default_factory=_generate_request_id)
    checkpoint_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Populated lazily by :meth:`compute_expires_at` once the async_timeout is known.
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(
                seconds=self.async_timeout + _DEFAULT_EXPIRES_GRACE_SECONDS
            )


@dataclass
class RunResult:
    """Three-state run outcome (Phase 2 breaking change replaces TaskResult).

    Phase 1 defines the dataclass so HITL types are coherent, but
    ``DeepResearch.run()`` still returns ``TaskResult`` — see the roadmap
    for the v1.3.0 → v1.4.0 migration.
    """

    task_id: str
    status: Literal["completed", "failed", "awaiting_human"] = "completed"
    answer: str | None = None
    boxed_answer: str = ""
    duration_seconds: float = 0.0
    log_path: Path | None = None
    error: str | None = None
    turns: int = 0
    tool_calls: int = 0
    error_type: str | None = None
    perf_metrics: dict | None = None
    checkpoints: list | None = None

    # HITL extension (only populated when status == "awaiting_human").
    checkpoint_id: str | None = None
    pending_human_request: PendingHumanRequest | None = None

    @property
    def success(self) -> bool:
        return self.status == "completed" and self.error is None
