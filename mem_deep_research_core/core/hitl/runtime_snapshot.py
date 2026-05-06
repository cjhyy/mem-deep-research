"""RuntimeSnapshot build/restore primitives.

Phase 0 responsibility: assemble a :class:`RuntimeSnapshot` from the pieces
that already expose ``snapshot()`` / ``state_snapshot()`` / etc., and restore
those pieces from a snapshot.

The functions here are deliberately loose — callers pass the module
instances they want captured, and missing sources are silently skipped so
unit tests (and Phase 0 golden tests) can build snapshots piecemeal.

In Phase 2 ``MainLoopRunner`` will wire these into its ``_build_runtime_snapshot``
/ ``_restore_runtime_snapshot`` methods and add the tool-boundary cursor.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from mem_deep_research_core.core.hitl.types import (
    RUNTIME_SNAPSHOT_SCHEMA_VERSION,
    RuntimeSnapshot,
)


def build_snapshot(
    *,
    task_description: str = "",
    message_history: list[dict] | None = None,
    turn_count: int = 0,
    session_memory: dict | None = None,
    todo_state: dict | None = None,
    last_assistant_text: str = "",
    task_failed: bool = False,
    tool_calls_executed: int = 0,
    reflection_pending: bool = False,
    adaptive_pending: bool = False,
    effective_mode: str = "",
    reasoning_effort: str | None = None,
    context_manager: Any | None = None,
    monitor: Any | None = None,
    inline_skill_selector: Any | None = None,
    llm_client: Any | None = None,
    include_sub_agent_contextvar: bool = True,
    framework_version: str = "",
    # Phase 2 placeholders — callers may already pass these; we retain the shape.
    assistant_response_text: str = "",
    current_tool_calls: list[dict] | None = None,
    current_tool_index: int = 0,
    completed_tool_results: list[tuple[str, str]] | None = None,
    effective_arguments: dict | None = None,
    pending_human_request: Any | None = None,
) -> RuntimeSnapshot:
    """Assemble a RuntimeSnapshot from the pieces currently in memory.

    Any source left as None is simply omitted — the corresponding snapshot
    field stays at its default. That makes Phase 0 unit tests easy: pass
    only what you want to exercise.
    """
    snap = RuntimeSnapshot(
        schema_version=RUNTIME_SNAPSHOT_SCHEMA_VERSION,
        framework_version=framework_version,
        checkpoint_created_at=datetime.now(timezone.utc),
        task_description=task_description,
        message_history=list(message_history or []),
        turn_count=turn_count,
        session_memory=dict(session_memory or {}),
        todo_state=dict(todo_state) if todo_state else None,
        last_assistant_text=last_assistant_text,
        task_failed=task_failed,
        tool_calls_executed=tool_calls_executed,
        reflection_pending=reflection_pending,
        adaptive_pending=adaptive_pending,
        effective_mode=effective_mode,
        reasoning_effort=reasoning_effort,
        assistant_response_text=assistant_response_text,
        current_tool_calls=list(current_tool_calls or []),
        current_tool_index=current_tool_index,
        completed_tool_results=list(completed_tool_results or []),
        effective_arguments=dict(effective_arguments) if effective_arguments else None,
        pending_human_request=pending_human_request,
    )

    if context_manager is not None:
        snap.context_manager_state = context_manager.snapshot()
    if monitor is not None:
        snap.monitor_state = monitor.state_snapshot()
    if inline_skill_selector is not None:
        snap.inline_skill_state = inline_skill_selector.snapshot()

    contextvar_state: dict[str, dict[str, Any]] = {}
    if llm_client is not None and hasattr(llm_client, "save_contextvar_state"):
        contextvar_state["llm_provider"] = llm_client.save_contextvar_state()
    if include_sub_agent_contextvar:
        # Imported lazily so tests that don't touch sub-agent code avoid the cost.
        from mem_deep_research_core.core.sub_agent_runner import (
            save_sub_agent_contextvar_state,
        )

        contextvar_state["sub_agent_runner"] = save_sub_agent_contextvar_state()
    snap.contextvar_state = contextvar_state

    return snap


class SnapshotSchemaMismatch(RuntimeError):
    """Raised when the stored snapshot schema version no longer matches."""


def restore_snapshot(
    snap: RuntimeSnapshot,
    *,
    context_manager: Any | None = None,
    monitor: Any | None = None,
    inline_skill_selector: Any | None = None,
    llm_client: Any | None = None,
    strict_schema: bool = True,
) -> None:
    """Rehydrate live modules from a RuntimeSnapshot.

    Raises SnapshotSchemaMismatch if the snapshot schema no longer
    matches the current code version and ``strict_schema=True``. The HITL
    design calls for a migration path here; Phase 0 just fails loudly.
    """
    if strict_schema and snap.schema_version != RUNTIME_SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotSchemaMismatch(
            f"RuntimeSnapshot schema {snap.schema_version} does not match "
            f"current schema {RUNTIME_SNAPSHOT_SCHEMA_VERSION}."
        )

    if context_manager is not None and snap.context_manager_state:
        context_manager.restore(snap.context_manager_state)
    if monitor is not None and snap.monitor_state:
        monitor.restore_state(snap.monitor_state)
    if inline_skill_selector is not None and snap.inline_skill_state:
        inline_skill_selector.restore(snap.inline_skill_state)

    ctxvar = snap.contextvar_state or {}
    if llm_client is not None and "llm_provider" in ctxvar and hasattr(
        llm_client, "restore_contextvar_state"
    ):
        llm_client.restore_contextvar_state(ctxvar["llm_provider"])
    if "sub_agent_runner" in ctxvar:
        from mem_deep_research_core.core.sub_agent_runner import (
            restore_sub_agent_contextvar_state,
        )

        restore_sub_agent_contextvar_state(ctxvar["sub_agent_runner"])


def clone_snapshot(snap: RuntimeSnapshot) -> RuntimeSnapshot:
    """Shallow structural clone useful for golden tests (does not deep-copy module state)."""
    return replace(snap)
