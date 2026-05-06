"""Checkpoint store for durable HITL suspend.

Responsibility: persist a ``RuntimeSnapshot`` + its ``PendingHumanRequest``
when async HITL kicks in, and reload the pair when the approver resumes.

Phase 2 ships a filesystem implementation — one JSON file per checkpoint
under ``<output_dir>/pending/<checkpoint_id>.json``. Phase 3 adds
pluggable Redis / Postgres backends via the same Protocol.

Serialization rules (Phase 2):
- All dataclasses → dict via ``dataclasses.asdict``.
- ``datetime`` → ISO-8601 string (UTC).
- ``set`` → sorted list (keyed by insertion would lose determinism).
- ``OffloadRecord`` / ``ToolCallRecord`` / ``MonitoringState`` round-trip
  through ``dataclasses.asdict`` + ``<cls>(**payload)`` so live module
  state stays authoritative.
- ``pending_human_request.expires_at`` is recomputed on load so grace
  period math stays consistent even if the clock drifted.

Non-goals (deferred):
- Encryption at rest (deployment-layer concern per design doc).
- Cross-process locking for concurrent resume of the same checkpoint
  (filesystem store is single-writer — Phase 3 addresses multi-process).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mem_deep_research_core.core.context_manager import OffloadRecord, ToolCallRecord
from mem_deep_research_core.core.hitl.types import (
    RUNTIME_SNAPSHOT_SCHEMA_VERSION,
    HumanDecision,
    PendingHumanRequest,
    RuntimeSnapshot,
)
from mem_deep_research_core.core.monitoring import EscalationAction

logger = logging.getLogger("mem_deep_research")


def _generate_checkpoint_id() -> str:
    return f"ckpt_{uuid.uuid4().hex[:16]}"


# ======================================================================
# Serialization helpers
# ======================================================================


def _encode(obj: Any) -> Any:
    """Custom JSON encoder supporting datetime / set / dataclass / Path / Enum."""
    if isinstance(obj, datetime):
        return {"__type__": "datetime", "value": obj.isoformat()}
    if isinstance(obj, set):
        return {"__type__": "set", "value": sorted(obj)}
    if isinstance(obj, Path):
        return {"__type__": "path", "value": str(obj)}
    if isinstance(obj, EscalationAction):
        return {"__type__": "escalation_action", "value": obj.value}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            "__type__": "dataclass",
            "class": f"{obj.__class__.__module__}:{obj.__class__.__name__}",
            "fields": {k: _encode(v) for k, v in dataclasses.asdict(obj).items()},
        }
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        encoded = [_encode(v) for v in obj]
        return encoded if isinstance(obj, list) else {"__type__": "tuple", "value": encoded}
    return obj


def _decode(obj: Any) -> Any:
    """Inverse of :func:`_encode`."""
    if isinstance(obj, dict):
        kind = obj.get("__type__")
        if kind == "datetime":
            return datetime.fromisoformat(obj["value"])
        if kind == "set":
            return set(obj["value"])
        if kind == "path":
            return Path(obj["value"])
        if kind == "escalation_action":
            return EscalationAction(obj["value"])
        if kind == "tuple":
            return tuple(_decode(v) for v in obj["value"])
        if kind == "dataclass":
            return _decode_dataclass(obj["class"], obj["fields"])
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


# Dataclass classes the filesystem store is willing to reconstruct.
# Keeping this explicit avoids ``import_module`` on arbitrary checkpoint
# payloads — a small security / blast-radius win.
_DATACLASS_REGISTRY: dict[str, type] = {
    f"{OffloadRecord.__module__}:{OffloadRecord.__name__}": OffloadRecord,
    f"{ToolCallRecord.__module__}:{ToolCallRecord.__name__}": ToolCallRecord,
    f"{PendingHumanRequest.__module__}:{PendingHumanRequest.__name__}": PendingHumanRequest,
    f"{HumanDecision.__module__}:{HumanDecision.__name__}": HumanDecision,
}


def _decode_dataclass(class_path: str, fields: dict) -> Any:
    cls = _DATACLASS_REGISTRY.get(class_path)
    if cls is None:
        raise ValueError(
            f"Checkpoint references unknown dataclass {class_path}. "
            f"Register it in _DATACLASS_REGISTRY."
        )
    decoded_fields = {k: _decode(v) for k, v in fields.items()}
    return cls(**decoded_fields)


# ======================================================================
# Snapshot envelope (what actually hits disk)
# ======================================================================


def _snapshot_to_payload(snap: RuntimeSnapshot) -> dict:
    """Project a RuntimeSnapshot to a JSON-safe dict."""
    return {
        "schema_version": snap.schema_version,
        "framework_version": snap.framework_version,
        "checkpoint_created_at": _encode(snap.checkpoint_created_at),
        "task_description": snap.task_description,
        "message_history": _encode(snap.message_history),
        "turn_count": snap.turn_count,
        "session_memory": _encode(snap.session_memory),
        "todo_state": _encode(snap.todo_state),
        "last_assistant_text": snap.last_assistant_text,
        "task_failed": snap.task_failed,
        "tool_calls_executed": snap.tool_calls_executed,
        "assistant_response_text": snap.assistant_response_text,
        "current_tool_calls": _encode(snap.current_tool_calls),
        "current_tool_index": snap.current_tool_index,
        "completed_tool_results": _encode(snap.completed_tool_results),
        "effective_arguments": _encode(snap.effective_arguments),
        "reflection_pending": snap.reflection_pending,
        "adaptive_pending": snap.adaptive_pending,
        "effective_mode": snap.effective_mode,
        "reasoning_effort": snap.reasoning_effort,
        "context_manager_state": _encode(snap.context_manager_state),
        "monitor_state": _encode(snap.monitor_state),
        "inline_skill_state": _encode(snap.inline_skill_state),
        "contextvar_state": _encode(snap.contextvar_state),
        "pending_human_request": _encode(snap.pending_human_request),
    }


def _payload_to_snapshot(payload: dict) -> RuntimeSnapshot:
    """Inverse of :func:`_snapshot_to_payload`."""
    return RuntimeSnapshot(
        schema_version=payload["schema_version"],
        framework_version=payload.get("framework_version", ""),
        checkpoint_created_at=_decode(payload["checkpoint_created_at"]),
        task_description=payload.get("task_description", ""),
        message_history=_decode(payload.get("message_history") or []),
        turn_count=payload.get("turn_count", 0),
        session_memory=_decode(payload.get("session_memory") or {}),
        todo_state=_decode(payload.get("todo_state")),
        last_assistant_text=payload.get("last_assistant_text", ""),
        task_failed=payload.get("task_failed", False),
        tool_calls_executed=payload.get("tool_calls_executed", 0),
        assistant_response_text=payload.get("assistant_response_text", ""),
        current_tool_calls=_decode(payload.get("current_tool_calls") or []),
        current_tool_index=payload.get("current_tool_index", 0),
        completed_tool_results=_decode(payload.get("completed_tool_results") or []),
        effective_arguments=_decode(payload.get("effective_arguments")),
        reflection_pending=payload.get("reflection_pending", False),
        adaptive_pending=payload.get("adaptive_pending", False),
        effective_mode=payload.get("effective_mode", ""),
        reasoning_effort=payload.get("reasoning_effort"),
        context_manager_state=_decode(payload.get("context_manager_state") or {}),
        monitor_state=_decode(payload.get("monitor_state") or {}),
        inline_skill_state=_decode(payload.get("inline_skill_state") or {}),
        contextvar_state=_decode(payload.get("contextvar_state") or {}),
        pending_human_request=_decode(payload.get("pending_human_request")),
    )


# ======================================================================
# Protocol + Filesystem impl
# ======================================================================


@runtime_checkable
class CheckpointStore(Protocol):
    """Durable store for ``RuntimeSnapshot`` pending human decision."""

    async def save(self, snapshot: RuntimeSnapshot) -> str: ...

    async def load(self, checkpoint_id: str) -> RuntimeSnapshot: ...

    async def delete(self, checkpoint_id: str) -> None: ...

    async def list_checkpoints(self) -> list[str]: ...


class FilesystemCheckpointStore:
    """JSON-per-checkpoint store under ``<root>/pending/<id>.json``.

    Writes are atomic (tempfile + os.replace). Reads are lazy — the full
    payload is parsed on ``load`` only. Phase 2 does not implement
    cross-process locking; the single-writer assumption matches the
    single-DeepResearch-instance model.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root) / "pending"
        self._root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, checkpoint_id: str) -> Path:
        # Guard against path traversal — checkpoint ids come from us, but
        # resume is an external entrypoint so validate defensively.
        if "/" in checkpoint_id or ".." in checkpoint_id or not checkpoint_id:
            raise ValueError(f"Illegal checkpoint id: {checkpoint_id!r}")
        return self._root / f"{checkpoint_id}.json"

    async def save(self, snapshot: RuntimeSnapshot) -> str:
        if snapshot.schema_version != RUNTIME_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"Refusing to checkpoint schema {snapshot.schema_version} "
                f"under current code schema {RUNTIME_SNAPSHOT_SCHEMA_VERSION}."
            )
        checkpoint_id = _generate_checkpoint_id()
        if snapshot.pending_human_request is not None:
            snapshot.pending_human_request.checkpoint_id = checkpoint_id
        payload = _snapshot_to_payload(snapshot)

        target = self._path_for(checkpoint_id)
        # Atomic write via tempfile in the same directory (same FS for os.replace).
        fd, tmp_path = tempfile.mkstemp(
            prefix=f"{checkpoint_id}.", suffix=".json.tmp", dir=self._root
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_path, target)
        except Exception:
            # Cleanup the tempfile on failure so we don't leak
            # partially-written payloads across runs.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info(
            "[CheckpointStore] Saved checkpoint %s (schema=%s, path=%s)",
            checkpoint_id,
            snapshot.schema_version,
            target,
        )
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> RuntimeSnapshot:
        target = self._path_for(checkpoint_id)
        if not target.exists():
            raise KeyError(f"Checkpoint {checkpoint_id} not found at {target}")
        with target.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        snap = _payload_to_snapshot(payload)
        logger.info(
            "[CheckpointStore] Loaded checkpoint %s (schema=%s)",
            checkpoint_id,
            snap.schema_version,
        )
        return snap

    async def delete(self, checkpoint_id: str) -> None:
        target = self._path_for(checkpoint_id)
        try:
            target.unlink()
            logger.info("[CheckpointStore] Deleted checkpoint %s", checkpoint_id)
        except FileNotFoundError:
            logger.debug(
                "[CheckpointStore] Delete %s: already gone", checkpoint_id
            )

    async def list_checkpoints(self) -> list[str]:
        return sorted(p.stem for p in self._root.glob("*.json"))

    async def sweep_expired(self, *, now: datetime | None = None) -> list[str]:
        """Delete checkpoints whose pending request has passed ``expires_at``.

        Returns the list of deleted checkpoint ids — callers typically log
        or emit them to the transcript.
        """
        now = now or datetime.now(timezone.utc)
        deleted: list[str] = []
        for cid in await self.list_checkpoints():
            try:
                snap = await self.load(cid)
            except Exception as exc:
                logger.warning(
                    "[CheckpointStore] Skipping unparseable checkpoint %s during sweep: %s",
                    cid,
                    exc,
                )
                continue
            req = snap.pending_human_request
            if req is None or req.expires_at is None:
                continue
            expires = req.expires_at
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if expires < now:
                await self.delete(cid)
                deleted.append(cid)
        if deleted:
            logger.info(
                "[CheckpointStore] Swept %d expired checkpoints: %s", len(deleted), deleted
            )
        return deleted
