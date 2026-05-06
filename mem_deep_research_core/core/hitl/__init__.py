"""HITL / Durable execution infrastructure.

Phase 0 (current): data types, RuntimeSnapshot build/restore primitives.
Phase 1: synchronous HITL — wait_for_human() + GuardrailError path.
Phase 2: asynchronous HITL — PendingHumanException + checkpoint store + resume.
Phase 3: production hardening (pluggable stores, batch pending, audit trail).

Public surface at Phase 0 is intentionally narrow: RuntimeSnapshot + its
build/restore helpers. The main loop does not yet consume them; the
contract exists so downstream phases have something stable to target.
"""

from mem_deep_research_core.core.hitl.exceptions import (
    HitlRejectedError,
    PendingHumanException,
)
from mem_deep_research_core.core.hitl.runtime_snapshot import (
    build_snapshot,
    restore_snapshot,
)
from mem_deep_research_core.core.hitl.types import (
    RUNTIME_SNAPSHOT_SCHEMA_VERSION,
    HumanDecision,
    PendingHumanRequest,
    RunResult,
    RuntimeSnapshot,
)

__all__ = [
    "RUNTIME_SNAPSHOT_SCHEMA_VERSION",
    "HitlRejectedError",
    "HumanDecision",
    "PendingHumanException",
    "PendingHumanRequest",
    "RunResult",
    "RuntimeSnapshot",
    "build_snapshot",
    "restore_snapshot",
]
