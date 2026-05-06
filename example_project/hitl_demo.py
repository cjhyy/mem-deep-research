"""HITL (Human-in-the-Loop) demo — three flows in one runnable script.

This demo doesn't hit the LLM; it exercises the HITL surface
(``RuntimeFacade.wait_for_human`` + ``HitlRejectedError`` + checkpoint
suspend) directly so you can see the contract without paying for tokens.

Run:
    python example_project/hitl_demo.py

What it shows:
  1. APPROVE — a tool gated by ``wait_for_human``; an in-process approver
     supplies ``HumanDecision(approved=True)``; the tool runs normally with
     the approved arguments.
  2. REJECT (tool_error strategy) — approver returns
     ``approved=False``; the framework injects a tool-error message so the
     LLM (in a real run) would react.
  3. REJECT (abort_task strategy) — same rejection but with
     ``cfg.hitl.rejection_strategy="abort_task"`` → ``HitlRejectedError``
     bubbles out and the task ends with ``status=failed``.
  4. TIMEOUT → SUSPEND → RESUME — approver doesn't reply in time;
     ``PendingHumanException`` fires, framework persists a checkpoint, the
     demo loads it, delivers a late decision, and resumes from the tool
     boundary.

Patterns to copy into your project:
  - The approval hook (``approval_gate`` below) — gate tools by name, call
    ``ctx.runtime.wait_for_human``, translate decisions to actions.
  - The CLI approver — a coroutine that watches ``pending_store`` and
    delivers decisions. In production, replace with Slack / email / web UI.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from mem_deep_research_core import GuardrailError
from mem_deep_research_core.core.hitl import (
    HitlRejectedError,
    HumanDecision,
    PendingHumanException,
    PendingHumanRequest,
)
from mem_deep_research_core.core.hitl.checkpoint_store import (
    FilesystemCheckpointStore,
)
from mem_deep_research_core.core.hitl.pending_store import InMemoryPendingStore
from mem_deep_research_core.core.hitl.runtime_facade import (
    RuntimeFacade,
    set_current_runtime,
)
from mem_deep_research_core.core.hooks import HookContext, HookRegistry
from mem_deep_research_core.core.transcript import Transcript

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("hitl_demo")


# ---------------------------------------------------------------------------
# Approval hook (the pattern you'd register in your project's hooks.py)
# ---------------------------------------------------------------------------

HIGH_RISK_TOOLS = {"send_email", "execute_sql", "delete_file"}


def make_approval_hook(rejection_strategy: str = "tool_error"):
    """Return a hook function that gates HIGH_RISK_TOOLS via wait_for_human.

    In a real project this lives in ``hooks.py`` and registers itself with
    ``hooks.register("on_tool_start")``. For the demo we register manually.

    On rejection, we honour ``rejection_strategy``:
      - "tool_error" (default): raise ``GuardrailError``; tool is replaced
        with an error message and the LLM continues
      - "abort_task": raise ``HitlRejectedError``; pipeline aborts the task
    """

    async def approval_gate(ctx: HookContext, next_fn):
        if ctx.tool_name not in HIGH_RISK_TOOLS:
            return await next_fn(ctx)

        log.info(f"[hook] gating {ctx.tool_name} via wait_for_human")
        runtime = ctx.runtime
        if runtime is None:
            # No HITL runtime attached — fail closed.
            raise GuardrailError("hitl_not_configured", "no runtime facade")

        try:
            decision = await runtime.wait_for_human(
                prompt=f"Approve {ctx.tool_name}?",
                payload={"tool": ctx.tool_name, "args": ctx.arguments},
                sync_timeout=2.0,
                async_timeout=60.0,
                tags=["risky_tool"],
            )
        except PendingHumanException:
            # Phase 2 path: timeout → durable suspend. Re-raise so the main
            # loop catches and persists a checkpoint.
            raise

        if not decision.approved:
            if rejection_strategy == "abort_task":
                # Caller wants the whole task to fail on rejection.
                raise HitlRejectedError(
                    PendingHumanRequest(prompt="rejected", tool_call_id=""),
                    decision,
                )
            raise GuardrailError(
                "manual_rejection",
                decision.reason or "user rejected",
            )

        # Approver may have edited the args; merge them into ctx.arguments
        # so the downstream tool sees the override.
        if decision.payload and "args" in decision.payload:
            ctx.arguments.update(decision.payload["args"])

        return await next_fn(ctx)

    return approval_gate


# ---------------------------------------------------------------------------
# Mock approvers — stand-ins for Slack / email / CLI prompts
# ---------------------------------------------------------------------------


async def mock_approver(facade: RuntimeFacade, decision: HumanDecision, *, delay: float = 0.1):
    """Watch pending_store, deliver ``decision`` to the first open request."""
    for _ in range(int(delay / 0.01) + 50):
        ids = list(facade.pending_store._futures.keys())  # type: ignore[attr-defined]
        if ids:
            await asyncio.sleep(delay)
            await facade.pending_store.put_decision(ids[0], decision)
            log.info(
                f"[approver] delivered {decision.approved=} reason={decision.reason!r} "
                f"to request {ids[0]}"
            )
            return
        await asyncio.sleep(0.01)
    raise AssertionError("no pending request appeared within wait window")


# ---------------------------------------------------------------------------
# Demo flows — each runs the approval hook through wait_for_human directly
# ---------------------------------------------------------------------------


async def _wait_for_human_via_hook(
    facade: RuntimeFacade,
    *,
    tool_name: str,
    arguments: dict,
    rejection_strategy: str = "tool_error",
):
    """Drive the approval hook chain just like ToolExecutor would."""
    hooks = facade.hooks
    assert hooks is not None
    hooks.clear()
    hooks.register_fn("on_tool_start", make_approval_hook(rejection_strategy))
    hooks.set_default("on_tool_start", lambda ctx: ctx.arguments)

    token = set_current_runtime(facade)
    try:
        return await hooks.call(
            "on_tool_start",
            HookContext(
                hook_name="on_tool_start",
                tool_name=tool_name,
                arguments=dict(arguments),
            ),
        )
    finally:
        from mem_deep_research_core.core.hitl.runtime_facade import _runtime_var

        _runtime_var.reset(token)


async def flow_approve(facade: RuntimeFacade) -> None:
    print("\n=== Flow 1: APPROVE ===")
    approver_task = asyncio.create_task(
        mock_approver(
            facade,
            HumanDecision(
                approved=True,
                reason="reviewed by alice",
                payload={"args": {"recipient": "ops@example.com"}},
                decided_by="alice@example.com",
            ),
        )
    )
    final_args = await _wait_for_human_via_hook(
        facade, tool_name="send_email", arguments={"recipient": "nobody@example.com"}
    )
    await approver_task
    print(f"  → tool would run with arguments: {final_args}")


async def flow_reject_tool_error(facade: RuntimeFacade) -> None:
    print("\n=== Flow 2: REJECT (tool_error strategy) ===")
    approver_task = asyncio.create_task(
        mock_approver(
            facade,
            HumanDecision(approved=False, reason="not safe", decided_by="bob@example.com"),
        )
    )
    try:
        await _wait_for_human_via_hook(
            facade, tool_name="execute_sql", arguments={"query": "DROP TABLE users"}
        )
    except GuardrailError as e:
        print(f"  → GuardrailError raised: {e} (LLM would see [HITL rejected] tool result)")
    await approver_task


async def flow_reject_abort_task(facade: RuntimeFacade) -> None:
    print("\n=== Flow 3: REJECT (abort_task strategy) ===")
    approver_task = asyncio.create_task(
        mock_approver(
            facade,
            HumanDecision(approved=False, reason="must not run", decided_by="carol@example.com"),
        )
    )
    try:
        await _wait_for_human_via_hook(
            facade,
            tool_name="delete_file",
            arguments={"path": "/etc/passwd"},
            rejection_strategy="abort_task",
        )
    except HitlRejectedError as e:
        print(f"  → HitlRejectedError: {e.decision.decided_by} | {e.decision.reason}")
        print("  → Task would terminate with status=failed (no LLM react)")
    await approver_task


async def flow_timeout_resume(facade: RuntimeFacade, checkpoint_dir: Path) -> None:
    print("\n=== Flow 4: TIMEOUT → SUSPEND → RESUME ===")
    facade.bind_tool_context(tool_call_id="tc_demo_resume", turn_number=1)

    # No approver this time — the wait will time out.
    try:
        await facade.wait_for_human(
            prompt="Approve send_email?",
            payload={"tool": "send_email", "args": {"recipient": "nobody"}},
            sync_timeout=0.05,  # short; trigger fast suspend
            async_timeout=300.0,
            tags=["resume_demo"],
        )
        print("  unexpected: timeout didn't fire")
        return
    except PendingHumanException as pending:
        print(f"  → suspended; request_id={pending.request.request_id}")

        # Persist a checkpoint with a hand-built RuntimeSnapshot so we can
        # demo the resume path without dragging in a full MainLoopRunner.
        from mem_deep_research_core.core.hitl import build_snapshot

        snapshot = build_snapshot(
            task_description="demo resume",
            pending_human_request=pending.request,
        )
        store = FilesystemCheckpointStore(checkpoint_dir)
        ckpt_id = await store.save(snapshot)
        print(f"  → checkpoint saved: {ckpt_id}")

        # Some time later (could be hours), the approver decides.
        decision = HumanDecision(
            approved=True,
            reason="reviewed offline",
            decided_by="dave@example.com",
            payload={"args": {"recipient": "ops@example.com"}},
        )
        loaded = await store.load(ckpt_id)
        print(
            f"  → resumed; loaded snapshot for request "
            f"{loaded.pending_human_request.request_id}"
        )
        print(f"  → would deliver decision: approved={decision.approved} "
              f"args_override={decision.payload}")
        await store.delete(ckpt_id)
        print("  → checkpoint deleted (consumed)")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def main() -> None:
    transcript = Transcript()
    facade = RuntimeFacade(
        hooks=HookRegistry(),
        pending_store=InMemoryPendingStore(),
        transcript=transcript,
        agent_name="demo",
    )

    with tempfile.TemporaryDirectory(prefix="hitl_demo_ckpt_") as tmpdir:
        await flow_approve(facade)
        await flow_reject_tool_error(facade)
        await flow_reject_abort_task(facade)
        await flow_timeout_resume(facade, Path(tmpdir))

    print("\n=== Audit trail (transcript) ===")
    for evt in transcript._events:
        if evt.event_type.startswith("hitl_"):
            data = {k: evt.data.get(k) for k in ("request_id", "approved", "reason", "decided_by")}
            print(f"  {evt.event_type}: {data}")


if __name__ == "__main__":
    asyncio.run(main())
