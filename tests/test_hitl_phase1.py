"""Phase 1 HITL acceptance tests.

These exercise the public surface of synchronous HITL:

- ``RuntimeFacade.wait_for_human()`` across approve / reject / timeout.
- Arguments modified by the approver become "effective arguments" — the value
  the tool executor uses after ``on_tool_start``.
- Sub-agent context forces the sync-only path (no durable suspend).
- ``PendingHumanException`` propagates through the hook chain, the tool
  executor, and the sub-agent runner.

Phase 2 will add end-to-end ``awaiting_human`` → ``resume`` tests; Phase 1
stops at "the pieces compose correctly in isolation".
"""

from __future__ import annotations

import asyncio

import pytest

from mem_deep_research_core import GuardrailError
from mem_deep_research_core.core.hitl import (
    HumanDecision,
    PendingHumanException,
    PendingHumanRequest,
)
from mem_deep_research_core.core.hitl.pending_store import InMemoryPendingStore
from mem_deep_research_core.core.hitl.runtime_facade import (
    RuntimeFacade,
    get_current_runtime,
    set_current_runtime,
)
from mem_deep_research_core.core.hooks import HookContext, HookRegistry
from mem_deep_research_core.core.sub_agent_runner import _is_sub_agent_var


# ======================================================================
# wait_for_human: approve / reject / timeout
# ======================================================================


class TestWaitForHumanSyncPaths:
    @pytest.mark.asyncio
    async def test_approve_returns_decision(self):
        facade = RuntimeFacade(hooks=HookRegistry(), pending_store=InMemoryPendingStore())
        facade.bind_tool_context(tool_call_id="tc_1", turn_number=4)

        async def approver():
            # Wait until the request is registered, then approve.
            for _ in range(50):
                open_ids = list(facade.pending_store._futures.keys())  # type: ignore[attr-defined]
                if open_ids:
                    await facade.pending_store.put_decision(
                        open_ids[0],
                        HumanDecision(approved=True, reason="ok", payload={"args": {"q": "new"}}),
                    )
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("No pending request appeared")

        approver_task = asyncio.create_task(approver())
        decision = await facade.wait_for_human(
            "Approve search?",
            payload={"tool": "search", "args": {"q": "old"}},
            sync_timeout=2.0,
        )
        await approver_task

        assert decision.approved is True
        assert decision.payload == {"args": {"q": "new"}}
        assert decision.decided_at is not None

    @pytest.mark.asyncio
    async def test_reject_returns_decision(self):
        facade = RuntimeFacade(hooks=HookRegistry(), pending_store=InMemoryPendingStore())

        async def approver():
            for _ in range(50):
                open_ids = list(facade.pending_store._futures.keys())  # type: ignore[attr-defined]
                if open_ids:
                    await facade.pending_store.put_decision(
                        open_ids[0], HumanDecision(approved=False, reason="manual reject")
                    )
                    return
                await asyncio.sleep(0.01)

        approver_task = asyncio.create_task(approver())
        decision = await facade.wait_for_human("Allow send_email?", sync_timeout=2.0)
        await approver_task

        assert decision.approved is False
        assert decision.reason == "manual reject"

    @pytest.mark.asyncio
    async def test_timeout_in_main_agent_raises_pending_human_exception(self):
        """Phase 2 upgrade: main-agent timeout is a durable-suspend signal."""
        facade = RuntimeFacade(hooks=HookRegistry(), pending_store=InMemoryPendingStore())

        with pytest.raises(PendingHumanException) as excinfo:
            await facade.wait_for_human("never answered", sync_timeout=0.1)
        # Request stays open for resume delivery.
        assert facade.pending_store.has(excinfo.value.request.request_id)

    @pytest.mark.asyncio
    async def test_on_await_human_fires_on_request(self):
        hooks = HookRegistry()
        seen = []

        async def notifier(ctx, next_fn):
            seen.append(ctx.extra["request"].prompt)
            return await next_fn(ctx)

        hooks.register_fn("on_await_human", notifier)
        facade = RuntimeFacade(hooks=hooks, pending_store=InMemoryPendingStore())

        # Main-agent timeout raises PendingHumanException in Phase 2; the
        # on_await_human hook fires unconditionally when the request is put.
        with pytest.raises(PendingHumanException):
            await facade.wait_for_human("approve me?", sync_timeout=0.1)
        assert seen == ["approve me?"]


# ======================================================================
# Effective arguments: hook modifies args; modification must stick
# ======================================================================


class TestEffectiveArguments:
    @pytest.mark.asyncio
    async def test_hook_mutates_arguments_in_place(self):
        """Approval hook updates ctx.arguments — downstream must see the update."""
        hooks = HookRegistry()
        captured: dict = {}

        async def approval_hook(ctx, next_fn):
            # Simulate approver rewriting recipient.
            ctx.arguments.update({"recipient": "ops@example.com"})
            return await next_fn(ctx)

        def terminal(ctx):
            captured.update(ctx.arguments)
            return ctx.arguments

        hooks.register_fn("on_tool_start", approval_hook)
        hooks.set_default("on_tool_start", terminal)

        result = await hooks.call(
            "on_tool_start",
            HookContext(
                hook_name="on_tool_start",
                tool_name="send_email",
                arguments={"recipient": "nobody@example.com", "body": "hi"},
            ),
        )
        assert result == {"recipient": "ops@example.com", "body": "hi"}
        assert captured == {"recipient": "ops@example.com", "body": "hi"}


# ======================================================================
# Sub-agent HITL restriction
# ======================================================================


class TestSubAgentRestriction:
    @pytest.mark.asyncio
    async def test_sub_agent_wait_for_human_still_sync(self):
        """Phase 1 treats sub-agent and main-agent timeouts the same (TimeoutError).

        Phase 2 will diverge: non-sub-agent timeouts raise PendingHumanException
        for durable suspend; sub-agent timeouts stay on the sync path. Phase 1
        just verifies that the context var is visible inside wait_for_human
        and that the wait does terminate on timeout.
        """
        token = _is_sub_agent_var.set(True)
        try:
            facade = RuntimeFacade(
                hooks=HookRegistry(), pending_store=InMemoryPendingStore()
            )
            with pytest.raises(asyncio.TimeoutError):
                await facade.wait_for_human("sub-agent prompt", sync_timeout=0.05)
        finally:
            _is_sub_agent_var.reset(token)


# ======================================================================
# PendingHumanException propagation contract
# ======================================================================


class TestPendingHumanExceptionPropagation:
    @pytest.mark.asyncio
    async def test_not_swallowed_by_async_hook_chain(self):
        hooks = HookRegistry()
        request = PendingHumanRequest(
            prompt="pending", hook_point="on_tool_start", tool_call_id="tc_2"
        )

        def raising(ctx, next_fn):
            raise PendingHumanException(request)

        hooks.register_fn("on_tool_start", raising)
        with pytest.raises(PendingHumanException) as excinfo:
            await hooks.call(
                "on_tool_start", HookContext(hook_name="on_tool_start")
            )
        assert excinfo.value.request is request

    def test_not_swallowed_by_sync_hook_chain(self):
        hooks = HookRegistry()
        request = PendingHumanRequest(
            prompt="pending", hook_point="on_system_prompt_build"
        )

        def raising(ctx, next_fn):
            raise PendingHumanException(request)

        hooks.register_fn("on_system_prompt_build", raising)
        with pytest.raises(PendingHumanException):
            hooks.call_sync(
                "on_system_prompt_build",
                HookContext(hook_name="on_system_prompt_build"),
            )

    @pytest.mark.asyncio
    async def test_subsequent_hooks_do_not_run_after_pending_raise(self):
        """Hook fallthrough swallows regular errors; PendingHumanException must NOT fall through."""
        hooks = HookRegistry()
        request = PendingHumanRequest(prompt="pending")
        trailing_ran = []

        def raising(ctx, next_fn):
            raise PendingHumanException(request)

        def trailing(ctx, next_fn):
            trailing_ran.append(True)
            return next_fn(ctx)

        # Higher priority runs first — raising hook fires before trailing.
        hooks.register_fn("on_tool_start", raising, priority=10)
        hooks.register_fn("on_tool_start", trailing, priority=1)

        with pytest.raises(PendingHumanException):
            await hooks.call(
                "on_tool_start", HookContext(hook_name="on_tool_start")
            )
        assert trailing_ran == []


# ======================================================================
# HookContext.runtime resolves from ContextVar
# ======================================================================


class TestHookContextRuntime:
    def test_ctx_runtime_returns_current_facade(self):
        facade = RuntimeFacade()
        token = set_current_runtime(facade)
        try:
            ctx = HookContext(hook_name="on_tool_start")
            assert ctx.runtime is facade
        finally:
            from mem_deep_research_core.core.hitl.runtime_facade import _runtime_var

            _runtime_var.reset(token)

    def test_ctx_runtime_is_none_when_unset(self):
        # Guard against leakage from other tests by explicitly clearing.
        assert get_current_runtime() is None
        ctx = HookContext(hook_name="on_tool_start")
        assert ctx.runtime is None
