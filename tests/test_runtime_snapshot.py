"""Phase 0 golden tests: RuntimeSnapshot round-trip + hook async contract.

Each module that participates in HITL durable execution (ContextManager,
ExecutionMonitor, InlineSkillSelector, LLM providers, sub_agent_runner)
must be able to dump its state and reload it so the restored instance is
equivalent to the original. The hook system must also accept sync and
async hooks through the new async ``call()`` entry point.

These tests are the contract; Phase 2 leans on them whenever snapshot
shape evolves.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from mem_deep_research_core.core.context_manager import (
    ContextManager,
    ContextManagerConfig,
    OffloadRecord,
    ToolCallRecord,
)
from mem_deep_research_core.core.hitl import (
    RUNTIME_SNAPSHOT_SCHEMA_VERSION,
    RuntimeSnapshot,
    build_snapshot,
    restore_snapshot,
)
from mem_deep_research_core.core.hitl.runtime_snapshot import SnapshotSchemaMismatch
from mem_deep_research_core.core.hooks import HookContext, HookRegistry
from mem_deep_research_core.core.monitoring import (
    EscalationAction,
    ExecutionMonitor,
    MonitoringConfig,
)
from mem_deep_research_core.core.sub_agent_runner import (
    _is_sub_agent_var,
    restore_sub_agent_contextvar_state,
    save_sub_agent_contextvar_state,
)
from mem_deep_research_core.llm.provider_client_base import _temperature_override_var
from mem_deep_research_core.llm.providers.deepseek_openrouter_client import (
    _native_tool_name_map_var,
    _pending_tool_list_var,
)
from pathlib import Path
import tempfile

from mem_deep_research_core.skills.inline_selector import InlineSkillSelector
from mem_deep_research_core.skills.matcher import SkillMatcher


def _make_matcher() -> SkillMatcher:
    """Build an empty SkillMatcher rooted at a throwaway temp directory."""
    return SkillMatcher(skills_dir=Path(tempfile.mkdtemp(prefix="snapshot_skills_")))


def _make_base_provider():
    """Construct an LLMProviderClientBase instance bypassing the heavy __post_init__.

    save/restore_contextvar_state uses no instance state, so we only need
    ``super()`` resolution to work, which requires a real (subclass) instance.
    """
    from mem_deep_research_core.llm.provider_client_base import LLMProviderClientBase

    class _TestProvider(LLMProviderClientBase):
        # Fill the abstract surface with no-op stubs.
        def _create_client(self, config):
            return None

        async def _create_message(self, *args, **kwargs):
            return None

        def process_llm_response(self, *args, **kwargs):
            return "", True

        def extract_tool_calls_info(self, *args, **kwargs):
            return [], []

        def update_message_history(self, *args, **kwargs):
            return None

    return _TestProvider.__new__(_TestProvider)


def _make_deepseek_provider():
    from mem_deep_research_core.llm.providers.deepseek_openrouter_client import (
        DeepSeekOpenRouterClient,
    )

    return DeepSeekOpenRouterClient.__new__(DeepSeekOpenRouterClient)


# ======================================================================
# ContextManager snapshot / restore
# ======================================================================


class TestContextManagerSnapshot:
    def test_snapshot_restore_round_trip(self):
        cm = ContextManager(config=ContextManagerConfig())
        cm._current_turn = 7
        cm._offload_registry["ref_a"] = OffloadRecord(
            ref="ref_a", turn=3, char_count=12345, tool_names=["search"]
        )
        cm._dedup_cache["hash_a"] = ToolCallRecord(
            tool_name="search",
            arguments_hash="hash_a",
            arguments={"q": "x"},
            turn=3,
            result_hash="rh",
            result_brief="brief",
        )
        cm._compacted_turns.update({1, 2, 3})

        snap = cm.snapshot()

        fresh = ContextManager(config=ContextManagerConfig())
        fresh.restore(snap)

        assert fresh._current_turn == 7
        assert fresh._offload_registry["ref_a"].char_count == 12345
        assert fresh._offload_registry["ref_a"].tool_names == ["search"]
        assert fresh._dedup_cache["hash_a"].tool_name == "search"
        assert fresh._compacted_turns == {1, 2, 3}

    def test_snapshot_is_independent_after_mutation(self):
        cm = ContextManager(config=ContextManagerConfig())
        cm._offload_registry["r1"] = OffloadRecord(ref="r1", turn=0, char_count=1)
        snap = cm.snapshot()
        cm._offload_registry["r2"] = OffloadRecord(ref="r2", turn=0, char_count=2)

        # Snapshot was taken before r2 was added; restoring gives a CM without r2.
        fresh = ContextManager(config=ContextManagerConfig())
        fresh.restore(snap)
        assert "r1" in fresh._offload_registry
        assert "r2" not in fresh._offload_registry


# ======================================================================
# ExecutionMonitor state snapshot / restore
# ======================================================================


class TestExecutionMonitorSnapshot:
    def test_state_round_trip(self):
        mon = ExecutionMonitor(config=MonitoringConfig())
        mon.state.consecutive_empty_turns = 2
        mon.state.response_loop_escalation_count = 1
        mon.state.response_hash_window = ["h1", "h2"]
        mon.state.tool_loop_retry_count = 1
        mon.state.stall_warned = True
        mon.state.attempted_strategies = ["tried_A", "tried_B"]
        mon._last_loop_action = EscalationAction.INJECT_HINT
        mon._soft_timeout_fired = True

        snap = mon.state_snapshot()

        fresh = ExecutionMonitor(config=MonitoringConfig())
        fresh.restore_state(snap)

        assert fresh.state.consecutive_empty_turns == 2
        assert fresh.state.response_loop_escalation_count == 1
        assert fresh.state.response_hash_window == ["h1", "h2"]
        assert fresh.state.tool_loop_retry_count == 1
        assert fresh.state.stall_warned is True
        assert fresh.state.attempted_strategies == ["tried_A", "tried_B"]
        assert fresh._last_loop_action == EscalationAction.INJECT_HINT
        assert fresh._soft_timeout_fired is True


# ======================================================================
# InlineSkillSelector snapshot / restore
# ======================================================================


class TestInlineSkillSelectorSnapshot:
    def test_snapshot_round_trip(self):
        sel = InlineSkillSelector(matcher=_make_matcher())
        sel._pending_skills = ["skill_a", "skill_b"]
        sel._first_turn = False
        sel._touched_files = ["foo.py", "bar.py"]

        snap = sel.snapshot()

        fresh = InlineSkillSelector(matcher=_make_matcher())
        fresh.restore(snap)

        assert fresh._pending_skills == ["skill_a", "skill_b"]
        assert fresh._first_turn is False
        assert fresh._touched_files == ["foo.py", "bar.py"]


# ======================================================================
# ContextVar save / restore contract
# ======================================================================


class TestContextVarRoundTrip:
    def test_base_provider_temperature_round_trip(self):
        """LLMProviderClientBase.save/restore manages only _temperature_override_var."""
        provider = _make_base_provider()

        _temperature_override_var.set(0.42)
        state = provider.save_contextvar_state()
        _temperature_override_var.set(None)
        provider.restore_contextvar_state(state)
        assert _temperature_override_var.get(None) == 0.42

    def test_deepseek_contextvars_round_trip(self):
        """DeepSeekOpenRouterClient extends the base state with its two extra vars."""
        provider = _make_deepseek_provider()

        _temperature_override_var.set(0.8)
        _pending_tool_list_var.set([{"name": "t"}])
        _native_tool_name_map_var.set({"canonical": "native"})

        state = provider.save_contextvar_state()

        _temperature_override_var.set(None)
        _pending_tool_list_var.set(None)
        _native_tool_name_map_var.set(None)

        provider.restore_contextvar_state(state)

        assert _temperature_override_var.get(None) == 0.8
        assert _pending_tool_list_var.get(None) == [{"name": "t"}]
        assert _native_tool_name_map_var.get(None) == {"canonical": "native"}

    def test_sub_agent_contextvar_round_trip(self):
        _is_sub_agent_var.set(True)
        state = save_sub_agent_contextvar_state()
        _is_sub_agent_var.set(False)
        restore_sub_agent_contextvar_state(state)
        assert _is_sub_agent_var.get(False) is True


# ======================================================================
# build_snapshot / restore_snapshot end-to-end
# ======================================================================


class TestRuntimeSnapshotRoundTrip:
    def test_end_to_end_round_trip(self):
        cm = ContextManager(config=ContextManagerConfig())
        cm._current_turn = 5
        cm._offload_registry["r"] = OffloadRecord(ref="r", turn=1, char_count=10)

        mon = ExecutionMonitor(config=MonitoringConfig())
        mon.state.consecutive_empty_turns = 3
        mon._last_loop_action = EscalationAction.WARN

        sel = InlineSkillSelector(matcher=_make_matcher())
        sel._pending_skills = ["alpha"]
        sel._first_turn = False

        _temperature_override_var.set(0.6)
        _is_sub_agent_var.set(False)

        snap = build_snapshot(
            message_history=[{"role": "user", "content": "hi"}],
            turn_count=5,
            last_assistant_text="answer",
            effective_mode="deep",
            reasoning_effort="high",
            context_manager=cm,
            monitor=mon,
            inline_skill_selector=sel,
            framework_version="test",
        )

        # Perturb everything before restore to prove restoration works.
        cm_fresh = ContextManager(config=ContextManagerConfig())
        mon_fresh = ExecutionMonitor(config=MonitoringConfig())
        sel_fresh = InlineSkillSelector(matcher=_make_matcher())
        _temperature_override_var.set(None)
        _is_sub_agent_var.set(True)

        restore_snapshot(
            snap,
            context_manager=cm_fresh,
            monitor=mon_fresh,
            inline_skill_selector=sel_fresh,
        )

        assert cm_fresh._current_turn == 5
        assert "r" in cm_fresh._offload_registry
        assert mon_fresh.state.consecutive_empty_turns == 3
        assert mon_fresh._last_loop_action == EscalationAction.WARN
        assert sel_fresh._pending_skills == ["alpha"]
        assert sel_fresh._first_turn is False
        assert snap.turn_count == 5
        assert snap.effective_mode == "deep"
        assert snap.reasoning_effort == "high"
        assert snap.schema_version == RUNTIME_SNAPSHOT_SCHEMA_VERSION
        # sub_agent_var round-tripped via contextvar_state.
        assert _is_sub_agent_var.get(False) is False

    def test_schema_mismatch_raises(self):
        snap = build_snapshot()
        mismatched = replace(snap, schema_version=RUNTIME_SNAPSHOT_SCHEMA_VERSION + 7)
        with pytest.raises(SnapshotSchemaMismatch):
            restore_snapshot(mismatched)


# ======================================================================
# Hook async/sync interop
# ======================================================================


class TestHookAsyncContract:
    def test_async_call_runs_sync_hook_natively(self):
        """Sync hooks run on the event loop thread directly (no to_thread)."""
        registry = HookRegistry()
        calls = []

        def sync_hook(ctx, next_fn):
            calls.append("sync")
            return next_fn(ctx)

        registry.register_fn("on_agent_start", sync_hook)
        registry.set_default("on_agent_start", lambda ctx: "result")

        result = asyncio.run(
            registry.call("on_agent_start", HookContext(hook_name="on_agent_start"))
        )
        assert result == "result"
        assert calls == ["sync"]

    def test_async_call_awaits_async_hook(self):
        registry = HookRegistry()
        events = []

        async def async_hook(ctx, next_fn):
            events.append("before")
            inner = next_fn(ctx)
            if asyncio.iscoroutine(inner):
                inner = await inner
            events.append("after")
            return inner

        registry.register_fn("on_agent_start", async_hook)
        registry.set_default("on_agent_start", lambda ctx: "done")

        result = asyncio.run(
            registry.call("on_agent_start", HookContext(hook_name="on_agent_start"))
        )
        assert result == "done"
        assert events == ["before", "after"]

    def test_call_sync_rejects_async_hook(self):
        registry = HookRegistry()
        stray = []

        async def async_hook(ctx, next_fn):
            return "should not run"

        def wrapping_hook(ctx, next_fn):
            # Materialise the coroutine and close it so the rejection path sees
            # an awaitable without pytest warning about a never-awaited coroutine.
            coro = async_hook(ctx, next_fn)
            stray.append(coro)
            return coro

        registry.register_fn("on_agent_start", wrapping_hook)
        try:
            with pytest.raises(RuntimeError, match="async result"):
                registry.call_sync(
                    "on_agent_start", HookContext(hook_name="on_agent_start")
                )
        finally:
            for coro in stray:
                coro.close()

    def test_on_suspend_and_on_resume_are_supported(self):
        registry = HookRegistry()
        assert "on_suspend" in registry.SUPPORTED_HOOKS
        assert "on_resume" in registry.SUPPORTED_HOOKS
        # Registration smoke-test — no TypeError.
        registry.register_fn("on_suspend", lambda ctx, nxt: nxt(ctx))
        registry.register_fn("on_resume", lambda ctx, nxt: nxt(ctx))
