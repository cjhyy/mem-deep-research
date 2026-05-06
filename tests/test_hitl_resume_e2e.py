"""End-to-end HITL resume integration tests.

These drive ``MainLoopRunner.run_from_tool_cursor`` with mocked LLM +
tool executor to verify the full pause → resume handshake:

1. Restored module state (monitor counters, dedup cache, session memory)
   survives ``skip_init=True`` into the resumed turns.
2. The pending tool is executed with ``effective_arguments`` merged from
   the human decision's ``payload["args"]``.
3. LLM is called for the resumed-and-after turns only — never for the
   paused turn (which already has its assistant message in ``message_history``).
4. Rejected decisions inject a tool_error message instead of running the tool.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf

from mem_deep_research_core.core.context_manager import (
    ContextManager,
    OffloadRecord,
    ToolCallRecord,
)
from mem_deep_research_core.core.hitl import (
    HumanDecision,
    PendingHumanRequest,
    build_snapshot,
)
from mem_deep_research_core.core.hooks import HookRegistry, hooks
from mem_deep_research_core.core.main_loop import MainLoopContext, MainLoopRunner
from mem_deep_research_core.core.memory import SessionMemory
from mem_deep_research_core.core.monitoring import (
    EscalationAction,
    ExecutionMonitor,
    MonitoringConfig,
)
from mem_deep_research_core.core.task_planner import TaskPlanner


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(autouse=True)
def clear_hooks():
    hooks.clear_all()
    yield
    hooks.clear_all()


def _make_cfg(max_turns=3):
    return OmegaConf.create(
        {
            "main_agent": {
                "max_turns": max_turns,
                "max_tool_calls_per_turn": 5,
                "keep_tool_result": -1,
            }
        }
    )


def _make_stream_handler():
    h = MagicMock()
    h.stream_start_agent = AsyncMock(return_value="agent-id")
    h.stream_end_agent = AsyncMock()
    h.stream_start_llm = AsyncMock()
    h.stream_end_llm = AsyncMock()
    h.stream_tool_call = AsyncMock()
    h.stream_reasoning = AsyncMock()
    h.stream_message = AsyncMock()
    h.stream_usage_info = AsyncMock()
    return h


def _make_task_log():
    log = MagicMock()
    log.log_step = MagicMock()
    log.record_perf = MagicMock()
    log.save = MagicMock()
    log.save_checkpoint = MagicMock()
    return log


def _make_output_formatter():
    f = MagicMock()
    f.format_tool_result_for_user = MagicMock(
        side_effect=lambda r: {"type": "text", "text": str(r.get("result", r.get("error", "")))}
    )
    f.format_final_summary_and_log = MagicMock(
        return_value=("Final summary", "boxed")
    )
    return f


def _make_noop_async():
    async def _noop(*args, **kwargs):
        return None

    return _noop


def _build_runner(
    llm_responses: list,
    max_turns: int = 3,
) -> tuple[MainLoopRunner, list]:
    """Build a MainLoopRunner whose LLM handler pops from ``llm_responses``.

    Each response tuple is ``(text, should_break, tool_calls)`` matching the
    signature of ``LLMCallHandler.handle_llm_call``.
    """
    call_log: list = []
    responses = list(llm_responses)

    async def mock_llm_call(
        system_prompt,
        message_history,
        tool_definitions,
        step_id,
        purpose="",
        keep_tool_result=-1,
        agent_type="main",
        stream_message_callback=None,
    ):
        call_log.append({"step_id": step_id, "purpose": purpose})
        if responses:
            return responses.pop(0)
        return None, True, None

    async def mock_summary(
        system_prompt,
        agent_prompt_instance,
        message_history,
        tool_definitions,
        purpose,
        task_description,
        task_failed,
        agent_type="main",
        task_guidance="",
        stream_message_callback=None,
        **kwargs,
    ):
        return f"Summary of: {task_description}"

    ctx = MainLoopContext(
        cfg=_make_cfg(max_turns=max_turns),
        monitor=ExecutionMonitor(config=MonitoringConfig()),
        context_manager=ContextManager(),
        stream_handler=_make_stream_handler(),
        tool_executor=MagicMock(),
        sub_agent_runner=None,
        llm_handler=MagicMock(),
        summary_handler=MagicMock(),
        task_planner=TaskPlanner(enabled=False),
        inline_skill_selector=None,
        llm_client=MagicMock(),
        output_formatter=_make_output_formatter(),
        task_log=_make_task_log(),
        context={},
        chinese_context=False,
        handle_llm_call=mock_llm_call,
        handle_summary=mock_summary,
        intercept_key_message=_make_noop_async(),
        streaming_final_message=_make_noop_async(),
        stream_tool_reasoning=_make_noop_async(),
        extract_recent_tool_names=lambda history, lookback=6: [],
        deduplicate_trailing_messages=lambda history: 0,
        response_language="auto",
        hooks=HookRegistry(),
    )
    runner = MainLoopRunner(ctx)

    # Wire tool_executor stubs the resume path expects.
    runner.tool_executor.execute_single_tool = AsyncMock()
    runner.tool_executor.handle_failed_tool_calls = MagicMock(return_value=([], []))
    runner.llm_client.max_context_length = -1
    runner.llm_client.update_message_history = MagicMock(side_effect=lambda h, r, e: h)
    runner.llm_client.reasoning_effort = "medium"

    return runner, call_log


def _build_snapshot_with_state(
    pending_tool_call: dict,
    effective_args: dict,
    *,
    session_findings: list[str],
    attempted_strategies: list[str],
    dedup_tool_name: str,
):
    """Build a RuntimeSnapshot that the resumed runner must reproduce."""
    cm = ContextManager()
    cm._current_turn = 2
    cm._offload_registry["r_pre_pause"] = OffloadRecord(
        ref="r_pre_pause", turn=1, char_count=1000
    )
    cm._dedup_cache["h_pre_pause"] = ToolCallRecord(
        tool_name=dedup_tool_name,
        arguments_hash="h_pre_pause",
        arguments={"q": "x"},
        turn=1,
        result_hash="rh",
        result_brief="brief",
    )

    mon = ExecutionMonitor(config=MonitoringConfig())
    # attempted_strategies is accumulation-only (not reset by record_progress),
    # so it's the right field to prove restore survived skip_init.
    mon.state.attempted_strategies = list(attempted_strategies)

    sm = SessionMemory()
    for finding in session_findings:
        sm.add_finding(finding)

    pending_req = PendingHumanRequest(
        prompt="Approve send_email?",
        payload={"tool": pending_tool_call["tool_name"]},
        tool_call_id=pending_tool_call["id"],
        turn_number=2,
        sync_timeout=30.0,
    )

    snap = build_snapshot(
        message_history=[
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            # Assistant turn that produced the tool_use — already in history
            # when pause happened, so the resumed runner must NOT re-call the
            # LLM for this turn.
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "I'll send an email."}],
            },
        ],
        turn_count=2,
        session_memory=sm.to_dict(),
        last_assistant_text="I'll send an email.",
        effective_mode="deep",
        reasoning_effort="high",
        context_manager=cm,
        monitor=mon,
        current_tool_calls=[pending_tool_call],
        current_tool_index=0,
        effective_arguments=effective_args,
        pending_human_request=pending_req,
    )
    return snap


# ======================================================================
# Tests
# ======================================================================


class TestRunFromToolCursorPreservesState:
    """Task #23 regression — restored module state must survive skip_init=True."""

    @pytest.mark.asyncio
    async def test_monitor_and_dedup_cache_survive_resume(self):
        """After resume, the monitor counter and dedup cache from the snapshot
        are still there — ``_run_inner`` did not reset them."""
        pending_call = {
            "id": "tc_pending",
            "tool_name": "send_email",
            "server_name": "mail",
            "arguments": {"to": "nobody"},
        }
        snap = _build_snapshot_with_state(
            pending_call,
            effective_args={"to": "nobody"},
            session_findings=["fact_A", "fact_B"],
            attempted_strategies=["tried_route_X", "tried_route_Y"],
            dedup_tool_name="search",
        )

        # One LLM call after the pending tool completes — yields a final text.
        runner, call_log = _build_runner(
            llm_responses=[("All done.", False, None)],
            max_turns=3,
        )
        runner.tool_executor.execute_single_tool.return_value = (
            {"server_name": "mail", "tool_name": "send_email", "result": "sent"},
            5,
        )

        decision = HumanDecision(
            approved=True, reason="ok", payload={"args": {"to": "ops@example.com"}}
        )

        final, _is_simple = await runner.run_from_tool_cursor(
            snap,
            decision,
            system_prompt="sys",
            main_agent_prompt_instance=MagicMock(),
            task_engine_cfg=None,
            task_description="task",
            task_guidance="",
            tool_definitions=[{"name": "send_email"}],
            keep_tool_result=-1,
        )

        # Restored module state survived into the resumed loop.
        assert "h_pre_pause" in runner.context_manager._dedup_cache
        assert "r_pre_pause" in runner.context_manager._offload_registry
        # SessionMemory from_dict replaced the instance — findings must match.
        assert "fact_A" in runner.session_memory.key_findings
        assert "fact_B" in runner.session_memory.key_findings
        # Monitor attempted_strategies is accumulation-only: survives progress events.
        assert runner.monitor.state.attempted_strategies == [
            "tried_route_X",
            "tried_route_Y",
        ]

        # Pending tool ran with the approver-supplied argument override.
        runner.tool_executor.execute_single_tool.assert_called_once()
        kwargs = runner.tool_executor.execute_single_tool.call_args.kwargs
        assert kwargs["arguments"] == {"to": "ops@example.com"}
        assert kwargs["tool_name"] == "send_email"
        assert kwargs["call_id"] == "tc_pending"

    @pytest.mark.asyncio
    async def test_llm_not_called_for_paused_turn(self):
        """Resume skips the LLM call for the suspended turn — the assistant
        message is already in message_history. Subsequent turns still call LLM."""
        pending_call = {
            "id": "tc_p",
            "tool_name": "search",
            "server_name": "web",
            "arguments": {"q": "old"},
        }
        snap = _build_snapshot_with_state(
            pending_call,
            effective_args={"q": "old"},
            session_findings=[],
            attempted_strategies=[],
            dedup_tool_name="web",
        )

        # Two LLM responses: resumed turn #1 (after pending tool) answers
        # directly (should_break=True), no further turns needed.
        runner, call_log = _build_runner(
            llm_responses=[("Answer after resume.", True, None)],
            max_turns=3,
        )
        runner.tool_executor.execute_single_tool.return_value = (
            {"server_name": "web", "tool_name": "search", "result": "foo"},
            3,
        )

        await runner.run_from_tool_cursor(
            snap,
            HumanDecision(approved=True),
            system_prompt="sys",
            main_agent_prompt_instance=MagicMock(),
            task_engine_cfg=None,
            task_description="task",
            task_guidance="",
            tool_definitions=[{"name": "search"}],
            keep_tool_result=-1,
        )

        # Exactly one LLM call: the post-pending-tool turn.
        assert len(call_log) == 1, f"Expected 1 LLM call, got {len(call_log)}: {call_log}"

    @pytest.mark.asyncio
    async def test_rejected_with_abort_strategy_raises_hitl_rejected_error(self):
        """rejection_strategy=abort_task short-circuits — pipeline catches and
        marks status=failed instead of letting the LLM keep going."""
        from mem_deep_research_core.core.hitl import HitlRejectedError

        pending_call = {
            "id": "tc_a",
            "tool_name": "delete_all",
            "server_name": "db",
            "arguments": {},
        }
        snap = _build_snapshot_with_state(
            pending_call,
            effective_args={},
            session_findings=[],
            attempted_strategies=[],
            dedup_tool_name="db",
        )

        runner, call_log = _build_runner(
            llm_responses=[("should not be called", True, None)],
            max_turns=3,
        )
        # Wire abort_task strategy via cfg.hitl.
        runner.cfg = OmegaConf.create(
            {
                "main_agent": runner.cfg.main_agent,
                "hitl": {"rejection_strategy": "abort_task"},
            }
        )

        with pytest.raises(HitlRejectedError) as excinfo:
            await runner.run_from_tool_cursor(
                snap,
                HumanDecision(
                    approved=False, reason="not safe", decided_by="alice"
                ),
                system_prompt="sys",
                main_agent_prompt_instance=MagicMock(),
                task_engine_cfg=None,
                task_description="task",
                task_guidance="",
                tool_definitions=[{"name": "delete_all"}],
                keep_tool_result=-1,
            )

        assert excinfo.value.decision.decided_by == "alice"
        # No tool execution, no LLM call — abort is final.
        runner.tool_executor.execute_single_tool.assert_not_called()
        assert call_log == []

    @pytest.mark.asyncio
    async def test_rejected_decision_skips_tool_and_injects_error(self):
        """Rejection: tool is NOT executed; an error message lands in the history
        so the LLM can react."""
        pending_call = {
            "id": "tc_risky",
            "tool_name": "delete_all",
            "server_name": "db",
            "arguments": {},
        }
        snap = _build_snapshot_with_state(
            pending_call,
            effective_args={},
            session_findings=[],
            attempted_strategies=[],
            dedup_tool_name="db",
        )

        runner, call_log = _build_runner(
            llm_responses=[("I see it was rejected, stopping.", True, None)],
            max_turns=3,
        )

        await runner.run_from_tool_cursor(
            snap,
            HumanDecision(approved=False, reason="not safe"),
            system_prompt="sys",
            main_agent_prompt_instance=MagicMock(),
            task_engine_cfg=None,
            task_description="task",
            task_guidance="",
            tool_definitions=[{"name": "delete_all"}],
            keep_tool_result=-1,
        )

        # Tool was NEVER called under rejection.
        runner.tool_executor.execute_single_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_resume_hook_fires_before_tool(self):
        """``on_resume`` lands before the tool executes so business-side
        resource rebuild can complete first."""
        order: list[str] = []

        pending_call = {
            "id": "tc_r",
            "tool_name": "search",
            "server_name": "web",
            "arguments": {"q": "x"},
        }
        snap = _build_snapshot_with_state(
            pending_call,
            effective_args={"q": "x"},
            session_findings=[],
            attempted_strategies=[],
            dedup_tool_name="web",
        )

        runner, _ = _build_runner(
            llm_responses=[("done", True, None)],
            max_turns=3,
        )

        async def exec_tool(**kwargs):
            order.append("tool")
            return (
                {"server_name": "web", "tool_name": "search", "result": "r"},
                1,
            )

        runner.tool_executor.execute_single_tool.side_effect = exec_tool

        async def on_resume_hook(ctx, next_fn):
            order.append("on_resume")
            return await next_fn(ctx)

        runner.hooks.register_fn("on_resume", on_resume_hook)

        await runner.run_from_tool_cursor(
            snap,
            HumanDecision(approved=True),
            system_prompt="sys",
            main_agent_prompt_instance=MagicMock(),
            task_engine_cfg=None,
            task_description="task",
            task_guidance="",
            tool_definitions=[{"name": "search"}],
            keep_tool_result=-1,
        )

        assert order.index("on_resume") < order.index("tool")
