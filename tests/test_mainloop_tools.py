"""
Integration tests — Main loop + tool execution chain.

Covers:
- MainLoopRunner with LLM returning tool_calls → ToolExecutor executes → result in history → next turn
- Tool dedup across turns (ContextManager dedup cache)
- Tool execution error propagation
- Multi-tool per turn execution
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from omegaconf import OmegaConf

from mem_deep_research_core.core.context_manager import ContextManager
from mem_deep_research_core.core.hooks import HookRegistry, hooks
from mem_deep_research_core.core.main_loop import MainLoopContext, MainLoopRunner
from mem_deep_research_core.core.monitoring import ExecutionMonitor, MonitoringConfig
from mem_deep_research_core.core.task_planner import TaskPlanner
from mem_deep_research_core.core.tool_executor import ToolExecutor
from mem_deep_research_core.core.tool_result_formatter import ToolResultFormatter


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(autouse=True)
def clear_hooks():
    hooks.clear_all()
    yield
    hooks.clear_all()


def _make_cfg(max_turns=5, max_tool_calls=5, keep_tool_result=-1):
    return OmegaConf.create(
        {
            "main_agent": {
                "max_turns": max_turns,
                "max_tool_calls_per_turn": max_tool_calls,
                "keep_tool_result": keep_tool_result,
            }
        }
    )


def _make_stream_handler():
    handler = MagicMock()
    handler.stream_start_agent = AsyncMock(return_value="agent-id")
    handler.stream_end_agent = AsyncMock()
    handler.stream_start_llm = AsyncMock()
    handler.stream_end_llm = AsyncMock()
    handler.stream_tool_call = AsyncMock()
    handler.stream_reasoning = AsyncMock()
    handler.stream_message = AsyncMock()
    handler.stream_usage_info = AsyncMock()
    return handler


def _make_task_log():
    log = MagicMock()
    log.log_step = MagicMock()
    log.record_perf = MagicMock()
    log.save = MagicMock()
    log.save_checkpoint = MagicMock()
    return log


def _make_output_formatter():
    formatter = MagicMock()
    formatter.format_tool_result_for_user = MagicMock(
        return_value={"type": "text", "text": "formatted result"}
    )
    formatter.format_final_summary_and_log = MagicMock(
        return_value=("Final summary", "Boxed answer")
    )
    return formatter


async def _noop(*args, **kwargs):
    return None


def _build_runner(
    llm_responses: list[tuple[str | None, bool, list | None]],
    max_turns: int = 5,
    tool_executor=None,
):
    """Build MainLoopRunner with mocked LLM and optional real-ish ToolExecutor.

    Args:
        llm_responses: List of (response_text, should_break, tool_calls).
        tool_executor: If provided, use this ToolExecutor; otherwise mock.
    """
    call_log = []
    responses = list(llm_responses)

    async def mock_llm_call(
        system_prompt, message_history, tool_definitions, step_id,
        purpose="", keep_tool_result=-1, agent_type="main",
        stream_message_callback=None,
    ):
        call_log.append({
            "step_id": step_id,
            "history_len": len(message_history),
        })
        if responses:
            return responses.pop(0)
        return None, True, None

    async def mock_summary(*args, **kwargs):
        return "Summary"

    mock_llm_client = MagicMock()
    mock_llm_client.max_context_length = -1
    mock_llm_client.update_message_history = MagicMock(side_effect=lambda h, r, e: h)

    ctx = MainLoopContext(
        cfg=_make_cfg(max_turns=max_turns),
        monitor=ExecutionMonitor(config=MonitoringConfig()),
        context_manager=ContextManager(),
        stream_handler=_make_stream_handler(),
        tool_executor=tool_executor or MagicMock(),
        sub_agent_runner=None,
        llm_handler=MagicMock(),
        summary_handler=MagicMock(),
        task_planner=TaskPlanner(enabled=False),
        inline_skill_selector=None,
        llm_client=mock_llm_client,
        output_formatter=_make_output_formatter(),
        task_log=_make_task_log(),
        context={},
        chinese_context=False,
        handle_llm_call=mock_llm_call,
        handle_summary=mock_summary,
        intercept_key_message=_noop,
        streaming_final_message=_noop,
        stream_tool_reasoning=_noop,
        extract_recent_tool_names=lambda h, lookback=6: [],
        deduplicate_trailing_messages=lambda h: 0,
        response_language="English",
        hooks=HookRegistry(),
    )
    runner = MainLoopRunner(ctx)
    return runner, call_log


# ============================================================
# Tests: Main loop executes tool calls
# ============================================================


class TestMainLoopWithToolCalls:
    """Main loop should execute tool calls returned by LLM and feed results back."""

    @pytest.mark.asyncio
    async def test_tool_call_then_final_answer(self):
        """Turn 1: LLM returns tool call → Turn 2: LLM returns final answer."""
        tool_calls_turn1 = [
            [{"server_name": "calc", "tool_name": "add", "arguments": {"a": 1, "b": 2}, "id": "c1"}],
            [],  # empty second batch
        ]

        runner, call_log = _build_runner([
            ("Let me calculate...", False, tool_calls_turn1),
            ("The answer is 3.", False, None),  # final answer, no tools
        ])

        # MainLoop calls execute_single_tool directly (not execute_tool_calls)
        runner.tool_executor.execute_single_tool = AsyncMock(
            return_value=(
                {"result": "3", "server_name": "calc", "tool_name": "add"},
                50,  # duration_ms
            )
        )
        runner.tool_executor.handle_failed_tool_calls = MagicMock(return_value=([], []))

        result, _ = await runner.run(
            system_prompt="sys",
            message_history=[{"role": "user", "content": [{"type": "text", "text": "1+2?"}]}],
            tool_definitions=[{"name": "add"}],
            main_agent_prompt_instance=MagicMock(),
            task_engine_cfg=None,
            task_description="1+2?",
            task_guidance="",
            keep_tool_result=-1,
        )

        # Result comes from mock_summary, not directly from LLM response
        assert result is not None
        assert len(call_log) == 2  # Two LLM calls: tool turn + final answer turn
        # Second call should have more history (tool result added)
        assert call_log[1]["history_len"] > call_log[0]["history_len"]

    @pytest.mark.asyncio
    async def test_multiple_tools_per_turn(self):
        """LLM returns multiple tool calls in a single turn."""
        tool_calls = [
            [
                {"server_name": "calc", "tool_name": "add", "arguments": {"a": 1, "b": 2}, "id": "c1"},
                {"server_name": "calc", "tool_name": "mul", "arguments": {"a": 3, "b": 4}, "id": "c2"},
            ],
            [],
        ]

        runner, call_log = _build_runner([
            ("Computing both...", False, tool_calls),
            ("Results: 3 and 12.", False, None),
        ])

        call_count = 0

        async def mock_execute_single(server_name, tool_name, arguments, call_id, agent_name="main"):
            nonlocal call_count
            call_count += 1
            return {"result": str(call_count), "server_name": server_name, "tool_name": tool_name}, 30

        runner.tool_executor.execute_single_tool = mock_execute_single
        runner.tool_executor.handle_failed_tool_calls = MagicMock(return_value=([], []))

        result, _ = await runner.run(
            system_prompt="sys",
            message_history=[{"role": "user", "content": [{"type": "text", "text": "calc"}]}],
            tool_definitions=[{"name": "add"}, {"name": "mul"}],
            main_agent_prompt_instance=MagicMock(),
            task_engine_cfg=None,
            task_description="calc",
            task_guidance="",
            keep_tool_result=-1,
        )

        assert len(call_log) == 2
        assert call_count == 2  # Both tools were executed

    @pytest.mark.asyncio
    async def test_tool_error_does_not_crash_loop(self):
        """Tool execution error should be fed back to LLM, not crash the loop."""
        tool_calls = [
            [{"server_name": "srv", "tool_name": "fail_tool", "arguments": {}, "id": "c1"}],
            [],
        ]

        runner, call_log = _build_runner([
            ("Trying tool...", False, tool_calls),
            ("Tool failed, but I know the answer.", False, None),
        ])

        runner.tool_executor.execute_single_tool = AsyncMock(
            return_value=(
                {"error": "connection timeout", "server_name": "srv", "tool_name": "fail_tool"},
                100,
            )
        )
        runner.tool_executor.handle_failed_tool_calls = MagicMock(return_value=([], []))

        result, _ = await runner.run(
            system_prompt="sys",
            message_history=[{"role": "user", "content": [{"type": "text", "text": "do it"}]}],
            tool_definitions=[{"name": "fail_tool"}],
            main_agent_prompt_instance=MagicMock(),
            task_engine_cfg=None,
            task_description="do it",
            task_guidance="",
            keep_tool_result=-1,
        )

        # Should complete without crashing
        assert result is not None
        assert len(call_log) == 2


# ============================================================
# Tests: ToolExecutor unit tests
# ============================================================


class TestToolExecutorUnit:
    """Unit tests for ToolExecutor.execute_single_tool."""

    @pytest.mark.asyncio
    async def test_execute_single_tool_success(self):
        """Successful tool execution returns result and duration."""
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool_call = AsyncMock(
            return_value={"result": "42", "server_name": "calc", "tool_name": "add"}
        )

        executor = ToolExecutor(
            tool_manager=mock_tool_manager,
            output_formatter=_make_output_formatter(),
            tool_result_formatter=ToolResultFormatter(context={}, hooks=HookRegistry()),
            context={},
            hook_registry=HookRegistry(),
        )

        result, duration_ms = await executor.execute_single_tool(
            server_name="calc",
            tool_name="add",
            arguments={"a": 1, "b": 2},
            call_id="c1",
        )

        assert result["result"] == "42"
        assert duration_ms >= 0
        mock_tool_manager.execute_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_single_tool_exception(self):
        """Tool raising exception returns error dict, not crash."""
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool_call = AsyncMock(
            side_effect=ConnectionError("server down")
        )

        executor = ToolExecutor(
            tool_manager=mock_tool_manager,
            output_formatter=_make_output_formatter(),
            tool_result_formatter=ToolResultFormatter(context={}, hooks=HookRegistry()),
            context={},
            retry_max=0,  # No retries for test speed
            hook_registry=HookRegistry(),
        )

        result, duration_ms = await executor.execute_single_tool(
            server_name="srv",
            tool_name="broken",
            arguments={},
            call_id="c1",
        )

        assert "error" in result
        assert "server down" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_tool_calls_respects_max(self):
        """execute_tool_calls should only process up to max_tool_calls."""
        mock_tool_manager = MagicMock()
        mock_tool_manager.execute_tool_call = AsyncMock(
            return_value={"result": "ok"}
        )

        executor = ToolExecutor(
            tool_manager=mock_tool_manager,
            output_formatter=_make_output_formatter(),
            tool_result_formatter=ToolResultFormatter(context={}, hooks=HookRegistry()),
            context={},
            hook_registry=HookRegistry(),
        )

        calls = [
            {"server_name": "s", "tool_name": f"t{i}", "arguments": {}, "id": f"c{i}"}
            for i in range(10)
        ]

        data, results, exceeded = await executor.execute_tool_calls(
            tool_calls=calls,
            max_tool_calls=3,
        )

        assert exceeded is True
        assert len(results) == 3  # Only 3 processed


# ============================================================
# Tests: Tool call dedup via ContextManager
# ============================================================


class TestToolCallDedup:
    """ContextManager should detect duplicate tool calls across turns."""

    def test_dedup_detects_identical_call(self):
        """Identical tool_name + arguments → cached on second attempt."""
        cm = ContextManager()
        cm.set_turn(1)

        calls_turn1 = [{"server_name": "s", "tool_name": "search", "arguments": {"q": "python"}, "id": "c1"}]
        # First call: not in cache, should pass through
        to_exec, cached = cm.filter_duplicate_calls(calls_turn1)
        assert len(to_exec) == 1
        assert len(cached) == 0

        # Register the result to populate cache
        cm.register_tool_results(
            calls_turn1,
            [("c1", {"type": "text", "text": "search results"})],
            turn=1,
        )

        cm.set_turn(2)
        calls_turn2 = [{"server_name": "s", "tool_name": "search", "arguments": {"q": "python"}, "id": "c2"}]
        to_exec, cached = cm.filter_duplicate_calls(calls_turn2)
        assert len(to_exec) == 0  # Filtered as duplicate
        assert len(cached) == 1  # Cached result returned

    def test_dedup_different_args_not_duplicate(self):
        """Different arguments → not duplicate."""
        cm = ContextManager()
        cm.set_turn(1)

        calls1 = [{"server_name": "s", "tool_name": "search", "arguments": {"q": "python"}, "id": "c1"}]
        cm.filter_duplicate_calls(calls1)
        cm.register_tool_results(calls1, [("c1", {"type": "text", "text": "r1"})], turn=1)

        cm.set_turn(2)
        calls2 = [{"server_name": "s", "tool_name": "search", "arguments": {"q": "rust"}, "id": "c2"}]
        to_exec, cached = cm.filter_duplicate_calls(calls2)
        assert len(to_exec) == 1  # Different args, not cached
        assert len(cached) == 0
