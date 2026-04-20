"""
Integration tests — verify end-to-end component wiring.

Uses mocked LLM clients to test the full pipeline without real API calls.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf

from mem_deep_research_core.core.answer_handler import post_process_final_answer
from mem_deep_research_core.core.constants import (
    FALLBACK_NO_ANSWER,
    SUB_AGENT_PREFIX,
    TAG_TASK_PLAN,
)
from mem_deep_research_core.core.context_manager import ContextManager
from mem_deep_research_core.core.hooks import HookRegistry, hooks
from mem_deep_research_core.core.main_loop import MainLoopContext, MainLoopRunner
from mem_deep_research_core.core.monitoring import ExecutionMonitor, MonitoringConfig
from mem_deep_research_core.core.task_planner import SubQuestion, TaskPlan, TaskPlanner

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(autouse=True)
def clear_hooks():
    """Clear all hooks before each test."""
    hooks.clear_all()
    yield
    hooks.clear_all()


def _make_mock_cfg(max_turns=3, max_tool_calls=5, keep_tool_result=-1):
    """Create a minimal OmegaConf config for MainLoopRunner."""
    return OmegaConf.create(
        {
            "main_agent": {
                "max_turns": max_turns,
                "max_tool_calls_per_turn": max_tool_calls,
                "keep_tool_result": keep_tool_result,
            }
        }
    )


def _make_mock_stream_handler():
    """Create a mock StreamHandler with all async methods."""
    handler = MagicMock()
    handler.stream_start_agent = AsyncMock(return_value="agent-id-123")
    handler.stream_end_agent = AsyncMock()
    handler.stream_start_llm = AsyncMock()
    handler.stream_end_llm = AsyncMock()
    handler.stream_tool_call = AsyncMock()
    handler.stream_reasoning = AsyncMock()
    handler.stream_message = AsyncMock()
    handler.stream_usage_info = AsyncMock()
    return handler


def _make_mock_task_log():
    """Create a mock TaskTracer."""
    log = MagicMock()
    log.log_step = MagicMock()
    log.record_perf = MagicMock()
    log.save = MagicMock()
    log.save_checkpoint = MagicMock()
    return log


def _make_mock_output_formatter():
    """Create a mock OutputFormatter."""
    formatter = MagicMock()
    formatter.format_tool_result_for_user = MagicMock(
        return_value={"type": "text", "text": "formatted result"}
    )
    formatter.format_final_summary_and_log = MagicMock(
        return_value=("Final summary", "Boxed answer")
    )
    return formatter


def _make_noop_async():
    async def _noop(*args, **kwargs):
        return None

    return _noop


# ============================================================
# MainLoopRunner Integration Tests
# ============================================================


class TestMainLoopRunnerIntegration:
    """Test MainLoopRunner with mocked LLM — verifies full turn loop."""

    def _build_runner(
        self,
        llm_responses: list[tuple[str, bool, list | None]],
        max_turns=3,
    ) -> tuple[MainLoopRunner, list]:
        """Build a MainLoopRunner with mocked LLM call handler.

        Args:
            llm_responses: List of (response_text, should_break, tool_calls) tuples.
                           Each call to handle_llm_call pops from front.
        """
        call_log = []
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

        mock_prompt_instance = MagicMock()
        mock_prompt_instance.generate_summarize_prompt = MagicMock(return_value="summarize prompt")

        ctx = MainLoopContext(
            cfg=_make_mock_cfg(max_turns=max_turns),
            monitor=ExecutionMonitor(config=MonitoringConfig()),
            context_manager=ContextManager(),
            stream_handler=_make_mock_stream_handler(),
            tool_executor=MagicMock(),
            sub_agent_runner=None,
            llm_handler=MagicMock(),
            summary_handler=MagicMock(),
            task_planner=TaskPlanner(enabled=False),
            inline_skill_selector=None,
            llm_client=MagicMock(),
            output_formatter=_make_mock_output_formatter(),
            task_log=_make_mock_task_log(),
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
        return runner, call_log

    @pytest.mark.asyncio
    async def test_simple_response_no_tools(self):
        """LLM responds without tool calls and no tools were ever executed —
        fast-path: treat as direct answer and return immediately (1 LLM call)."""
        runner, call_log = self._build_runner(
            [
                ("The answer is 42.", False, None),  # No tool calls → direct answer
            ]
        )

        result, is_simple = await runner.run(
            system_prompt="You are helpful.",
            message_history=[
                {"role": "user", "content": [{"type": "text", "text": "What is 42?"}]}
            ],
            tool_definitions=[],
            main_agent_prompt_instance=MagicMock(),
            task_engine_cfg=None,
            task_description="What is 42?",
            task_guidance="",
            keep_tool_result=-1,
        )

        assert "42" in result
        assert is_simple  # truthy for simple responses (no tool calls)
        assert len(call_log) == 1  # fast-path: single LLM call

    @pytest.mark.asyncio
    async def test_max_turns_reached(self):
        """Verify loop terminates after max_turns."""
        # LLM always returns tool calls — should hit max_turns
        tool_calls = [
            [{"server_name": "calc", "tool_name": "add", "arguments": {"a": 1}, "id": "c1"}],
            [],
        ]
        runner, call_log = self._build_runner(
            [("thinking...", False, tool_calls)] * 5,
            max_turns=3,
        )

        # Mock tool execution
        runner.tool_executor.execute_single_tool = AsyncMock(
            return_value=({"server_name": "calc", "tool_name": "add", "result": "2"}, None)
        )
        runner.tool_executor.execute_tool_calls = AsyncMock(
            return_value=([], [("c1", {"type": "text", "text": "result"})], False)
        )
        runner.tool_executor.handle_failed_tool_calls = MagicMock(return_value=([], []))
        runner.llm_client.max_context_length = -1
        runner.llm_client.update_message_history = MagicMock(side_effect=lambda h, r, e: h)

        result, _ = await runner.run(
            system_prompt="sys",
            message_history=[{"role": "user", "content": [{"type": "text", "text": "task"}]}],
            tool_definitions=[{"name": "add"}],
            main_agent_prompt_instance=MagicMock(),
            task_engine_cfg=None,
            task_description="task",
            task_guidance="",
            keep_tool_result=-1,
        )

        assert len(call_log) <= 3  # Should not exceed max_turns

    @pytest.mark.asyncio
    async def test_llm_failure_terminates(self):
        """LLM returning None should terminate the loop."""
        runner, call_log = self._build_runner(
            [
                (None, True, None),  # LLM failed
            ]
        )

        result, _ = await runner.run(
            system_prompt="sys",
            message_history=[{"role": "user", "content": [{"type": "text", "text": "task"}]}],
            tool_definitions=[],
            main_agent_prompt_instance=MagicMock(),
            task_engine_cfg=None,
            task_description="task",
            task_guidance="",
            keep_tool_result=-1,
        )

        # Should produce a summary even on failure
        assert result is not None


# ============================================================
# Hook Integration Tests
# ============================================================


class TestHookIntegration:
    """Test hooks fire correctly during MainLoopRunner execution."""

    @pytest.mark.asyncio
    async def test_on_agent_start_fires(self):
        """on_agent_start hook should fire at the beginning of run()."""
        hook_calls = []

        @hooks.register("on_agent_start", priority=10)
        def capture(ctx, original_fn):
            hook_calls.append({"query": ctx.query})
            return original_fn(ctx)

        # Need to set default for the hook
        hooks.set_default("on_agent_start", lambda ctx: None)

        async def mock_llm(*args, **kwargs):
            return "answer", False, None

        async def mock_summary(*args, **kwargs):
            return "summary"

        ctx = MainLoopContext(
            cfg=_make_mock_cfg(max_turns=1),
            monitor=ExecutionMonitor(config=MonitoringConfig()),
            context_manager=ContextManager(),
            stream_handler=_make_mock_stream_handler(),
            tool_executor=MagicMock(),
            sub_agent_runner=None,
            llm_handler=MagicMock(),
            summary_handler=MagicMock(),
            task_planner=TaskPlanner(enabled=False),
            inline_skill_selector=None,
            llm_client=MagicMock(),
            output_formatter=_make_mock_output_formatter(),
            task_log=_make_mock_task_log(),
            context={},
            chinese_context=False,
            handle_llm_call=mock_llm,
            handle_summary=mock_summary,
            intercept_key_message=_make_noop_async(),
            streaming_final_message=_make_noop_async(),
            stream_tool_reasoning=_make_noop_async(),
            extract_recent_tool_names=lambda h, lookback=6: [],
            deduplicate_trailing_messages=lambda h: 0,
            response_language="English",
            hooks=hooks,
        )

        runner = MainLoopRunner(ctx)
        await runner.run(
            system_prompt="sys",
            message_history=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            tool_definitions=[],
            main_agent_prompt_instance=MagicMock(),
            task_engine_cfg=None,
            task_description="hello",
            task_guidance="",
            keep_tool_result=-1,
        )

        assert len(hook_calls) == 1
        assert hook_calls[0]["query"] == "hello"


# ============================================================
# Language Detection Integration Tests
# ============================================================


class TestLanguageDetection:
    """Test response_language auto-detection."""

    @pytest.mark.asyncio
    async def test_auto_detect_chinese(self):
        """Auto mode should detect Chinese from query."""

        async def mock_llm(*args, **kwargs):
            return "这是回答", False, None

        async def mock_summary(*args, **kwargs):
            return "中文摘要"

        hooks.set_default("on_agent_start", lambda ctx: None)
        hooks.set_default("on_agent_end", lambda ctx: None)
        hooks.set_default("on_turn_start", lambda ctx: None)
        hooks.set_default("on_turn_end", lambda ctx: None)

        ctx = MainLoopContext(
            cfg=_make_mock_cfg(max_turns=1),
            monitor=ExecutionMonitor(config=MonitoringConfig()),
            context_manager=ContextManager(),
            stream_handler=_make_mock_stream_handler(),
            tool_executor=MagicMock(),
            sub_agent_runner=None,
            llm_handler=MagicMock(),
            summary_handler=MagicMock(),
            task_planner=TaskPlanner(enabled=False),
            inline_skill_selector=None,
            llm_client=MagicMock(),
            output_formatter=_make_mock_output_formatter(),
            task_log=_make_mock_task_log(),
            context={},
            chinese_context=False,
            handle_llm_call=mock_llm,
            handle_summary=mock_summary,
            intercept_key_message=_make_noop_async(),
            streaming_final_message=_make_noop_async(),
            stream_tool_reasoning=_make_noop_async(),
            extract_recent_tool_names=lambda h, lookback=6: [],
            deduplicate_trailing_messages=lambda h: 0,
            response_language="auto",
            hooks=hooks,
        )

        runner = MainLoopRunner(ctx)
        await runner.run(
            system_prompt="sys",
            message_history=[
                {"role": "user", "content": [{"type": "text", "text": "请解释量子计算的基本原理"}]}
            ],
            tool_definitions=[],
            main_agent_prompt_instance=MagicMock(),
            task_engine_cfg=None,
            task_description="请解释量子计算的基本原理",
            task_guidance="",
            keep_tool_result=-1,
        )

        # After auto-detection, Chinese query should set response_language to Chinese
        assert runner.response_language == "Chinese"
        assert runner.chinese_context is True

    @pytest.mark.asyncio
    async def test_auto_detect_english(self):
        """Auto mode should detect English from query."""

        async def mock_llm(*args, **kwargs):
            return "answer", False, None

        async def mock_summary(*args, **kwargs):
            return "summary"

        hooks.set_default("on_agent_start", lambda ctx: None)
        hooks.set_default("on_agent_end", lambda ctx: None)
        hooks.set_default("on_turn_start", lambda ctx: None)
        hooks.set_default("on_turn_end", lambda ctx: None)

        ctx = MainLoopContext(
            cfg=_make_mock_cfg(max_turns=1),
            monitor=ExecutionMonitor(config=MonitoringConfig()),
            context_manager=ContextManager(),
            stream_handler=_make_mock_stream_handler(),
            tool_executor=MagicMock(),
            sub_agent_runner=None,
            llm_handler=MagicMock(),
            summary_handler=MagicMock(),
            task_planner=TaskPlanner(enabled=False),
            inline_skill_selector=None,
            llm_client=MagicMock(),
            output_formatter=_make_mock_output_formatter(),
            task_log=_make_mock_task_log(),
            context={},
            chinese_context=False,
            handle_llm_call=mock_llm,
            handle_summary=mock_summary,
            intercept_key_message=_make_noop_async(),
            streaming_final_message=_make_noop_async(),
            stream_tool_reasoning=_make_noop_async(),
            extract_recent_tool_names=lambda h, lookback=6: [],
            deduplicate_trailing_messages=lambda h: 0,
            response_language="auto",
            hooks=hooks,
        )

        runner = MainLoopRunner(ctx)
        await runner.run(
            system_prompt="sys",
            message_history=[
                {"role": "user", "content": [{"type": "text", "text": "Explain quantum computing"}]}
            ],
            tool_definitions=[],
            main_agent_prompt_instance=MagicMock(),
            task_engine_cfg=None,
            task_description="Explain quantum computing",
            task_guidance="",
            keep_tool_result=-1,
        )

        assert runner.response_language == "English"
        assert runner.chinese_context is False


# ============================================================
# Answer Handler Integration Tests
# ============================================================


class TestAnswerHandler:
    """Test post_process_final_answer."""

    @pytest.mark.asyncio
    async def test_with_answer(self):
        """Valid answer should be formatted."""
        cfg = OmegaConf.create(
            {
                "main_agent": {
                    "output_process": {
                        "final_answer_extraction": False,
                    }
                }
            }
        )

        formatter = _make_mock_output_formatter()
        task_log = _make_mock_task_log()

        summary, boxed = await post_process_final_answer(
            cfg=cfg,
            final_answer_text="The answer is 42.",
            task_description="What is the answer?",
            message_history=[],
            system_prompt="sys",
            chinese_context=False,
            task_log=task_log,
            output_formatter=formatter,
            llm_client=MagicMock(),
        )

        assert summary == "Final summary"
        assert boxed == "Boxed answer"
        formatter.format_final_summary_and_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_without_answer(self):
        """Empty answer should use fallback."""
        cfg = OmegaConf.create(
            {
                "main_agent": {
                    "output_process": {
                        "final_answer_extraction": False,
                    }
                }
            }
        )

        formatter = _make_mock_output_formatter()
        task_log = _make_mock_task_log()

        await post_process_final_answer(
            cfg=cfg,
            final_answer_text="",
            task_description="task",
            message_history=[],
            system_prompt="sys",
            chinese_context=False,
            task_log=task_log,
            output_formatter=formatter,
            llm_client=MagicMock(),
        )

        # Should have called format with FALLBACK_NO_ANSWER
        call_args = formatter.format_final_summary_and_log.call_args
        assert call_args[0][0] == FALLBACK_NO_ANSWER

    @pytest.mark.asyncio
    async def test_final_answer_hook_can_override_answer(self):
        """on_final_answer hook should run before final formatting."""
        cfg = OmegaConf.create(
            {
                "main_agent": {
                    "output_process": {
                        "final_answer_extraction": False,
                    }
                }
            }
        )

        formatter = _make_mock_output_formatter()
        task_log = _make_mock_task_log()
        registry = HookRegistry()

        def final_answer_hook(ctx, original_fn):
            return f"[Judged] {ctx.result}"

        registry.register_fn("on_final_answer", final_answer_hook)

        await post_process_final_answer(
            cfg=cfg,
            final_answer_text="The answer is 42.",
            task_description="What is the answer?",
            message_history=[],
            system_prompt="sys",
            chinese_context=False,
            task_log=task_log,
            output_formatter=formatter,
            llm_client=MagicMock(),
            hooks=registry,
        )

        call_args = formatter.format_final_summary_and_log.call_args
        assert call_args[0][0] == "[Judged] The answer is 42."


# ============================================================
# Task Planner Integration Tests
# ============================================================


class TestTaskPlannerIntegration:
    """Test TaskPlanner template loading."""

    def test_research_plan_context_string(self):
        """TaskPlan.to_context_string should use templates."""
        plan = TaskPlan(
            main_question="How does photosynthesis work?",
            sub_questions=[
                SubQuestion(id=1, question="What is light reaction?", priority="high"),
                SubQuestion(id=2, question="What is dark reaction?", priority="medium"),
            ],
        )

        context = plan.to_context_string()
        assert "How does photosynthesis work?" in context
        assert "What is light reaction?" in context
        assert "What is dark reaction?" in context
        assert "!!!" in context  # high priority marker


# ============================================================
# Constants Integration Tests
# ============================================================


class TestConstantsIntegration:
    """Verify constants are used consistently."""

    def test_sub_agent_prefix(self):
        assert SUB_AGENT_PREFIX == "agent-"
        assert "agent-researcher".startswith(SUB_AGENT_PREFIX)

    def test_tag_task_plan(self):
        assert TAG_TASK_PLAN == "[TASK PLAN]"
