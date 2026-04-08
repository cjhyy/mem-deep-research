"""
Tests for hook injection chain.

Covers:
- Consumer modules accept hooks via constructor, use them instead of global
- MainLoopRunner uses ctx.hooks
- ToolResultFormatter, PromptBuilder, LLMCallHandler, MessageInterceptor, SubAgentRunner
  all respect injected hooks
- No module falls back to global singleton — hooks must be explicitly passed
"""

from unittest.mock import MagicMock

import pytest

from mem_deep_research_core.core.hooks import HookContext, HookRegistry
from mem_deep_research_core.core.hooks import hooks as global_hooks


@pytest.fixture(autouse=True)
def clean_global():
    global_hooks.clear_all()
    yield
    global_hooks.clear_all()


# ============================================================
# ToolResultFormatter
# ============================================================


class TestToolResultFormatterHooks:
    def test_uses_injected_hooks(self):
        """ToolResultFormatter should use injected hooks, not global."""
        from mem_deep_research_core.core.tool_result_formatter import ToolResultFormatter

        custom = HookRegistry()
        calls = []

        custom.set_default("on_tool_result_format", lambda ctx: ctx.formatted_result)
        custom.register_fn(
            "on_tool_result_format",
            lambda ctx, fn: calls.append("custom") or fn(ctx),
        )

        formatter = ToolResultFormatter(context={}, hooks=custom)

        # Global should have no hooks
        assert not global_hooks.has_hooks("on_tool_result_format")

        # Call summarize — should trigger custom hook
        formatter.summarize_tool_result(
            tool_name="t", tool_result={"result": "ok"}, duration_ms=100
        )
        assert "custom" in calls

    def test_hooks_required(self):
        """ToolResultFormatter requires hooks keyword argument."""
        from mem_deep_research_core.core.tool_result_formatter import ToolResultFormatter

        with pytest.raises(TypeError, match="hooks"):
            ToolResultFormatter(context={})


# ============================================================
# PromptBuilder
# ============================================================


class TestPromptBuilderHooks:
    def test_uses_injected_hooks(self):
        """PromptBuilder should use injected hooks, not global."""
        from mem_deep_research_core.core.prompt_builder import PromptBuilder
        from mem_deep_research_core.utils.external_loader import ConfigLoader
        from omegaconf import OmegaConf

        custom = HookRegistry()
        cfg = OmegaConf.create({"main_agent": {"prompt": {}}})
        builder = PromptBuilder(
            cfg=cfg, context={}, chinese_context=False,
            hooks=custom, config_loader=ConfigLoader(),
        )
        assert builder._hooks is custom
        assert builder._hooks is not global_hooks

    def test_hooks_and_config_loader_required(self):
        from mem_deep_research_core.core.prompt_builder import PromptBuilder
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"main_agent": {"prompt": {}}})
        with pytest.raises(TypeError):
            PromptBuilder(cfg=cfg, context={}, chinese_context=False)


# ============================================================
# LLMCallHandler
# ============================================================


class TestLLMCallHandlerHooks:
    def test_uses_injected_hooks(self):
        """LLMCallHandler should use injected hooks."""
        from mem_deep_research_core.core.llm_call_handler import LLMCallHandler

        custom = HookRegistry()
        mock_client = MagicMock()
        handler = LLMCallHandler(
            main_llm_client=mock_client,
            hooks=custom,
        )
        assert handler._hooks is custom
        assert handler._hooks is not global_hooks

    def test_none_hooks_creates_fresh_instance(self):
        """When hooks=None, LLMCallHandler creates a fresh HookRegistry (not global)."""
        from mem_deep_research_core.core.llm_call_handler import LLMCallHandler

        mock_client = MagicMock()
        handler = LLMCallHandler(main_llm_client=mock_client)
        assert handler._hooks is not global_hooks
        assert isinstance(handler._hooks, HookRegistry)


# ============================================================
# MessageInterceptorHandler
# ============================================================


class TestMessageInterceptorHooks:
    def test_uses_injected_hooks(self):
        from mem_deep_research_core.core.message_interceptor import MessageInterceptorHandler

        custom = HookRegistry()
        handler = MessageInterceptorHandler(hooks=custom)
        assert handler._hooks is custom

    def test_hooks_required(self):
        from mem_deep_research_core.core.message_interceptor import MessageInterceptorHandler

        with pytest.raises(TypeError, match="hooks"):
            MessageInterceptorHandler()


# ============================================================
# SubAgentRunner
# ============================================================


class TestSubAgentRunnerHooks:
    def test_uses_injected_hooks(self):
        from mem_deep_research_core.core.sub_agent_runner import SubAgentRunner
        from mem_deep_research_core.utils.external_loader import ConfigLoader
        from omegaconf import OmegaConf

        custom = HookRegistry()
        cfg = OmegaConf.create({"main_agent": {"max_turns": 3}})
        runner = SubAgentRunner(
            sub_agent_tool_managers={},
            sub_agent_llm_client=MagicMock(),
            output_formatter=MagicMock(),
            cfg=cfg,
            task_log=MagicMock(),
            hooks=custom,
            config_loader=ConfigLoader(),
        )
        assert runner._hooks is custom

    def test_hooks_and_config_loader_required(self):
        from mem_deep_research_core.core.sub_agent_runner import SubAgentRunner
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"main_agent": {"max_turns": 3}})
        with pytest.raises(TypeError):
            SubAgentRunner(
                sub_agent_tool_managers={},
                sub_agent_llm_client=MagicMock(),
                output_formatter=MagicMock(),
                cfg=cfg,
                task_log=MagicMock(),
            )


# ============================================================
# ContextManager
# ============================================================


class TestContextManagerHooks:
    def test_accepts_hooks(self):
        from mem_deep_research_core.core.context_manager import ContextManager

        custom = HookRegistry()
        cm = ContextManager(hooks=custom)
        assert cm._hooks is custom

    def test_none_hooks_stays_none(self):
        """ContextManager stores None when not given hooks (checked at call site)."""
        from mem_deep_research_core.core.context_manager import ContextManager

        cm = ContextManager()
        assert cm._hooks is None


# ============================================================
# MainLoopRunner — ctx.hooks propagation
# ============================================================


class TestMainLoopRunnerHooks:
    def test_uses_ctx_hooks(self):
        """MainLoopRunner should use hooks from MainLoopContext, not global."""
        from mem_deep_research_core.core.main_loop import MainLoopContext, MainLoopRunner
        from mem_deep_research_core.core.context_manager import ContextManager
        from mem_deep_research_core.core.monitoring import ExecutionMonitor, MonitoringConfig
        from mem_deep_research_core.core.task_planner import TaskPlanner
        from unittest.mock import AsyncMock
        from omegaconf import OmegaConf

        custom = HookRegistry()

        ctx = MainLoopContext(
            cfg=OmegaConf.create({"main_agent": {"max_turns": 1, "max_tool_calls_per_turn": 5, "keep_tool_result": -1}}),
            monitor=ExecutionMonitor(config=MonitoringConfig()),
            context_manager=ContextManager(),
            stream_handler=MagicMock(),
            tool_executor=MagicMock(),
            sub_agent_runner=None,
            llm_handler=MagicMock(),
            summary_handler=MagicMock(),
            task_planner=TaskPlanner(enabled=False),
            inline_skill_selector=None,
            llm_client=MagicMock(),
            output_formatter=MagicMock(),
            task_log=MagicMock(),
            context={},
            chinese_context=False,
            handle_llm_call=AsyncMock(),
            handle_summary=AsyncMock(),
            intercept_key_message=AsyncMock(),
            streaming_final_message=AsyncMock(),
            stream_tool_reasoning=AsyncMock(),
            extract_recent_tool_names=lambda h, lookback=6: [],
            deduplicate_trailing_messages=lambda h: 0,
            response_language="auto",
            hooks=custom,
        )

        runner = MainLoopRunner(ctx)
        assert runner.hooks is custom
        assert runner.hooks is not global_hooks

    def test_hooks_none_raises(self):
        """MainLoopRunner should raise if hooks is None."""
        from mem_deep_research_core.core.main_loop import MainLoopContext, MainLoopRunner
        from mem_deep_research_core.core.context_manager import ContextManager
        from mem_deep_research_core.core.monitoring import ExecutionMonitor, MonitoringConfig
        from mem_deep_research_core.core.task_planner import TaskPlanner
        from unittest.mock import AsyncMock
        from omegaconf import OmegaConf

        ctx = MainLoopContext(
            cfg=OmegaConf.create({"main_agent": {"max_turns": 1, "max_tool_calls_per_turn": 5, "keep_tool_result": -1}}),
            monitor=ExecutionMonitor(config=MonitoringConfig()),
            context_manager=ContextManager(),
            stream_handler=MagicMock(),
            tool_executor=MagicMock(),
            sub_agent_runner=None,
            llm_handler=MagicMock(),
            summary_handler=MagicMock(),
            task_planner=TaskPlanner(enabled=False),
            inline_skill_selector=None,
            llm_client=MagicMock(),
            output_formatter=MagicMock(),
            task_log=MagicMock(),
            context={},
            chinese_context=False,
            handle_llm_call=AsyncMock(),
            handle_summary=AsyncMock(),
            intercept_key_message=AsyncMock(),
            streaming_final_message=AsyncMock(),
            stream_tool_reasoning=AsyncMock(),
            extract_recent_tool_names=lambda h, lookback=6: [],
            deduplicate_trailing_messages=lambda h: 0,
            response_language="auto",
            hooks=None,
        )

        with pytest.raises(ValueError, match="hooks is required"):
            MainLoopRunner(ctx)
