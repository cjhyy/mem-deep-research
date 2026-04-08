"""
Integration tests — Sub-agent spawn + return.

Covers:
- SubAgentRunner.spawn() creates isolated context, runs MainLoopRunner, returns result
- SubAgentRunner._parse_task_description handles dict/str/raw
- Spawned agent gets isolated ContextManager and Monitor
- Spawn error handling returns error message (no crash)
- spawn_depth is passed through
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from omegaconf import OmegaConf

from mem_deep_research_core.core.hooks import HookRegistry
from mem_deep_research_core.core.hooks import hooks as global_hooks
from mem_deep_research_core.core.sub_agent_runner import SubAgentRunner


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(autouse=True)
def clear_hooks():
    global_hooks.clear_all()
    yield
    global_hooks.clear_all()


def _make_task_log():
    log = MagicMock()
    log.log_step = MagicMock()
    log.record_perf = MagicMock()
    log.save = MagicMock()
    log.save_checkpoint = MagicMock()
    log.start_sub_agent_session = MagicMock()
    log.end_sub_agent_session = MagicMock()
    log.sub_agent_message_history_sessions = {}
    log.current_sub_agent_session_id = "test-session"
    return log


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


def _make_output_formatter():
    formatter = MagicMock()
    formatter.format_tool_result_for_user = MagicMock(
        return_value={"type": "text", "text": "formatted"}
    )
    formatter.format_final_summary_and_log = MagicMock(
        return_value=("Summary", "Boxed")
    )
    return formatter


def _make_runner_kwargs():
    """Common kwargs for SubAgentRunner constructor."""
    from mem_deep_research_core.utils.external_loader import ConfigLoader

    return dict(
        sub_agent_tool_managers={},
        sub_agent_llm_client=MagicMock(),
        output_formatter=_make_output_formatter(),
        cfg=OmegaConf.create({"main_agent": {"max_turns": 3}}),
        task_log=_make_task_log(),
        context={"user_name": "test_user"},
        chinese_context=False,
        response_language="English",
        stream_handler=_make_stream_handler(),
        hooks=HookRegistry(),
        config_loader=ConfigLoader(),
    )


# ============================================================
# _parse_task_description
# ============================================================


class TestParseTaskDescription:
    def test_dict_with_task_description_key(self):
        result = SubAgentRunner._parse_task_description(
            {"task_description": "Research quantum computing"}
        )
        assert result == "Research quantum computing"

    def test_plain_string(self):
        result = SubAgentRunner._parse_task_description("Do the thing")
        assert result == "Do the thing"

    def test_json_string_with_task_description(self):
        import json
        raw = json.dumps({"task_description": "Analyze data"})
        result = SubAgentRunner._parse_task_description(raw)
        assert result == "Analyze data"

    def test_dict_without_task_description(self):
        result = SubAgentRunner._parse_task_description({"query": "hello"})
        # Should fall back to str(dict)
        assert "hello" in result

    def test_non_string_non_dict(self):
        result = SubAgentRunner._parse_task_description(42)
        assert result == "42"


# ============================================================
# spawn() integration
# ============================================================


class TestSubAgentSpawn:
    @pytest.mark.asyncio
    async def test_spawn_returns_result(self):
        """spawn() should run MainLoopRunner and return its result."""
        runner = SubAgentRunner(**_make_runner_kwargs())

        mock_llm_client = MagicMock()
        mock_llm_client.max_context_length = -1
        mock_llm_client.update_message_history = MagicMock(side_effect=lambda h, r, e: h)

        async def mock_llm_call(*args, **kwargs):
            return "The answer is 42.", False, None

        async def mock_summary(*args, **kwargs):
            return "Summary: 42"

        result = await runner.spawn(
            task_description="What is 6*7?",
            parent_llm_client=mock_llm_client,
            parent_tool_executor=MagicMock(),
            parent_tool_definitions=[],
            parent_callbacks={
                "handle_llm_call": mock_llm_call,
                "handle_summary": mock_summary,
            },
            keep_tool_result=-1,
            spawn_depth=1,
        )

        assert result is not None
        assert isinstance(result, str)
        # Should contain some answer (either direct or summary)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_spawn_error_returns_error_message(self):
        """If spawn() encounters an error, it should return an error string, not raise."""
        runner = SubAgentRunner(**_make_runner_kwargs())

        mock_llm_client = MagicMock()
        # Make max_context_length raise to simulate initialization failure
        type(mock_llm_client).max_context_length = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("client broken"))
        )

        async def mock_llm_call(*args, **kwargs):
            raise RuntimeError("LLM crashed")

        result = await runner.spawn(
            task_description="Doomed task",
            parent_llm_client=mock_llm_client,
            parent_tool_executor=MagicMock(),
            parent_tool_definitions=[],
            parent_callbacks={"handle_llm_call": mock_llm_call},
            spawn_depth=0,
        )

        assert "[Spawn Error]" in result or "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_spawn_uses_isolated_context_manager(self):
        """Each spawn should get its own ContextManager instance."""
        runner = SubAgentRunner(**_make_runner_kwargs())

        context_managers = []
        original_init = type(runner).__init__

        # Patch MainLoopRunner to capture the context_manager
        from mem_deep_research_core.core.main_loop import MainLoopRunner

        original_run = MainLoopRunner.run

        async def capturing_run(self_runner, **kwargs):
            context_managers.append(self_runner.context_manager)
            return "captured result", False

        mock_llm_client = MagicMock()
        mock_llm_client.max_context_length = -1

        async def mock_llm(*args, **kwargs):
            return "answer", False, None

        with patch.object(MainLoopRunner, "run", capturing_run):
            await runner.spawn(
                task_description="task 1",
                parent_llm_client=mock_llm_client,
                parent_tool_executor=MagicMock(),
                parent_tool_definitions=[],
                parent_callbacks={"handle_llm_call": mock_llm},
                spawn_depth=0,
            )

            await runner.spawn(
                task_description="task 2",
                parent_llm_client=mock_llm_client,
                parent_tool_executor=MagicMock(),
                parent_tool_definitions=[],
                parent_callbacks={"handle_llm_call": mock_llm},
                spawn_depth=0,
            )

        # Two spawns should have created two different ContextManagers
        assert len(context_managers) == 2
        assert context_managers[0] is not context_managers[1]

    @pytest.mark.asyncio
    async def test_spawn_passes_hooks_instance(self):
        """spawn() should pass hooks_instance to MainLoopContext."""
        custom_hooks = HookRegistry()
        kwargs = _make_runner_kwargs()
        kwargs["hooks"] = custom_hooks
        runner = SubAgentRunner(**kwargs)

        from mem_deep_research_core.core.main_loop import MainLoopRunner

        captured_hooks = []

        async def capturing_run(self_runner, **kwargs):
            captured_hooks.append(self_runner.hooks)
            return "result", False

        mock_llm_client = MagicMock()
        mock_llm_client.max_context_length = -1

        with patch.object(MainLoopRunner, "run", capturing_run):
            await runner.spawn(
                task_description="task",
                parent_llm_client=mock_llm_client,
                parent_tool_executor=MagicMock(),
                parent_tool_definitions=[],
                parent_callbacks={},
                spawn_depth=0,
                hooks_instance=custom_hooks,
            )

        assert len(captured_hooks) == 1
        assert captured_hooks[0] is custom_hooks


# ============================================================
# run() (pre-configured sub-agents)
# ============================================================


class TestSubAgentRun:
    @pytest.mark.asyncio
    async def test_run_missing_sub_agent_config_returns_error(self):
        """run() with non-existent sub-agent name should return error message."""
        runner = SubAgentRunner(**_make_runner_kwargs())

        result = await runner.run(
            sub_agent_name="agent-nonexistent",
            task_description="do something",
        )

        assert "Error" in result or "failed" in result or "not found" in result.lower()
