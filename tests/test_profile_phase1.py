"""Phase 1 Profile 系统测试

验证：
1. Profile ABC 默认钩子行为（pass-through）
2. StandardProfile 等价于基类默认
3. Profile registry 解析（字符串 / class / instance / None）
4. Custom profile 注册
5. MainLoopRunner 正确调用 profile 钩子（mock profile 验证时序）
"""

import pytest

from mem_deep_research_core.core.profiles import (
    Profile,
    StandardProfile,
    list_profiles,
    register_profile,
    resolve_profile,
)


# =========================================================
# 1. Profile 基类默认行为
# =========================================================


class TestProfileDefaults:
    """Base Profile 的默认钩子全部 pass-through，不影响主循环行为。"""

    @pytest.mark.asyncio
    async def test_on_agent_start_returns_none(self):
        p = StandardProfile()
        assert await p.on_agent_start(ctx=_FakeCtx()) is None

    @pytest.mark.asyncio
    async def test_build_initial_system_prompt_passthrough(self):
        p = StandardProfile()
        assert await p.build_initial_system_prompt("hello", ctx=_FakeCtx()) == "hello"

    @pytest.mark.asyncio
    async def test_on_turn_start_returns_none(self):
        p = StandardProfile()
        assert await p.on_turn_start(ctx=_FakeCtx()) is None

    @pytest.mark.asyncio
    async def test_should_inject_reflection_is_false(self):
        p = StandardProfile()
        assert await p.should_inject_reflection(ctx=_FakeCtx()) is False

    @pytest.mark.asyncio
    async def test_build_reflection_prompt_returns_none(self):
        p = StandardProfile()
        assert await p.build_reflection_prompt(ctx=_FakeCtx()) is None

    @pytest.mark.asyncio
    async def test_on_llm_response_passthrough(self):
        p = StandardProfile()
        result = await p.on_llm_response("assistant text", ctx=_FakeCtx())
        assert result == "assistant text"

    @pytest.mark.asyncio
    async def test_on_before_tools_passthrough(self):
        p = StandardProfile()
        calls = [{"name": "x"}]
        result = await p.on_before_tools(calls, ctx=_FakeCtx())
        assert result == calls

    @pytest.mark.asyncio
    async def test_on_after_tools_returns_none(self):
        p = StandardProfile()
        assert await p.on_after_tools([], ctx=_FakeCtx()) is None

    @pytest.mark.asyncio
    async def test_should_run_verify_is_false(self):
        p = StandardProfile()
        assert await p.should_run_verify(ctx=_FakeCtx()) is False

    @pytest.mark.asyncio
    async def test_run_verify_returns_none(self):
        p = StandardProfile()
        assert await p.run_verify(ctx=_FakeCtx()) is None

    @pytest.mark.asyncio
    async def test_build_final_answer_returns_last_text(self):
        p = StandardProfile()
        result = await p.build_final_answer("final", [], ctx=_FakeCtx())
        assert result == "final"

    def test_default_config_empty(self):
        assert StandardProfile.default_config() == {}

    def test_snapshot_structure(self):
        """StandardProfile snapshot 包含 name + strategies 字典（每个 strategy 状态为空）。"""
        snap = StandardProfile().snapshot()
        assert snap["name"] == "standard"
        assert "strategies" in snap
        # 默认 2 个 strategy（offload_evidence + summary_evidence），均无状态
        assert set(snap["strategies"].keys()) == {"offload_evidence", "summary_evidence"}
        assert all(s == {} for s in snap["strategies"].values())

    def test_restore_noop(self):
        StandardProfile().restore({"any": "state"})  # should not raise


class _FakeCtx:
    """Minimal ProfileContext stub for default-behavior tests."""
    turn_number = 0
    task_description = ""
    mode = "standard"
    tool_calls_executed = 0
    assistant_response_text = ""
    last_assistant_text = ""
    message_history: list = []
    session_memory = None
    todo_tracker = None
    context_manager = None
    llm_client = None
    hooks = None


# =========================================================
# 2. Profile registry
# =========================================================


class TestResolveProfile:
    def test_none_returns_standard(self):
        p = resolve_profile(None)
        assert isinstance(p, StandardProfile)

    def test_string_standard(self):
        p = resolve_profile("standard")
        assert isinstance(p, StandardProfile)

    def test_class_returns_instance(self):
        p = resolve_profile(StandardProfile)
        assert isinstance(p, StandardProfile)

    def test_instance_returned_as_is(self):
        inst = StandardProfile()
        assert resolve_profile(inst) is inst

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown profile name"):
            resolve_profile("nonexistent")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            resolve_profile(123)


class TestRegisterProfile:
    def test_custom_profile_registered(self):
        class MyProfile(Profile):
            name = "my_custom_profile"

        register_profile(MyProfile)
        try:
            p = resolve_profile("my_custom_profile")
            assert isinstance(p, MyProfile)
            assert "my_custom_profile" in list_profiles()
        finally:
            # Clean up
            from mem_deep_research_core.core.profiles import _PROFILE_REGISTRY
            _PROFILE_REGISTRY.pop("my_custom_profile", None)

    def test_non_profile_subclass_rejected(self):
        class NotAProfile:
            name = "x"

        with pytest.raises(TypeError):
            register_profile(NotAProfile)  # type: ignore[arg-type]

    def test_missing_name_rejected(self):
        class NamelessProfile(Profile):
            pass  # inherits name="base"

        with pytest.raises(ValueError, match="non-empty 'name'"):
            register_profile(NamelessProfile)


# =========================================================
# 3. Profile 钩子时序（集成测试，验证主循环调用点）
# =========================================================


class _TracingProfile(Profile):
    """记录所有钩子调用的 profile，用于验证时序。"""

    name = "tracing"

    def __init__(self):
        super().__init__(config=None)  # 使用基类构造初始化 extraction_strategies
        self.calls: list[str] = []

    async def on_agent_start(self, ctx):
        self.calls.append("on_agent_start")

    async def on_turn_start(self, ctx):
        self.calls.append(f"on_turn_start:turn={ctx.turn_number}")

    async def on_llm_response(self, text, ctx):
        self.calls.append(f"on_llm_response:turn={ctx.turn_number}")
        return text


class TestProfileHookInvocation:
    """验证主循环在正确位置调用 profile 钩子。

    使用 existing test_mainloop_tools fixtures 的风格：mock _handle_llm_call。
    """

    @pytest.mark.asyncio
    async def test_on_agent_start_called_once_before_loop(self):
        """on_agent_start 应在主循环开始前调用一次。"""
        from tests.test_mainloop_tools import _build_runner

        tracer = _TracingProfile()
        runner, _ = _build_runner([
            ("Direct answer.", True, None),  # stop_reason=end_turn → should_break=True
        ])
        runner.profile = tracer

        await runner.run(
            system_prompt="sys",
            message_history=[{"role": "user", "content": [{"type": "text", "text": "q"}]}],
            tool_definitions=[],
            main_agent_prompt_instance=_DummyPromptInstance(),
            task_engine_cfg=None,
            task_description="q",
            task_guidance="",
            keep_tool_result=-1,
        )

        # on_agent_start 必须在任何 on_turn_start 之前恰好调用一次
        assert tracer.calls.count("on_agent_start") == 1
        agent_start_idx = tracer.calls.index("on_agent_start")
        turn_start_events = [i for i, c in enumerate(tracer.calls) if c.startswith("on_turn_start")]
        if turn_start_events:
            assert agent_start_idx < turn_start_events[0]

    @pytest.mark.asyncio
    async def test_on_turn_start_called_each_turn(self):
        """每轮主循环开始时应调 on_turn_start 一次。"""
        from tests.test_mainloop_tools import _build_runner

        tracer = _TracingProfile()
        # Two turns: tool call → then final answer
        tool_calls = [
            [{"server_name": "calc", "tool_name": "add", "arguments": {}, "id": "c1"}],
            [],
        ]
        runner, _ = _build_runner([
            ("Calling...", False, tool_calls),
            ("Done.", True, None),
        ])
        runner.profile = tracer

        from unittest.mock import AsyncMock, MagicMock
        runner.tool_executor.execute_single_tool = AsyncMock(
            return_value=({"result": "3", "server_name": "calc", "tool_name": "add"}, 10)
        )
        runner.tool_executor.handle_failed_tool_calls = MagicMock(return_value=([], []))

        await runner.run(
            system_prompt="sys",
            message_history=[{"role": "user", "content": [{"type": "text", "text": "q"}]}],
            tool_definitions=[{"name": "add"}],
            main_agent_prompt_instance=_DummyPromptInstance(),
            task_engine_cfg=None,
            task_description="q",
            task_guidance="",
            keep_tool_result=-1,
        )

        turn_starts = [c for c in tracer.calls if c.startswith("on_turn_start")]
        assert len(turn_starts) == 2
        assert turn_starts[0] == "on_turn_start:turn=1"
        assert turn_starts[1] == "on_turn_start:turn=2"

    @pytest.mark.asyncio
    async def test_on_llm_response_called_per_response(self):
        """每次 LLM 响应返回后应调 on_llm_response。"""
        from tests.test_mainloop_tools import _build_runner

        tracer = _TracingProfile()
        runner, _ = _build_runner([
            ("Hello.", True, None),
        ])
        runner.profile = tracer

        await runner.run(
            system_prompt="sys",
            message_history=[{"role": "user", "content": [{"type": "text", "text": "q"}]}],
            tool_definitions=[],
            main_agent_prompt_instance=_DummyPromptInstance(),
            task_engine_cfg=None,
            task_description="q",
            task_guidance="",
            keep_tool_result=-1,
        )

        llm_calls = [c for c in tracer.calls if c.startswith("on_llm_response")]
        assert len(llm_calls) == 1

    @pytest.mark.asyncio
    async def test_default_profile_is_standard(self):
        """MainLoopRunner 未指定 profile 时默认 StandardProfile。"""
        from tests.test_mainloop_tools import _build_runner

        runner, _ = _build_runner([
            ("Ok.", True, None),
        ])
        assert isinstance(runner.profile, StandardProfile)


class _DummyPromptInstance:
    """Minimal prompt instance stub."""
    pass
