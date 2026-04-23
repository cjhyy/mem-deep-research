"""Observability 插槽系统测试

验证：
1. Base Observer Protocol 默认 no-op
2. Registry 串联多个 observer（嵌套语义 + 顺序）
3. 并发 tool call 下每个 observer 收到独立 ctx（不串号）
4. ctx 字段在 yield 后由 runtime 填充
5. Observer 异常不被吞（透传给上层）
6. 空 registry 零开销
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from mem_deep_research_core.observability import (
    AgentObserver,
    AgentRunContext,
    LLMCallContext,
    LLMObserver,
    ObserverRegistry,
    ToolCallContext,
    ToolObserver,
)


# =========================================================
# 1. Base Observer Protocol — 默认行为
# =========================================================


class TestDefaultObservers:
    @pytest.mark.asyncio
    async def test_tool_observer_default_noop(self):
        obs = ToolObserver()
        ctx = ToolCallContext(call_id="c1", tool_name="t", server_name="s", arguments={})
        async with obs.around_tool_call(ctx):
            pass  # body 执行正常

    @pytest.mark.asyncio
    async def test_agent_observer_default_noop(self):
        obs = AgentObserver()
        ctx = AgentRunContext(agent_name="a", agent_id="id1")
        async with obs.around_agent_run(ctx):
            pass

    @pytest.mark.asyncio
    async def test_llm_observer_default_noop(self):
        obs = LLMObserver()
        ctx = LLMCallContext(
            agent_name="a", turn_number=1, provider="x", model="m", messages_count=0
        )
        async with obs.around_llm_call(ctx):
            pass


# =========================================================
# 2. Registry — 串联 + 顺序
# =========================================================


class _TracingToolObserver(ToolObserver):
    def __init__(self, name: str, trace: list):
        self.name = name
        self.trace = trace

    @asynccontextmanager
    async def around_tool_call(self, ctx: ToolCallContext):
        self.trace.append(f"{self.name}:enter")
        try:
            yield
        finally:
            self.trace.append(f"{self.name}:exit")


class TestRegistryNesting:
    @pytest.mark.asyncio
    async def test_empty_registry_is_noop(self):
        r = ObserverRegistry()
        ctx = ToolCallContext(call_id="c1", tool_name="t", server_name="s", arguments={})
        trace = []
        async with r.around_tool_call(ctx):
            trace.append("body")
        assert trace == ["body"]
        assert r.empty

    @pytest.mark.asyncio
    async def test_multi_observer_nested_order(self):
        """List 顺序 = 外到内；先注册的是最外层。"""
        r = ObserverRegistry()
        trace = []
        r.register_tool(_TracingToolObserver("A", trace))
        r.register_tool(_TracingToolObserver("B", trace))
        r.register_tool(_TracingToolObserver("C", trace))

        ctx = ToolCallContext(call_id="c1", tool_name="t", server_name="s", arguments={})
        async with r.around_tool_call(ctx):
            trace.append("body")

        assert trace == [
            "A:enter", "B:enter", "C:enter",
            "body",
            "C:exit", "B:exit", "A:exit",
        ]

    @pytest.mark.asyncio
    async def test_exception_propagates_and_all_exit(self):
        """Body 抛异常时，所有已 enter 的 observer 都收到 __aexit__。"""
        r = ObserverRegistry()
        trace = []
        r.register_tool(_TracingToolObserver("A", trace))
        r.register_tool(_TracingToolObserver("B", trace))

        ctx = ToolCallContext(call_id="c1", tool_name="t", server_name="s", arguments={})
        with pytest.raises(ValueError, match="boom"):
            async with r.around_tool_call(ctx):
                trace.append("body")
                raise ValueError("boom")

        assert trace == ["A:enter", "B:enter", "body", "B:exit", "A:exit"]

    def test_register_wrong_type_rejected(self):
        r = ObserverRegistry()
        with pytest.raises(TypeError):
            r.register_tool("not an observer")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            r.register_agent(42)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            r.register_llm(None)  # type: ignore[arg-type]

    def test_register_chaining(self):
        r = ObserverRegistry()
        result = (
            r.register_tool(ToolObserver())
             .register_agent(AgentObserver())
             .register_llm(LLMObserver())
        )
        assert result is r
        assert not r.empty


# =========================================================
# 3. 并发下 ctx 独立（不串号）
# =========================================================


class _RecordingToolObserver(ToolObserver):
    def __init__(self):
        self.ctxs_seen: list[ToolCallContext] = []

    @asynccontextmanager
    async def around_tool_call(self, ctx: ToolCallContext):
        self.ctxs_seen.append(ctx)
        yield
        # yield 后读 ctx 已填充字段


class TestConcurrencyIsolation:
    @pytest.mark.asyncio
    async def test_concurrent_tool_calls_have_distinct_ctxs(self):
        """asyncio.gather 下每个 tool call 拿到独立 ctx 对象。"""
        r = ObserverRegistry()
        obs = _RecordingToolObserver()
        r.register_tool(obs)

        async def one_call(i: int):
            ctx = ToolCallContext(
                call_id=f"c{i}",
                tool_name="search",
                server_name="web",
                arguments={"q": f"query {i}"},
            )
            async with r.around_tool_call(ctx):
                await asyncio.sleep(0.01 * i)  # 模拟执行
                ctx.result = {"text": f"result {i}"}
                ctx.duration_ms = i * 10

        await asyncio.gather(*[one_call(i) for i in range(5)])

        assert len(obs.ctxs_seen) == 5
        # 每个 ctx 的 call_id 独立
        assert {c.call_id for c in obs.ctxs_seen} == {"c0", "c1", "c2", "c3", "c4"}
        # 每个 ctx 是独立实例（不是同一对象）
        assert len({id(c) for c in obs.ctxs_seen}) == 5
        # 每个 ctx 的 result 独立填充（验证 yield 后赋值可见）
        for c in obs.ctxs_seen:
            i = int(c.call_id[1:])
            assert c.result == {"text": f"result {i}"}
            assert c.duration_ms == i * 10


# =========================================================
# 4. Yield 后字段可见
# =========================================================


class TestContextFieldsAfterYield:
    @pytest.mark.asyncio
    async def test_observer_reads_filled_fields_after_body(self):
        """Observer 在 yield 之后（__aexit__ 前）能读到 runtime 填充的字段。"""
        captured = {}

        class Probe(ToolObserver):
            @asynccontextmanager
            async def around_tool_call(self, ctx):
                yield
                captured["result"] = ctx.result
                captured["error"] = ctx.error
                captured["duration_ms"] = ctx.duration_ms

        r = ObserverRegistry().register_tool(Probe())
        ctx = ToolCallContext(call_id="c1", tool_name="t", server_name="s", arguments={})
        async with r.around_tool_call(ctx):
            ctx.result = {"data": "ok"}
            ctx.duration_ms = 42

        assert captured == {"result": {"data": "ok"}, "error": None, "duration_ms": 42}

    @pytest.mark.asyncio
    async def test_observer_reads_error_on_exception(self):
        """Body 抛异常时，observer 仍能通过 __aexit__ 感知（但 ctx.error 需 runtime 填充）。"""
        captured = {}

        class Probe(ToolObserver):
            @asynccontextmanager
            async def around_tool_call(self, ctx):
                try:
                    yield
                except Exception as e:
                    captured["caught"] = str(e)
                    raise

        r = ObserverRegistry().register_tool(Probe())
        ctx = ToolCallContext(call_id="c1", tool_name="t", server_name="s", arguments={})
        with pytest.raises(ValueError):
            async with r.around_tool_call(ctx):
                raise ValueError("runtime failed")

        assert captured["caught"] == "runtime failed"


# =========================================================
# 5. DeepResearch / MainLoopRunner 集成
# =========================================================


class TestRuntimeIntegration:
    @pytest.mark.asyncio
    async def test_main_loop_runner_has_observer_registry(self):
        """MainLoopRunner 构造后 _observers 字段存在（默认空 registry）。"""
        from tests.test_mainloop_tools import _build_runner

        runner, _ = _build_runner([("ok", True, None)])
        assert hasattr(runner, "_observers")
        assert isinstance(runner._observers, ObserverRegistry)

    @pytest.mark.asyncio
    async def test_agent_observer_receives_final_answer(self):
        """跑一轮 main loop，观察 observer 的 ctx 被填充。"""
        from tests.test_mainloop_tools import _build_runner

        captured = {}

        class AgentProbe(AgentObserver):
            @asynccontextmanager
            async def around_agent_run(self, ctx):
                captured["before"] = {
                    "name": ctx.agent_name,
                    "task": ctx.task_description,
                    "final": ctx.final_answer,
                }
                yield
                captured["after"] = {
                    "name": ctx.agent_name,
                    "final": ctx.final_answer,
                }

        runner, _ = _build_runner([("Direct answer.", True, None)])
        runner._observers = ObserverRegistry().register_agent(AgentProbe())

        await runner.run(
            system_prompt="sys",
            message_history=[{"role": "user", "content": [{"type": "text", "text": "q"}]}],
            tool_definitions=[],
            main_agent_prompt_instance=MagicMock(),
            task_engine_cfg=None,
            task_description="q",
            task_guidance="",
            keep_tool_result=-1,
        )

        assert captured["before"]["task"] == "q"
        assert captured["before"]["final"] is None
        # 主循环跑完后，observer 应看到 final_answer 已被 runtime 填充
        assert captured["after"]["final"] is not None


# =========================================================
# 6. AgentRuntime 集成
# =========================================================


class TestAgentRuntimeIntegration:
    def test_agent_runtime_default_empty_registry(self):
        from mem_deep_research_core.core.agent_runtime import AgentRuntime

        rt = AgentRuntime()
        assert rt.observers is not None
        assert isinstance(rt.observers, ObserverRegistry)
        assert rt.observers.empty

    def test_agent_runtime_accepts_custom_registry(self):
        from mem_deep_research_core.core.agent_runtime import AgentRuntime

        custom = ObserverRegistry().register_tool(ToolObserver())
        rt = AgentRuntime(observers=custom)
        assert rt.observers is custom
        assert not rt.observers.empty
