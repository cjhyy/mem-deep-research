"""ObserverRegistry — 串联多个 observer，按注册顺序嵌套调用

AsyncExitStack 保证多个 observer 的 __aenter__ / __aexit__ 在同一 async 栈内，
OTel context 嵌套正确。List 顺序 = 外到内（先注册的是最外层 span）。
"""

from contextlib import AsyncExitStack, asynccontextmanager

from mem_deep_research_core.observability.base import (
    AgentObserver,
    AgentRunContext,
    LLMCallContext,
    LLMObserver,
    ToolCallContext,
    ToolObserver,
)


class ObserverRegistry:
    """持有多个 observer 实例，为 runtime 提供统一的 around_* 入口。

    空 registry（未注册任何 observer）时所有 around_* 上下文管理器零开销
    （AsyncExitStack 不做任何事），runtime 行为完全等价未接入观测。

    用法:
        registry = ObserverRegistry()
        registry.register_tool(LangfuseToolObserver())
        registry.register_agent(LangfuseAgentObserver())
        registry.register_llm(LangfuseLLMObserver())

        dr = DeepResearch(profile=..., observers=registry)
    """

    def __init__(self):
        self._tool_observers: list[ToolObserver] = []
        self._agent_observers: list[AgentObserver] = []
        self._llm_observers: list[LLMObserver] = []

    # ---- register API（链式）----

    def register_tool(self, obs: ToolObserver) -> "ObserverRegistry":
        if not isinstance(obs, ToolObserver):
            raise TypeError(f"Expected ToolObserver subclass, got {type(obs)!r}")
        self._tool_observers.append(obs)
        return self

    def register_agent(self, obs: AgentObserver) -> "ObserverRegistry":
        if not isinstance(obs, AgentObserver):
            raise TypeError(f"Expected AgentObserver subclass, got {type(obs)!r}")
        self._agent_observers.append(obs)
        return self

    def register_llm(self, obs: LLMObserver) -> "ObserverRegistry":
        if not isinstance(obs, LLMObserver):
            raise TypeError(f"Expected LLMObserver subclass, got {type(obs)!r}")
        self._llm_observers.append(obs)
        return self

    # ---- around_* 入口（runtime 调用）----

    @asynccontextmanager
    async def around_tool_call(self, ctx: ToolCallContext):
        """嵌套调用所有 tool observer。list 顺序 = 外到内。

        Exception 透传 —— 任一 observer 或 body 抛异常都向上传播；
        AsyncExitStack 保证所有已 enter 的 observer 的 __aexit__ 都被调用。
        """
        if not self._tool_observers:
            # 空 registry 快速路径
            yield
            return
        async with AsyncExitStack() as stack:
            for obs in self._tool_observers:
                await stack.enter_async_context(obs.around_tool_call(ctx))
            yield

    @asynccontextmanager
    async def around_agent_run(self, ctx: AgentRunContext):
        if not self._agent_observers:
            yield
            return
        async with AsyncExitStack() as stack:
            for obs in self._agent_observers:
                await stack.enter_async_context(obs.around_agent_run(ctx))
            yield

    @asynccontextmanager
    async def around_llm_call(self, ctx: LLMCallContext):
        if not self._llm_observers:
            yield
            return
        async with AsyncExitStack() as stack:
            for obs in self._llm_observers:
                await stack.enter_async_context(obs.around_llm_call(ctx))
            yield

    # ---- 辅助 ----

    @property
    def empty(self) -> bool:
        """True 表示没有任何 observer 注册（可用于跳过 ctx 构造优化）。"""
        return not (
            self._tool_observers or self._agent_observers or self._llm_observers
        )


__all__ = ["ObserverRegistry"]
