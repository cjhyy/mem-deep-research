"""Observability 插槽系统

框架暴露三个 async context manager 型插槽，业务项目写适配器接入 Langfuse /
OpenTelemetry / Datadog 等任何观测后端。

设计：docs/27-observability-slots.md

用法:
    from mem_deep_research_core.observability import (
        ObserverRegistry,
        ToolObserver, AgentObserver, LLMObserver,
        ToolCallContext, AgentRunContext, LLMCallContext,
    )

    class MyToolObserver(ToolObserver):
        @asynccontextmanager
        async def around_tool_call(self, ctx):
            # ... 你的观测逻辑 ...
            yield

    registry = ObserverRegistry().register_tool(MyToolObserver())
    dr = DeepResearch(..., observers=registry)
"""

from mem_deep_research_core.observability.base import (
    AgentObserver,
    AgentRunContext,
    LLMCallContext,
    LLMObserver,
    ToolCallContext,
    ToolObserver,
)
from mem_deep_research_core.observability.registry import ObserverRegistry

__all__ = [
    "ObserverRegistry",
    "ToolObserver",
    "AgentObserver",
    "LLMObserver",
    "ToolCallContext",
    "AgentRunContext",
    "LLMCallContext",
]
