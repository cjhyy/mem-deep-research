"""Observability 最小示例 — 用 stdout 打印 span 事件

演示如何实现 ToolObserver / AgentObserver / LLMObserver 并挂到 DeepResearch。
不依赖 langfuse / OpenTelemetry —— 生产环境把 stdout 换成对应 SDK 即可。

运行：
    python example_project/observability_example.py
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager

from mem_deep_research_core.observability import (
    AgentObserver,
    AgentRunContext,
    LLMCallContext,
    LLMObserver,
    ObserverRegistry,
    ToolCallContext,
    ToolObserver,
)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class StdoutAgentObserver(AgentObserver):
    @asynccontextmanager
    async def around_agent_run(self, ctx: AgentRunContext):
        _log(
            f"[agent:start] name={ctx.agent_name} id={ctx.agent_id} "
            f"profile={ctx.profile_name} mode={ctx.mode} parent={ctx.parent_agent_id}"
        )
        try:
            yield
        finally:
            _log(
                f"[agent:end  ] name={ctx.agent_name} id={ctx.agent_id} "
                f"turns={ctx.turns_executed} tool_calls={ctx.tool_calls_executed} "
                f"error={ctx.error!r} "
                f"final={(ctx.final_answer or '')[:60]!r}"
            )


class StdoutToolObserver(ToolObserver):
    @asynccontextmanager
    async def around_tool_call(self, ctx: ToolCallContext):
        _log(
            f"[tool:start] call_id={ctx.call_id} name={ctx.tool_name} "
            f"server={ctx.server_name} agent={ctx.agent_name} turn={ctx.turn_number}"
        )
        try:
            yield
        finally:
            err_or_ok = f"error={ctx.error!r}" if ctx.error else "ok"
            _log(
                f"[tool:end  ] call_id={ctx.call_id} name={ctx.tool_name} "
                f"{err_or_ok} duration_ms={ctx.duration_ms}"
            )


class StdoutLLMObserver(LLMObserver):
    @asynccontextmanager
    async def around_llm_call(self, ctx: LLMCallContext):
        _log(
            f"[llm:start ] agent={ctx.agent_name} turn={ctx.turn_number} "
            f"provider={ctx.provider} model={ctx.model} "
            f"messages={ctx.messages_count}"
        )
        try:
            yield
        finally:
            _log(
                f"[llm:end   ] agent={ctx.agent_name} turn={ctx.turn_number} "
                f"stop={ctx.stop_reason} usage={ctx.token_usage} "
                f"duration_ms={ctx.duration_ms} error={ctx.error!r}"
            )


def make_stdout_registry() -> ObserverRegistry:
    """Build a registry with the three stdout observers."""
    return (
        ObserverRegistry()
        .register_agent(StdoutAgentObserver())
        .register_tool(StdoutToolObserver())
        .register_llm(StdoutLLMObserver())
    )


async def main() -> None:
    """Run DeepResearch with stdout observability wired in.

    This example expects your project to have a proper agent.yaml; if you're
    just reading the code to learn the API, the structure matters more than
    the actual run.
    """
    from mem_deep_research_core import DeepResearch

    dr = DeepResearch.from_project("example_project", observers=make_stdout_registry())
    result = await dr.run("What's the capital of France?")
    print("\n=== Final answer ===")
    print(result.final_answer)


if __name__ == "__main__":
    asyncio.run(main())
