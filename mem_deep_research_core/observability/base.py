"""Observer 契约

框架暴露三个 observer 插槽：tool / agent / LLM。每个 observer 是 async context
manager，包裹对应操作的整段执行，保证 OTel / Langfuse / OpenTelemetry 的 context
栈正确（__aenter__ 和 __aexit__ 在同一 async 栈内）。

设计依据：docs/27-observability-slots.md

每次操作独立 Context 对象 —— 并发不串号；用 call_id / agent_id 作为唯一关联键。
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any


# =========================================================
# Context 对象（每次操作独立实例，observer 读 yield 后填充的字段）
# =========================================================


@dataclass
class ToolCallContext:
    """每次工具调用的观测上下文。

    Runtime 构造时填充 call_id / tool_name / server_name / arguments /
    agent_name / turn_number。Yield 之后由 runtime 填充 result / error /
    duration_ms。Observer 在 __aexit__ 或 yield 后读取已填充字段做记录。
    """

    call_id: str
    tool_name: str
    server_name: str
    arguments: dict
    agent_name: str = "main"
    turn_number: int = 0

    # Yield 之后填充
    result: Any = None           # dict | None：工具结果
    error: str | None = None     # 异常字符串（若有）
    duration_ms: int | None = None


@dataclass
class AgentRunContext:
    """每次 agent 运行（main / sub / spawn）的观测上下文。"""

    agent_name: str
    agent_id: str                # runtime 生成的 uuid
    parent_agent_id: str | None = None
    task_description: str = ""
    profile_name: str = ""
    mode: str = ""

    # Yield 之后填充
    final_answer: str | None = None
    turns_executed: int = 0
    tool_calls_executed: int = 0
    error: str | None = None


@dataclass
class LLMCallContext:
    """每次 LLM provider create_message 调用的观测上下文。"""

    agent_name: str
    turn_number: int
    provider: str
    model: str
    messages_count: int

    # Yield 之后填充
    response_text: str | None = None
    stop_reason: str | None = None
    token_usage: dict | None = None  # {"input_tokens": N, "output_tokens": N, "total_tokens": N}
    duration_ms: int | None = None
    error: str | None = None


# =========================================================
# Observer 基类
# =========================================================


class ToolObserver:
    """Tool 执行观测。默认空实现 —— 子类 override around_tool_call 即可。"""

    @asynccontextmanager
    async def around_tool_call(self, ctx: ToolCallContext):
        """包裹整个 tool 执行。

        Yield 之后 ctx.result / error / duration_ms 已由 runtime 填充。
        异常通过 __aexit__ 抛回，observer 自己决定是否记录。

        子类典型实现：
            async with langfuse.start_as_current_observation(
                name=ctx.tool_name, as_type="tool", input=ctx.arguments,
            ) as span:
                yield
                if ctx.error:
                    span.update(level="ERROR", status_message=ctx.error)
                else:
                    span.update(output=ctx.result, metadata={"duration_ms": ctx.duration_ms})
        """
        yield


class AgentObserver:
    """Agent 运行观测（main / sub / spawn）。"""

    @asynccontextmanager
    async def around_agent_run(self, ctx: AgentRunContext):
        """包裹整个 agent 生命周期。

        Yield 之后 ctx.final_answer / turns_executed / tool_calls_executed / error 已填充。
        """
        yield


class LLMObserver:
    """LLM provider.create_message 观测。"""

    @asynccontextmanager
    async def around_llm_call(self, ctx: LLMCallContext):
        """包裹一次 LLM 调用。

        Yield 之后 ctx.response_text / stop_reason / token_usage / duration_ms / error 已填充。
        """
        yield


__all__ = [
    "ToolCallContext",
    "AgentRunContext",
    "LLMCallContext",
    "ToolObserver",
    "AgentObserver",
    "LLMObserver",
]
