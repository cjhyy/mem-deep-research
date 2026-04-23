# Observability 接入指南

> 配套设计：`docs/27-observability-slots.md`
> 示例代码：`example_project/observability_example.py`

框架为 observability 提供三个插槽，业务项目写适配器接入 Langfuse / OpenTelemetry / Datadog / 其他后端。**框架本身不依赖任何 observability SDK**。

## 最小上手

```python
from contextlib import asynccontextmanager
from mem_deep_research_core.observability import (
    ObserverRegistry, ToolObserver, ToolCallContext,
)

class MyToolObserver(ToolObserver):
    @asynccontextmanager
    async def around_tool_call(self, ctx: ToolCallContext):
        print(f"tool {ctx.tool_name} start, args={ctx.arguments}")
        yield
        print(f"tool {ctx.tool_name} end, duration={ctx.duration_ms}ms")

from mem_deep_research_core import DeepResearch

dr = DeepResearch(
    ...,
    observers=ObserverRegistry().register_tool(MyToolObserver()),
)
```

## 三个插槽

| 插槽 | 包裹对象 | Context 字段 |
|------|---------|---------------|
| `ToolObserver.around_tool_call(ctx)` | 每次 tool 执行 | `call_id` / `tool_name` / `server_name` / `arguments` / `agent_name` / `turn_number`；yield 后填充 `result` / `error` / `duration_ms` |
| `AgentObserver.around_agent_run(ctx)` | Main agent / sub-agent 的整个 run | `agent_name` / `agent_id` / `parent_agent_id` / `task_description` / `profile_name` / `mode`；yield 后填充 `final_answer` / `turns_executed` / `tool_calls_executed` / `error` |
| `LLMObserver.around_llm_call(ctx)` | Provider 的 create_message | `agent_name` / `turn_number` / `provider` / `model` / `messages_count`；yield 后填充 `response_text` / `stop_reason` / `token_usage` / `duration_ms` / `error` |

## 几个核心原则

### Context 是独立实例

每次 tool call / agent run / LLM call 都构造一个**新的** Context 对象，所以并发调用下 observer 看到的 ctx 互不污染。用 `ctx.call_id` / `ctx.agent_id` 做关联键（不要用 `tool_name` / `agent_name`）。

### Yield 之后读填充字段

```python
@asynccontextmanager
async def around_tool_call(self, ctx):
    # yield 之前 ctx.result / ctx.duration_ms 还是 None
    with span_start(...) as span:
        yield
        # yield 之后 runtime 已经填充了 result / error / duration_ms
        span.update(output=ctx.result, duration=ctx.duration_ms)
```

### Observer 不吞异常

如果 body（runtime 的 tool 执行）抛异常，异常会透传给 `__aexit__`，observer 自己决定是否记录。**不要** silently swallow，否则上层 runtime 无法区分"失败"和"成功"。

```python
@asynccontextmanager
async def around_tool_call(self, ctx):
    try:
        yield
    except Exception as e:
        span.update(level="ERROR", status_message=str(e))
        raise  # ← 必须重新抛出
```

### 多 observer 并存

`ObserverRegistry` 可以同时注册多个 observer（Langfuse + Prometheus + stdout）：

```python
registry = (
    ObserverRegistry()
    .register_tool(LangfuseToolObserver())
    .register_tool(PrometheusToolObserver())   # 两个都会被调用，嵌套顺序 = list 顺序
    .register_agent(LangfuseAgentObserver())
    .register_llm(LangfuseLLMObserver())
)
```

## Langfuse 接入示例

```python
from contextlib import asynccontextmanager
from langfuse import Langfuse
from mem_deep_research_core.observability import (
    ToolObserver, AgentObserver, LLMObserver,
    ToolCallContext, AgentRunContext, LLMCallContext,
)

_langfuse = Langfuse()


class LangfuseToolObserver(ToolObserver):
    @asynccontextmanager
    async def around_tool_call(self, ctx: ToolCallContext):
        with _langfuse.start_as_current_observation(
            name=ctx.tool_name,
            as_type="tool",
            input=ctx.arguments,
        ) as span:
            try:
                yield
                if ctx.error:
                    span.update(level="ERROR", status_message=ctx.error)
                else:
                    span.update(
                        output=ctx.result,
                        metadata={"duration_ms": ctx.duration_ms},
                    )
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                raise


class LangfuseAgentObserver(AgentObserver):
    @asynccontextmanager
    async def around_agent_run(self, ctx: AgentRunContext):
        # Sub-agent 自动嵌套到当前 OTel context（父 agent 的 span）
        with _langfuse.start_as_current_observation(
            name=ctx.agent_name,
            as_type="agent",
            input=ctx.task_description,
        ) as span:
            try:
                yield
            finally:
                if ctx.error:
                    span.update(level="ERROR", status_message=ctx.error)
                else:
                    span.update(
                        output=ctx.final_answer[:5000] if ctx.final_answer else None,
                        metadata={
                            "profile": ctx.profile_name,
                            "mode": ctx.mode,
                            "turns": ctx.turns_executed,
                            "tool_calls": ctx.tool_calls_executed,
                        },
                    )


class LangfuseLLMObserver(LLMObserver):
    @asynccontextmanager
    async def around_llm_call(self, ctx: LLMCallContext):
        with _langfuse.start_as_current_observation(
            name="llm_call",
            as_type="generation",
            model=ctx.model,
        ) as span:
            try:
                yield
            finally:
                span.update(
                    output=ctx.response_text[:2000] if ctx.response_text else None,
                    usage_details=ctx.token_usage,
                    metadata={
                        "stop_reason": ctx.stop_reason,
                        "duration_ms": ctx.duration_ms,
                    },
                )
                if ctx.error:
                    span.update(level="ERROR", status_message=ctx.error)
```

业务入口：

```python
from mem_deep_research_core import DeepResearch
from mem_deep_research_core.observability import ObserverRegistry

registry = (
    ObserverRegistry()
    .register_tool(LangfuseToolObserver())
    .register_agent(LangfuseAgentObserver())
    .register_llm(LangfuseLLMObserver())
)

dr = DeepResearch(profile="deep_research", observers=registry)
```

## 业务根 span（外部 trace 绑定）

如果你需要把整个 DeepResearch 运行绑到已有的 trace（比如 HTTP 请求的 trace_id），在业务层自己起根 span，框架的 observer 会自动嵌套：

```python
# 业务层（tanka-assistant 风格）
with langfuse_client.start_as_current_observation(
    name="my_service_call",
    as_type="agent",
    trace_context={"trace_id": request.trace_id},
    input=request.query,
) as root:
    root.update_trace(
        user_id=request.user_id,
        session_id=request.session_id,
    )
    # 框架的 observers 会自动嵌套到 root 下
    result = await dr.run(request.query)
    root.update(output=result.final_answer)
```

## Observer 和 Hook 的区别

两者独立，职责不同：

| 维度 | Hook | Observer |
|------|------|----------|
| 触发形态 | 同步函数 `hook(ctx, next_fn)` | async context manager `around_*(ctx)` |
| 目的 | 业务层修改执行逻辑（改参数 / 改结果 / guardrail 拦截）| 业务层观测执行过程（log / trace / metric）|
| Context 栈支持 | 否 | 是（__aenter__ 和 __aexit__ 在同一 async 栈，OTel context 嵌套正确）|
| 并发安全 | 两次独立调用（on_tool_start / on_tool_end），缺关联键 | 每次调用独立 ctx 对象，天然隔离 |

两个并存，按需各自注册。

## 空 registry 零开销

未传 `observers=` 时，所有 `around_*` 快速返回（`AsyncExitStack` 不做任何事），框架行为完全等价。接入 observer 不会给不用它的用户带来性能负担。

## 更多

- 完整设计决策：`docs/27-observability-slots.md`
- 单文件示例：`example_project/observability_example.py`
