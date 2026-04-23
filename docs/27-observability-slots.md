# Observability 插槽契约

> 状态基线：2026-04-22
> 文档定位：框架为 observability 工具（Langfuse / OpenTelemetry / Datadog / ...）提供的最小扩展点契约；具体适配器由业务项目自己写。
> 配套阅读：`docs/26-memory-extraction-strategy.md`（strategy 层模式借鉴）

## 背景：tanka-assistant 目前的做法与其问题

`tanka-assistant/feature/relay-langfuse` 分支把 Langfuse 集成**直接打在 vendored 的框架源码里**（5 个文件改动）：

| 文件 | 改动性质 | 归属层 |
|------|---------|-------|
| `task_manager.py` | 业务层根 span（绑 trace_id / user_id / metadata） | **业务** — 完全不在框架范围 |
| `sub_agent_runner.py` | 子 agent 整体包 `as_type="agent"` 的 Langfuse span | **框架** — 缺 sub-agent 生命周期插槽 |
| `tool_executor.py` | 每次工具调用包 `as_type="tool"` 的 Langfuse span | **框架** — 能用现有 `on_tool_start/end` hook 但有配对问题 |
| `openai_compatible_client.py` | `from langfuse.openai import ...` 替换原生 openai；塞 `name="llm_call"` kwarg | **框架** — LLM 层直接绑死 Langfuse，粒度太粗 |
| `gpt_openai_client.py` | 同上 | 同上 |

### 当前集成的问题

1. **vendored 代码被污染**：框架升级 → patch 重做 → 永远冲突
2. **违反框架设计原则**：`CLAUDE.md` 明确要求"业务逻辑不内置在框架中，通过 hook 注入"
3. **LLM 层绑死**：`from langfuse.openai import` 让所有 provider 强依赖 langfuse，不用 langfuse 的用户被动接入
4. **`name="llm_call"` kwarg 是炸弹**：只有 langfuse.openai 识别该 kwarg 并 strip，降级到原生 openai 时会被 API 拒绝
5. **Tool span 配对问题**：hook 没有 `tool_call_id` 字段，用户 hook 无法关联 `on_tool_start` 和 `on_tool_end`（同 tool 并发调用时会串号）
6. **Sub-agent 级别没有生命周期插槽**：当前只有"每轮" hook，没有"整个 agent"的 `start/end`，只能直接改 `sub_agent_runner.py`
7. **Hook 不是 context manager 形态**：OTel/Langfuse span 依赖 `__enter__` / `__exit__` 在同一 async 栈内，拆成 `on_tool_start` + `on_tool_end` 两次独立 hook 会破坏 OTel context 栈

## 设计原则

1. **框架零依赖 observability 工具**：不 import langfuse / opentelemetry / datadog；所有 observer 都是业务层实现
2. **Context-manager 型插槽**：async context manager 包住整个操作（工具执行、子 agent 运行、LLM 调用），保证 OTel context 栈正确
3. **每个操作独立 observer 实例 / context**：并发不串号；用 `call_id` / `agent_id` / `request_id` 作为唯一 key
4. **Observer 可组合**：多个 observer 并行工作（Langfuse + Prometheus + Sentry），互不影响
5. **业务层适配器模式**：框架给契约，业务写 Langfuse / Datadog 适配器
6. **向后兼容**：未注入 observer 时框架行为不变；观测完全是可选能力

## 核心概念：Observer 插槽

Framework 暴露 3 个 observer 插槽：

```
Runtime 执行路径:
  ┌── ToolObserver.around_tool_call(ctx) ──┐
  │   (包裹整个 tool executor 执行)         │
  └────────────────────────────────────────┘

  ┌── AgentObserver.around_agent_run(ctx) ──┐
  │   (包裹 main agent / sub-agent 生命周期)│
  └────────────────────────────────────────┘

  ┌── LLMObserver.around_llm_call(ctx) ──────┐
  │   (包裹 LLM provider 的 create_message) │
  └───────────────────────────────────────────┘
```

每个 observer 是一个 **async context manager**，包裹具体操作的整个执行范围。

## 插槽契约

### 1. Observer Protocol

```python
# mem_deep_research_core/observability/base.py
from abc import ABC
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallContext:
    """每次工具调用的观测上下文（唯一 id，用于 observer 关联）。"""
    call_id: str           # 框架分配的 tool call 唯一 id（和 LLM 的 tool_use_id 一致）
    tool_name: str
    server_name: str
    arguments: dict
    agent_name: str        # "main" / "agent-xxx" / 用户自定义
    turn_number: int
    # 结果在 context manager 内 yield 之后由框架填充：
    result: dict | None = None
    error: str | None = None
    duration_ms: int | None = None


@dataclass
class AgentRunContext:
    """每次 agent 运行（main / sub / spawn）的观测上下文。"""
    agent_name: str        # "main" / "agent-xxx"
    agent_id: str          # 框架分配的 run 唯一 id
    parent_agent_id: str | None = None  # sub-agent 的父 agent
    task_description: str = ""
    profile_name: str = ""  # StandardProfile / DeepResearchProfile
    mode: str = ""          # quick / standard / deep
    # 结果在 yield 之后填充：
    final_answer: str | None = None
    turns_executed: int = 0
    tool_calls_executed: int = 0
    error: str | None = None


@dataclass
class LLMCallContext:
    """每次 LLM 调用的观测上下文。"""
    agent_name: str
    turn_number: int
    provider: str          # "anthropic" / "openai" / ...
    model: str
    messages_count: int
    # 结果 yield 后填充：
    response_text: str | None = None
    stop_reason: str | None = None
    token_usage: dict | None = None  # {"input": N, "output": N, "total": N}
    duration_ms: int | None = None
    error: str | None = None


class ToolObserver(ABC):
    @asynccontextmanager
    async def around_tool_call(self, ctx: ToolCallContext):
        """包裹整个工具执行。yield 之后 ctx.result / ctx.error / ctx.duration_ms 已填充。"""
        yield


class AgentObserver(ABC):
    @asynccontextmanager
    async def around_agent_run(self, ctx: AgentRunContext):
        """包裹整个 agent 生命周期。yield 之后 ctx.final_answer / turns_executed / ... 已填充。"""
        yield


class LLMObserver(ABC):
    @asynccontextmanager
    async def around_llm_call(self, ctx: LLMCallContext):
        """包裹 LLM provider 的 create_message 调用。yield 之后 ctx.response_text / token_usage / ... 已填充。"""
        yield
```

### 2. Composable Registry

```python
# mem_deep_research_core/observability/registry.py
class ObserverRegistry:
    """并行持有多个 observer，按注册顺序串联调用。"""

    def __init__(self):
        self._tool_observers: list[ToolObserver] = []
        self._agent_observers: list[AgentObserver] = []
        self._llm_observers: list[LLMObserver] = []

    def register_tool(self, obs: ToolObserver): ...
    def register_agent(self, obs: AgentObserver): ...
    def register_llm(self, obs: LLMObserver): ...

    @asynccontextmanager
    async def around_tool_call(self, ctx: ToolCallContext):
        """多个 tool observer 嵌套调用。list 顺序 = 外层到内层。"""
        async with AsyncExitStack() as stack:
            for obs in self._tool_observers:
                await stack.enter_async_context(obs.around_tool_call(ctx))
            yield

    # 类似 agent / llm
```

### 3. 框架注入点

**`AgentRuntime` 加可选 `observer_registry`**：

```python
class AgentRuntime:
    def __init__(
        self,
        ...,
        observers: ObserverRegistry | None = None,
    ):
        self.observers = observers or ObserverRegistry()  # 默认空 registry
```

**三处调用点**：

```python
# tool_executor.py
async def execute_single_tool(self, ..., call_id):
    ctx = ToolCallContext(call_id=call_id, tool_name=..., ...)
    async with self._observers.around_tool_call(ctx):
        try:
            result = await self._execute_with_retry(...)
            ctx.result = result
        except Exception as e:
            ctx.error = str(e)
            raise
        finally:
            ctx.duration_ms = ...
    return result, ctx.duration_ms


# sub_agent_runner.py / orchestrator.py (main agent 入口)
async def run(self, agent_name, task_description, ...):
    ctx = AgentRunContext(
        agent_name=agent_name,
        agent_id=str(uuid.uuid4()),
        parent_agent_id=...,
        task_description=task_description,
        profile_name=self.profile.name,
        mode=effective_mode,
    )
    async with self._observers.around_agent_run(ctx):
        try:
            final_answer = await self._run_inner(...)
            ctx.final_answer = final_answer
            ctx.turns_executed = ...
            ctx.tool_calls_executed = ...
        except Exception as e:
            ctx.error = str(e)
            raise
    return final_answer


# provider_client_base.py
async def create_message(self, ...):
    ctx = LLMCallContext(
        agent_name=agent_type,
        turn_number=step_id,
        provider=self.__class__.__name__,
        model=self.model_name,
        messages_count=len(message_history),
    )
    async with self._observers.around_llm_call(ctx):
        try:
            response = await self._do_create_message(...)
            ctx.response_text = ...
            ctx.stop_reason = ...
            ctx.token_usage = self.get_usage()
        except Exception as e:
            ctx.error = str(e)
            raise
    return response
```

### 4. DeepResearch 入口接受 observers

```python
dr = DeepResearch(
    profile="deep_research",
    observers=ObserverRegistry().register_tool(langfuse_tool_obs)
                                 .register_agent(langfuse_agent_obs)
                                 .register_llm(langfuse_llm_obs),
)
```

## 业务侧 — Langfuse 适配器的样子

**tanka-assistant 自己写**（不进框架）：

```python
# tanka_assistant/observability/langfuse_adapter.py
from contextlib import asynccontextmanager
from langfuse import Langfuse
from mem_deep_research_core.observability import (
    ToolObserver, AgentObserver, LLMObserver,
    ToolCallContext, AgentRunContext, LLMCallContext,
)

_langfuse = Langfuse()  # 或单例


class LangfuseAgentObserver(AgentObserver):
    @asynccontextmanager
    async def around_agent_run(self, ctx: AgentRunContext):
        trace_ctx = {"trace_id": _current_trace_id()} if ctx.parent_agent_id is None else None
        with _langfuse.start_as_current_observation(
            name=ctx.agent_name,
            as_type="agent",
            trace_context=trace_ctx,
            input=ctx.task_description,
        ) as span:
            try:
                yield
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
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                raise


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
                        output=ctx.result.get("result") if isinstance(ctx.result, dict) else ctx.result,
                        metadata={"duration_ms": ctx.duration_ms},
                    )
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                raise


class LangfuseLLMObserver(LLMObserver):
    @asynccontextmanager
    async def around_llm_call(self, ctx: LLMCallContext):
        with _langfuse.start_as_current_observation(
            name=f"llm_call",
            as_type="generation",
            input={"messages_count": ctx.messages_count, "provider": ctx.provider},
            model=ctx.model,
        ) as span:
            try:
                yield
                if ctx.error:
                    span.update(level="ERROR", status_message=ctx.error)
                else:
                    span.update(
                        output=ctx.response_text[:2000] if ctx.response_text else None,
                        usage_details=ctx.token_usage,
                        metadata={"stop_reason": ctx.stop_reason, "duration_ms": ctx.duration_ms},
                    )
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                raise


# tanka-assistant 入口
def make_observer_registry() -> ObserverRegistry:
    from mem_deep_research_core.observability import ObserverRegistry
    r = ObserverRegistry()
    r.register_tool(LangfuseToolObserver())
    r.register_agent(LangfuseAgentObserver())
    r.register_llm(LangfuseLLMObserver())
    return r
```

### tanka-assistant 的 `task_manager.py` 改造

```python
# tanka_assistant/deep_research/service/task_manager.py
from langfuse import Langfuse
from tanka_assistant.observability.langfuse_adapter import make_observer_registry

lf_client = Langfuse()

async def run_task(task_id, query, context):
    # 业务层自己管根 span（绑 trace_id / user_id）
    with lf_client.start_as_current_observation(
        name=f"relay_{config_name}",
        as_type="agent",
        trace_context={"trace_id": context["trace_id"].replace("-", "").lower()},
        input=context.get("original_query") or query,
    ) as root:
        root.update_trace(
            name=f"relay_{config_name}",
            user_id=context.get("user_id", ""),
            session_id=context.get("user_id", ""),
            metadata={"task_id": task_id, "project": project, "config_name": config_name},
        )

        # 框架通过 observers 接管 sub-span 嵌套
        dr = DeepResearch(
            profile="deep_research",
            observers=make_observer_registry(),
            ...
        )
        result = await dr.run(query, context=context)

        root.update(output=result.final_answer[:5000] if result.final_answer else None)
```

## 实施范围对比

### 框架需要改动的地方（v1.2.7 候选）

| 新增 | 具体位置 |
|------|---------|
| `mem_deep_research_core/observability/` 新目录 | `base.py`（Protocol + Context）+ `registry.py`（ObserverRegistry）|
| `AgentRuntime.observers` 字段 | `core/agent_runtime.py` |
| `execute_single_tool` 接入 `around_tool_call` | `core/tool_executor.py` — 原来几处 `_HAS_LANGFUSE / _lf_tool_cm` 全部删除，替换成 `async with self._observers.around_tool_call(ctx)` |
| sub_agent_runner / main agent 接入 `around_agent_run` | `core/sub_agent_runner.py` + `core/main_loop.py` |
| `LLMProviderClientBase.create_message` 接入 `around_llm_call` | `llm/provider_client_base.py` |
| `DeepResearch(observers=...)` 入口参数 | `deep_research.py` |

**框架代码变化范围**：新增 ~200 行（observability 目录）+ 修改现有 5 个文件各约 10 行（接入 context manager），总计 ~250 行可控。

### 业务项目需要改动的地方（tanka-assistant 本次 PR）

| 改动 | 性质 |
|------|------|
| 新增 `tanka_assistant/observability/langfuse_adapter.py` | 一个新文件，约 150 行 |
| 改 `task_manager.py` 仅用业务层根 span + 传 `observers=` 到 DeepResearch | 减少现有 70 行到约 40 行 |
| **删除** 所有 vendored 框架代码里的 langfuse 改动 | `mem_deep_research/mem_deep_research_core/` 下 4 文件恢复原样 |

## Hook 系统与 Observer 的关系

为什么新增 observer 而不是直接扩展现有 hook？

| 维度 | 现有 Hook | 新增 Observer |
|------|----------|-------------|
| 触发形态 | 同步函数 `hook(ctx, next_fn)` | async context manager |
| 调用时机 | 单点触发（on_tool_start 或 on_tool_end）| 包裹整段执行（__aenter__ / body / __aexit__）|
| Context 栈支持 | 否（两次独立 hook，OTel context 断裂）| 是（OTel / asyncio contextvars 在同一栈内）|
| 并发 | 不关联 start/end，缺 call_id | 每次调用独立 context 对象，天然隔离 |
| 设计目的 | 业务层修改参数 / 结果 | observability + tracing |

**两者并存，职责不同**：
- Hook = 业务层定制执行逻辑（改参数 / 改结果 / guardrail 拦截）
- Observer = 业务层观测执行过程（log / trace / metric）

## 实施分阶段

### Phase 1：Observer 基础设施（仅框架）

- `core/observability/base.py`：三个 Protocol + 三个 Context dataclass
- `core/observability/registry.py`：`ObserverRegistry` 串联多个 observer
- `core/observability/__init__.py`：导出 Protocol 和 Registry

**不改运行时代码**，只提供接口定义。版本化为 v1.2.7 候选。

### Phase 2：Runtime 接入

- `tool_executor.py`：`execute_single_tool` 包 `around_tool_call`
- `sub_agent_runner.py`：`run` 包 `around_agent_run`
- `main_loop.py` / `orchestrator.py`：main agent 入口包 `around_agent_run`
- `llm/provider_client_base.py`：`create_message` 包 `around_llm_call`
- `agent_runtime.py`：加 `observers` 字段
- `deep_research.py`：`DeepResearch(observers=...)` 入口参数

观察：**未传 observers 时行为完全等价**（空 registry 的 `AsyncExitStack` 不产生任何副作用）。

### Phase 3：测试 + 示例

- `tests/test_observability.py`：
  - Observer Protocol 默认 no-op
  - Registry 串联多个 observer
  - 并发工具调用下每个 observer 收到独立 ctx（不串号）
  - yield 后 ctx 字段已被 runtime 填充
  - Exception 透传（observer 不吞异常）
- `example_project/observability_example.py`：最小 observer 实现示例（stdout 打印，不依赖 langfuse）
- 文档：`docs/observability-guide.md`（用户接入指南）

### Phase 4（业务）：tanka-assistant 迁移

- 新增 `tanka_assistant/observability/langfuse_adapter.py`
- 改 `task_manager.py` 用框架 observer registry + 保留业务层根 span
- `mem_deep_research/` vendored 代码全部恢复原样
- 业务验证：trace 结构和现在一致（root → agent → tool / generation 嵌套）

## 关键决策

| # | 决策 | 结论 |
|---|------|------|
| 1 | Observer 接口形态 | async context manager（`around_*`），不是 start/end 对 |
| 2 | 粒度 | 三个插槽：tool / agent / LLM。不再细分 reflection / verify / summary（那些归 profile 钩子，observer 可从 agent 粒度读出）|
| 3 | 并发安全 | 每次调用独立 ctx 对象 + asyncio contextvars 继承保证 observer 能正确嵌套 |
| 4 | 组合 | `ObserverRegistry` 用 `AsyncExitStack` 串联多个 observer，顺序 = 外到内 |
| 5 | 异常处理 | Observer 不吞异常。Framework 异常透传给 observer 的 `__aexit__`，observer 自己决定是否记录 |
| 6 | 默认行为 | 空 registry 零开销（`AsyncExitStack` 什么都不做），可关断 |
| 7 | 业务侧根 span | 框架不管（tanka-assistant 自己用 langfuse 在 task_manager 起 root span，框架的 observer 自动嵌套）|
| 8 | LLM observer 必须性 | 可选。Langfuse 用户可以继续用 `langfuse.openai` 自动捕获（不需要 LLM observer）；想要更强控制才用 observer |
| 9 | Hook 系统是否调整 | 不动。Observer 和 hook 并存，职责分明 |
| 10 | Tool call_id 关联 | `ToolCallContext.call_id` 作为唯一关联键（等于 LLM 的 tool_use_id）|

## 关键风险

| 风险 | 缓解 |
|------|------|
| Observer 抛异常污染主流程 | 文档明确 observer 不吞异常；业务层 observer 自己写 try/except；框架不做一刀切兜底 |
| 多 observer 嵌套性能开销 | 空 observer 几乎零开销（一次 AsyncExitStack）；业务自己控 observer 数量 |
| OTel context 跨 async task 丢失 | `asyncio.gather` 下每个 task copy 当前 contextvars，observer 的 `start_as_current_observation` 在 task 内创建的 span 自动嵌套到父 task 的 context |
| LLM observer 和 provider-specific auto-instrument（如 langfuse.openai）冲突产生重复 span | 用户选一个方案；文档明确建议（只用 LLM observer 或只用 provider wrapper，不重复）|
| Observer 破坏 `_maybe_offload_result` 等 async 路径 | observer 只观察不修改；所有 context 字段是副本，observer 写入不影响 runtime 决策 |

## 不做的事情

- **不做内置 Langfuse / OpenTelemetry / Datadog adapter**：每个观测栈有自己的 SDK 特性和版本演进，框架维护成本太高；业务自己写 50-200 行适配器
- **不做 `BaseObserver` 基类继承**：Protocol + ABC 已足够，避免多继承复杂度
- **不做 span/metric 双轨协议**：observer 接口统一，内部想发 metric 还是 span 用户自己选
- **不做 observer 级别优先级**：按 registry list 顺序，用户需要精细控制就自己排
- **不做 hook 和 observer 的桥接层**：两者独立，用户按需各自注册
