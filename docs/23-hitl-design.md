# Human-in-the-Loop 设计

> 状态基线：2026-04-21
> 文档定位：同步/异步 HITL 最终设计方案，为 v1.3.0 / v1.4.0 HITL 能力提供实施依据
> 配套阅读：`docs/20-roadmap.md`（版本路线图）、`docs/22-profile-boundary.md`（Runtime/Profile 边界）、`docs/24-runtime-snapshot-design.md`（前置基础设施，待建）

## 目标

支持两种 HITL 形态，由框架基于响应时间**自适应切换**（用户不用提前决定）：

- **同步 HITL**：进程阻塞等待人类决定，适合 CLI / Jupyter / WebSocket 长连接
- **异步 HITL**：超过阈值后任务存盘、进程退出；人类在 UI / email / webhook 回复后由外部事件恢复任务

自动切换的关键：hook 里声明 `sync_timeout`，框架在超时后自动降级为异步路径，hook 代码无需分支处理。

## 设计原则

1. **优先 Runtime 一致性**：HITL 是 Runtime 层能力，不是 profile 专属，机制对所有 profile 都可用
2. **LLM 意图优先**：不做推断，只识别 LLM 通过 API（`stop_reason`）或 hook 抛异常明确表达的意图
3. **工具边界精确 suspend**：只在"某个具体工具真正执行前"的边界支持 durable suspend，不承诺任意 hook 点 continuation
4. **Breaking change 允许**：框架当前生态负担较轻，优先语义清晰，不为兼容旧 API 保留不优雅的形态

## 现状与缺口

### 已具备
- Hook 系统：`on_tool_start` / `on_before_llm_call` / `GuardrailError` 可实现同步拦截式 HITL
- Checkpoint / Resume 机制：轮级 checkpoint 已有

### 缺口
| 缺口 | 说明 |
|------|------|
| 循环级 suspend 原语 | 无 `await runtime.wait_for_human()` API |
| Hook 异步调用链 | `hooks.py:193-243` 的 `call()` 是纯同步实现，全仓 27 处调用点 |
| 工具边界级 resume | 现有 resume 是轮级，无法从"工具未执行的边界"恢复 |
| 完整 RuntimeSnapshot | 现有 checkpoint 漏掉 `offload_registry` / `dedup_cache` / `monitor_state` / ContextVar 类状态 / 主循环临时标志位 |
| 外部触发 resume | `resume_from` 只接 dict，无外部事件订阅机制 |
| Pending human request 存储 | 无持久化 pending state 的 store |

## Suspend / Resume 语义（v1.3.0 定版）

### 支持的 suspend 位置

**唯一官方保证点**：`on_tool_start`，即"某个具体工具真正执行前"。

其他 hook 点（`on_before_llm_call` / `on_after_llm_call` / `on_tool_filter`）**可以同步等待**（阻塞 + 超时），但**不保证 durable resume**。

### Resume 流程（关键契约）

Resume 不是 "replay 整个 turn"，而是"从工具执行游标继续"：

```
加载 snapshot
  → 跳过 LLM 调用
  → 跳过 on_tool_start hook re-entry（hook 不会因 resume 再次触发）
  → 用 effective_arguments 直接启动 pending tool
  → tool_result append 到 message_history
  → 进入下一轮正常循环
```

这个契约有一个前提：**触发 durable suspend 的 approval hook 必须是 terminal `on_tool_start` hook**。

- 也就是它在 `on_tool_start` 链中最后执行；timeout / pending 发生时，后面不再有依赖 `next_fn(ctx)` 的 hook
- 任何参数规范化、策略注入、前置 guardrail，都必须在它之前完成，并直接修改 `ctx.arguments`
- 如果业务想保留多段 pre-tool 逻辑，应该把这些逻辑内聚到 terminal approval hook 内，或保证 approval hook 是最低优先级

这样设计的好处：
- Pause 期间 message_history 里的"未配对 tool_use"不会被任何 LLM 调用看到，Anthropic API 兼容性天然保证
- 已执行工具不重跑，副作用不重复
- Hook 逻辑只跑一次，避免"hook 内有外部副作用导致 resume 后重复执行"

### 并发工具的 suspend 策略

**场景**：Turn N 并发调 3 个工具，2 个在 `on_tool_start` 抛 `PendingHumanException`。

**策略**：**等当前 batch 内所有并发工具完成或确认 pending 后，再整 task suspend**。

- 已完成的工具结果全部存入 snapshot（通过 offload ref 引用，snapshot 本身不含大结果内容）
- Pending 工具记录 effective_arguments 到 snapshot
- 进行中的工具允许自然完成（不主动 cancel，避免副作用不可追溯）
- Resume 时已完成的工具不重跑，pending 工具从 effective_arguments 启动

Phase 2 只支持"单 pending"（有多个 pending 时第一个决定，其他等 resume 后串行处理）。Phase 3 扩展为 batch 审批。

### 被拒绝的场景

```
hook 收到 decision.approved=False
  → raise GuardrailError("rejected", decision.reason)
  → 主循环 catch GuardrailError，注入错误 tool_result
  → LLM 下一轮基于拒绝信息继续
```

## 控制流契约

### `PendingHumanException` 必须透传

作为 runtime 控制流异常，不能被降级成普通 tool error。实现上必须像 `GuardrailError` 一样显式透传：

- `HookRegistry.call()` 不吞它
- `ToolExecutor.execute_single_tool()` 不包装成 `{error: ...}`
- 并发工具执行的 `asyncio.gather(..., return_exceptions=True)` 路径要识别并重新抛出
- 只有 `MainLoopRunner._execute_tools()` / 外层 `run()` 才允许 catch 并转成 `RunResult(status="awaiting_human")`

否则 async HITL 会退化成"工具报错后让 LLM 自己继续"，而不是 durable suspend。

### 参数权威来源

LLM 原始输出的 `call["arguments"]` 只是**候选参数**。经过 `on_tool_start` / HITL decision 改写后的参数才进入权威链路，但这里需要区分两层：

1. **effective arguments**：经过 hook / HITL 改写后的**占位符安全参数**，仍允许保留 `[SECURE:xxx]` 形式；这是 transcript、dedup cache、strategy summary、tool result registry、checkpoint snapshot 的唯一权威
2. **resolved execution arguments**：在真正执行工具前，由 SecureContext 把 `[SECURE:xxx]` 解析成真实值；只存在于内存，不写入 transcript / checkpoint

因此 SecureContext 的契约是：

- snapshot / pending request / 审计日志里只存 **effective arguments**
- SecureContext 解析发生在工具真正执行前的最后一步
- resume 时先恢复 effective arguments，再重新做一次 SecureContext 解析，得到 resolved execution arguments
- 如果 HITL decision payload 里引入了敏感值，必须在持久化前被转换成 `[SECURE:xxx]` 引用或被显式拒绝，不能把明文 secret 写入 checkpoint

### 子 Agent HITL 限制

Phase 2 明确**不支持子 agent 内部 HITL**：

- 子 agent（包括 `spawn_agent` 和命名 sub-agent）内部的 `wait_for_human()` 只做同步等待，超时直接抛 `GuardrailError`
- 父 agent 的 `on_tool_start`（包括 `spawn_agent` 边界）触发 HITL 合法，无限制
- 实现方式：子 agent 的 MainLoopRunner 入口设置 `_is_sub_agent_var` ContextVar，`wait_for_human()` 检测到时强制走同步等待路径

子 agent 内部 durable HITL 延后到 v1.4.0 workflow layer（子 agent 作为 workflow 节点，节点内 pause 由 workflow engine 管理）。

## API 设计

### Hook 作者侧

```python
from mem_deep_research_core.core.hooks import hooks, HookContext
from mem_deep_research_core import GuardrailError

HIGH_RISK_TOOLS = {"send_email", "execute_sql", "file_delete"}

@hooks.register("on_tool_start")
async def approval_gate(ctx: HookContext, next_fn):
    if ctx.tool_name not in HIGH_RISK_TOOLS:
        return await next_fn(ctx)

    decision = await ctx.runtime.wait_for_human(
        prompt=f"Approve {ctx.tool_name}?",
        payload={"tool": ctx.tool_name, "args": ctx.arguments, "turn": ctx.turn_number},
        sync_timeout=30.0,      # 同步等待上限，超过则自动降级为异步
        async_timeout=3600.0,   # 异步等待整体上限，超过算 task 失败
        tags=["approval", "risky_tool"],
    )

    if not decision.approved:
        raise GuardrailError("manual_rejection", decision.reason or "user rejected")

    # 人类可在 decision.payload 里修改参数
    if decision.payload and "args" in decision.payload:
        ctx.arguments.update(decision.payload["args"])

    return await next_fn(ctx)
```

### 框架入口

```python
# 三态返回
result: RunResult = await dr.run(query, context={...})

match result.status:
    case "completed":
        print(result.answer)
    case "failed":
        print(result.error)
    case "awaiting_human":
        notify_approver(
            checkpoint_id=result.checkpoint_id,
            pending=result.pending_human_request,
        )

# 外部恢复
result = await dr.resume_with_human_decision(
    checkpoint_id="abc123",
    decision=HumanDecision(
        approved=True,
        reason="reviewed by alice@example.com",
        payload={"args": {"recipient": "ops@example.com"}},
    ),
)
```

**Breaking change**：`DeepResearch.run()` 从返回 `TaskResult` 改为 `RunResult`。不做别名兼容。迁移路径写入 CHANGELOG：
- `RunResult` 设计为 **`TaskResult` 的超集**，保留现有公共字段：`task_id` / `answer` / `boxed_answer` / `status` / `duration_seconds` / `log_path` / `error` / `turns` / `tool_calls` / `error_type` / `perf_metrics` / `checkpoints`
- 新增 `.checkpoint_id` / `.pending_human_request`
- `result.answer` / `result.boxed_answer` / `result.error` 继续可用；HITL 只是在 `status == "awaiting_human"` 时多出恢复相关字段

### 数据类

```python
@dataclass
class HumanDecision:
    approved: bool
    reason: str | None = None
    payload: dict | None = None          # 人类修改后的参数或附加信息
    decided_by: str | None = None
    decided_at: datetime | None = None

@dataclass
class PendingHumanRequest:
    request_id: str
    checkpoint_id: str | None = None      # checkpoint 持久化完成后回填
    prompt: str
    payload: dict
    hook_point: str                      # 目前只有 "on_tool_start"
    turn_number: int
    tool_call_id: str                    # 定位到唯一的 in-flight tool
    sync_timeout: float
    async_timeout: float
    tags: list[str]
    created_at: datetime
    expires_at: datetime                 # created_at + async_timeout + grace_period

@dataclass
class RunResult:
    task_id: str
    status: Literal["completed", "failed", "awaiting_human"]
    answer: str | None = None
    boxed_answer: str = ""
    duration_seconds: float = 0.0
    log_path: Path | None = None
    error: str | None = None
    turns: int = 0
    tool_calls: int = 0
    error_type: str | None = None
    perf_metrics: dict | None = None
    checkpoints: list | None = None
    checkpoint_id: str | None = None
    pending_human_request: PendingHumanRequest | None = None

class PendingHumanException(Exception):
    """Runtime control flow exception — MUST NOT be swallowed by hooks/tool executor."""
    def __init__(self, request: PendingHumanRequest):
        self.request = request
```

## 核心实现机制

### 自动降级触发点

`runtime.wait_for_human()` 内部：

```python
async def wait_for_human(
    self, prompt, payload, sync_timeout, async_timeout, tags=None,
) -> HumanDecision:
    # 子 agent 禁用异步降级
    if _is_sub_agent_var.get(False):
        return await self._sync_wait_only(prompt, payload, sync_timeout, tags)

    request_id = generate_request_id()
    await self._pending_store.put(request_id, PendingHumanRequest(...))

    try:
        # 同步等待：外部 resume 会 set_result
        return await asyncio.wait_for(
            self._pending_store.wait_for_decision(request_id),
            timeout=sync_timeout,
        )
    except asyncio.TimeoutError:
        raise PendingHumanException(
            PendingHumanRequest(
                request_id=request_id,
                tool_call_id=self._current_tool_call_id,
                prompt=prompt, payload=payload, ...
            )
        )
```

### 主循环 catch 路径

```python
try:
    ... main loop ...
except PendingHumanException as e:
    # 保存完整 RuntimeSnapshot
    snapshot = self._build_runtime_snapshot(pending_request=e.request)
    checkpoint_id = await self._checkpoint_store.save(snapshot, e.request)
    e.request.checkpoint_id = checkpoint_id
    return RunResult(
        task_id=self.task_id,
        status="awaiting_human",
        answer=None,
        checkpoint_id=checkpoint_id,
        pending_human_request=e.request,
    )
```

### Resume 入口

```python
async def resume_with_human_decision(
    self, checkpoint_id: str, decision: HumanDecision,
) -> RunResult:
    snapshot = await self._checkpoint_store.load(checkpoint_id)
    request = snapshot.pending_human_request

    runner = MainLoopRunner(...)
    runner._restore_runtime_snapshot(snapshot)    # 含 ContextVar 恢复
    runner.inject_human_decision(request.request_id, decision)

    # 从工具边界继续，不重放整轮
    return await runner.run_from_tool_cursor()


def inject_human_decision(self, request_id: str, decision: HumanDecision):
    assert self._pending_request.request_id == request_id
    if not decision.approved:
        # Resume 后在工具执行位置直接抛 GuardrailError
        self._pending_rejection = GuardrailError(
            "rejected", decision.reason or "user rejected"
        )
        return
    # 把 decision 的 payload merge 到 effective_arguments
    if decision.payload and "args" in decision.payload:
        self._pending_tool_effective_arguments.update(decision.payload["args"])
```

### RuntimeSnapshot 字段清单

```python
@dataclass
class RuntimeSnapshot:
    # 元数据
    schema_version: int
    framework_version: str
    checkpoint_created_at: datetime

    # 对话 / 任务状态（现有 checkpoint 已覆盖）
    message_history: list[dict]
    turn_count: int
    session_memory: dict
    todo_state: dict | None
    last_assistant_text: str
    task_failed: bool
    tool_calls_executed: int

    # 工具边界游标（新增，支持工具边界级 resume）
    assistant_response_text: str
    current_tool_calls: list[dict]                # 当前 turn LLM 产出的完整 tool batch
    current_tool_index: int                       # suspend 时卡在哪个 tool
    completed_tool_results: list[tuple[str, str]] # (tool_call_id, offload_ref) — 不存完整结果
    effective_arguments: dict | None              # Phase 2 单 pending；Phase 3 演进为 dict[tool_call_id, dict]

    # 主循环临时标志（新增）
    reflection_pending: bool
    adaptive_pending: bool
    effective_mode: str                           # adaptive 路由后的终态
    reasoning_effort: str | None

    # 框架内部状态（新增）
    offload_registry: dict[str, OffloadRecord]
    dedup_cache: dict[str, ToolCallRecord]
    monitor_state: dict                           # ExecutionMonitor.state_snapshot()
    inline_skill_pending: list[str]               # InlineSkillSelector 待注入技能

    # ContextVar 类状态（新增，见下方契约）
    contextvar_state: dict[str, Any]              # {"temperature_override": 0.5, "deepseek_pending_tool_list": [...], ...}

    # HITL 请求
    pending_human_request: PendingHumanRequest
```

### ContextVar 恢复契约

ContextVar 不能用普通 snapshot 字段恢复（新进程的 ContextVar 初始值为空）。每个 ContextVar 持有者需暴露一对方法：

```python
# llm/provider_client_base.py
class LLMProviderClientBase:
    def save_contextvar_state(self) -> dict: ...
    def restore_contextvar_state(self, state: dict) -> None: ...
```

对应需要改造的位置：
- `llm/provider_client_base.py` — `_temperature_override_var`
- `llm/providers/deepseek_openrouter_client.py` — `_pending_tool_list_var` / `_native_tool_name_map_var`
- 子 agent 入口 — `_is_sub_agent_var`（新增）

### Resource 生命周期

MCP stdio 子进程、数据库连接、HTTP session 等 pause 时不能直接序列化。通过一对 hook 管理：

- **`on_suspend`**：进程退出前通知，各模块做资源清理（close tool session、flush pending writes 等）
- **`on_resume`**：新进程恢复后调用，重建外部资源（MCP 子进程重启、数据库连接池初始化）

context fingerprint 保证 resume 时 MCP server env 一致，冷启动成本只是子进程重建时间。

### 同步 HITL 与 checkpoint 的关系

同步 HITL 默认**不存 checkpoint**（进程崩溃 → 任务失败 → 用户重试）。

提供可选配置 `durable_sync_hitl: bool`，显式启用后同步路径也存 checkpoint。默认关闭，避免每个工具调用多 100ms+ 磁盘 I/O。

## 新增模块

```
mem_deep_research_core/core/
├── hitl/
│   ├── __init__.py
│   ├── types.py                 # HumanDecision / PendingHumanRequest / RunResult
│   ├── exceptions.py            # PendingHumanException
│   ├── pending_store.py         # Protocol + InMemory + Filesystem 实现
│   ├── checkpoint_store.py      # Protocol + Filesystem 实现（Phase 3 加 Redis/DB）
│   ├── runtime_snapshot.py      # RuntimeSnapshot 构建 + 恢复
│   └── runtime_facade.py        # Runtime 对外 API (wait_for_human, inject_human_decision)
```

## 修改点

| 文件 / 模块 | 改动 |
|------------|------|
| `core/hooks.py` | 调用链改 async，保留同步 hook 原生执行路径（检测后分发，不强制 `to_thread`） |
| `HookContext` | Runtime 通过 ContextVar 暴露（不侵入 dataclass） |
| `core/main_loop.py` `MainLoopRunner` | 外层 try/except `PendingHumanException`；`_build_runtime_snapshot()` / `_restore_runtime_snapshot()`；维护 in-flight tool cursor；新增 `run_from_tool_cursor()` |
| `core/main_loop.py` `_execute_tools()` | 并发 gather 识别 `PendingHumanException`，不 cancel 其他已启动工具，等 batch 完成后整体 suspend |
| `core/tool_executor.py` | `PendingHumanException` 透传，不降级为 tool error；返回 effective arguments |
| `core/context_manager.py` `ContextManager` | expose `get_offload_registry_snapshot()` / `restore_offload_registry()`；dedup cache 同理 |
| `core/monitoring.py` `ExecutionMonitor` | expose `state_snapshot()` / `restore_state()` |
| `core/sub_agent_runner.py` | 入口设 `_is_sub_agent_var=True`；try/except 放行 `PendingHumanException` 到父层（不吃） |
| `llm/provider_client_base.py` | 暴露 `save_contextvar_state` / `restore_contextvar_state` |
| `llm/providers/deepseek_openrouter_client.py` | 同上 |
| `DeepResearch` | `run()` 返回 `RunResult`（breaking）；新增 `resume_with_human_decision()` |

## 分阶段实施

### Phase 0：基础设施

不涉及 HITL 业务，纯构建底座。

- Hook 调用链 async 化 + 同步 hook 兼容适配
- `RuntimeSnapshot` 数据结构 + `build` / `restore` 机制
- 各模块暴露 `snapshot()` / `restore()` 方法（ContextManager / ExecutionMonitor / InlineSkillSelector）
- ContextVar 恢复契约（`save_contextvar_state` / `restore_contextvar_state`）
- `on_suspend` / `on_resume` 生命周期 hook
- Golden test 基础设施：任意运行时状态 → snapshot → restore → 状态等价

**产出**：基础设施稳定，`RuntimeSnapshot` 机制可用于 HITL 和未来的任意 durable execution 场景。

### Phase 1：同步 HITL

- `hitl/types.py` + `HumanDecision` / `PendingHumanRequest` / `RunResult` / `PendingHumanException`
- `runtime.wait_for_human()` 同步版（只 `asyncio.wait_for` + 超时抛 `TimeoutError`，不触发 snapshot）
- `on_await_human` hook 可选（业务自定义通知）
- 子 agent `_is_sub_agent_var` 检测 + 强制同步等待路径
- 超时或被拒 → `GuardrailError` 流入主循环已有逻辑
- `PendingHumanException` 透传契约在 Hook / ToolExecutor 落地

**验收**：
- 同步 hook 阻塞 + 批准 / 拒绝 / 超时三条路径
- hook 内修改 `arguments` 生效并成为 effective arguments
- 子 agent 内部 `wait_for_human` 不抛 `PendingHumanException`
- 透传测试：hook 抛 `PendingHumanException` 不被 hook/tool/sub-agent 层吞掉

### Phase 2：异步降级

- `PendingHumanException` 在主循环外层 catch + 存 checkpoint
- `CheckpointStore` Protocol + filesystem 实现（`output_dir/pending/<id>.json`）
- `PendingStore` Protocol + filesystem 实现
- `DeepResearch.run()` 返回 `RunResult` 三态（breaking）
- `DeepResearch.resume_with_human_decision()` 入口
- `run_from_tool_cursor()`：跳过 LLM / on_tool_start，直接从 effective_arguments 启动 pending tool
- `on_suspend` / `on_resume` 生命周期 hook 落地
- 并发工具 suspend 策略：等 batch 完成再整体 suspend，已完成结果通过 offload ref 存 snapshot

**验收**：
- 同 pending tool pause → 新进程 resume → 工具正确执行 + message_history 连贯
- Resume 后 LLM 调用次数 = Resume 前 + 新产生的，不重放
- Golden test：pause 前/resume 后 runtime 状态等价（除时间戳）
- 并发 3 工具、1 pending：其他 2 个自然完成，snapshot 含已完成结果的 offload ref
- `async_timeout` 到 → task 失败 + checkpoint 自动清理

### Phase 3：生产加固

- `CheckpointStore` / `PendingStore` 可插拔（Redis / Postgres 实现，至少留接口）
- `on_human_request_created` hook：业务侧通知 email / webhook / Slack
- 审批历史追溯（每个 request 的 created / decided / resumed 事件进 transcript）
- 过期 checkpoint / pending_request 的 sweeper
- `HumanDecision` 拒绝后策略可配（注入 tool_error vs abort 整 task）
- Batch pending：同 turn 多个 pending 一起批准
  - `effective_arguments` 字段演进为 `dict[tool_call_id, dict]`
  - `schema_version` bump，提供迁移
- Benchmark：pause / resume 耗时、snapshot 大小
- 文档 + example project

**验收**：
- 自定义 CheckpointStore 接入
- `on_human_request_created` hook 通知
- Example project e2e 通过
- 10 并发 task 的 pause/resume 无竞态

## 与 Roadmap 对齐

### v1.3.0
HITL 作为 Durable Execution 的第一个场景，和"统一结果生命周期"共享 `RuntimeSnapshot` 基础设施。完成标准新增：
> 同步 HITL 可用；异步 HITL 最小场景（单机 filesystem store）可用；RuntimeSnapshot 覆盖 offload / dedup / monitor / ContextVar 状态；Golden test 保证 snapshot 往返等价性。

### v1.4.0
Workflow layer 天然包含"人工确认"节点，HITL 基础设施直接被 workflow 节点调用。子 agent 内部 HITL 开放（节点内 pause 由 workflow engine 管理）。Phase 3 的 batch pending 是 workflow 并行分支的自然扩展。

### Profile 拆分（`docs/22`）
HITL 是 **Runtime 能力**，不是 profile 专属。`DeepResearchProfile` 可以在默认策略里定义"某类 tool 自动触发审批"，但机制在 runtime 层。

## 不做的事情

- **不做 webhook server**：框架只提供 `resume_with_human_decision()` API，webhook / UI 由用户项目实现
- **不做内置审批 UI**：example project 提供 CLI 审批器作为参考
- **不做多步多人审批流**：workflow layer（v1.4.0）负责
- **不做加密 / 访问控制**：checkpoint 里可能含敏感数据，由部署层负责加密
- **不做任意 hook 点 continuation**：只保证 `on_tool_start` 边界级 suspend

## 关键决策汇总

| # | 决策 | 结论 |
|---|------|------|
| 1 | Hook 系统 async 化方式 | 调用链改 async，保留同步 hook 原生执行路径（检测后分发，不强制 `to_thread`） |
| 2 | Runtime 注入方式 | ContextVar（无侵入） |
| 3 | 超时降级控制位置 | `wait_for_human()` 内部自动抛 `PendingHumanException`，hook 代码无需分支处理 |
| 4 | Checkpoint 存储 | Phase 1/2 filesystem；Phase 3 可插拔（Redis / Postgres） |
| 5 | 并发 pending 策略 | Phase 2 等 batch 完成后整体 suspend（"单 pending 决定"）；Phase 3 扩展 batch |
| 6 | `DeepResearch.run()` 返回值 | 直接 breaking，不做别名兼容，CHANGELOG 写清迁移 |
| 7 | Snapshot schema 版本化 | Phase 0 起引入 `schema_version`，字段增删走迁移 |
| 8 | Replay 粒度 | 工具边界级 resume（`on_tool_start` 边界），不支持任意 hook 点 continuation |
| 9 | 子 agent HITL | Phase 2 禁用子 agent 内部 durable HITL（只做同步等待 + 超时转 GuardrailError），v1.4.0 workflow layer 开放 |
| 10 | Resume 时 hook 是否重新执行 | **不重新执行**。Resume 跳过 LLM 调用、跳过 `on_tool_start` re-entry，直接从 effective_arguments 启动 pending tool |

## 关键风险

| 风险 | 严重度 | 缓解 |
|------|-------|------|
| Hook async 化影响现有用户 hook | 中 | 保留同步 hook 原生路径；文档给明确迁移 |
| RuntimeSnapshot 完整性有遗漏 | 高 | Phase 0 Golden test 保证往返等价；ContextVar 契约强制 |
| Resume 和原执行行为漂移 | 高 | 每次修改 snapshot schema 都补回归测试；`schema_version` 不兼容直接报错 |
| 子 agent 内 HITL 被误用 | 中 | 运行时检测 `_is_sub_agent_var`，强制降级为同步 + 超时 GuardrailError，文档明确限制 |
| 并发工具副作用不可追溯 | 中 | 已启动工具不主动 cancel，等自然完成；副作用记录到 transcript |
| Checkpoint 里敏感数据泄露 | 中 | 文档声明部署层负责加密；`_secure` 字段可选过滤 |

## 测试清单

### Phase 0
- 任意 runtime 状态 → snapshot → restore → 等价（除时间戳）
- ContextVar 往返：`_temperature_override_var` / `_pending_tool_list_var` / `_native_tool_name_map_var`
- 同步 hook 在 async 调用链中正确执行（不用 `to_thread`）
- `on_suspend` / `on_resume` hook 触发时机正确

### Phase 1
- 同步 hook 阻塞 + 批准返回
- 同步 hook 阻塞 + 拒绝 → `GuardrailError`
- 同步 hook sync_timeout → `TimeoutError`（phase 1 不抛 `PendingHumanException`）
- hook 内修改 arguments 成为 effective arguments
- 子 agent 内 `wait_for_human` 强制同步 + 超时 `GuardrailError`
- `PendingHumanException` 透传：hook / tool_executor / sub_agent_runner 不吞

### Phase 2
- sync_timeout 到 → `PendingHumanException` → 自动 checkpoint + `awaiting_human` 返回
- `resume_with_human_decision` 后工具正确执行
- Resume 后 runtime 状态和 pause 前等价（Golden test）
- Resume 后 LLM 调用次数 = 预期（不重放）
- 并发 3 工具、1 pending：其他 2 个自然完成，snapshot 含 offload ref
- `async_timeout` 到 → task 失败 + checkpoint 清理
- 多 task 的 pending request 隔离
- `schema_version` 不兼容时明确报错

### Phase 3
- 自定义 CheckpointStore / PendingStore 接入
- `on_human_request_created` 通知 hook
- 审批 trace 完整
- Batch pending：同 turn 多个 pending 一起批准
- Example project e2e
- 10 并发 task 的 pause/resume 性能符合预期
