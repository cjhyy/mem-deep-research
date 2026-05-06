# API 参考

## `DeepResearch`

```python
from mem_deep_research import DeepResearch
```

### 构造函数

```python
DeepResearch(
    llm_provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_turns: int = 20,
    max_tool_calls_per_turn: int = 10,
    temperature: float = 0.3,
    tools: list[str] | None = None,
    tool_blacklist: list | None = None,
    logs_dir: str | Path = "logs",
    response_language: str = "auto",
    chinese_context: bool = False,
    interceptor_preset: str = "default",
    hint_generation: bool = False,
    final_answer_extraction: bool = False,
    execution_mode: str = "auto",
    config: DictConfig | None = None,
    runtime: AgentRuntime | None = None,
)
```

说明：

- `config` 传入完整 OmegaConf 配置时，会覆盖前面的简写参数
- `runtime` 用于实例级 hooks / config loader 隔离

### 类方法

```python
DeepResearch.from_project(
    project_dir,
    config_name="agent",
    logs_dir=None,
    runtime=None,
)

DeepResearch.from_config_dir(
    config_dir,
    config_name="agent",
    logs_dir=None,
    env_file=None,
    runtime=None,
)
```

## 运行方法

### 异步运行

```python
result = await dr.run(
    task: str,
    context: dict | None = None,
    on_progress: Callable | None = None,
    stream_queue: Any | None = None,
)
```

### 同步运行

```python
result = dr.run_sync(task, context=None)
```

### 批量运行

```python
results = await dr.run_batch(
    tasks: list[str],
    parallel: bool = False,
    max_concurrent: int = 3,
)
```

### 恢复执行

```python
result = await dr.resume(
    log_path,
    context=None,
    stream_queue=None,
)

result = dr.resume_sync(log_path, context=None)
```

### 其他方法

```python
tools = await dr.list_tools()

report = await dr.validate()

await dr.close()
```

`validate()` 返回：

```python
{
    "valid": bool,
    "errors": list[str],
    "warnings": list[str],
    "tools_count": int,
}
```

## `TaskResult`

```python
@dataclass
class TaskResult:
    task_id: str
    answer: str
    boxed_answer: str = ""
    status: str = "completed"  # "completed" | "failed" | "awaiting_human"
    duration_seconds: float = 0.0
    log_path: Path | None = None
    error: str | None = None
    turns: int = 0
    tool_calls: int = 0
    error_type: str | None = None
    perf_metrics: dict | None = None
    checkpoints: list | None = None

    # v1.3.0 HITL — populated only when status == "awaiting_human"
    checkpoint_id: str | None = None
    pending_human_request: PendingHumanRequest | None = None

    @property
    def success(self) -> bool: ...
```

`status` 三态：

- `completed` — 任务正常完成，`answer` 是最终答案。
- `failed` — 任务失败，`error` 含错误信息，`error_type` 含分类。
- `awaiting_human` — 任务在工具边界 suspend 等待人工决定（v1.3.0+）。`checkpoint_id` 标识持久化的快照，`pending_human_request` 描述要决定什么。`answer` 为空。

## HITL 数据类（v1.3.0+）

```python
from mem_deep_research_core import HumanDecision, PendingHumanRequest

@dataclass
class HumanDecision:
    approved: bool
    reason: str | None = None
    payload: dict | None = None       # {"args": {...}} 可覆盖工具参数
    decided_by: str | None = None
    decided_at: datetime | None = None

@dataclass
class PendingHumanRequest:
    prompt: str                        # 给审批者看的提示
    payload: dict                      # 上下文（工具名、当前参数等）
    hook_point: str = "on_tool_start"  # 触发点
    turn_number: int = 0
    tool_call_id: str = ""
    sync_timeout: float = 30.0         # 同步等待秒数；超时则 suspend
    async_timeout: float = 3600.0      # 异步等待整体上限
    tags: list[str] = []
    request_id: str                    # 自动生成
    checkpoint_id: str | None = None
    created_at: datetime
    expires_at: datetime
```

## HITL Resume 入口（v1.3.0+）

```python
result = await dr.run("query")

if result.status == "awaiting_human":
    # 通知审批者... 拿到 decision 后：
    decision = HumanDecision(
        approved=True,
        reason="reviewed by alice",
        payload={"args": {"recipient": "ops@example.com"}},  # 可选：覆盖工具参数
    )
    final = await dr.resume_with_human_decision(
        checkpoint_id=result.checkpoint_id,
        decision=decision,
        # task_description 可不传，从 checkpoint 自动恢复
    )
    # final.status == "completed" 或再次 "awaiting_human"
```

参数：

- `checkpoint_id` — 来自 suspended `TaskResult.checkpoint_id`
- `decision: HumanDecision` — `approved=False` 时注入 tool error，下一轮 LLM 看到拒绝信息继续
- `task_description: str | None` — 一般不传（自动从 snapshot 解析）；仅迁移 pre-v1.3.0 checkpoint 时显式传入
- `context` / `stream_queue` — 同 `run()`

## `AgentRuntime`

```python
runtime = AgentRuntime()
dr = DeepResearch(runtime=runtime)
```

用途：

- 隔离 hooks
- 隔离 config loader
- 在 `from_project()` 场景加载项目级 `hooks.py`

## Hook API

项目目录下可放置 `hooks.py`，`from_project()` 会自动加载。

```python
from mem_deep_research_core.core.hooks import hooks, HookContext

@hooks.register("on_tool_end", priority=10)
def my_hook(ctx: HookContext, original_fn):
    result = original_fn(ctx)
    return result
```

### `HookContext` 常用字段

```python
HookContext(
    hook_name: str,
    query: str | None = None,
    result: Any | None = None,
    context: dict | None = None,
    tool_name: str | None = None,
    server_name: str | None = None,
    arguments: dict | None = None,
    tool_result: dict | None = None,
    duration_ms: int | None = None,
    turn_number: int | None = None,
    tool_calls_count: int | None = None,
    tool_calls_batch: list | None = None,
    compact_action: str | None = None,
    extra: dict = {},
)
```

### 当前支持的主要 hooks

- `on_agent_start`
- `on_agent_end`
- `on_turn_start`
- `on_turn_end`
- `on_tool_start`
- `on_tool_end`
- `on_tool_filter`
- `on_system_prompt_build`
- `on_summarize_prompt_build`
- `on_tool_result_format`
- `on_thinking_generate`
- `on_env_inject`
- `on_message_intercept`
- `on_before_llm_call`
- `on_after_llm_call`
- `on_context_compact`
- `on_reflection_build`
- `on_route_classify`
- `on_route_apply`
- `on_result_offload`
- `on_result_restore`
- `on_query_compile`
- `on_suspend` / `on_resume` (v1.3.0+) — durable execution lifecycle (HITL / process restart)
- `on_human_request_created` / `on_human_request_decided` / `on_human_request_resumed` (v1.3.0+) — HITL audit trail
- `on_await_human` (v1.3.0, **deprecated** alias of `on_human_request_created`, removed in v1.4.0)

## HITL 拒绝策略 + HitlRejectedError (v1.3.0+)

`cfg.hitl.rejection_strategy` 控制 `HumanDecision(approved=False)` 的行为：

| 值 | 行为 |
|---|---|
| `"tool_error"` (默认) | 注入 `[HITL rejected]` tool result，LLM 反应继续。适合"被拒可降级"场景 |
| `"abort_task"` | 抛 `HitlRejectedError`，pipeline 翻译成 `status=failed` + `error=HITL rejected by ...`，不再调 LLM。适合"被拒必须终止"场景（高敏感金融、删除等） |

```python
from mem_deep_research_core import HitlRejectedError  # 从 hitl.exceptions

try:
    result = await dr.resume_with_human_decision(checkpoint_id, decision)
except HitlRejectedError as e:
    # 仅 abort_task 路径会到这里 — pipeline 已转成 status=failed，
    # 这里通常不需要再 catch，除非自己组装 RuntimeFacade
    pass
```

`HitlRejectedError` 同 `PendingHumanException`，被 hook 系统视为 runtime control-flow exception — 不会被 hook fallthrough 吞掉。

## HITL Transcript 审计

`Transcript` 自动写入三类 HITL 事件（`EventType`）：`HITL_REQUEST_CREATED` / `HITL_REQUEST_DECIDED` / `HITL_REQUEST_RESUMED`，共享 `request_id` 串联同一审批的全生命周期。配合上面三个 hook 可对接合规系统 / Slack 通知。

## 异步上下文管理

```python
async with DeepResearch.from_project("./my_project") as dr:
    result = await dr.run("你的任务")
```
