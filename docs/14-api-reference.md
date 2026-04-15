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
    status: str = "completed"
    duration_seconds: float = 0.0
    log_path: Path | None = None
    error: str | None = None
    turns: int = 0
    tool_calls: int = 0
    error_type: str | None = None
    perf_metrics: dict | None = None
    checkpoints: list | None = None

    @property
    def success(self) -> bool: ...
```

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

## 异步上下文管理

```python
async with DeepResearch.from_project("./my_project") as dr:
    result = await dr.run("你的任务")
```
