# API 参考

## DeepResearch — 主入口

**文件**: `deep_research.py`

```python
from mem_deep_research import DeepResearch
```

### 构造方法

```python
DeepResearch(
    llm_provider: str = None,       # "anthropic" | "openai" | "openrouter" | "deepseek"
    model: str = None,              # 模型名称
    api_key: str = None,            # API Key（自动检测 provider）
    tools: list[str] = None,        # 工具列表 ["tool-calculator", ...]
    max_turns: int = 20,            # 最大轮次
    execution_mode: str = "auto",   # "auto" | "quick" | "standard" | "deep"
    response_language: str = "auto",# "auto" | "Chinese" | "English" | ...
    output_dir: str = "logs/",      # 日志输出目录
    context: dict = None,           # 运行时上下文（支持 _secure 字段）
    config_override: dict = None,   # 覆盖任意配置项
)
```

### 类方法

```python
# 从项目目录加载（推荐）— 自动发现 config/、hooks.py、.env
dr = DeepResearch.from_project("./my_project")

# 从配置目录加载
dr = DeepResearch.from_config_dir("./config")
```

### 实例方法

```python
# 异步运行
result: ResearchResult = await dr.run(
    query: str,                     # 研究任务
    context: dict = None,           # 运行时上下文（覆盖构造时的 context）
    stream_queue: asyncio.Queue = None,  # 流式事件队列
)

# 同步运行
result: ResearchResult = dr.run_sync(query, context=None)

# 批量运行
results: list[TaskResult] = await dr.run_batch(
    tasks: list[str],
    parallel: bool = False,       # 是否并行执行
    max_concurrent: int = 3,      # 最大并发数
)

# 从中断点恢复（需要之前的日志文件）
result: TaskResult = await dr.resume(
    log_path: str | Path,         # 之前任务的日志文件路径
    context: dict = None,
    stream_queue: asyncio.Queue = None,
)

# 同步恢复
result: TaskResult = dr.resume_sync(log_path, context=None)

# 校验配置
dr.validate()  # 无返回，不合法时抛 ConfigValidationError

# 列出可用工具
tools: list[dict] = await dr.list_tools()

# 关闭资源
await dr.close()
```

### Context Manager

```python
async with DeepResearch.from_project("./my_project") as dr:
    result = await dr.run("你的任务")
# 自动调用 close()
```

---

## ResearchResult — 执行结果

```python
@dataclass
class ResearchResult:
    task_id: str                    # 任务 ID
    answer: str                     # 最终答案
    boxed_answer: str = ""          # 格式化答案（如有）
    status: str = "completed"       # "completed" | "failed"
    duration_seconds: float = 0.0   # 执行耗时
    log_path: Path | None = None    # 日志文件路径
    error: str | None = None        # 错误信息

    # v0.3 执行详情
    turns: int = 0                  # 实际执行轮次
    tool_calls: int = 0             # 工具调用总数
    error_type: str | None = None   # "llm_error" | "tool_error" | "config_error" | "timeout"
    perf_metrics: dict | None = None
    checkpoints: list | None = None

    @property
    def success(self) -> bool       # status == "completed" and error is None
```

---

## 异常体系

```python
from mem_deep_research_core import (
    MemDeepResearchError,       # 基类
    # 工具异常
    ToolError,                  # 工具基类
    ToolNotFoundError,          # 工具未找到
    ServerNotFoundError,        # MCP 服务器未找到
    ToolExecutionError,         # 工具执行失败
    ToolConnectionError,        # 无法连接工具服务器
    ToolTimeoutError,           # 工具执行超时
    # LLM 异常
    LLMError,                   # LLM 基类
    LLMProviderNotFoundError,   # Provider 未找到
    LLMAPIError,                # API 调用错误
    LLMRateLimitError,          # 频率限制
    LLMResponseParseError,      # 响应解析错误
    ContextLimitError,          # 上下文超限
    # 配置异常
    ConfigurationError,         # 配置基类
    ConfigNotFoundError,        # 配置文件未找到
    ConfigValidationError,      # 配置验证失败
    MissingEnvVarError,         # 缺少环境变量
    # Pipeline 异常
    PipelineError,              # Pipeline 基类
    MaxTurnsExceededError,      # 超过最大轮次
    TaskCancelledError,         # 任务被取消
    GuardrailError,             # 护栏校验失败
    # 解析异常
    ParseError,                 # 解析基类
    JSONParseError,             # JSON 解析错误
    ToolCallParseError,         # 工具调用解析错误
)
```

### 异常层级

```
MemDeepResearchError
├── ToolError
│   ├── ToolNotFoundError
│   ├── ServerNotFoundError
│   ├── ToolExecutionError
│   ├── ToolConnectionError
│   └── ToolTimeoutError
├── LLMError
│   ├── LLMProviderNotFoundError
│   ├── LLMAPIError
│   ├── LLMRateLimitError
│   ├── LLMResponseParseError
│   └── ContextLimitError
├── ConfigurationError
│   ├── ConfigNotFoundError
│   ├── ConfigValidationError
│   └── MissingEnvVarError
├── PipelineError
│   ├── MaxTurnsExceededError
│   ├── TaskCancelledError
│   └── GuardrailError
└── ParseError
    ├── JSONParseError
    └── ToolCallParseError
```

### 异常属性

所有异常继承 `MemDeepResearchError`，提供：

- `message: str` — 错误信息
- `details: dict` — 结构化错误详情

各子类额外属性：

```python
# ToolNotFoundError
e.tool_name, e.server_name

# ServerNotFoundError
e.server_name, e.available_servers

# ToolExecutionError
e.tool_name, e.server_name, e.__cause__

# ToolTimeoutError
e.tool_name, e.timeout_seconds

# LLMAPIError
e.provider, e.status_code, e.response

# LLMRateLimitError
e.provider, e.retry_after

# GuardrailError
e.guardrail_name
```

---

## Hook API

**文件**: `core/hooks.py`

```python
from mem_deep_research_core.core.hooks import hooks, HookContext
```

### 注册

```python
# 装饰器
@hooks.register("on_tool_end", priority=10)
def my_hook(ctx: HookContext, original_fn):
    result = original_fn(ctx)  # 调用原逻辑
    return result              # 返回结果

# 直接注册
hooks.register("on_agent_start", priority=10)(my_func)
```

### HookContext

```python
@dataclass
class HookContext:
    hook_name: str              # 钩子名称
    tool_name: str = ""         # 当前工具名（on_tool_* 钩子）
    tool_arguments: dict = None # 工具参数
    result: Any = None          # 结果（on_tool_end 等）
    context: dict = None        # 运行时上下文
    extra: dict = None          # 额外数据
```

### 管理

```python
hooks.has_hooks("on_tool_end")     # 检查是否有注册的钩子
hooks.list_hooks()                 # 列出所有已注册钩子
hooks.clear("on_tool_end")        # 清除指定钩子
hooks.clear()                     # 清除所有钩子
```

### 全部钩子

| Hook | 时机 | 可修改内容 |
|------|------|-----------|
| `on_agent_start` | Agent 开始 | — |
| `on_agent_end` | Agent 结束 | — |
| `on_turn_start` | 每轮开始 | — |
| `on_turn_end` | 每轮结束 | — |
| `on_tool_start` | 工具调用前 | `arguments` |
| `on_tool_end` | 工具调用后 | `tool_result` |
| `on_tool_filter` | 去重后、执行前 | `tool_calls_batch` |
| `on_system_prompt_build` | system prompt 生成后 | 返回值（str） |
| `on_summarize_prompt_build` | 摘要 prompt 生成后 | 返回值（str） |
| `on_tool_result_format` | 工具结果格式化 | 返回值 |
| `on_thinking_generate` | thinking 描述生成 | 返回值 |
| `on_env_inject` | MCP 环境变量注入 | `server_params` |
| `on_message_intercept` | 消息拦截处理 | — |
| `on_before_llm_call` | LLM 调用前 | raise `GuardrailError` 阻止 |
| `on_after_llm_call` | LLM 调用后 | raise `GuardrailError` 拒绝 |
| `on_context_compact` | 上下文压缩时 | — |
| `on_reflection_build` | 反思 prompt 生成 | 返回值 |

---

## Provider 注册表

| 短名称 | 类名 | 环境变量 |
|--------|------|---------|
| `anthropic` | `ClaudeAnthropicClient` | `ANTHROPIC_API_KEY` |
| `openai` | `GPTOpenAIClient` | `OPENAI_API_KEY` |
| `openrouter` | `ClaudeOpenRouterClient` | `OPENROUTER_API_KEY` |
| `deepseek` | `DeepSeekOpenRouterClient` | `DEEPSEEK_API_KEY` |

API Key 自动检测规则：

| Key 前缀 | 自动检测 Provider |
|----------|------------------|
| `sk-ant-` | `anthropic` |
| `sk-or-` | `openrouter` |
| `sk-` | `openai` |
| 其他 | `openrouter`（默认） |

---

## 配置模型

**文件**: `config_schema.py`

所有配置通过 Pydantic 模型校验。关键模型：

### MainAgentConfig

```python
class MainAgentConfig(BaseModel):
    prompt: PromptConfig                    # Prompt 配置
    llm: LLMConfig                         # LLM 配置
    tool_config: list[str] = []            # 工具列表
    tool_blacklist: list = []              # 工具黑名单
    max_turns: int = 20                    # 最大轮次 (≥1)
    max_tool_calls_per_turn: int = 10      # 每轮最大工具调用数 (≥1)
    keep_tool_result: int = -1             # 保留工具结果数 (-1=全部)
    execution_mode: Literal["auto", "quick", "standard", "deep"] = "auto"
    max_concurrent_subagents: int = 3      # 最大并行子 Agent 数 (≥1)
    response_language: str = "auto"
    task_engine: TaskEngineConfig           # 深度研究/反思配置
    todo_tracker: TodoTrackerConfig
    context_manager: ContextManagerConfig
    monitoring: MonitoringConfigSchema
    skill_selection: SkillSelectionConfig
    interceptor: InterceptorConfig
```

### ContextManagerConfig

```python
class ContextManagerConfig(BaseModel):
    enable_dedup: bool = True              # 跨轮次去重
    enable_compact: bool = True            # Level 1 摘要替换
    compact_at_ratio: float = 0.6          # [0,1] L1 触发阈值
    summarize_at_ratio: float = 0.8        # [0,1] L2 触发阈值 (必须 > compact_at_ratio)
    compact_keep_recent: int = 3           # 保留最近 N 轮
    max_dedup_cache_size: int = 200        # 去重缓存上限
    result_offload_threshold: int = 5000   # 结果卸载阈值 (0=禁用)
    chars_per_token: float = 3.5           # 无 tiktoken 时的估算比例
```

### LLMConfig

```python
class LLMConfig(BaseModel):
    provider_class: str                    # Provider 类名
    model_name: str                        # 模型名称
    temperature: float = 0.3               # [0, 2]
    max_tokens: int = 32000
    max_context_length: int = -1           # -1=不限制
    enable_streaming: bool = True
```
