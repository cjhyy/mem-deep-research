# Mem Deep Research

可扩展的 AI Agent 端到端任务完成框架。基于 MCP 工具协议，支持多 LLM 提供商。

## 特性

- **MCP 工具系统** — 本地 (stdio)、远程 HTTP (streamable-http)、SSE 三种传输模式
- **内置工具** — 计算器、文件系统读写、代码执行器、搜索引擎
- **三级上下文管理** — Observation Masking → LLM 摘要压缩 → 二分裁剪，自动防爆
- **执行监控** — 三级升级策略 (WARN → INJECT_HINT → TERMINATE)，循环检测 + 超时控制
- **Hook 系统** — 17 个生命周期钩子，所有行为可自定义
- **多语言支持** — `response_language: auto` 自动检测回答语言
- **子 Agent** — 复杂任务自动分解，子 Agent 复用主循环，上下文隔离
- **记忆系统** — SessionMemory（短期）+ LongTermMemory（跨 session 持久化）
- **任务追踪** — TodoTracker 独立于消息历史，上下文压缩时不丢失
- **四种执行模式** — quick / standard / deep / auto，按需选择
- **任务恢复** — `resume()` 从中断点恢复执行
- **Skill 系统** — 规则匹配 / LLM 选择 / Inline 三种模式
- **隐私保护** — `_secure` 字段自动占位符替换
- **多 LLM 支持** — Anthropic、OpenAI、OpenRouter、DeepSeek 等

## 快速开始

### 1. 安装

```bash
git clone https://github.com/cjhyy/mem-deep-research.git
cd mem-deep-research
pip install -e .
```

### 2. 创建项目

最快的方式：复制 `example_project`。

```bash
cp -r example_project my_project
cd my_project
```

编辑 `.env`，填入你的 API Key：

```bash
OPENROUTER_API_KEY=your-key-here
```

### 3. 运行

```bash
python run.py "量子计算的基本原理是什么？"
```

搞定。框架会自动检测语言，用中文回答。

### 4. 代码调用

```python
from mem_deep_research import DeepResearch

# 方式 1: 从项目目录加载（推荐）
dr = DeepResearch.from_project("./my_project")
result = await dr.run("你的研究任务")
print(result.answer)

# 方式 2: 纯代码配置
dr = DeepResearch(
    llm_provider="openrouter",
    model="anthropic/claude-sonnet-4",
    api_key="your-key",
    tools=["tool-calculator"],
)
result = await dr.run("123 * 456 + 789")

# 同步调用
result = dr.run_sync("你的任务")

# 批量运行
results = await dr.run_batch(["任务1", "任务2"], parallel=True, max_concurrent=3)

# 从中断点恢复
result = await dr.resume("logs/task_xxx.json")
```

## 项目结构

```
my_project/
├── config/
│   ├── agent.yaml              # Agent 配置（LLM、工具、参数）
│   ├── tool/                   # 自定义工具配置（MCP YAML）
│   ├── skills/definitions/     # 自定义 Skill（Markdown）
│   └── prompts/                # 自定义 Prompt 模板
├── hooks.py                    # 生命周期钩子（自动加载）
├── .env                        # API 密钥
└── run.py                      # 入口脚本
```

## 配置

### 最小配置

```yaml
# config/agent.yaml
main_agent:
  llm:
    provider_class: "ClaudeOpenRouterClient"
    model_name: "anthropic/claude-sonnet-4"
    openrouter_api_key: "${oc.env:OPENROUTER_API_KEY}"
  tool_config:
    - tool-calculator
  max_turns: 10
```

### 完整配置

```yaml
main_agent:
  prompt:
    agent_type: main             # main | worker
    tool_format: xml             # xml | native
    presets: [task_completion]    # 可选: task_completion, task_planning, time_sensitive

  llm:
    provider_class: "ClaudeOpenRouterClient"
    model_name: "anthropic/claude-sonnet-4"
    temperature: 0.3
    max_tokens: 32000
    max_context_length: -1       # -1 = 不限制
    openrouter_api_key: "${oc.env:OPENROUTER_API_KEY}"

  tool_config:
    - tool-calculator
    - tool-filesystem            # 文件读写
    - tool-searching-serper      # 需要 SERPER_API_KEY

  max_turns: 20
  max_tool_calls_per_turn: 10
  keep_tool_result: -1           # -1 全部保留

  response_language: auto        # auto | Chinese | English | Japanese | ...

  task_engine:
    enabled: false               # 配置 deep 行为参数，不再单独决定 auto 路由
    reflection_interval: 5

  context_manager:
    enable_dedup: true
    compact_at_ratio: 0.6
    summarize_at_ratio: 0.8

  monitoring:
    enable_loop_detection: true
    max_total_time: 600.0

  skill_selection:
    enabled: true
    method: inline               # rules | llm | inline

  interceptor:
    preset: default              # default | verbose | minimal | debug

# 子 Agent（可选）
sub_agents:
  agent-researcher:
    llm:
      provider_class: "ClaudeOpenRouterClient"
      model_name: "anthropic/claude-sonnet-4"
      openrouter_api_key: "${oc.env:OPENROUTER_API_KEY}"
    tool_config: [tool-searching-serper]
    max_turns: 10
```

## 自定义工具

在 `config/tool/` 下添加 YAML 文件：

```yaml
# 本地工具（stdio）
name: "tool-my-script"
tool_command: "python"
args: ["tools/my_server.py"]
env:
  MY_API_KEY: "${oc.env:MY_API_KEY}"
```

```yaml
# 远程工具（HTTP）
name: "tool-remote-api"
url: "https://api.example.com/mcp"
transport: "streamable-http"
headers:
  Authorization: "Bearer ${oc.env:API_TOKEN}"
```

然后在 `agent.yaml` 中引用：

```yaml
tool_config:
  - tool-calculator
  - tool-my-script
  - tool-remote-api
```

## Hook 系统

`hooks.py` 放在项目根目录，`from_project()` 自动加载。

```python
from mem_deep_research_core.core.hooks import hooks, HookContext

# 记录每次工具调用
@hooks.register("on_tool_end", priority=10)
def log_tool(ctx: HookContext, original_fn):
    print(f"Tool {ctx.tool_name} finished in {ctx.duration_ms}ms")
    return original_fn(ctx)

# 修改 system prompt
@hooks.register("on_system_prompt_build", priority=50)
def customize_prompt(ctx: HookContext, original_fn):
    prompt = original_fn(ctx)
    return prompt + "\n\nAlways cite sources."

# Guardrail — 阻止危险操作
from mem_deep_research_core import GuardrailError

@hooks.register("on_before_llm_call", priority=10)
def guardrail(ctx: HookContext, original_fn):
    if "DELETE" in str(ctx.extra.get("messages", [])):
        raise GuardrailError("safety", "Blocked: SQL DELETE detected")
    return original_fn(ctx)
```

### 全部可用 Hook

| Hook | 时机 | 可修改 |
|------|------|--------|
| `on_agent_start` / `on_agent_end` | Agent 生命周期 | — |
| `on_turn_start` / `on_turn_end` | 每轮开始/结束 | — |
| `on_tool_start` / `on_tool_end` | 工具调用前/后 | arguments / tool_result |
| `on_tool_filter` | 去重后、执行前 | tool_calls_batch |
| `on_system_prompt_build` | system prompt 生成后 | 返回值 |
| `on_summarize_prompt_build` | 摘要 prompt 生成后 | 返回值 |
| `on_tool_result_format` | 工具结果格式化 | 返回值 |
| `on_thinking_generate` | thinking 描述生成 | 返回值 |
| `on_before_llm_call` / `on_after_llm_call` | LLM 调用前后 | raise GuardrailError |
| `on_env_inject` | MCP 环境变量注入 | server_params |
| `on_message_intercept` | 消息拦截处理 | — |
| `on_context_compact` | 上下文压缩 | — |
| `on_reflection_build` | 反思 prompt 生成 | 返回值 |

## 隐私保护 (SecureContext)

`context` 中的 `_secure` 字段在 system prompt 中自动显示为占位符，工具调用时自动还原：

```python
result = await dr.run("查询用户信息", context={
    "user_name": "Alice",           # LLM 可见
    "_secure": {
        "user_id": "real-123",      # LLM 看到 [SECURE:user_id]
        "api_token": "secret",      # 工具调用时自动替换回真实值
    }
})
```

## LLM Provider

| Provider | 类名 | API Key 环境变量 |
|----------|------|----------------|
| OpenRouter (Claude) | `ClaudeOpenRouterClient` | `OPENROUTER_API_KEY` |
| Anthropic 直连 | `ClaudeAnthropicClient` | `ANTHROPIC_API_KEY` |
| OpenAI | `GPTOpenAIClient` | `OPENAI_API_KEY` |
| DeepSeek | `DeepSeekOpenRouterClient` | `DEEPSEEK_API_KEY` |

## 架构概览

```
DeepResearch.run(query)
  → AgentFactory → Pipeline → Orchestrator
    → PromptBuilder 构建 system prompt
    → MainLoopRunner 执行主循环:
        路由 effective_mode → LLM 调用 → 工具执行 → Context 管理 → 监控检查
        (子 Agent 复用同一 MainLoopRunner，隔离上下文)
    → SummaryHandler 生成最终摘要
  → TaskResult
```

详细文档见 [`docs/`](docs/) 目录：

| 文档 | 内容 |
|------|------|
| [00-architecture](docs/00-architecture.md) | 架构概览 |
| [01-quick-start](docs/01-quick-start.md) | 快速开始 |
| [02-configuration](docs/02-configuration.md) | 配置系统 |
| [03-core-modules](docs/03-core-modules.md) | 核心模块 |
| [04-context-management](docs/04-context-management.md) | 上下文管理 |
| [05-monitoring](docs/05-monitoring.md) | 执行监控 |
| [06-llm-providers](docs/06-llm-providers.md) | LLM Provider |
| [07-tool-system](docs/07-tool-system.md) | 工具系统 |
| [08-hook-and-secure-context](docs/08-hook-and-secure-context.md) | Hook 与隐私保护 |
| [09-prompt-system](docs/09-prompt-system.md) | Prompt 系统 |
| [10-skill-system](docs/10-skill-system.md) | Skill 系统 |
| [11-development](docs/11-development.md) | 开发指南 |
| [12-memory-and-todo](docs/12-memory-and-todo.md) | 记忆与任务追踪 |
| [13-execution-modes](docs/13-execution-modes.md) | 执行模式与语言控制 |
| [14-api-reference](docs/14-api-reference.md) | API 参考 |
| [15-technical-roadmap](docs/15-technical-roadmap.md) | 历史技术 Roadmap（存档） |
| [16-dual-mode-execution-plan](docs/16-dual-mode-execution-plan.md) | Dual-mode 当前状态 |
| [17-repo-architecture-review](docs/17-repo-architecture-review.md) | 仓库架构评估与演进建议 |
| [18-offload-evidence-optimization](docs/18-offload-evidence-optimization.md) | Offload 与 Evidence 联动优化方案 |
| [20-roadmap](docs/20-roadmap.md) | 当前版本路线图（权威） |
| [21-industry-framework-analysis](docs/21-industry-framework-analysis.md) | 业界框架分析与后续优化方向 |

## 开发

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## License

Apache License 2.0
