# Mem Deep Research

可扩展的 AI Agent 框架，专注于深度研究任务。基于 MCP 工具协议，支持多 LLM 提供商。

## 特性

- **MCP 工具系统** — 本地 (stdio)、远程 HTTP (streamable-http)、SSE 三种传输模式
- **三级上下文管理** — Observation Masking → LLM 摘要压缩 → 二分裁剪，自动防爆
- **执行监控** — 三级升级策略 (WARN → INJECT_HINT → TERMINATE)，循环检测 + 超时控制
- **Hook 系统** — 17 个生命周期钩子，所有行为可自定义
- **多语言支持** — `response_language: auto` 自动检测回答语言
- **子 Agent** — 复杂任务自动分解，子 Agent 复用主循环，上下文隔离
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
    presets: [research]          # 可选: research, time_sensitive, research_planning

  llm:
    provider_class: "ClaudeOpenRouterClient"
    model_name: "anthropic/claude-sonnet-4"
    temperature: 0.3
    max_tokens: 32000
    max_context_length: -1       # -1 = 不限制
    openrouter_api_key: "${oc.env:OPENROUTER_API_KEY}"

  tool_config:
    - tool-calculator
    - tool-searching-serper      # 需要 SERPER_API_KEY

  max_turns: 20
  max_tool_calls_per_turn: 10
  keep_tool_result: -1           # -1 全部保留

  response_language: auto        # auto | Chinese | English | Japanese | ...

  deep_research:
    enabled: true
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
@hooks.register("on_before_llm_call", priority=10)
def guardrail(ctx: HookContext, original_fn):
    from mem_deep_research_core.exceptions import GuardrailError
    if "DELETE" in str(ctx.extra.get("messages", [])):
        raise GuardrailError("Blocked: SQL DELETE detected")
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
| `on_before_llm_call` / `on_after_llm_call` | LLM 调用前后 | raise GuardrailError |
| `on_env_inject` | MCP 环境变量注入 | server_params |
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
  → Pipeline → AgentFactory → Orchestrator
    → PromptBuilder 构建 system prompt
    → MainLoopRunner 执行主循环:
        LLM 调用 → 工具执行 → Context 管理 → 监控检查
        (子 Agent 复用同一 MainLoopRunner，隔离上下文)
    → SummaryHandler 生成最终摘要
  → ResearchResult
```

详细文档见 [`docs/`](docs/) 目录。

## 开发

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v     # 248 tests
```

## License

Apache License 2.0
