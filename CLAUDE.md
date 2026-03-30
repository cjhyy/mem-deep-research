# Mem Deep Research Framework

可扩展的 AI Agent 框架，专注于深度研究任务。基于 MCP 工具协议，支持多 LLM 提供商。

## 项目结构

```
mem_deep_research_core/              # 框架核心代码
├── deep_research.py                 # 主入口 (DeepResearch 类)
├── config_schema.py                 # Pydantic 配置验证
├── exceptions.py                    # 异常定义
├── core/                            # 核心模块
│   ├── orchestrator.py              # Agent 编排器（主循环 _run_main_loop）
│   ├── context_manager.py           # 上下文管理（masking + dedup + source registry）
│   ├── window_strategy.py           # 窗口压缩策略（ObservationMasking/LLMSummarize/BinaryReduction）
│   ├── hooks.py                     # 钩子系统（HookRegistry, HookContext）
│   ├── secure_context.py            # 隐私数据保护（_secure 字段 → 占位符）
│   ├── tool_executor.py             # 工具执行器
│   ├── llm_call_handler.py          # LLM 调用 + 重试
│   ├── sub_agent_runner.py          # 子 Agent 生命周期
│   ├── stream_handler.py            # SSE 流式输出
│   ├── monitoring.py                # 执行监控 + 循环检测
│   ├── user_context.py              # 用户上下文构建
│   ├── task_planner.py              # LLM 任务分解
│   ├── message_interceptor.py       # 消息拦截
│   └── answer_handler.py            # 最终答案提取
├── llm/                             # LLM 客户端
│   ├── provider_client_base.py      # Provider 基类
│   └── providers/                   # 8 个 Provider 实现
├── prompts/                         # Prompt 系统
│   ├── agent_prompt.py              # AgentPrompt 统一类
│   ├── template_loader.py           # 模板加载器
│   └── templates/                   # Markdown 模板（base/presets/extraction/...）
├── tool/                            # 工具模块
│   ├── manager.py                   # ToolManager（MCP 工具管理）
│   └── mcp_servers/                 # 内置 MCP 服务器
├── skills/                          # Skill 系统
│   ├── matcher.py                   # 规则匹配 + 注入
│   ├── llm_selector.py              # LLM Skill 选择
│   └── inline_selector.py           # Inline Skill 选择（零额外开销）
└── utils/
    ├── external_loader.py           # 配置加载器（全局 config_loader/external_loader）
    ├── tool_utils.py                # 工具辅助
    └── stream_parsing_utils.py      # 流式解析（StructuredTagExtractor, TextInterceptor）
config/                              # 框架默认配置
├── agent_example.yaml               # Agent 配置示例
├── tool/                            # 内置工具配置 YAML
└── skills/definitions/              # Skill 定义
tests/                               # 单元测试
```

## 核心概念

### 执行流程

```
DeepResearch.run(query)
  → Pipeline.run()
    → AgentFactory 创建 Orchestrator + LLM Client + ToolManager
      → Orchestrator._run_main_loop():
          while turn < max_turns:
            1. Monitor.pre_turn_check()
            2. LLM 调用
            3. Monitor.post_turn_check() (循环检测)
            4. Inline Skill: 解析 <next_skills>
            5. 解析工具调用 + 跨轮次去重
            6. 执行工具（SecureContext 自动解占位符）
            7. Context 管理（ObservationMasking → LLMSummarize → BinaryReduction）
            8. Hook: on_turn_end
            9. 反思检查点
  → 生成最终摘要 → ResearchResult
```

### 使用方式

```python
from mem_deep_research import DeepResearch

# 方式 1: 从项目目录加载
dr = DeepResearch.from_project("./my_project")
result = await dr.run("研究任务")

# 方式 2: 代码配置
dr = DeepResearch(
    llm_provider="anthropic",
    model="claude-sonnet-4-20250514",
    api_key="your-key",
)
result = await dr.run("任务")

# 同步
result = dr.run_sync("任务")
```

### 项目目录结构

用户项目通过 `DeepResearch.from_project()` 加载：

```
my_project/
├── config/
│   ├── agent.yaml              # Agent 配置（LLM、工具、参数）
│   ├── tool/                   # 项目级工具配置（覆盖框架默认）
│   ├── skills/definitions/     # 项目级 Skill 定义
│   └── prompts/                # 自定义 Prompt 模板
├── hooks.py                    # 项目钩子（自动加载）
├── .env                        # API 密钥
└── run.py                      # 入口脚本
```

## 关键系统详解

### 1. 钩子系统 (`core/hooks.py`)

全局注册表 `hooks = HookRegistry()`，支持的钩子：

| 钩子 | 时机 | 可修改 |
|------|------|--------|
| `on_agent_start` | Agent 开始 | — |
| `on_agent_end` | Agent 结束 | — |
| `on_turn_start` | 每轮开始 | — |
| `on_turn_end` | 每轮结束 | — |
| `on_tool_start` | 工具调用前 | arguments |
| `on_tool_end` | 工具调用后 | tool_result |
| `on_tool_result_format` | 结果格式化 | 返回值 |
| `on_thinking_generate` | thinking 描述 | 返回值 |
| `on_env_inject` | MCP 环境变量 | server_params |

用法：
```python
from mem_deep_research_core.core.hooks import hooks, HookContext

@hooks.register("on_tool_end", priority=10)
def my_hook(ctx: HookContext, original_fn):
    result = original_fn(ctx)   # 调用原逻辑
    return modified_result      # 或完全替换
```

### 2. SecureContext (`core/secure_context.py`)

context dict 中的 `_secure` 字段自动在 system prompt 中显示为 `[SECURE:xxx]` 占位符，工具调用前自动替换回真实值。

```python
context = {
    "user_name": "张三",          # 正常显示
    "_secure": {
        "user_id": "real-123",    # system prompt 中显示 [SECURE:user_id]
        "org_id": "org-456",      # 工具调用时自动替换回 "org-456"
    }
}
```

关键函数：
- `get_display_value(ctx, field)` — system prompt 用，_secure 字段返回占位符
- `get_real_value(ctx, field)` — 工具执行用，始终返回真实值
- `resolve_placeholders_in_args(args, ctx)` — 递归替换工具参数中的占位符

### 3. 上下文管理 (`core/context_manager.py` + `core/window_strategy.py`)

三级窗口压缩策略，通过 `WindowStrategyPipeline` 组合：

| 级别 | 策略 | 触发条件 | LLM 成本 |
|------|------|---------|---------|
| L1 | ObservationMasking | token 占比 > 60% | 零 |
| L2 | LLMSummarize | token 占比 > 80% | 一次 LLM 调用 |
| L3 | BinaryReduction | token 占比 > 95% | 零 |

支持自定义策略：继承 `WindowStrategy` ABC，实现 `should_trigger()` + `apply()`。

### 4. Skill 系统 (`skills/`)

三种选择方式（配置 `skill_selection.method`）：

| method | 说明 | 开销 |
|--------|------|------|
| `rules` | 基于 keywords/tools/context 打分 | 零 |
| `llm` | 额外 LLM 调用选择 | 一次轻量 LLM |
| `inline` | LLM 在回复中声明 `<next_skills>` | 零 |

Skill 定义格式（`config/skills/definitions/*.md`）：
```markdown
---
name: search_strategy
type: knowledge
description: "搜索策略指南"
triggers:
  keywords: ["搜索", "查找"]
  tools_mentioned: ["semantic_search"]
metadata:
  priority: 10
---
# Skill 内容（注入 system prompt）
```

### 5. 工具系统 (`tool/manager.py`)

基于 MCP 协议，支持三种传输：
- **stdio**: 本地进程（`npx`, `python` 脚本）
- **streamable-http**: HTTP 远程服务
- **sse**: Server-Sent Events

工具配置 YAML 示例：
```yaml
name: tool-searching
tool_command: npx
args: ["-y", "@anthropic/tool-searching"]
env:
  SERPER_API_KEY: "${oc.env:SERPER_API_KEY,}"
```

### 6. Prompt 系统 (`prompts/`)

`AgentPrompt` 统一类，通过配置组合模板：

```yaml
prompt:
  agent_type: main           # main | worker
  tool_format: xml           # xml | native
  presets: [research, time_sensitive]  # 可选预设
  custom_system_template: my_prompt    # 自定义模板
```

模板位于 `prompts/templates/`，用 `{{variable}}` 占位符，`PromptTemplateLoader` 加载渲染。

### 7. LLM 客户端 (`llm/`)

所有 provider 继承 `LLMProviderClientBase`，关键方法：
- `call_llm(system_prompt, messages, tool_definitions)` — 调用 LLM
- `update_message_history(history, tool_results, exceeded)` — 更新消息历史
- `_estimate_tokens(text)` — token 估算

## 配置说明 (`config_schema.py`)

```yaml
main_agent:
  prompt:
    agent_type: main
    tool_format: xml
    presets: []
  llm:
    provider_class: "ClaudeOpenRouterClient"
    model_name: "anthropic/claude-sonnet-4"
    temperature: 0.3
    max_tokens: 32000
    max_context_length: 128000    # -1 = 不限制
    keep_tool_result: 5           # -1 全部保留, N 保留最近 N 个
  tool_config: [tool-reading, tool-searching]
  max_turns: 20
  max_tool_calls_per_turn: 10
  chinese_context: false
  skill_selection:
    enabled: true
    method: inline               # rules | llm | inline
    max_skills: 3
  context_manager:
    enable_dedup: true
    enable_compact: true
    compact_at_ratio: 0.6
    summarize_at_ratio: 0.8
    compact_keep_recent: 3
  deep_research:
    enabled: false
    reflection_interval: 5
    auto_planning: false
```

## 开发规范

- Python 3.12+，全异步设计
- 测试运行：`python -m pytest tests/ -v`
- 配置验证用 Pydantic，运行时配置用 OmegaConf (Hydra)
- 工具遵循 MCP (Model Context Protocol) 规范
- 框架无状态，所有定制通过项目级 config + hooks.py 注入
- 新增核心功能需在 `config_schema.py` 添加配置项并设合理默认值
