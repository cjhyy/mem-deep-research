# Mem Deep Research Framework

可扩展的 AI Agent 框架，专注于深度研究任务。基于 MCP 工具协议，支持多 LLM 提供商。

## 项目结构

```
mem_deep_research_core/              # 框架核心代码
├── deep_research.py                 # 主入口 (DeepResearch 类)
├── config_schema.py                 # Pydantic 配置验证
├── exceptions.py                    # 异常定义
├── core/                            # 核心模块
│   ├── orchestrator.py              # Agent 编排器（组件初始化 + 任务协调）
│   ├── main_loop.py                 # 主执行循环 (MainLoopRunner + MainLoopContext)
│   ├── prompt_builder.py            # Prompt 构建（system prompt + skill 注入 + hint）
│   ├── constants.py                 # 框架常量 + 工具函数（单一来源）
│   ├── context_manager.py           # 上下文管理（masking + dedup + source registry）
│   ├── window_strategy.py           # 窗口压缩策略（ObservationMasking/LLMSummarize/BinaryReduction）
│   ├── hooks.py                     # 钩子系统（HookRegistry, HookContext）
│   ├── secure_context.py            # 隐私数据保护（_secure 字段 → 占位符）
│   ├── tool_executor.py             # 工具执行器
│   ├── llm_call_handler.py          # LLM 调用 + 重试 + 摘要生成
│   ├── sub_agent_runner.py          # 子 Agent（复用 MainLoopRunner，隔离上下文）
│   ├── stream_handler.py            # SSE 流式输出
│   ├── monitoring.py                # 执行监控 + 循环检测
│   ├── task_planner.py              # LLM 任务分解（模板化）
│   ├── message_interceptor.py       # 消息拦截
│   ├── user_context.py              # 用户上下文构建（可选工具类，通过 hook 注入）
│   ├── answer_handler.py            # 最终答案后处理
│   └── interceptor_config.py        # 拦截器配置 + 预设
├── llm/                             # LLM 客户端
│   ├── provider_client_base.py      # Provider 基类
│   └── providers/                   # 多 Provider 实现
├── prompts/                         # Prompt 系统
│   ├── agent_prompt.py              # AgentPrompt 统一类
│   ├── template_loader.py           # 模板加载器
│   └── templates/                   # Markdown 模板（base/presets/planning/reflection/...）
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
tests/                               # 测试
```

## 核心概念

### 执行流程

```
DeepResearch.run(query)
  → Pipeline.run()
    → AgentFactory 创建 Orchestrator + LLM Client + ToolManager
      → Orchestrator.run_main_agent():
          1. PromptBuilder 构建 system prompt + skill 注入
          2. 创建 MainLoopRunner
          → MainLoopRunner.run():
              while turn < max_turns:
                1. on_agent_start hook（首轮）+ 语言自动检测
                2. Monitor.pre_turn_check()
                3. LLM 调用（on_before_llm_call / on_after_llm_call guardrail）
                4. Monitor.post_turn_check()（循环检测 + 升级策略）
                5. Inline Skill: 解析 <next_skills>
                6. 解析工具调用 + 跨轮次去重
                7. 执行工具（SecureContext 自动解占位符）
                   - 普通工具 → ToolExecutor
                   - 子 Agent → SubAgentRunner → MainLoopRunner（隔离上下文）
                8. Context 管理（ObservationMasking → LLMSummarize → BinaryReduction）
                9. Hook: on_turn_end
               10. 反思检查点（deep_research 模式）
          → 生成最终摘要（SummaryHandler）
      → post_process_final_answer → ResearchResult
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
| `on_tool_filter` | 工具去重后、执行前 | tool_calls_batch |
| `on_system_prompt_build` | system prompt 生成后 | 返回值 |
| `on_summarize_prompt_build` | summarize prompt 生成后 | 返回值 |
| `on_tool_result_format` | 结果格式化 | 返回值 |
| `on_thinking_generate` | thinking 描述 | 返回值 |
| `on_env_inject` | MCP 环境变量 | server_params |
| `on_message_intercept` | 消息拦截处理 | — |
| `on_before_llm_call` | LLM 调用前验证 | raise GuardrailError 阻止 |
| `on_after_llm_call` | LLM 调用后验证 | raise GuardrailError 拒绝 |
| `on_context_compact` | context 压缩时 | — |
| `on_reflection_build` | 反思 prompt 生成 | 返回值 |

用法：
```python
from mem_deep_research_core.core.hooks import hooks, HookContext

@hooks.register("on_tool_end", priority=10)
def my_hook(ctx: HookContext, original_fn):
    result = original_fn(ctx)   # 调用原逻辑
    return modified_result      # 或完全替换
```

业务逻辑（如用户身份注入）通过 hook 实现，不内置在框架中：
```python
from mem_deep_research_core.core.user_context import UserContextBuilder

@hooks.register("on_system_prompt_build", priority=50)
def inject_user_context(ctx, original_fn):
    prompt = original_fn(ctx)
    builder = UserContextBuilder(ctx.context, chinese_context=False)
    identity = builder.build_user_identity_context()
    return prompt + f"\n\n{identity}" if identity else prompt
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

### 3. 上下文管理 (`core/context_manager.py` + `core/window_strategy.py`)

三级窗口压缩策略，通过 `WindowStrategyPipeline` 组合：

| 级别 | 策略 | 触发条件 | LLM 成本 |
|------|------|---------|---------|
| L1 | ObservationMasking | token 占比 > 60% | 零 |
| L2 | LLMSummarize | token 占比 > 80% | 一次 LLM 调用 |
| L3 | BinaryReduction | token 占比 > 95% | 零 |

支持自定义策略：继承 `WindowStrategy` ABC，实现 `should_trigger()` + `apply()`。

### 4. 子 Agent (`core/sub_agent_runner.py`)

子 Agent 复用 `MainLoopRunner` 执行，与主 Agent 共享同一套能力（context 管理、监控、hook），但上下文完全隔离。

配置方式（可选）：
```yaml
sub_agents:
  agent-researcher:           # 名称必须以 "agent-" 前缀开头
    prompt:
      agent_type: worker
    llm:
      provider_class: "ClaudeOpenRouterClient"
      model_name: "anthropic/claude-sonnet-4"
    tool_config: [tool-searching-serper]
    max_turns: 10
```

不配置 `sub_agents` 时，主 Agent 独立完成所有任务。

### 5. 语言控制

通过 `response_language` 配置控制回答语言：

```yaml
main_agent:
  response_language: auto      # auto | Chinese | English | Japanese | ...
```

| 值 | 行为 |
|---|------|
| `auto`（默认） | 从 query 自动检测语言 |
| `Chinese` | 思维链 + 回答都用中文 |
| `English` | 全英文 |

`chinese_context: true` 向后兼容，等同 `response_language: Chinese`。
自定义检测逻辑可通过 `on_agent_start` hook 覆盖。

### 6. Skill 系统 (`skills/`)

三种选择方式（配置 `skill_selection.method`）：

| method | 说明 | 开销 |
|--------|------|------|
| `rules` | 基于 keywords/tools/context 打分 | 零 |
| `llm` | 额外 LLM 调用选择 | 一次轻量 LLM |
| `inline` | LLM 在回复中声明 `<next_skills>` | 零 |

### 7. 工具系统 (`tool/manager.py`)

基于 MCP 协议，支持三种传输：
- **stdio**: 本地进程（`npx`, `python` 脚本）
- **streamable-http**: HTTP 远程服务
- **sse**: Server-Sent Events

### 8. Prompt 系统 (`prompts/`)

`AgentPrompt` 统一类，通过配置组合模板：

```yaml
prompt:
  agent_type: main           # main | worker
  tool_format: xml           # xml | native
  presets: [research, time_sensitive]  # 可选预设
  custom_system_template: my_prompt    # 自定义模板
```

模板位于 `prompts/templates/`，用 `{{variable}}` 占位符，`PromptTemplateLoader` 加载渲染。

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
    max_context_length: -1        # -1 = 不限制
  tool_config: [tool-calculator, tool-searching-serper]
  max_turns: 20
  max_tool_calls_per_turn: 10
  keep_tool_result: -1            # -1 全部保留, N 保留最近 N 个
  response_language: auto          # auto | Chinese | English | ...
  add_message_id: true
  skill_selection:
    enabled: true
    method: inline                 # rules | llm | inline
    max_skills: 3
  context_manager:
    enable_dedup: true
    enable_compact: true
    compact_at_ratio: 0.6
    summarize_at_ratio: 0.8
    compact_keep_recent: 3
    max_dedup_cache_size: 200
  monitoring:
    enable_loop_detection: true
    loop_escalation_terminate_threshold: 3
    max_total_time: 600.0
    temperature_boost: 0.3
    temperature_boost_cap: 1.0
  deep_research:
    enabled: false
    reflection_interval: 5
    auto_planning: false
  interceptor:
    preset: default                # default | verbose | minimal | debug

sub_agents: null                   # 可选，子 Agent 配置
output_dir: logs/
```

## 开发规范

- Python 3.12+，全异步设计
- 测试运行：`python -m pytest tests/ -v`
- 配置验证用 Pydantic，运行时配置用 OmegaConf (Hydra)
- 工具遵循 MCP (Model Context Protocol) 规范
- 框架无状态，所有定制通过项目级 config + hooks.py 注入
- 业务逻辑不内置在框架中，通过 hook 注入（如用户身份、benchmark 特定逻辑）
- 新增核心功能需在 `config_schema.py` 添加配置项并设合理默认值
- 所有硬编码值统一管理在 `core/constants.py`
