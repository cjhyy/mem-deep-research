# Example Project

演示如何使用 Mem Deep Research 框架构建研究型 Agent。

## 快速开始

```bash
# 1. 安装框架（在项目根目录）
pip install -e ..

# 2. 配置 API 密钥
cp .env.example .env
# 编辑 .env，填入你的 API Key

# 3. 运行
python run.py "123 * 456 + 789 等于多少？"
```

## 项目结构

```
example_project/
├── config/
│   ├── agent.yaml              # 默认配置（OpenRouter + Claude）
│   ├── agent_anthropic.yaml    # Anthropic 直连配置
│   ├── agent_minimal.yaml      # 最小配置
│   ├── tool/                   # 项目级自定义工具
│   │   └── tool-custom-api.yaml
│   ├── skills/
│   │   └── definitions/
│   │       └── analysis_guide.md   # 自定义 Skill
│   └── prompts/
│       └── custom_system.md    # 自定义 Prompt 模板
├── hooks.py                    # 钩子（自动加载）
├── run.py                      # 入口脚本
├── run_advanced.py             # 高级用法示例
├── .env.example                # 环境变量模板
├── .env                        # 你的 API 密钥（不提交）
├── logs/                       # 运行日志
└── README.md
```

## 配置说明

### 切换配置

```bash
# 使用默认 OpenRouter 配置
python run.py "你的任务"

# 使用 Anthropic 直连
python run.py "你的任务" --config agent_anthropic

# 使用最小配置
python run.py "你的任务" --config agent_minimal
```

### 配置文件说明

| 文件 | LLM Provider | 工具 | 适用场景 |
|------|-------------|------|---------|
| `agent.yaml` | OpenRouter (Claude) | calculator + search | 通用研究 |
| `agent_anthropic.yaml` | Anthropic 直连 | calculator + search | 需要 cache_control |
| `agent_minimal.yaml` | OpenRouter (Claude) | calculator | 快速测试 |

## 自定义工具

在 `config/tool/` 目录下添加 YAML 文件即可注册自定义工具：

```yaml
# config/tool/tool-my-api.yaml
name: "tool-my-api"
url: "http://localhost:8080/mcp"
transport: "streamable-http"
headers:
  Authorization: "Bearer ${oc.env:MY_API_TOKEN}"
```

然后在 `config/agent.yaml` 的 `tool_config` 中引用：
```yaml
tool_config:
  - tool-calculator
  - tool-my-api
```

## 自定义 Skill

在 `config/skills/definitions/` 下添加 Markdown 文件：

```markdown
---
name: my_skill
type: knowledge
description: "技能描述"
triggers:
  keywords: ["关键词"]
metadata:
  priority: 10
---
# Skill 内容（会注入到 system prompt）
```

## 钩子系统

`hooks.py` 会被 `from_project()` 自动加载。可用钩子：

- `on_agent_start` / `on_agent_end` — Agent 生命周期
- `on_turn_start` / `on_turn_end` — 每轮开始/结束
- `on_tool_start` / `on_tool_end` — 工具调用前/后
- `on_tool_filter` — 工具调用过滤（去重后，执行前）
- `on_system_prompt_build` — 修改 system prompt
- `on_tool_result_format` — 自定义工具结果格式
- `on_before_llm_call` / `on_after_llm_call` — LLM 调用前后（可做 guardrail）
- `on_env_inject` — MCP 环境变量注入
