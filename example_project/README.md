# Example Project

一个通用 Agent，能处理从简单问候到小时级深度研究的任何任务。

## 快速开始

```bash
# 1. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 OPENROUTER_API_KEY

# 2. 安装框架
cd .. && pip install -e . && cd example_project

# 3. 运行
python run.py "你好"
```

## 使用示例

```bash
# 快速回答（quick 模式 — 不调工具，直接回答）
python run.py "你好"
python run.py "什么是机器学习？" --quick

# 计算（standard 模式 — 调用工具）
python run.py "123 * 456 + 789"

# 研究（deep 模式 — 多轮搜索 + 反思 + 任务追踪）
python run.py "研究量子计算的最新进展" --deep

# 框架自动选择模式（auto — 默认）
python run.py "分析 2024 年 AI Agent 领域的关键技术突破"
```

## 框架自动处理的事情

| 能力 | 说明 |
|------|------|
| **执行模式自动选择** | 简单问答 → quick，需要工具 → standard，task_engine → deep |
| **语言自动检测** | 中文问题中文答，英文问题英文答 |
| **任务追踪** | 复杂任务自动维护 todo list，context 压缩不丢 |
| **子 Agent** | LLM 自主决定是否 spawn 子 agent 处理子任务 |
| **上下文管理** | 三级压缩防爆，大结果自动卸载到文件 |
| **循环检测** | 检测重复响应，自动升级策略或终止 |
| **记忆系统** | SessionMemory 追踪发现，LongTermMemory 跨 session 积累 |
| **Skill 渐进加载** | 按需注入 skill，省 token |

## 项目结构

```
example_project/
├── config/
│   ├── agent.yaml              # 主配置（auto 模式，覆盖所有场景）
│   ├── agent_anthropic.yaml    # Anthropic 直连
│   ├── agent_minimal.yaml      # 最小配置（快速测试）
│   ├── tool/                   # 自定义工具
│   ├── skills/definitions/     # 自定义 Skill
│   └── prompts/                # 自定义 Prompt 模板
├── hooks.py                    # 生命周期钩子（自动加载）
├── run.py                      # 入口脚本
├── .env                        # API 密钥
└── logs/                       # 运行日志 + 卸载结果 + 长期记忆
    ├── offloaded_results/      # 大工具结果文件
    └── memory/                 # 长期记忆存储
```

## 添加搜索能力

取消 `agent.yaml` 中 `tool-searching-serper` 的注释，并在 `.env` 中配置 `SERPER_API_KEY`：

```yaml
tool_config:
  - tool-calculator
  - tool-searching-serper
```

这样 Agent 就能搜索互联网、阅读网页，处理真实的研究任务。

## 自定义

### 添加工具

在 `config/tool/` 下加 YAML 文件，然后在 `agent.yaml` 的 `tool_config` 中引用。

### 添加 Skill

在 `config/skills/definitions/` 下加 Markdown 文件。LLM 会按需加载。

### 修改行为

编辑 `hooks.py`。所有 Agent 行为都可通过 17 个 hook 自定义。
