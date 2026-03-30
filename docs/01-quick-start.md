# 快速开始

## 安装

```bash
pip install mem-deep-research
```

从源码安装：

```bash
git clone https://github.com/cjhyy/mem-deep-research.git
cd mem-deep-research
pip install -e .
```

## 环境要求

- Python 3.12+
- 至少一个 LLM Provider 的 API Key

## 最快上手方式

```bash
# 复制示例项目
cp -r example_project my_project
cd my_project

# 配置 API Key
echo "OPENROUTER_API_KEY=your-key" > .env

# 运行
python run.py "你好"                                    # 快速回答
python run.py "123 * 456 + 789"                         # 计算
python run.py "什么是量子计算"                            # 知识问答
python run.py "研究 AI Agent 框架的最新进展" --deep       # 深度研究
```

## 三种使用方式

### 方式 1: 从项目目录加载（推荐）

```python
from mem_deep_research import DeepResearch

dr = DeepResearch.from_project("./my_project")
result = await dr.run("你的研究任务")

print(result.answer)
print(result.status)          # "completed" 或 "failed"
print(result.duration_seconds)
```

### 方式 2: 代码配置

```python
from mem_deep_research import DeepResearch

dr = DeepResearch(
    llm_provider="openrouter",
    model="anthropic/claude-sonnet-4",
    api_key="your-key",
    tools=["tool-calculator", "tool-searching-serper"],
)
result = await dr.run("你的研究任务")

# 同步调用
result = dr.run_sync("你的任务")
```

### 方式 3: CLI

```bash
python run.py "你的任务"
python run.py "任务" --deep      # 强制深度研究模式
python run.py "任务" --flash     # 强制快速回答模式
python run.py "任务" --config agent_anthropic  # 切换配置
```

## 最小配置

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

## 完整配置

```yaml
main_agent:
  llm:
    provider_class: "ClaudeOpenRouterClient"
    model_name: "anthropic/claude-sonnet-4"
    temperature: 0.3
    max_tokens: 32000
    openrouter_api_key: "${oc.env:OPENROUTER_API_KEY}"

  tool_config:
    - tool-calculator
    - tool-searching-serper

  execution_mode: auto              # auto | flash | standard | deep
  max_turns: 30
  max_concurrent_subagents: 3
  response_language: auto           # auto | Chinese | English | ...

  todo_tracker:
    enabled: true                   # 任务追踪

  deep_research:
    enabled: true
    reflection_interval: 5

  context_manager:
    enable_dedup: true
    compact_at_ratio: 0.6
    summarize_at_ratio: 0.8
    result_offload_threshold: 5000  # 大结果卸载到文件

  monitoring:
    enable_loop_detection: true
    max_total_time: 1800.0          # 30 分钟

  skill_selection:
    enabled: true
    method: inline
    progressive: true               # 按需加载 skill
```

## 框架自动提供的能力

无需额外配置，框架自动处理：

- **执行模式自动选择** — 简单问答 → flash，需要工具 → standard，deep_research → deep
- **语言自动检测** — 中文问题中文答，英文问题英文答
- **内置 spawn_agent 工具** — LLM 可自主 spawn 子 agent 处理子任务
- **内置 update_todo 工具** — LLM 可管理任务列表追踪进度
- **三级 context 压缩** — 自动管理 token，不会爆
- **循环检测** — 检测重复响应，自动升级策略或终止
- **SessionMemory** — 自动追踪关键发现、已用策略
- **LongTermMemory** — 跨 session 积累知识

## 项目目录结构

```
my_project/
├── config/
│   ├── agent.yaml              # Agent 配置（必需）
│   ├── tool/                   # 自定义工具配置（可选）
│   ├── skills/definitions/     # 自定义 Skill（可选）
│   └── prompts/                # 自定义 Prompt 模板（可选）
├── hooks.py                    # 生命周期钩子（可选，自动加载）
├── .env                        # API 密钥
└── run.py                      # 入口脚本
```

## ResearchResult

```python
@dataclass
class ResearchResult:
    task_id: str                    # 唯一任务 ID
    answer: str                     # 最终答案
    boxed_answer: str               # 格式化答案
    status: str                     # "completed" | "failed"
    duration_seconds: float         # 执行时长
    log_path: Optional[Path]        # 日志路径
    error: Optional[str]            # 错误信息
```
