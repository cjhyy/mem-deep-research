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

开发环境：

```bash
pip install -e ".[dev]"
```

## 环境要求

- Python 3.12+
- 至少一个 LLM Provider 的 API Key

## 三种使用方式

### 方式 1: 从项目目录加载（推荐）

```python
from mem_deep_research import DeepResearch

dr = DeepResearch.from_project("./my_project")
result = await dr.run("研究 AI Agent 框架的最新进展")

print(result.answer)
print(result.status)        # "completed" 或 "failed"
print(result.duration_seconds)
```

### 方式 2: 代码配置

```python
from mem_deep_research import DeepResearch

dr = DeepResearch(
    llm_provider="anthropic",
    model="claude-sonnet-4-20250514",
    api_key="your-api-key",
)
result = await dr.run("你的研究任务")
```

### 方式 3: CLI

```bash
# 创建新项目
mem-deep-research init my_project

# 运行研究（在项目目录中）
python run.py "你的研究任务"
```

## 同步 API

```python
result = dr.run_sync("你的研究任务")
```

## 批量执行

```python
tasks = [
    "研究 LLM 推理能力",
    "研究 MCP 协议最新进展",
    "研究 AI Agent 安全性",
]
results = await dr.run_batch(tasks, parallel=True, max_concurrent=3)
```

## 项目目录结构

```
my_project/
├── config/
│   ├── agent.yaml              # Agent 配置（必需）
│   ├── tool/                   # 工具配置（可选，覆盖框架默认）
│   ├── skills/definitions/     # 自定义 Skill 定义（可选）
│   └── prompts/                # 自定义 Prompt 模板（可选）
├── hooks.py                    # 生命周期钩子（可选，自动加载）
├── .env                        # API 密钥（可选）
└── run.py                      # 入口脚本
```

## 最小 agent.yaml 配置

```yaml
main_agent:
  llm:
    provider_class: "ClaudeOpenRouterClient"
    model_name: "anthropic/claude-sonnet-4"
    temperature: 0.3
    max_tokens: 32000
    openrouter_api_key: "${oc.env:OPENROUTER_API_KEY}"

  tool_config:
    - tool-searching-serper

  max_turns: 20
```

## 环境变量

在 `.env` 文件中配置 API 密钥：

```bash
# LLM Provider
OPENROUTER_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
DEEPSEEK_API_KEY=your_key

# 工具
SERPER_API_KEY=your_key
```

## ResearchResult 结构

```python
@dataclass
class ResearchResult:
    task_id: str                    # 唯一任务 ID
    answer: str                     # 最终研究答案
    boxed_answer: str               # 格式化答案（可选）
    status: str                     # "completed" | "failed"
    duration_seconds: float         # 执行时长
    log_path: Optional[Path]        # 任务日志路径
    error: Optional[str]            # 错误信息

    @property
    def success(self) -> bool:      # 是否成功完成
        return self.status == "completed" and not self.error
```

## 上下文传递

可以传递用户上下文供工具使用：

```python
context = {
    "user_name": "张三",
    "timezone": "Asia/Shanghai",
    "_secure": {                     # 敏感字段，LLM 不可见
        "user_id": "uid-123",
        "api_token": "secret-456",
    }
}
result = await dr.run("研究任务", context=context)
```

## 进度回调

```python
async def on_progress(event):
    print(f"[{event['type']}] {event.get('data', {})}")

result = await dr.run("研究任务", on_progress=on_progress)
```
