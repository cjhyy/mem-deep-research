# 执行模式与语言控制

## 执行模式

**文件**: `config_schema.py` · `core/main_loop.py` · `core/orchestrator.py`

```yaml
main_agent:
  execution_mode: auto     # auto | quick | standard | deep
```

### 四种模式

| 模式 | 循环 | 工具 | 反思 | 子 Agent | 适用场景 |
|------|------|------|------|---------|---------|
| `quick` | 少轮 | 是 | 否 | 否 | 简单问答、计算、翻译 |
| `standard` | 多轮 | 是 | 否 | 否 | 需要工具辅助的一般任务 |
| `deep` | 多轮 | 是 | 是 | 是 | 复杂研究、多步骤分析 |
| `auto` | — | — | — | — | 自动选择（见下方） |

### auto 模式选择逻辑

```
if task_engine.enabled == true:
    → deep 模式
else:
    → standard 模式
```

`auto` 是默认值。如果你在配置中启用了 `task_engine`，`auto` 会选择 `deep`；否则选择 `standard`。

### quick 模式

限制最大轮次的快速执行模式，可调用工具但不做反思。适用于：

- 简单问答
- 计算任务
- 文本翻译/改写

### standard 模式

多轮执行循环（最多 `max_turns` 轮），每轮可调用工具。包含：

- LLM 调用 + 工具执行
- 上下文管理（三级压缩）
- 监控检查（循环检测 + 超时）
- Skill 注入

### deep 模式

在 `standard` 基础上增加：

- **反思检查点**: 每 `reflection_interval` 轮注入反思 prompt，让 LLM 评估进度
- **任务分解**: 可选 `auto_planning`，在开始前用 LLM 分解研究计划
- **子 Agent**: LLM 可通过 `spawn_agent` 工具派发子任务
- **TodoTracker**: 自动启用任务追踪
- **SessionMemory**: 自动追踪关键发现

```yaml
main_agent:
  execution_mode: deep     # 或 auto + task_engine.enabled: true
  task_engine:
    enabled: true
    reflection_interval: 5   # 每 5 轮反思一次
    auto_planning: false     # 是否在开始前自动分解任务
```

---

## 语言控制

**文件**: `config_schema.py` · `core/main_loop.py`

```yaml
main_agent:
  response_language: auto    # auto | Chinese | English | Japanese | ...
```

### 工作方式

| 值 | 行为 |
|---|------|
| `auto`（默认） | 从 query 自动检测语言，用同语言回答 |
| `Chinese` | 思维链 + 回答全部使用中文 |
| `English` | 全英文 |
| 其他语言名 | 支持任何语言名称（如 `Japanese`、`Korean`） |

### 自动检测

`auto` 模式在首轮调用时检测 query 语言。检测逻辑使用正则匹配 CJK 字符比例：

- CJK 字符 > 10%: 中文
- 否则: 英文

### 向后兼容

旧配置 `chinese_context: true` 等同于 `response_language: Chinese`。

### 自定义检测

通过 `on_agent_start` Hook 覆盖自动检测结果：

```python
@hooks.register("on_agent_start", priority=50)
def force_language(ctx: HookContext, original_fn):
    result = original_fn(ctx)
    ctx.extra["response_language"] = "Chinese"  # 强制中文
    return result
```

---

## StreamHandler — 流式输出

**文件**: `core/stream_handler.py`

通过 asyncio.Queue 向外部推送 SSE 事件，支持实时展示 Agent 执行进度。

### 事件类型

| 事件 | 触发时机 | 数据 |
|------|---------|------|
| `start_of_workflow` | 任务开始 | `task_id` |
| `start_of_agent` | Agent/子Agent 开始 | `agent_name`, `agent_id` |
| `end_of_agent` | Agent/子Agent 结束 | `agent_id` |
| `message` | LLM 文本输出 | `content`, `agent_id` |
| `tool_call` | 工具调用 | `tool_name`, `arguments` |
| `tool_result` | 工具结果 | `tool_name`, `result` |
| `status` | 状态变更 | `status`, `message` |
| `end_of_workflow` | 任务结束 | `result` |

### 使用方式

```python
import asyncio
from mem_deep_research import DeepResearch

queue = asyncio.Queue()
dr = DeepResearch.from_project("./my_project")

# 传入 queue 接收流式事件
result = await dr.run("研究任务", stream_queue=queue)

# 在另一个 coroutine 中消费事件
async def consume():
    while True:
        event = await queue.get()
        if event.get("type") == "end_of_workflow":
            break
        print(event)
```

### 与 FastAPI 集成

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/research/stream")
async def stream_research(query: str):
    queue = asyncio.Queue()
    dr = DeepResearch.from_project("./my_project")

    async def generate():
        task = asyncio.create_task(dr.run(query, stream_queue=queue))
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") == "end_of_workflow":
                break
        await task

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 结果卸载

**文件**: `core/context_manager.py`

大块工具结果自动写到文件，消息历史只保留摘要引用，节省上下文空间。

```yaml
main_agent:
  context_manager:
    result_offload_threshold: 5000   # 超过 5000 字符卸载到文件，0=禁用
    result_offload_dir: ""           # 空=使用 output_dir
```

### 工作流程

1. 工具返回结果 > `result_offload_threshold` 字符
2. 完整结果写入 `{output_dir}/offloaded/{tool_name}_{turn}.txt`
3. 消息历史中替换为摘要引用：`[Result offloaded to file: path. First 200 chars: ...]`
4. LLM 可通过文件路径读取完整内容（如果配备文件读取工具）
