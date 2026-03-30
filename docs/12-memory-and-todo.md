# 记忆与任务追踪

框架提供两层记忆机制和独立的任务追踪系统，保证关键信息在上下文压缩时不丢失。

## SessionMemory — 短期记忆

**文件**: `core/memory.py` · **作用域**: 单次 `run()` 调用

每次运行自动创建，追踪四类结构化信息：

| 字段 | 说明 | 上限 |
|------|------|------|
| `key_findings` | LLM 回复中提取的关键发现 | 20 条 |
| `attempted_strategies` | 已尝试的工具调用策略 | 15 条 |
| `sources` | 从工具结果中提取的引用来源 | 30 条 |
| `sub_agent_results` | 子 Agent 返回的结果摘要 | 无限制（按 agent_name 去重） |

### 工作机制

1. **自动提取**: `extract_from_tool_result()` 从工具结果的 `url`/`title`/`snippet` 字段中自动提取来源
2. **去重**: `add_finding()` 和 `add_strategy()` 自动跳过已存在的条目
3. **注入**: `to_context_string()` 生成 `[SESSION MEMORY]` 标记块，注入到消息历史
4. **保护**: 上下文压缩时 `[SESSION MEMORY]` 块不会被裁掉

### API

```python
from mem_deep_research_core.core.memory import SessionMemory

memory = SessionMemory()

# 手动添加
memory.add_finding("量子退相干是量子计算的主要障碍")
memory.add_strategy("使用 serper 搜索 'quantum decoherence solutions'")
memory.add_source(url="https://arxiv.org/abs/...", title="Quantum Error Correction")
memory.add_sub_agent_result("agent-researcher", "找到 3 篇相关论文...")

# 从工具结果自动提取
memory.extract_from_tool_result("serper_search", {"url": "...", "title": "...", "snippet": "..."})

# 生成注入文本
context_text = memory.to_context_string()  # "[SESSION MEMORY]\n\n## Key Findings So Far\n..."

# 状态查询
memory.is_empty()  # True/False
```

### 注入格式

```markdown
[SESSION MEMORY]

## Key Findings So Far
- 量子退相干是量子计算的主要障碍
- 表面码是目前最有前途的纠错方案

## Attempted Strategies
- 使用 serper 搜索 'quantum decoherence solutions'

## Sources Collected
- [Quantum Error Correction](https://arxiv.org/abs/...) — Surface codes achieve...

## Sub-Agent Results
### agent-researcher
找到 3 篇相关论文...
```

---

## LongTermMemory — 长期记忆

**文件**: `core/memory.py` · **作用域**: 跨 session 持久化

基于 JSON 文件存储，支持关键词检索。通过 Hook 集成到 Agent 生命周期。

### 存储格式

文件位于 `{storage_path}/memory.json`：

```json
[
  {
    "key": "user_prefers_chinese",
    "value": "用户偏好中文回答，习惯简洁风格",
    "metadata": {"source": "conversation", "topic": "preference"},
    "timestamp": 1711234567.89,
    "access_count": 3
  }
]
```

### API

```python
from mem_deep_research_core.core.memory import LongTermMemory

memory = LongTermMemory(storage_path="memory/", max_entries=1000)

# 存储（key 已存在则更新）
memory.store("user_prefers_chinese", "用户偏好中文回答", {"source": "conversation"})

# 召回（关键词匹配，返回 top_k 条）
entries = memory.recall("用户偏好", top_k=5)
for entry in entries:
    print(f"{entry.key}: {entry.value}")

# 按 metadata 过滤
entries = memory.recall("", metadata_filter={"topic": "preference"})

# 删除
memory.forget("user_prefers_chinese")  # -> True/False

# 列出全部
all_entries = memory.list_all()

# 去重（合并 key 相同的条目，保留最新）
memory.deduplicate()

# 清空
memory.clear()
```

### 召回算法

`recall()` 使用简单关键词匹配评分：

1. 将 query 拆分为词
2. 在每个 entry 的 `key + value` 中检查每个词是否出现
3. 按匹配词数降序排列，同分时按 timestamp 降序
4. 返回 top_k 条结果

可通过 Hook 替换为向量检索：

```python
from mem_deep_research_core.core.hooks import hooks, HookContext

@hooks.register("on_agent_start", priority=5)
def inject_memory(ctx: HookContext, original_fn):
    entries = memory.recall(ctx.extra.get("query", ""), top_k=5)
    if entries:
        ctx.extra["memory_context"] = "\n".join(e.value for e in entries)
    return original_fn(ctx)
```

### 线程安全

`store()`、`forget()`、`clear()`、`deduplicate()` 使用 `threading.Lock` 保护。`recall()` 和 `list_all()` 为只读操作，不持锁。

---

## TodoTracker — 任务追踪

**文件**: `core/todo_tracker.py` · **作用域**: 单次 `run()` 调用

独立于 `message_history` 的任务状态管理。上下文截断时状态不丢失，每轮自动重新注入。

### 启用方式

```yaml
main_agent:
  todo_tracker:
    enabled: true      # 也会随 deep_research.enabled 自动启用
```

### 工作流程

1. 框架自动注册内置工具 `update_todo`
2. LLM 通过调用 `update_todo` 创建/更新任务
3. 每轮开始时，TodoTracker 将当前状态注入为 `[TASK PROGRESS]` 消息
4. 上下文压缩时 `[TASK PROGRESS]` 块不会被裁掉

### update_todo 工具

LLM 可用的四个 action：

| action | 必须参数 | 可选参数 | 说明 |
|--------|---------|---------|------|
| `add` | `task` | `priority` | 添加新任务 |
| `start` | `task_id` | — | 标记任务为进行中 |
| `complete` | `task_id` | `result` | 标记任务为已完成 |
| `list` | — | — | 列出所有任务 |

```json
{"action": "add", "task": "搜索量子计算论文", "priority": "high"}
{"action": "start", "task_id": 1}
{"action": "complete", "task_id": 1, "result": "找到 5 篇论文"}
```

### 任务状态机

```
pending  ──→  in_progress  ──→  completed
   │                               ▲
   └───────────────────────────────┘
```

- `pending` (⬜) — 待处理
- `in_progress` (🔄) — 进行中
- `completed` (✅) — 已完成

### 注入格式

```
[TASK PROGRESS]

Progress: 50% (1/2 completed)

✅ [1] [!!!] 搜索量子计算论文 → 找到 5 篇论文
🔄 [2] [!!] 总结论文要点

Currently working on: #2 总结论文要点

Use the update_todo tool to update task status when you start or complete a task.
```

### 代码 API

```python
from mem_deep_research_core.core.todo_tracker import TodoTracker

tracker = TodoTracker(enabled=True)

# 通过工具接口操作
tracker.update_from_tool_call({"action": "add", "task": "搜索论文", "priority": "high"})
tracker.update_from_tool_call({"action": "start", "task_id": 1})
tracker.update_from_tool_call({"action": "complete", "task_id": 1, "result": "已完成"})

# 状态查询
tracker.has_pending_work   # bool: 是否有未完成任务
tracker.progress           # float: 0.0~1.0
tracker.is_empty           # bool

# 注入到消息历史
msg = tracker.build_injection_message(turn=5)
if msg:
    message_history.append(msg)

# 序列化/反序列化
data = tracker.to_dict()
tracker = TodoTracker.from_dict(data, enabled=True)

# 重置
tracker.reset()
```

---

## 配置参考

```yaml
main_agent:
  # TodoTracker
  todo_tracker:
    enabled: false                 # 也会随 deep_research.enabled 自动启用

  # Deep Research (自动启用 todo_tracker + session memory)
  deep_research:
    enabled: false
    reflection_interval: 5
    auto_planning: false
```

## 与上下文管理的交互

SessionMemory 和 TodoTracker 生成的注入块都带有特殊标记：

- `[SESSION MEMORY]` — SessionMemory 注入块
- `[TASK PROGRESS]` — TodoTracker 注入块

这些标记块在上下文三级压缩（ObservationMasking / LLMSummarize / BinaryReduction）中受到保护，不会被裁掉。这是框架保证研究进度不丢失的核心机制。
