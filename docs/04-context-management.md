# 上下文管理

上下文管理是框架的核心子系统之一，负责工具调用去重、上下文窗口压缩和引用追踪。

## 三级压缩策略

```
┌──────────────────────────────────────────────────────────┐
│ L1: ObservationMasking        token 占比 ≥ 60%          │
│     零 LLM 成本               替换工具结果为单行摘要     │
├──────────────────────────────────────────────────────────┤
│ L2: LLMSummarize              token 占比 ≥ 80%          │
│     一次 LLM 调用             压缩旧对话为结构化摘要     │
├──────────────────────────────────────────────────────────┤
│ L3: BinaryReduction           token 占比 ≥ 95%          │
│     零 LLM 成本               紧急删除中间消息           │
└──────────────────────────────────────────────────────────┘
```

### L1: ObservationMasking（观察遮蔽）

- **触发条件**: token 占比 ≥ `compact_at_ratio`（默认 0.6）
- **策略**: 将旧轮次的工具结果替换为单行摘要
- **保留**: 最近 `compact_keep_recent` 轮（默认 3）不压缩
- **成本**: 零 LLM 调用
- **实现**: 遍历消息历史，找到 tool_result 块，替换为 `[已压缩: {tool_name} 结果摘要]`

```python
# 压缩前
{"role": "user", "content": "[tool_result] 大量搜索结果内容..."}

# 压缩后
{"role": "user", "content": "[已压缩: web_search 返回 5 条结果]"}
```

### L2: LLMSummarize（LLM 摘要）

- **触发条件**: token 占比 ≥ `summarize_at_ratio`（默认 0.8）
- **策略**: 调用 LLM 将旧对话历史压缩为结构化摘要
- **成本**: 一次轻量 LLM 调用
- **输出**: `[RESEARCH CONTEXT SUMMARY]` 块，替换旧消息

```python
# 压缩后的消息历史
[
    {"role": "user", "content": "[RESEARCH CONTEXT SUMMARY]\n已完成搜索...\n关键发现..."},
    # 最近几轮保持原样
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "[tool_result] ..."},
]
```

### L3: BinaryReduction（二进制缩减）

- **触发条件**: token 占比 ≥ 95%
- **策略**: 保留首条消息 + 尾部消息，删除中间一半
- **成本**: 零 LLM 调用
- **用途**: 紧急情况下防止上下文溢出

## 工具调用去重

ContextManager 跟踪所有工具调用的参数和结果，实现跨轮次去重：

```python
# 去重流程
to_execute, cached_results = context_manager.filter_duplicate_calls(tool_calls)

# to_execute: 需要实际执行的调用
# cached_results: 从缓存返回的重复调用结果
```

### 渐进式升级

| 重复次数 | 行为 |
|---------|------|
| 第 1 次 | 警告 LLM 已调用过相同参数 |
| 第 2 次 | 强烈警告，建议换策略 |
| 第 3+ 次 | 直接返回缓存结果，不再执行 |

## 引用追踪

`SourceRegistry` 自动从工具结果中提取引用信息：

```python
class SourceRecord:
    url: str           # 来源 URL
    title: str         # 来源标题
    tool_name: str     # 使用的工具
    turn: int          # 所在轮次
```

引用摘要会在最终答案生成前注入消息历史。

## WindowStrategy 接口

支持自定义压缩策略：

```python
from mem_deep_research_core.core.window_strategy import WindowStrategy, WindowContext

class MyStrategy(WindowStrategy):
    def should_trigger(self, ctx: WindowContext) -> bool:
        """判断是否应执行压缩"""
        return ctx.token_ratio > 0.7

    def apply(self, messages: list, ctx: WindowContext) -> CompressResult:
        """执行压缩，直接修改 messages 列表"""
        # ... 自定义压缩逻辑
        return CompressResult(
            messages_affected=n,
            tokens_saved=saved,
            action_label="my_strategy",
        )
```

### WindowContext 数据

```python
@dataclass
class WindowContext:
    current_turn: int
    max_turns: int
    token_count: int
    max_tokens: int
    token_ratio: float              # token_count / max_tokens
    message_history: list           # 只读引用
    system_prompt: str
    call_registry: dict             # 工具调用记录
    compacted_turns: set            # 已压缩的轮次
    estimate_tokens_fn: Callable    # token 估算函数
```

### WindowStrategyPipeline

```python
from mem_deep_research_core.core.window_strategy import WindowStrategyPipeline

# 使用默认三级策略
pipeline = WindowStrategyPipeline.default_strategies()

# 自定义策略组合
pipeline = WindowStrategyPipeline(strategies=[
    ObservationMaskingStrategy(threshold=0.5),
    MyCustomStrategy(),
    BinaryReductionStrategy(threshold=0.9),
])
```

## 配置参考

```yaml
context_manager:
  enable_dedup: true          # 启用工具调用去重
  enable_compact: true        # 启用上下文压缩
  compact_at_ratio: 0.6       # L1 触发阈值
  summarize_at_ratio: 0.8     # L2 触发阈值
  compact_keep_recent: 3      # 保留最近轮数
```

## 结果生命周期（Result Lifecycle）

工具结果从产生到压缩到恢复的完整链路。v1.3.0 把这条链作为 Runtime Contract 收口，每个阶段有明确的状态转换：

```text
                  ┌──────────────────────────────────────────────────┐
                  │  ToolExecutor.execute_single_tool                │
                  │  → format → tool_result_for_llm dict             │
                  └──────────────────────────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  ContextManager.backup_large_result(result, tool, turn)     │
        │                                                             │
        │  if len(result) >= result_offload_threshold:                │
        │      write to <offload_dir>/<ref>.txt                       │
        │      OffloadRecord(state="backed_up") in registry           │
        │      message["_offload_refs"] = [ref]                       │
        └─────────────────────────────────────────────────────────────┘
                                       │
                              (turns accumulate)
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  ContextManager.finalize_offload_candidates(...)            │
        │                                                             │
        │  for msg older than keep_recent:                            │
        │      replace content with [OFFLOADED:ref|chars] marker      │
        │      msg["_type"] = MT.OFFLOADED                            │
        │      OffloadRecord.state := "offloaded"                     │
        └─────────────────────────────────────────────────────────────┘
                                       │
                       (orthogonal: token pressure)
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  ContextManager.apply_compact / apply_summarize             │
        │  → ObservationMasking → LLMSummarize → BinaryReduction      │
        │  (only operates on message TEXT; offload registry untouched)│
        └─────────────────────────────────────────────────────────────┘
                                       │
                       (LLM requests detail back)
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  read_result(ref)  →  ContextManager.restore_single_file    │
        │  reads the .txt file, returns full text. Marker stays put;  │
        │  the LLM sees the read_result tool call's response.         │
        └─────────────────────────────────────────────────────────────┘
                                       │
                          (process restart / HITL resume)
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  ContextManager.restore_offloaded_content(message_history)  │
        │  →  scan for [OFFLOADED:...] markers, optionally reload     │
        │  →  _rebuild_registry_from_history rebuilds OffloadRecord   │
        │     from message sidecar fields, even if registry was lost. │
        └─────────────────────────────────────────────────────────────┘
```

关键不变量（被 e2e 测试 `tests/test_result_lifecycle_e2e.py` 锁定）：

- 每个 disk 上的 `.txt` 文件对应一条 registry entry，反之亦然 — 没有 orphan 文件，没有 dangling registry。
- `_offload_refs` sidecar 字段是 durable handle 列表；`OFFLOADED:` marker 是状态升级信号。冷启动 (`_rebuild_registry_from_history`) 同时看到两者时，marker 优先（`state="offloaded"`）。
- offload 与 compact 正交：observation masking 只改 message TEXT，不动 offload registry；offload 只改大结果消息的 `content` 和 `_type`，不动其他消息。
- `read_result(ref)` 在任何 marker 状态下都能拉回原文（只要 `.txt` 仍在 disk 上）。
- HITL resume / process restart 后通过 `RuntimeSnapshot.context_manager_state`（含 `offload_registry` / `dedup_cache` / `compacted_turns`）一次性恢复，无需重读 disk；冷启动的 `_rebuild_registry_from_history` 是 fallback。

代码引用：
- `mem_deep_research_core/core/context_manager.py` — `backup_large_result` / `finalize_offload_candidates` / `apply_compact` / `restore_single_file` / `_rebuild_registry_from_history` / `merge_offload_registry`
- `mem_deep_research_core/core/main_loop.py::_maybe_offload_result` — 每轮工具结果走的 offload 入口
- `tests/test_result_lifecycle_e2e.py` — 7-stage 端到端回归

## ContextManager API

```python
class ContextManager:
    # Token 估算
    def estimate_tokens(self, text: str) -> int
    def get_context_ratio(self, system_prompt, message_history, max_context_length) -> float

    # 去重
    def filter_duplicate_calls(self, calls) -> tuple[list, list]
    def register_tool_results(self, tool_calls, tool_results_with_id, turn)

    # 压缩（由 manage_context 统一调用）
    def manage_context(self, message_history, current_turn, system_prompt, max_context_length) -> str

    # 引用
    def source_registry -> SourceRegistry
```
