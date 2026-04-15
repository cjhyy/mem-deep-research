# Offload 与 Evidence 联动优化方案

本文档描述一个拟议中的上下文优化方案，用于解决当前 `result_offload`、`microcompact`、`ObservationMasking` 与 `evidence` 之间职责分散、语义不一致的问题。

目标不是“再加一层 placeholder”，而是把旧工具结果统一收敛为：

```text
[OFFLOADED:toolmsg_abcd1234.txt|25000]

Evidence:
- ...
- ...

Full content: read_result("toolmsg_abcd1234.txt")
```

## 目标语义

期望行为：

- 最近 `compact_keep_recent` 轮的工具结果始终以完整文本保留在 `message_history`
- 超出最近窗口的旧工具结果，不再保留完整文本，而是替换为 `OFFLOADED marker`
- `OFFLOADED marker` 内联该消息对应的 evidence
- LLM 可以通过 `read_result(ref)` 回捞完整原文
- evidence 与 offload 通过同一个 `ref` 绑定，而不是只靠 `turn` 猜测
- 不引入额外的独立 LLM 调用

非目标：

- 不再保留 `[microcompact] ...` 这类通用 placeholder 作为常规产物
- 不要求每条工具结果都单独做一次额外 summarize
- 不要求按单个 tool call 级别精确切分 evidence

## 当前实现的主要问题

### 1. 即时 offload 无法表达滑动窗口

当前 `offload_large_result()` 在工具结果刚返回时就决定是否替换 history。它只知道“当前轮”，不知道未来还有多少轮，因此无法表达“最近 N 轮始终保留完整结果”的滑动窗口语义。

结果是：

- `result_offload_min_turn` 只能表示“前 N 轮不 offload”
- 不能表示“任意时刻最近 N 轮不 offload”

### 2. 旧结果清扫与 offload 不是同一条链路

当前有两条彼此割裂的链路：

- `offload_large_result()`：工具返回时写文件并可能立即替换内容
- `microcompact()` / `ObservationMaskingStrategy`：后续轮次再清理旧消息

如果即时 offload 已经把原文替换掉，后续滑动窗口策略就失去了操作完整结果的机会。

### 3. evidence 现在是“全局记忆”，不是“消息附件”

当前 evidence 主要进入 `SessionMemory.evidence_items`，再以 `[SESSION MEMORY] -> ## Evidence Ledger` 的形式每轮注入上下文。

这有两个问题：

- evidence 和某条具体被 offload 的消息没有稳定的一一对应关系
- `EvidenceItem.offload_ref` 虽然存在，但主路径并没有稳定填充它

### 4. Provider 会按“消息级”合并工具结果

Anthropic / OpenAI-compatible provider 会把一轮多个 tool result 合并成一条 `TOOL_RESULT` 消息。因此“即将被清扫”的真实粒度应该是 message，而不是单个 tool call。

## 设计原则

### 1. 先备份，再替换

工具结果进入上下文时先完整保留，同时立刻写文件备份，拿到稳定 `ref`。真正替换 history 的时机延后到滑动窗口清扫阶段。

### 2. 以 message 为主键

滑动窗口保护和 offload 替换都按“消息级”而不是“tool call 级”处理。一个 `ref` 对应一条将被清扫的 `TOOL_RESULT` 消息。

### 3. evidence 必须与 ref 绑定

evidence 不是悬浮在 session 上的“全局事实集合”，而是某条即将 offload 的消息的摘要附属物。主关联键是 `offload_ref`。

### 4. 不额外发起 LLM 调用

evidence 提取借用正常主回合完成，而不是单独调用一个轻量模型。

## 推荐流程

### Phase 1：工具结果落入 history 时只做备份

当工具结果写入 `message_history` 时：

1. 保留完整原文
2. 如果超过 `result_offload_threshold`，立刻写到 offload 存储
3. 生成稳定 `ref`，例如 `toolmsg_abcd1234.txt`
4. 把 `ref` 挂到该条 `TOOL_RESULT` 消息的 metadata 上

此阶段不替换 history 文本。

推荐消息 metadata：

```python
{
    "role": "user",
    "_type": MT.TOOL_RESULT,
    "_offload_ref": "toolmsg_abcd1234.txt",
    "_offload_chars": 25000,
    "_offload_state": "backed_up",
    "_offload_prepared": False,
    "content": [{"type": "text", "text": "...完整原文..."}],
}
```

说明：

- 这里的 `ref` 应统一使用逻辑引用，不使用绝对路径
- `read_result(ref)` 再通过 `ContextManager` 映射到真实存储位置

### Phase 2：`prepare_offload_candidates()` 在下一轮 LLM 前准备候选

在每轮主 LLM 调用前执行 `prepare_offload_candidates()`。

它只做三件事：

1. 找出“这轮之后将滑出 `compact_keep_recent` 窗口”的旧 `TOOL_RESULT` 消息
2. 将这些消息标记为 `pending_evidence`
3. 注入一条 sidecar prompt，要求模型在本轮正常回答时顺手返回这些消息的 evidence

注意：

- 这一步不清扫消息
- 候选消息在本轮仍保持完整原文，确保模型看得到
- 这一步也不发起额外 LLM 调用

推荐 sidecar prompt 结构：

```text
[OFFLOAD PREP]

The following tool-result messages will be offloaded after this turn.
While completing your normal reasoning, emit evidence blocks for any candidate
you used in this turn.

Output format:
<offload_evidence ref="toolmsg_abcd1234.txt">
- fact 1
- fact 2
</offload_evidence>
```

推荐新增一个临时消息类型，例如 `MT.OFFLOAD_PREP`。它不需要受压缩保护，只服务当前一轮。

### Phase 3：解析本轮 assistant 输出中的 evidence

主 LLM 在正常回合里，除正常回答和工具调用外，可以额外输出：

```xml
<offload_evidence ref="toolmsg_abcd1234.txt">
- 关键事实 1
- 关键事实 2
</offload_evidence>
```

主循环解析这些块，并写入 `offload_registry[ref]`：

```python
{
    "toolmsg_abcd1234.txt": {
        "turn": 3,
        "char_count": 25000,
        "state": "evidence_ready",
        "evidence": [
            "关键事实 1",
            "关键事实 2",
        ],
    }
}
```

如需保留全局 ledger，可同步写入 `SessionMemory.evidence_items`，但 `offload marker` 展示内容应以 `ref` 对应的 evidence 为准。

### Phase 4：本轮结束后再真正清扫

在本轮 assistant 响应已经产生之后，再把本轮准备好的候选消息替换成 `MT.OFFLOADED`：

```text
[OFFLOADED:toolmsg_abcd1234.txt|25000]

Evidence:
- 关键事实 1
- 关键事实 2

Full content: read_result("toolmsg_abcd1234.txt")
```

替换后的消息建议：

```python
{
    "role": "user",
    "_type": MT.OFFLOADED,
    "_offload_ref": "toolmsg_abcd1234.txt",
    "_offload_chars": 25000,
    "content": [{"type": "text", "text": rendered_marker}],
}
```

这样：

- 最近窗口内仍有完整原文
- 超窗后只保留 `OFFLOADED marker`
- evidence 与 ref 保持一一对应

## 失败与兜底策略

常规路径下不再生成通用 placeholder。

推荐兜底顺序：

1. 有 `ref` 且有 evidence：输出 `OFFLOADED + Evidence`
2. 有 `ref` 但本轮未产出 evidence：输出裸 `OFFLOADED marker`
3. 备份写文件失败：保留原文，不执行 offload

可选优化：

- 增加 `offload_evidence_grace_turns`，允许某些候选在 evidence 缺失时再多保留 1 轮
- 但默认实现可以先不加，保持机制简单

## 为什么不推荐继续使用 `microcompact placeholder`

当前 `microcompact()` 的替代文本形如：

```text
[microcompact] search: 25000 chars — ...
```

问题在于：

- 没有稳定 `ref`
- 不能直接回捞完整内容
- 与 evidence 没有结构化关联
- 对 resume / read_result / final summary 都不够友好

因此在本方案里，`microcompact` 不再输出通用 placeholder，而是直接输出 `OFFLOADED marker`。

## 数据结构建议

### 1. `ToolCallRecord` 继续保存完整结果

`ToolCallRecord.result_full` 仍建议保留原始完整结果，用于 dedup、debug、回退和测试。

但额外增加：

```python
offload_ref: str = ""
message_ref: str = ""
```

说明：

- `offload_ref`：真实 offload 文件引用
- `message_ref`：对应哪条 tool-result message

### 2. 新增 `OffloadRecord`

建议在 `ContextManager` 中增加一个显式注册表：

```python
@dataclass
class OffloadRecord:
    ref: str
    turn: int
    char_count: int
    message_ref: str
    tool_names: list[str]
    state: str  # backed_up | pending_evidence | offloaded
    evidence: list[str] = field(default_factory=list)
```

这样可以避免从 marker 文本反向解析业务语义。

### 3. 统一 ref 规范

推荐只在 LLM、marker、`read_result()` 中暴露逻辑 ref，例如：

```text
toolmsg_abcd1234.txt
```

而不是：

- `/abs/path/to/offloaded_results/toolmsg_abcd1234.txt`
- 推导式 `turn3_search_25000chars.txt`

统一逻辑 ref 后：

- prompt 更干净
- 恢复逻辑更简单
- 路径遍历防护更明确

## 配置收敛建议

保留：

- `result_offload_threshold`
- `result_offload_dir`
- `compact_keep_recent`

建议删除：

- `result_offload_min_turn`
- `result_offload_context_ratio`

原因：

- “最近 N 轮保留完整内容”应统一由滑动窗口决定
- 是否清扫旧结果，应由 `compact_keep_recent` 和消息状态决定
- 即时 offload 不应再基于“当前上下文宽裕”做临时判断

可选新增：

```yaml
context_manager:
  result_offload_threshold: 5000
  compact_keep_recent: 5
  offload_evidence_grace_turns: 0
```

## 实现落点建议

优先改动文件：

- `mem_deep_research_core/core/main_loop.py`
- `mem_deep_research_core/core/context_manager.py`
- `mem_deep_research_core/core/constants.py`
- `mem_deep_research_core/core/memory.py`
- `tests/test_context_manager.py`
- `tests/test_compact_offload.py`
- `tests/test_mainloop_tools.py`

建议改法：

### `main_loop.py`

- 在每轮 LLM 调用前调用 `prepare_offload_candidates()`
- 注入 `OFFLOAD PREP` sidecar prompt
- 解析 `<offload_evidence ref=\"...\">...</offload_evidence>`
- 在本轮完成后触发真正的消息替换

### `context_manager.py`

- `offload_large_result()` 改为“只备份并返回 ref”
- 新增 `prepare_offload_candidates()`
- 新增 `finalize_offload_candidates()`
- `microcompact()` 改为基于 `ref` 输出 `MT.OFFLOADED`

### `constants.py`

- 新增 `MT.OFFLOAD_PREP`
- 增加 evidence tag 常量和 marker 渲染辅助函数

### `memory.py`

- `EvidenceItem` 保留 `offload_ref`
- `SessionMemory` 继续做全局 evidence ledger，但不再承担唯一绑定职责

## 测试建议

至少覆盖以下场景：

1. 最近 `N` 轮完整保留
2. 第 `N+1` 轮之前的旧结果被替换为 `OFFLOADED marker`
3. `prepare_offload_candidates()` 不会提前清扫消息
4. 主回合无额外 LLM 调用
5. evidence 能按 `ref` 正确绑定到 marker
6. Anthropic / OpenAI-compatible 的“合并 tool result 消息”路径
7. GPT native 每个 tool result 单独一条消息的路径
8. 无 evidence 时降级为裸 marker
9. `read_result(ref)` 可回捞完整原文
10. resume 后 `OFFLOADED marker` 仍能恢复

## 方案收益

实现完成后，框架会得到一条更清晰的生命周期：

```text
完整工具结果
  -> 写文件备份
  -> 最近 N 轮完整保留
  -> 下一轮借主 LLM 产 evidence
  -> 替换为 OFFLOADED marker
  -> read_result(ref) 回捞原文
```

相比当前实现，优势是：

- 真正实现“最近 N 轮完整可见”
- evidence 与 offload 形成稳定绑定
- 不增加额外 LLM 调用
- 不再依赖通用 placeholder
- 对 dedup、resume、verify、summary 都更一致

## 相关文档

- [04-context-management](./04-context-management.md)
- [12-memory-and-todo](./12-memory-and-todo.md)
- [15-technical-roadmap](./15-technical-roadmap.md)
