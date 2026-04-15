# Arena Review: Offload 机制四阶段流水线重构

**日期：** 2026-04-14  
**主题：** review 当前代码（offload 机制重构）  
**参与模型：** Claude Opus / GPT-5.4 / Gemini 3.1-pro-preview

---

## 变更摘要

本次变更将 `context_manager.py` 中的 offload 机制从旧的单步替换重构为四阶段流水线：

1. **backup_large_result** — 将大工具结果备份到磁盘
2. **prepare_offload_candidates** — 按滑动窗口标记即将滑出的消息为 `pending_evidence` 状态
3. **OFFLOAD_PREP sidecar 注入**（`main_loop.py`）— 让 LLM 提取 evidence
4. **finalize_offload_candidates** — 将已提取 evidence 的消息替换为 OFFLOADED 占位符

同时新增：
- `microcompact` 方法（窗口压缩/兜底流程）
- `update_offload_evidence`（绑定 evidence）
- `restore_offloaded_content` / `restore_single_file`（按需恢复）
- `_generate_offload_ref` 改为 `toolmsg_{uuid8}.txt` 格式
- `config_schema.py` 移除部分旧配置项
- 新增 `tests/test_compact_offload.py` 覆盖局部方法

---

## 总体评估

三位审查者一致认为重构方向正确，但在实现细节上发现了多个高风险问题：

- OFFLOAD_PREP 消息可能未被清理导致上下文污染
- 状态机缺乏失败回滚机制
- 路径遍历安全防护不一致
- prepare 与 finalize 的 cutoff 计算存在争议（有意设计 vs off-by-one bug）
- 端到端集成测试严重不足

---

## 优点

### 四阶段流水线架构设计合理
**支持：** Claude Opus、GPT-5.4、Gemini 3.1-pro-preview | **置信度：** 0.90

将 offload 从单步操作拆分为 backup → prepare → evidence extraction → finalize 四个阶段，实现了在替换前通过 LLM 提取关键信息（evidence）的能力，避免了信息的无损丢失。这种设计使得 offload 后的占位符仍能携带有价值的摘要信息，显著提升了上下文管理的质量。

### restore_single_file 具备路径安全校验
**支持：** GPT-5.4、Claude Opus、Gemini 3.1-pro-preview | **置信度：** 0.95

`restore_single_file` 方法实现了 `realpath` + 前缀校验来防止路径遍历攻击，说明开发者具有安全意识。这为后续统一安全路径提供了良好的参考实现。

### 新增了针对局部方法的单元测试
**支持：** GPT-5.4、Gemini 3.1-pro-preview | **置信度：** 0.85

`tests/test_compact_offload.py` 覆盖了 backup、finalize、`restore_single_file` 等核心方法的基本行为，为重构提供了一定的安全网。

---

## 待改进项

### 缺少主循环级别的端到端集成测试
**支持：** GPT-5.4、Claude Opus、Gemini 3.1-pro-preview | **置信度：** 0.90

当前测试仅覆盖 `ContextManager` 的局部方法，缺少验证完整链路的集成测试：

```
大工具结果 → 备份 → 注入 sidecar → LLM 输出 <offload_evidence> → finalize 替换 → read_result 恢复
```

本次重构的核心复杂度在于 `main_loop` 与 `context_manager` 之间的多阶段协作，仅有单元测试无法有效防止集成层断链，也无法覆盖 LLM 调用失败/重试等异常场景。

### read_result 工具描述中的 ref 示例已过时
**支持：** Gemini 3.1-pro-preview、Claude Opus | **置信度：** 0.95

`read_result` 工具的 description 中给出的示例仍为旧格式 `'turn3_search_15000chars.txt'`，但重构后 `_generate_offload_ref` 生成的实际格式已变为 `'toolmsg_{uuid8}.txt'`。示例不一致会降低 LLM 正确使用该工具的概率。

### finalize_offload_candidates 与 microcompact 的职责边界需要澄清
**支持：** Gemini 3.1-pro-preview | **反对：** GPT-5.4、Claude Opus | **置信度：** 0.65

Gemini 认为 `finalize_offload_candidates` 与 `microcompact` 逻辑冗余，属于死代码应删除；但 GPT-5.4 和 Claude Opus 均反对，认为 `finalize` 承担的是 evidence 驱动的有条件替换，与 `microcompact` 的兜底压缩不等价。无论最终结论如何，两个方法的职责边界和调用时序需要在代码注释或文档中明确说明。

---

## 风险

### [HIGH] OFFLOAD_PREP sidecar 消息可能未被清理，导致上下文持续污染
**支持：** Gemini 3.1-pro-preview、Claude Opus | **反对：** GPT-5.4 | **置信度：** 0.85

`main_loop` 中为 LLM 提取 evidence 而注入的 `_type: MT.OFFLOAD_PREP` 消息，在 LLM 调用完成后可能没有被显式移除。如果 `microcompact` 也不处理该类型，每轮都会累积一条过期的 offload 指令，浪费 token 并可能导致 LLM 在不该提取 evidence 时尝试提取。

### [HIGH] 状态机缺乏失败回滚，pending_evidence 状态可能滞留
**支持：** GPT-5.4、Claude Opus | **置信度：** 0.85

`prepare_offload_candidates` 将记录状态从 `backed_up` 改为 `pending_evidence`，但如果后续 LLM 调用失败、超时、被 `context_limit` 截断或异常中断，没有任何代码将状态回滚。状态滞留不会导致"永久不 offload"，但会导致 evidence 丢失，降低 offload 后占位符的信息质量。

### [HIGH] restore_offloaded_content 缺少路径遍历防护
**支持：** GPT-5.4、Claude Opus、Gemini 3.1-pro-preview | **置信度：** 0.93

`restore_single_file` 已实现 `realpath` + 前缀校验，但批量恢复方法 `restore_offloaded_content` 直接将 marker 中的 `file_name` 拼接到 `offload_dir` 后读取，未做等价的路径规范化校验。在 checkpoint 恢复场景下，`message_history` 中的 marker 可能被篡改，存在读取 `offload_dir` 外部任意文件的安全风险。

### [MEDIUM] prepare 与 finalize 的 cutoff 计算差异存在争议
**支持：** Claude Opus、GPT-5.4 | **反对：** Gemini 3.1-pro-preview | **置信度：** 0.70

- `prepare_offload_candidates` 使用 `cutoff_turn = current_turn - keep_recent + 1`
- `finalize_offload_candidates` 使用 `cutoff_turn = current_turn - keep_recent`（差 1）

Claude Opus 和 GPT-5.4 认为这是 off-by-one bug，会导致边界轮次消息延迟一轮才被 finalize，在最后一轮或中断场景下可能永远不被替换。Gemini 明确反对，认为这是滑动窗口的有意设计：prepare 标记即将滑出的消息使其在当前轮保留以便 LLM 提取 evidence，下一轮才由 finalize/microcompact 替换。**需要开发者确认设计意图。**

### [MEDIUM] 主循环中 offload 流程的集成完整性存疑
**支持：** GPT-5.4 | **反对：** Claude Opus、Gemini 3.1-pro-preview | **置信度：** 0.55

GPT-5.4 通过代码搜索未能在 `main_loop` 中找到 `prepare_offload_candidates`/`finalize_offload_candidates` 的稳定调用点，怀疑功能未完整集成。Gemini 则指出 `main_loop` 中确实存在 OFFLOAD_PREP 注入等集成逻辑，认为该发现可能是搜索不充分导致的误报。综合来看，集成逻辑可能存在但需要更完整的调用链追踪来确认。

### [LOW] update_offload_evidence 的赋值语义存在争议
**支持：** Gemini 3.1-pro-preview、Claude Opus | **反对：** GPT-5.4 | **置信度：** 0.60

`update_offload_evidence` 使用直接赋值 `record.evidence = evidence`，在同一 ref 多次出现 `<offload_evidence>` 时会覆盖已有内容。Claude Opus 同意应改为 extend + 去重。但 GPT-5.4 反对，认为 evidence 可能就是"最终版本"语义，盲目 extend 可能引入重复或旧证据残留。这取决于 evidence 的设计意图是累积还是替换。

---

## ~~待确认问题~~ ✅ 全部已确认

### Q1: prepare 与 finalize 的 cutoff 差异是有意的滑动窗口设计还是 off-by-one bug？
✅ **有意设计。** prepare 用 `N-K+1` 提前 1 轮标记，让 LLM 在当前轮还能看到完整内容并输出 evidence；finalize 用 `N-K` 在下一轮才替换。已在代码中添加详细注释说明。最后一轮 pending_evidence 滞留无影响（后续无 LLM 调用）；prepare 中已加重入检测兜底。

### Q2: evidence 的语义是累积追加还是最终替换？
✅ **最终替换。** 每个 ref 只在一轮中被标记为 pending_evidence，LLM 在该轮输出一次 `<offload_evidence>` 即为最终版本。如果 LLM 未输出，下一轮 prepare 会重新纳入并再次请求。赋值语义正确。

### Q3: OFFLOAD_PREP 消息是否在某个未被发现的路径中被清理？
✅ **已有清理，并补充了异常路径。** 正常路径：main_loop 1141-1145 行在 LLM 调用后显式过滤 `MT.OFFLOAD_PREP`。异常路径：新增了 context_limit/error continue 之前的清理逻辑。

---

## 行动项

| 优先级 | 行动 | 状态 |
|--------|------|------|
| **HIGH** | 修复 `restore_offloaded_content` 的路径遍历安全漏洞（复用 `restore_single_file` 的 `realpath` + 前缀校验逻辑） | ✅ 已添加 `realpath + startswith` 校验 |
| **HIGH** | 确认并修复 OFFLOAD_PREP 消息的清理逻辑 | ✅ 正常路径已有清理（main_loop 1141-1145）；新增 LLM 失败路径的清理（context_limit/error continue 之前） |
| **HIGH** | 明确 prepare/finalize cutoff 差异的设计意图并添加代码注释 | ✅ 有意设计：prepare 提前 1 轮标记让 LLM 提取 evidence，finalize 下一轮替换。已加详细注释 |
| **MEDIUM** | 为状态机添加失败回滚机制 | ✅ prepare 中加重入检测：`pending_evidence` 状态的记录重新纳入候选 |
| **MEDIUM** | 添加端到端集成测试覆盖完整 offload 链路 | ⬜ 后续补充 |
| **LOW** | 更新 `read_result` 工具描述中的 ref 示例格式 | ✅ 改为 `toolmsg_abcd1234.txt` |
