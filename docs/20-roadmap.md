# Mem Deep Research Framework — 迭代 Roadmap

> **权威版本**：本文档是唯一权威 Roadmap，取代 doc-15、doc-17、doc-19。
> 旧文档已降级为历史参考，顶部标注 `[DEPRECATED — 见 doc-20]`。
>
> 生成时间：2026-04-17
> 最后更新：2026-04-20（Phase 1 bug 状态核对 + 新发现项）
> 基于：Arena 三模型评审（Claude Opus / Gemini 3.1-pro-preview / DeepSeek-R1）

## Phase 1 进度快照（2026-04-20）

| Bug | 状态 | 证据 |
|-----|------|------|
| BUG-01 Grace Turn 绕过 context 管理 | ✅ 已修复 | `main_loop.py:1281-1306` — 注入 nudge 前 token 预算检查 + 满阈值走 `manage_context` |
| BUG-02 Nudge 去重破坏角色交替 | ✅ 已修复 | `main_loop.py:1265-1266` — 仅检查末尾一条 |
| BUG-03 常量注释误导 | ✅ 已修复 | `constants.py:41-45` — 注释已明确 "N=2 means 1 nudge + 1 break" |
| BUG-04 子 Agent 配置继承不完整 | ✅ 已修复 | `sub_agent_runner.py:104-175` — inherit_with_override 四层合并 |
| BUG-05 Registry merge 静默丢弃 | ✅ 已修复 | `context_manager.py:655-674` — collision WARNING + 覆盖写入，不丢条目 |
| BUG-06 MCP 会话跨任务串号 | ⚠️ 设计争议 | `tool/manager.py:286-378` — context fingerprint 机制已修复串号；但默认 `_default_env_inject` 自动将顶层 context 注入 env 的设计本身可议（见 BUG-09） |
| BUG-07 配置契约 fail-fast | ✅ 已修复 | `deep_research.py:297-341` — critical 字段缺失抛 `ConfigValidationError`，catch-all 已移除 |
| BUG-08 Fast-path 与 Grace Turn 冲突 | ✅ 已修复 | `main_loop.py:1246` — 加 `is_quick_mode` gate |

**额外完成（Review 发现）**：
- BUG-10 `main_loop.py:2017` agent_calls gather 加 `return_exceptions=True` + 异常归一化
- BUG-11 `agent_factory.py:357-384` close() 异常隔离，所有 tool_manager 必定尝试清理
- BUG-12 `todo_tracker.py` / `file_state_cache.py` / `transcript.py` 加 `threading.Lock`，iteration 快照化
- BUG-14 `response_language` 加 Pydantic `field_validator` 软验证
- BUG-16 `llm_call_handler.py:478` 硬编码 0.6 → `CONTEXT_REDUCTION_TARGET_RATIO`
- BUG-17 命名 sub-agent offload 孤儿：`SubAgentRunner.run()` 接受 `parent_context_manager`，finally 块 merge registry

**复核结论**：
- BUG-13 scrape_max_length：**非 bug**，配置优先级已正确
- BUG-15 `@file` 白名单：**复核已实现**，多层防护已在 input_compiler.py + config_schema.py 中
- BUG-18~21（Review agent 汇报）：**均非 bug**，属已有设计或误判

**测试**：543 passed，零 regression。

---

## 当前状态评估

框架已完成从功能扩张期向**运行时收敛期**的过渡节点。核心执行内核（主循环、上下文管理、子 Agent、Offload 流水线）功能已就位，但存在四类系统性风险需要在继续演进前修复：

| 风险维度 | 代表问题 | 影响 | 2026-04-20 状态 |
|---------|---------|------|-----------|
| 运行时状态一致性 | Grace Turn 绕过上下文管理 | 长任务 Token 溢出 | ✅ 已修复 |
| 多任务隔离失效 | MCP 会话串号、子 Agent 配置继承不完整 | 静默行为偏离 | ✅ 串号已修（fingerprint）；子 Agent 已修（inherit_with_override） |
| 配置契约分裂 | 多条读取路径、校验非 fail-fast | 调试成本极高 | ⚠️ critical 字段 fail-fast 完成；多路径读取未收口 |
| 并发安全 | 三个模块缺锁、gather 缺 return_exceptions、close 不隔离 | 资源泄漏、死锁 | ✅ Review 发现项全部修复 |
| 质量保障缺失 | 端到端集成测试严重不足 | 修复引入回归无法感知 | ⚠️ Phase 2 承接 |

---

## 版本规划总览

```
Phase 0  v1.2.3-prep   规划收敛与对齐          ~1 周
Phase 1  v1.2.3        高优稳定性修复           ~3-4 周
Phase 2  v1.3.0        契约收敛与架构整理        ~4-6 周
Phase 3  v1.4.0        质量体系与长期演进        ~4-6 周
```

---

## Phase 0：规划收敛与对齐（v1.2.3-prep）

**目标**：消除规划碎片化，建立单一权威执行框架。

### 任务清单

| 优先级 | 任务 | 完成标准 |
|--------|------|---------|
| P0 | 合并 doc-15/17/19 为本文档（doc-20） | 旧文档顶部标注 DEPRECATED |
| P0 | 统一版本号体系（v1.2.3 / v1.3.0 / v1.4.0） | CHANGELOG 和 pyproject.toml 一致 |
| P0 | 决策配置契约修复策略（方案 A vs 方案 B） | 技术方案文档落地 |
| P1 | 审查 `@file` 指令安全边界 | 安全评估报告，确认是否需提升至 Phase 1 |

### 配置契约修复策略决策

**推荐方案 B（一次性收口）**，区分两类字段：
- **核心必需字段**（如 `llm.provider_class`、`llm.model_name`）：缺失时 `_validate_config()` 直接 `raise ConfigurationError`，阻断初始化
- **可选字段**：缺失时 `WARNING` 日志 + 自动填充默认值，不阻断

理由：框架处于 v1.2.x 早期，用户基数有限，一次性收口的迁移成本可接受。

---

## Phase 1：v1.2.3 高优稳定性修复

**目标**：修复所有已确认的高优运行时风险，每个修复必须附带回归测试。

**入口条件**：Phase 0 完成，配置策略已决策。

### Bug 修复清单

#### BUG-01 Grace Turn 绕过上下文管理 🔴 ✅ 已修复（2026-04-20）

**位置**：`main_loop.py:1279-1283`

**问题**：Grace Turn 路径仅执行 microcompact，跳过完整 `manage_context` 流水线（含 `prepare_offload_candidates` / `finalize_offload_candidates`）。长任务中 Grace Turn 发生在 token 接近上限时，下一轮 LLM 调用直接触发 context limit 错误。

**修复方案**：
```python
# Grace Turn 注入 nudge 前，先做 token 预算检查
if self._token_ratio() > self.cfg.context_manager.compact_at_ratio:
    await self.context_manager.manage_context(self.message_history)
self.message_history.append(nudge_msg)
continue
```

**验收标准**：
- [ ] 上下文接近阈值时触发 Grace Turn，`manage_context` 被正确调用（mock 验证）
- [ ] Grace Turn 后 token 占比回落到阈值以下

---

#### BUG-02 Nudge 去重逻辑破坏消息角色交替 🔴 ✅ 已修复

**位置**：`main_loop.py`（nudge 去重的反向扫描 + `list.pop(i)`）

**问题**：当 LLM 停滞后恢复、再次停滞时，去重逻辑反向遍历整个历史并删除早期轮次的 nudge（role=user）。这可能导致历史中出现连续两条 `assistant` 消息，违反 Anthropic API 的严格角色交替规则，引发 400 Bad Request。

**修复方案**：移除反向扫描去重逻辑，改为仅检查 `message_history[-1]` 是否已为 nudge，避免重复注入：
```python
# 替换复杂的反向扫描
last = self.message_history[-1] if self.message_history else None
if last and last.get("_type") == MT.NO_TOOL_NUDGE:
    pass  # 已有 nudge，不重复注入
else:
    self.message_history.append(nudge_msg)
```

**验收标准**：
- [ ] 停滞→恢复→再次停滞场景下，消息历史角色序列合法（user/assistant 严格交替）
- [ ] 连续停滞场景下不会注入重复 nudge

---

#### BUG-03 常量命名语义误导（off-by-one）🟡 ✅ 已修复

**位置**：`constants.py`

**问题**：`MAX_CONSECUTIVE_NO_TOOL_TURNS=2` 注释声称"gets up to this many grace turns"，实际只有 1 次 grace turn（第 2 次直接 break）。

**修复方案**：
```python
# 修改注释，明确语义
MAX_CONSECUTIVE_NO_TOOL_TURNS = 2  # 连续 N 次无工具调用后终止；前 N-1 次注入 nudge，第 N 次 break
```
或重命名为 `MAX_NO_TOOL_TURNS_BEFORE_TERMINATION`。

**验收标准**：
- [ ] 常量名或注释准确反映"1 次 grace turn"的实际行为

---

#### BUG-04 子 Agent 配置继承不完整 🔴 ✅ 已修复（2026-04-20）

**位置**：`sub_agent_runner.py`

**问题**：`SubAgentRunner` 创建独立 `ContextManager` 时，`offload_dir` 已继承（已修复），但 `compact_keep_recent`、`enable_dedup`、`window strategy` 等配置未继承，退回默认值，导致子 Agent 的 compact/offload 行为与主链不一致。

**修复方案**：实现 `inherit_with_override` 模式：
- 默认从父级 `ContextManager` 继承完整配置
- 子 Agent 配置中显式声明的字段允许覆盖
- 设计原则：子 Agent 通常执行更短任务，可允许更激进的 compact 策略，但必须是显式配置而非静默退回默认值

**验收标准**：
- [ ] 子 Agent 的 `compact_keep_recent`、`enable_dedup` 与主链一致（未显式覆盖时）
- [ ] 显式覆盖的子 Agent 配置生效

---

#### BUG-05 Registry Merge 静默丢弃条目 🔴 ✅ 已修复

**位置**：`context_manager.py`（`merge_offload_registry`）

**问题**：`if ref not in self._offload_registry` 跳过冲突，被丢弃的 ref 对应文件永远不会被清理，产生孤儿文件，且无任何日志警告。

**修复方案**：
```python
def merge_offload_registry(self, child_registry: dict) -> None:
    for ref, path in child_registry.items():
        if ref in self._offload_registry:
            logger.warning(f"Offload registry collision on ref={ref!r}, child entry dropped. "
                           f"Existing: {self._offload_registry[ref]}, Child: {path}")
        else:
            self._offload_registry[ref] = path
```

**验收标准**：
- [ ] 冲突时输出 WARNING 日志
- [ ] 无冲突时子 Agent 所有条目正确 merge

---

#### BUG-06 MCP 会话缓存导致多任务串号 🔴 ⚠️ 串号已修复，设计可议

**位置**：`tool/manager.py`

**原问题**：`ToolManager` 缓存 server 级持久会话，stdio transport 的上下文注入仅在首次创建 session 时发生。单进程串行执行多个研究任务时，后续任务的环境变量可能沿用旧值。

**当前状态**（2026-04-20 核对）：
- **串号已修复**：`_compute_context_fingerprint`（`tool/manager.py:237-256`）为每个 stdio session 记录 env 注入的指纹；`_get_or_create_session`（`tool/manager.py:286-320`）检测到 context 变化时自动 invalidate + 重建 session。
- **双重防线**：工具调用时通过 `_mcp_context` arguments（`tool/manager.py:783-795`）每次传递当前 context，MCP server 即使不读 env 也能拿到正确身份。
- **设计争议（延伸为 BUG-09）**：默认 `_default_env_inject`（`tool/manager.py:62-83`）把顶层 context 全部塞成 env。这个行为本身导致 fingerprint/invalidate 机制的必要性。如果采用"env 仅注入静态 secret、用户身份只走 `_mcp_context` arguments"的纯设计，fingerprint 机制可以整体移除。

**验收标准**：
- [x] 多任务串行执行时，每个任务的 MCP 环境变量独立（fingerprint 保证）
- [ ] 设计决策：是否移除默认 context→env 注入（见 BUG-09）

---

#### BUG-07 配置契约 fail-fast 🟡 ✅ 已修复（2026-04-20，方案 B）

**位置**：`config_schema.py`、`_validate_config()`

**问题**：核心字段缺失时仅记录日志，不阻断初始化，导致框架以错误配置静默运行。

**修复方案**：按 Phase 0 决策的方案 B 实施，核心必需字段缺失时 `raise ConfigurationError`。

**验收标准**：
- [ ] `llm.provider_class` / `llm.model_name` 缺失时初始化阶段直接报错
- [ ] 可选字段缺失时 WARNING + 自动填充默认值

---

#### BUG-08 快速路径与 Grace Turn 首轮冲突 🟡 ✅ 已修复（2026-04-20）

**位置**：`main_loop.py`

**问题**：`total_tool_calls_executed == 0` 时快速路径直接 break，跳过 Grace Turn 恢复机会。对于强制要求工具调用的 Agent（如搜索 Agent），首轮偶发遗忘调用工具会直接终止任务。

**修复方案**：根据 `execution_mode` 条件性启用快速路径：
- `quick` 模式：保留快速路径（直接回答场景合理）
- `standard` / `deep` 模式：禁用快速路径，允许 Grace Turn 恢复

**验收标准**：
- [ ] `deep` 模式下首轮无工具调用时触发 Grace Turn 而非直接退出
- [ ] `quick` 模式下快速路径行为不变

---

### Phase 1 测试补全清单

| 场景 | 测试文件 | 优先级 |
|------|---------|--------|
| Grace Turn 后 LLM 恢复调用工具（计数器重置） | `test_mainloop_tools.py` | P0 |
| 停滞→恢复→再次停滞，消息角色序列合法性 | `test_mainloop_tools.py` | P0 |
| Grace Turn 发生时上下文接近阈值，`manage_context` 被调用 | `test_mainloop_tools.py` | P0 |
| 子 Agent 配置继承（`compact_keep_recent` 与主链一致） | `test_sub_agent.py` | P1 |
| Registry merge 冲突时 WARNING 日志输出 | `test_context_manager.py` | P1 |
| 多任务串行执行时 MCP 环境变量独立 | `test_tool_manager.py` | P1 |
| `deep` 模式首轮无工具调用触发 Grace Turn | `test_mainloop_tools.py` | P1 |
| 中文 nudge 文本路径（`chinese_context=True`） | `test_mainloop_tools.py` | P2 |

---

### 2026-04-20 Review 新发现

#### BUG-09 默认 context→env 注入设计争议 🟡

**位置**：`tool/manager.py:62-83`（`_default_env_inject`）

**问题**：默认注入逻辑把 context 顶层所有 string 字段塞成 MCP stdio subprocess 的环境变量。这带来：
1. 用户身份（user_id, org_id 等）被 bake 进长期存活的 stdio 子进程 env
2. 需要依赖 fingerprint 机制（`tool/manager.py:237-378`）做切换时 invalidate + rebuild
3. 进程列表可能泄露敏感字段
4. 用户理想的设计是：用户身份只通过 tool arguments 的 `_mcp_context` 传递（`tool/manager.py:783-795` 已实现），env 仅用于静态 secret（API key 等启动时固定的值）

**修复方案（选项 A — 推荐）**：移除 `_default_env_inject` 中对 context 的自动遍历注入，仅保留 `TASK_ID` 等系统字段：
```python
def _default_env_inject(ctx):
    server_params = ctx.server_params
    if TASK_CONTEXT_VAR.get() is not None:
        server_params.env["TASK_ID"] = TASK_CONTEXT_VAR.get()
    return server_params
```
同时可以移除 `_compute_context_fingerprint` / `_session_context_fingerprints` / context-change invalidate 逻辑（约 50-80 行简化）。

**替代（选项 C）**：加配置开关 `main_agent.tool_manager.auto_env_inject: false`，默认关闭；保持向后兼容。

**验收标准**：
- [ ] 方案决策（A vs C）
- [ ] `_mcp_context` arguments 路径文档化为唯一推荐的身份传递方式
- [ ] 若选 A：stdio session 可长期复用，无需 invalidate

---

#### BUG-10 asyncio.gather 缺 return_exceptions 🔴 ✅ 已修复（2026-04-20）

**位置**：`main_loop.py:2017-2040`

**问题**：`agent_calls` 路径的 `asyncio.gather` 无 `return_exceptions=True`，CancelledError 传播导致其他并发子 Agent 被取消，MCP 会话泄漏，`_sub_agent_semaphore` permit 不释放 → 后续 spawn 死锁。

**修复**：加 `return_exceptions=True` + 异常归一化为错误元组，保证 offload/transcript 和信号量释放照常完成。

---

#### BUG-11 AgentFactory.close() 异常未隔离 🔴 ✅ 已修复（2026-04-20）

**位置**：`agent_factory.py:357-384`

**问题**：任一 tool_manager `close_sessions()` 抛错就中断后续清理，遗留 subprocess/pipe。

**修复**：逐个 try/except，记录失败集合，保证所有 tool_manager 都被尝试清理。

---

#### BUG-12 TodoTracker / FileStateCache / Transcript 未加锁 🟡 ✅ 已修复（2026-04-20）

**位置**：`todo_tracker.py`、`file_state_cache.py`、`transcript.py`

**问题**：`SessionMemory` 已加 `threading.Lock`，但这三个模块未加。主 Agent 与子 Agent 并发访问时 list/dict 可能 race。

**修复**：三个模块统一加 `threading.Lock`，iteration 路径改为 snapshot。

---

#### BUG-13 scrape_max_length schema/env 不一致 🟡

**位置**：`config_schema.py:248` vs `orchestrator.py:297-300`

**问题**：`scrape_max_length` 在 `MonitoringConfigSchema` 中定义，但运行时通过 `os.getenv("SCRAPE_MAX_LENGTH", ...)` 读取，YAML 配置被忽略。

**修复方向**：在 `MainAgentConfig` 中暴露此字段，读取链统一走 config。

---

#### BUG-14 response_language 未用 Literal 约束 🟢

**位置**：`config_schema.py:354`

**问题**：`response_language: str` 接受任意字符串，容易拼写错误后静默走 fallback。

**修复方向**：改为 `Literal["auto", "Chinese", "English", ...]`，明确可选值集合。

---

#### BUG-15 @file 路径无白名单 🟡 ✅ 复核已实现

**位置**：`input_compiler.py`

**复核结论（2026-04-20）**：多层防护已存在，原 review 误判。

已实现的防护：
1. **敏感文件黑名单**（`input_compiler.py:68-79, 183-192`）：`.env`、SSH keys 等无条件拒绝
2. **可选 allowlist**（`input_compiler.py:87, 194-206`）：配置 `input_process.file_ref_allowed_dirs` 后只读白名单目录
3. **配置字段已暴露**（`config_schema.py:139-142`）：`InputProcessConfig.file_ref_allowed_dirs`
4. **realpath 规范化**（`input_compiler.py:179`）：防符号链接绕过
5. **50KB 大小上限**（`input_compiler.py:213-215`）：防大文件 OOM

默认不限制是刻意设计（向后兼容 CLI 场景）。多租户生产环境在 agent.yaml 显式配置：
```yaml
main_agent:
  input_process:
    file_ref_allowed_dirs: [/app/data, /app/uploads]
```

---

#### BUG-16 硬编码 0.6 未用 CONTEXT_REDUCTION_TARGET_RATIO 🟢 ✅ 已修复（2026-04-20）

**位置**：`llm_call_handler.py:478`

**修复**：替换为 `CONTEXT_REDUCTION_TARGET_RATIO` 常量。

---

#### BUG-17 命名 Sub-Agent offload 孤儿 🔴 ✅ 已修复（2026-04-20）

**位置**：`sub_agent_runner.py` `_run_configured_sub_agent`（`run` 方法）

**问题**：命名 sub-agent（`sub_agents.<name>` 配置的）创建独立 `ContextManager` 并可能产生 offload 文件，但 `finally` 块**不调用** `merge_offload_registry`（只有 spawn 路径做了）。主 Agent 的 `cleanup_offload_files()` 只看自己的 registry，**命名 sub-agent 的 offload 文件全部变孤儿**。

**修复**：
1. `SubAgentRunner.run()` 新增 `parent_context_manager` keyword 参数
2. 创建 sub-agent `ContextManager` 时传入 parent（已触发 inherit_with_override + 共享 offload_dir）
3. `finally` 块调用 `parent_context_manager.merge_offload_registry(context_manager)`，merge 失败仅 WARNING
4. 主 loop 调用点（`main_loop.py:2028`）传入 `self.context_manager`

**验收标准**：
- [x] 命名 sub-agent 产生的 offload 文件能被主 agent `cleanup_offload_files` 清理
- [x] merge 失败不阻塞 sub-agent 返回

---

### BUG-13 scrape_max_length 复核（2026-04-20）

核对代码后发现原 Review 报告误报：`orchestrator.py:292-300` 已经是"YAML config → env → default"的正确读取顺序。`ensure_dict(cfg.main_agent.get("monitoring", {}))` 不会被 Pydantic default 污染。**标记为非 bug，已关闭**。

---

### BUG-14 response_language Literal 复核（2026-04-20）

由于检测和用户自定义都可能返回任意语言名（`detect_language_by_chars` 仅覆盖四种，LLM 可能输出更多），改为**软验证**：`field_validator` 对未知值打 WARNING 但不阻塞，已知语言列表集中在 `_KNOWN_RESPONSE_LANGUAGES`。

---

### Phase 1 出口标准

- [x] BUG-01 ~ BUG-05、BUG-07、BUG-08 修复完成
- [x] Review 新发现 BUG-10、BUG-11、BUG-12 修复完成
- [ ] BUG-06 / BUG-09 设计决策
- [ ] BUG-13 ~ BUG-16 修复
- [ ] 全部回归测试通过（当前 543 passed）
- [ ] `ruff check` 零错误
- [ ] 版本号统一为 `v1.2.3`
- [ ] CHANGELOG 更新，包含配置迁移指南（BUG-07 方案 B 涉及 breaking change）

---

## Phase 2：v1.3.0 契约收敛与架构整理

**目标**：统一框架内部数据契约，建立端到端集成测试框架，为架构演进提供安全网。

**入口条件**：Phase 1 完成，所有高优 Bug 已修复且有回归测试。

### 任务清单

#### 契约统一

| 任务 | 说明 | 优先级 |
|------|------|--------|
| 统一 `tool_definitions` 数据结构 | 消除 MCP server 风格和 builtin tool 单体 dict 的混用，提供统一 `ToolDefinition` 类型 | P0 |
| 定义 `RuntimeSnapshot` 最小字段集 | 支持主/子 Agent 间状态传递和跨会话 resume 的最小接口（不含 SessionMemory 深度重构） | P0 |
| 收口结果生命周期契约 | 明确 tool result 从产生→offload→restore 的完整状态机，文档化并加断言保护 | P1 |
| 理清 Fast-path 与 Grace Turn 状态机 | 消除两者的逻辑冲突，用状态机图文档化互斥关系 | P1 |

#### 语言同步修复

| 任务 | 说明 |
|------|------|
| 统一 `orchestrator.py` 和 `main_loop.py` 的语言同步逻辑 | `auto` 模式下检测后同步更新 `PromptBuilder`、`inline_skill_selector` 等组件 |
| 验证 `on_hook_offload_evidence_prep` 返回非 string 时的 WARNING 日志 | 低成本改善 hook 实现者的调试体验 |

#### 端到端测试框架

| 任务 | 说明 | 优先级 |
|------|------|--------|
| 搭建 E2E 测试基础设施（mock LLM + mock MCP server） | 为 `DeepResearch.run()` 提供可测试的环境 | P0 |
| 覆盖核心链路：run → offload → cleanup | 验证完整生命周期 | P0 |
| 覆盖异常场景：LLM 调用失败、token 溢出、子 Agent 异常退出 | 至少 3 个异常场景 | P1 |
| 覆盖 Grace Turn 与反思轮的交互 | 防止无限循环风险 | P1 |

#### 架构整理（轻量）

| 任务 | 说明 |
|------|------|
| `MainLoopRunner` 接口抽取 | 仅抽取策略接口（`ModeResolver`、`ResultLifecycleManager`），实现不变；为 Phase 3 拆分做准备 |
| `parent_context_manager` 参数类型注解 | 加 `ContextManager \| None` 注解，启用静态分析 |
| 条件性：`@file` 安全边界加固 | 如 Phase 0 安全审查确认风险，引入 Project-root Allowlist 机制 |

### Phase 2 出口标准

- [ ] `ToolDefinition` 统一数据结构，无特殊分支处理
- [ ] `RuntimeSnapshot` 接口已定义并支持基本 resume 场景
- [ ] E2E 测试覆盖核心链路 + 至少 3 个异常场景
- [ ] Fast-path 与 Grace Turn 无状态冲突（E2E 测试验证）
- [ ] `MainLoopRunner` 策略接口已声明（`ModeResolver` 等接口定义）

---

## Phase 3：v1.4.0 质量体系与长期演进

**目标**：建立可持续的质量保障体系，完成架构拆分，为记忆系统和高级 Agent 能力奠定基础。

**入口条件**：Phase 2 完成，E2E 测试就位，接口已抽取。

### 任务清单

#### 质量体系

| 任务 | 说明 |
|------|------|
| 建立分层 benchmark 体系 | 单元 → 集成 → E2E → 性能，CI 门禁全部通过才能合入 |
| 完善公共入口测试覆盖 | `DeepResearch.run()`、`AgentFactory.run_batch()` 等 |
| 规范 release 工程流程 | 版本号 bump 自动化、CHANGELOG 生成、tag 创建、CI/CD 门禁 |

#### 架构演进

| 任务 | 说明 | 风险 |
|------|------|------|
| `MainLoopRunner` 完整策略拆分 | 基于 Phase 2 抽取的接口，拆分为 `ModeResolver`、`ResultLifecycleManager` 等独立模块 | 高风险重构，需 E2E 测试作为安全网 |
| `RuntimeSnapshot` 完整实现 | 支持跨会话 resume，含 `SessionMemory` 基础序列化 | 序列化兼容性问题 |
| 温度覆盖 `ContextVar` 隔离审查 | 确认多客户端/父子 Agent 间是否存在串扰，必要时改为实例字段 | 需确认子 Agent 调用方式（`create_task` vs `await`） |

#### 长期演进（RFC 阶段）

| 方向 | 说明 |
|------|------|
| 记忆系统深度重构 | `SessionMemory` + `LongTermMemory` 的统一接口、持久化策略、跨 session 检索 |
| 高级 Agent 能力 | 多 Agent 协作协议、Agent 间通信、动态工具发现 |

### Phase 3 出口标准

- [ ] CI 门禁包含分层测试（单元 + 集成 + E2E），全部通过才能合入
- [ ] `RuntimeSnapshot` 支持完整跨会话 resume（E2E 测试验证）
- [ ] `MainLoopRunner` 已拆分为至少 3 个独立模块，各模块有独立单元测试
- [ ] Release 流程自动化（版本号 bump、CHANGELOG 生成、tag 创建）
- [ ] 记忆系统重构 RFC 完成评审

---

## 待决策事项（Open Questions）

| 问题 | 决策期限 | 影响范围 |
|------|---------|---------|
| 配置契约修复策略：方案 A（渐进迁移）vs 方案 B（一次性收口） | Phase 0 | Phase 1 实施方式 |
| `@file` 指令是否存在安全越权风险 | Phase 0 | 是否提升至 Phase 1 |
| 子 Agent 配置继承：完全继承 vs `inherit_with_override` | Phase 1 启动前 | BUG-04 设计方向 |
| `RuntimeSnapshot` 最小字段集范围 | Phase 1 完成后 | Phase 2 工作量 |
| `MainLoopRunner` 拆分深度与时机 | Phase 2 完成后 | Phase 3 风险评估 |
| 温度覆盖 `ContextVar` 串扰风险 | Phase 3 启动前 | 是否需要架构变更 |

---

## 历史规划文档归档说明

| 文档 | 状态 | 说明 |
|------|------|------|
| `docs/15-technical-roadmap.md` | DEPRECATED | 使用 v1.2.3/v1.3/v1.4 版本号体系，已被本文档取代 |
| `docs/17-repo-architecture-review.md` | DEPRECATED | 使用 P0/P1/P2 优先级体系，已被本文档取代 |
| `docs/19-framework-issues-and-iteration-plan.md` | DEPRECATED | 使用阶段体系，已被本文档取代 |
| `docs/arena-review-offload-pipeline.md` | 历史参考 | Offload 流水线专项 Arena Review，保留作为背景材料 |

---

*本文档由 Arena 三模型评审（Claude Opus / Gemini 3.1-pro-preview / DeepSeek-R1）综合分析生成，2026-04-17。*
