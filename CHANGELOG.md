# Changelog

## v1.3.0 (2026-04-28)

**HITL (Human-in-the-Loop) — durable suspend / resume on tool boundaries**

Adds first-class human-approval support to the runtime. Approval hooks call
`ctx.runtime.wait_for_human(...)` from inside `on_tool_start`; the framework
either delivers the decision in-process or persists a `RuntimeSnapshot`
checkpoint and returns `awaiting_human`. The same checkpoint is later
delivered to `DeepResearch.resume_with_human_decision(...)` to continue
execution from the exact tool boundary.

Phase 0 / 1 / 2 from `docs/23-hitl-design.md` are all in this release.

### Phase 0: Snapshot infrastructure (foundation)

- **Hook system async-aware**: `HookRegistry.call()` is now `async`;
  synchronous hooks still execute natively (no `to_thread`). `call_sync()`
  preserves the sync entry point for hook sites that can't go async.
  All 27 framework call sites migrated.
- **`on_suspend` / `on_resume` lifecycle hooks** for resource
  cleanup/rebuild around durable execution.
- **Module snapshot/restore contract**: `ContextManager`, `ExecutionMonitor`,
  `InlineSkillSelector` each expose `snapshot()` / `restore()` for runtime
  state capture.
- **ContextVar save/restore**: `LLMProviderClientBase` /
  `DeepSeekOpenRouterClient` / `sub_agent_runner` expose
  `save_contextvar_state` / `restore_contextvar_state` so framework-owned
  ContextVars survive a process restart.
- **`RuntimeSnapshot` dataclass + `build_snapshot` / `restore_snapshot`**
  primitives in `core/hitl/runtime_snapshot.py` with `schema_version=1`.

### Phase 1: Synchronous HITL surface

- **`HumanDecision` / `PendingHumanRequest` / `RunResult`** data types
  (`core/hitl/types.py`).
- **`PendingHumanException`** runtime control-flow exception
  (`core/hitl/exceptions.py`); transparently propagated through hook
  chain, tool executor, sub-agent runner, and concurrent `asyncio.gather`
  fan-in paths.
- **`PendingStore` Protocol + `InMemoryPendingStore`** —
  `Future`-based rendezvous between `wait_for_human` and the approver.
- **`RuntimeFacade.wait_for_human(...)`** — main HITL entry point;
  short-circuits to auto-approved when `cfg.hitl.enabled=False`.
- **`HookContext.runtime`** resolves the current facade via ContextVar so
  `HookContext` stays invariant.
- **`on_await_human` hook** for business-side notification (Slack / email
  / webhook).
- **Sub-agent restriction**: `_is_sub_agent_var=True` forces
  synchronous-only path; durable suspend is disallowed inside sub-agents
  (design-doc Phase 2 contract). Deferred to v1.4.0 workflow layer.

### Phase 2: Asynchronous HITL + checkpoint/resume

- **`FilesystemCheckpointStore`** with atomic writes (`tempfile` +
  `os.replace`), path-traversal guards, and `sweep_expired()` for stale
  request cleanup.
- **`wait_for_human` timeout → `PendingHumanException`** (main agent
  path); the request stays open in the pending store so resume can
  deliver the decision after a process restart.
- **`MainLoopRunner._build_runtime_snapshot` / `_restore_runtime_snapshot`**
  + `_HitlLiveState` dataclass updated each turn.
- **Outer `try/except PendingHumanException`** in `MainLoopRunner.run`
  builds the snapshot, fires `on_suspend`, and re-raises to the pipeline
  layer for persistence.
- **`Pipeline.execute_task_pipeline`** catches and persists; returns 6-tuple
  with `(answer, boxed, log_path, status, checkpoint_id, pending_request)`.
- **Concurrent batch drain-then-suspend**:
  `_execute_regular_tools_concurrent` waits for the rest of the batch and
  records completed tools' offload refs before raising the first pending.
- **`MainLoopRunner.run_from_tool_cursor(snapshot, decision)`** resumes
  from the tool cursor: skips the LLM call for the paused turn, skips
  `on_tool_start` re-entry, runs the pending tool with
  `effective_arguments` merged from the approver's `decision.payload`,
  then re-enters the normal turn loop with `skip_init=True` so restored
  module state survives.
- **`DeepResearch.resume_with_human_decision(...)`** /
  **`AgentFactory.resume_with_human_decision(...)`** /
  **`execute_hitl_resume_pipeline`** — three-layer resume entry. Auto-resolves
  `task_description` from the snapshot.
- **`HitlConfig`** in `config_schema.py`:
  - `enabled` (default True)
  - `checkpoint_dir` — explicit override; falls back to `output_dir` →
    log directory.
  - `sweep_on_start` — runs `sweep_expired_checkpoints` on
    `AgentFactory.initialize()`.
- **`AgentFactory.sweep_expired_checkpoints()`** public API for
  out-of-band cleanup schedulers.

### TaskResult extensions

`TaskResult` (and the internal `AgentTaskResult`) gain two HITL fields:

- `checkpoint_id`: populated when `status == "awaiting_human"`.
- `pending_human_request`: the `PendingHumanRequest` describing what the
  approver must decide on.

`status` adds the `"awaiting_human"` state alongside `completed` /
`failed`. Existing call sites that only check `status == "completed"`
continue to work; treat `awaiting_human` separately when wiring HITL.

### Tests

- 33 HITL tests across `test_runtime_snapshot.py` (Phase 0 golden, 13),
  `test_hitl_phase1.py` (sync surface, 11), `test_hitl_phase2.py`
  (checkpoint store + config wiring, 14).
- 4 end-to-end resume integration tests in `test_hitl_resume_e2e.py`
  drive `run_from_tool_cursor` with mocked LLM + tool executor and
  verify module state survives, the LLM is not called for the paused
  turn, rejection injects a tool error, and `on_resume` fires before
  the tool executes.
- 708 / 708 passing.

### Migration notes

- No breaking change to existing call sites. `DeepResearch.run()` still
  returns `TaskResult`; the new `awaiting_human` status is an additive
  third state. Apps that only consume `completed` / `failed` continue
  to work.
- v1.4.0 will rename `TaskResult` to `RunResult` as a breaking change
  (per `docs/23-hitl-design.md`); using the new HITL fields today is
  forward-compatible.

### Known limitations (deferred to Phase 3 / v1.4.0)

- Concurrent batch's `_offload_refs` extraction relies on the field-name
  convention from `_maybe_offload_result`; not yet decoupled into a
  ContextManager API.
- `CheckpointStore` / `PendingStore` are Protocols but only one
  implementation each ships (filesystem / in-memory). Pluggable
  Redis / Postgres backends are Phase 3.
- Batch HITL approval (multiple pendings on one turn) is single-pending
  in v1.3.0; first decides, others wait for resume. Phase 3 evolves
  `effective_arguments` to `dict[tool_call_id, dict]`.
- `on_human_request_created` audit event + transcript wiring deferred
  to Phase 3.

## v1.2.6 (2026-04-22)

**Profile 架构落地 — 通用 Agent Runtime + 可插拔 Profile / Strategy 层**

这是一个向后兼容的架构新增版本。把之前散落在主循环里 29 处 `is_deep_mode` / `is_quick_mode` 研究专属分支，重构成可插拔的 `Profile` 抽象，并把"大工具结果细节保鲜"的三种机制统一到 `MemoryExtractionStrategy` 层。

定位转向：从"研究型 Agent 框架"向"通用 Agent Runtime + research 作为高级执行 profile"演进。research 行为由 `DeepResearchProfile` 聚合并保持等价，其他 profile 可零侵入接入。

### Profile 抽象（`core/profiles/`）

- **`Profile` ABC**：10 个生命周期钩子覆盖 agent_start / turn_start / reflection / LLM 响应 / pre-post tool / verify / final answer；默认 pass-through
- **`StandardProfile`**：通用 agent profile，所有 lifecycle 钩子空实现
- **`DeepResearchProfile`**：聚合研究专属决策
  - `should_inject_reflection` / `should_run_verify` / `should_create_task_plan` / `should_process_inline_skills`：mode 感知的 policy 决策
  - `needs_final_summary`：deep + 有工具调用时强制 summary，其他情况遵循用户配置
  - 配置字段：`reflection_enabled` / `enable_verify` / `generate_summary` / `auto_task_plan`
- **Registry**：`resolve_profile` 接受 str / class / instance / None；`register_profile` 支持自定义 profile
- **Orchestrator** 按 `execution_mode` 自动路由：`deep` / `auto` / `task_engine.enabled` → `DeepResearchProfile`，其他 → `StandardProfile`

### Memory Extraction Strategy 层（`memory_extraction/`）

统一"长任务细节保鲜"的可插拔扩展点，所有 strategy 通过 `profile.extraction_strategies` 组合：

- **4 个触发点**：`on_llm_response` / `on_tool_result` / `on_compact` / `on_offload`
- **3 个默认 strategy**（StandardProfile 含前两个，DeepResearchProfile 全含）
  - `OffloadEvidenceStrategy`：`<offload_evidence ref="...">` 绑定到 offload registry（所有 profile）
  - `SummaryEvidenceStrategy`：LLM 压缩 summary 的 `## Evidence` 段抽取（所有 profile）
  - `EvidenceTagStrategy`：自由 `<evidence>` tag 抽取到 session_memory（仅 DeepResearch）
- **2 个 opt-in strategy**（用户按需配置）
  - `FactExtractionStrategy`：工具结果回来后用轻量 LLM 抽 facts，内置 `(tool, content_hash)` 去重集合，resume-safe
  - `SummarizeOnCompactStrategy`：LLMSummarize 的整段 summary 作为 compact_anchor 存 session_memory（LangGraph / Mastra "memory is summary" 风格）
- **Vector store / RAG 接入**不作为内置 strategy 提供（每个 vector store client / embedding model / chunk 策略差异太大，框架给不出真正通用的抽象）。用户继承 `MemoryExtractionStrategy` 在项目内实现，通过 `register_strategy` 注册即可，参考 `docs/26-memory-extraction-strategy.md` 的"用户自定义 Strategy 示例"章节
- **Snapshot / Restore**：每个 strategy 独立 state，Profile 递归聚合，为后续 HITL resume 准备
- **Registry**：`resolve_strategy` / `register_strategy` / `list_strategies` 支持自定义

### Runtime 改造

- `MainLoopRunner` 加 `profile` 字段 + `_build_profile_ctx` / `_build_extraction_ctx` helper
- **29 处 `is_deep_mode` / `is_quick_mode` 功能性分支 → 1 处**（仅保留 adaptive 路由的 runtime 状态同步）
- 具体迁移：
  - `main_loop.py:807` task planner injection → `profile.should_create_task_plan`
  - `main_loop.py:1209` inline skill selection → `profile.should_process_inline_skills`
  - `main_loop.py:1668` reflection checkpoint → `profile.should_inject_reflection`
  - `main_loop.py:1780` verify checkpoint → `profile.should_run_verify`
  - `main_loop.py:1801` summary policy → `profile.needs_final_summary`
- `_maybe_offload_result` 转 async，triggers `profile.run_strategies_on_offload`
- `context_manager` / `window_strategy` 加 profile 注入路径，LLMSummarize 触发 `on_compact` strategy 链
- Runtime 统一对最终输出做 tag 清理（`<evidence>` / `<offload_evidence>`），strategy 只读写 session_memory

### 配置 API

```python
# 内置 profile
dr = DeepResearch(profile="deep_research", profile_config={...})

# 自定义 profile
dr = DeepResearch(profile=MyProfile(), profile_config={...})

# strategy 追加（保留默认）
profile_config={"extraction_strategies_extra": [FactExtractionStrategy(...)]}

# strategy 完全覆盖
profile_config={"extraction_strategies": [MyCustomStrategy(...)]}
```

### 默认行为调整（潜在影响）

**Offload 默认关闭** — `DEFAULT_RESULT_OFFLOAD_THRESHOLD` 从 `5000` 改为 `0`（关闭）：

- **原因**：offload 对环境有预期（output_dir 可写、文件系统可用），不应默认打开
- **示例项目**：`example_project/config/*.yaml` 里一直显式设 `result_offload_threshold: 5000`，不受影响
- **你需要做什么**：如果你之前依赖默认值，显式在配置里加 `main_agent.context_manager.result_offload_threshold: 5000`（或其他合适的字节数）

**`[OFFLOAD PREP]` sidecar 注入改为跟随 `OffloadEvidenceStrategy` 的存在**：

- 原来：只要有 offload 候选就注入 sidecar prompt 要求 LLM 产 `<offload_evidence>`，无论 profile 是否会抽取
- 现在：只有 profile 的 `extraction_strategies` 里含 `OffloadEvidenceStrategy` 时才注入
- **原因**：如果没有 strategy 消费 tag，sidecar 白占 prompt tokens
- **StandardProfile 和 DeepResearchProfile 的默认 strategies 都含 `OffloadEvidenceStrategy`**，所以 98% 用户无感
- 仅当你显式用 `extraction_strategies=[...]` 覆盖且不含 `OffloadEvidenceStrategy` 时，sidecar 才不再注入（这是期望行为）

### 向后兼容

- 既有 `execution_mode` / `task_engine` / `generate_summary` / `task_engine.enabled` 配置不变，通过 `Orchestrator` 路由自动映射到合适的 profile
- Research 场景行为 100% 等价（`DeepResearchProfile` 默认带 `[OffloadEvidence, SummaryEvidence, EvidenceTag]` strategies，对应原硬编码抽取集合）
- 543 + 95 个新测试 = 638 tests pass, zero regression

### 设计文档

- `docs/21-industry-framework-analysis.md`：业界框架对比 + 定位转向依据
- `docs/22-profile-boundary.md`：Runtime / Profile 边界盘点（9 个核心模块全量分类）
- `docs/25-profile-contract.md`：Profile 契约设计（10 个决策汇总）
- `docs/26-memory-extraction-strategy.md`：Strategy 层最终设计（10 个决策 + 3 阶段实施）

### Roadmap 调整

原 `docs/20-roadmap.md` 把 Profile 拆分放在 v1.4.0，但工程顺序上"先拆 profile 再收口 contract"更合理（profile 明确了 runtime contract 的消费方边界）。因此本次提前实施了原 v1.4.0 的 Profile 主题。原 v1.3.0 的 "Runtime Contract 收敛"（统一结果生命周期 / 配置契约全量收口 / 端到端回归测试）重排到下一个 minor 版本。

### 测试

**638 passed**（v1.2.5 时 543；Phase 2a +33，Phase 2b +18，Phase 2c +17，Phase 1 测试修正 +1）

## v1.2.5 (2026-04-20)

**循环退出机制对齐 Claude Code** — 移除 grace turn，使用 `stop_reason` / `finish_reason` 作为唯一退出信号。

### 背景

v1.2.3 引入的 grace turn 机制（`771fcf7`）让框架在 LLM 不再调工具时主动注入 nudge 再问一轮，本意是救援 LLM 中途误判完成的场景。实际 trade-off 反了 —— 100% 任务多等 ~6s 换 <1% 的补救收益，且补救效果本身存疑。本次回归 v1.2.2 之前的设计思路，并对齐 Claude Code 的做法：**信任 LLM 通过 API 明确表达的意图**。

### 破坏性变化（Breaking）

- **删除 `MAX_CONSECUTIVE_NO_TOOL_TURNS` 常量** — 不再按"连续无 tool 轮次"终止
- **删除 `MT.NO_TOOL_NUDGE` 消息类型** — nudge 机制整体移除
- **删除 grace turn 所有逻辑**（`main_loop.py` 约 75 行）
- 框架不再主动注入 nudge，LLM 不调工具就是完成信号

**迁移**：依赖 grace turn 行为的用户几乎没有（机制只存在于 v1.2.3/v1.2.4）；如果项目有测试断言"3 次 LLM 调用（含 grace 确认）"，需改为"2 次"。

### Provider 层改动

四个 provider 的 `process_llm_response` 从"永远返回 `should_break=False`"改为按 API 字段返回真实值：

- `claude_anthropic_client.py`: `should_break = stop_reason != "tool_use"`
- `openai_compatible_client.py`: `should_break = finish_reason != "tool_calls"`
- `gpt_openai_client.py`: 同上
- `deepseek_openrouter_client.py`: 委托基类，无需改动

### Main Loop 改动

- 删除 grace turn 整块（counter + nudge + token 预算检查 + microcompact 回退）
- 新增防御性检查：`should_break=True` 但响应里仍带 `tool_use` block（不规范 provider），先执行工具再退出
- 反思豁免检查前移：反思轮 LLM 只输出反思文字（`stop_reason=end_turn`），不应被 should_break 提前截断，改为在 should_break 之前 continue 给 LLM 下一轮

### 相关 Bug 影响

BUG-01 / BUG-02 / BUG-03 / BUG-08 在 v1.2.4 的修复代码全部删除 —— 源头（grace turn）不存在，这些 bug 不再可能发生。docs/20-roadmap.md 标记为 "🗑️ 代码移除"。

### Verification

- 543 tests pass，零 regression
- 简单任务延迟下降：一次 tool call 的查询从 ~20s 缩到 ~14s（省 1 次 LLM 调用）

## v1.2.4 (2026-04-20)

**Phase 1 稳定性修复** — 基于代码 Review 定位的运行时安全与配置契约修复。

### Runtime Safety
- **BUG-01 Grace Turn context budget check**: 注入 nudge 前按 token 比率检查，超阈值走完整 `manage_context`（含 summarize），否则降级为 microcompact
- **BUG-08 Fast-path gating**: `total_tool_calls_executed==0` 早退只在 `quick` 模式生效；`standard`/`deep` 模式走 grace turn 恢复
- **BUG-10 gather safety**: `agent_calls` gather 加 `return_exceptions=True` + 异常归一化，防 CancelledError 泄漏 MCP 会话和信号量
- **BUG-11 AgentFactory.close() isolation**: 每个 tool_manager close 隔离 try/except，一个失败不阻塞其他清理
- **BUG-12 concurrency locks**: TodoTracker / FileStateCache / Transcript 加 `threading.Lock`，迭代路径快照化

### Sub-Agent Lifecycle
- **BUG-04 inherit_with_override**: 子 Agent ContextManager 四层合并（defaults ← main_agent ← parent 快照 ← sub_agent 覆盖），stale YAML key 防御过滤
- **BUG-17 named sub-agent offload orphan**: `SubAgentRunner.run()` 接受 `parent_context_manager`；finally 合并 offload registry，命名 sub-agent 的 offload 文件能被父 Agent 的 cleanup 清理

### Config Contract
- **BUG-07 fail-fast (scheme B)**: critical 字段（`llm.provider_class` / `llm.model_name`）缺失抛 `ConfigValidationError`；non-critical 字段 WARNING + 默认值；catch-all 已移除
- **BUG-14 response_language 软校验**: `field_validator` 对未知语言 WARNING 但不阻塞，保留检测 fallback 和自定义语言空间
- **BUG-16 constant extraction**: `llm_call_handler.py` 硬编码 `0.6` 替换为 `CONTEXT_REDUCTION_TARGET_RATIO`

### Docs
- `docs/20-roadmap.md`: Phase 1 状态快照 + BUG-01~17 核验记录 + BUG-06/09/15 设计决策留档
- 543 tests 全过，零 regression

## v1.2.3 (2026-04-17)

**Runtime isolation & sub-agent lifecycle** — 见对应 commit。

## v1.2.2 (2026-04-14)

**Evidence extraction & sliding window offload** — 见对应 commit。

## v1.2.1 (2026-04-10)

**on_final_answer hook & generate_summary config** — 见对应 commit。

## v1.2.0 (2026-04-08)

**Phase 1 内核收敛完成** — 运行时隔离、消息类型系统、稳定性修复、集成测试骨架。

### Runtime Isolation (`AgentRuntime`)
- 新增 `core/agent_runtime.py`：轻量容器，持有实例级 `HookRegistry` + `ConfigLoader`
- 重构 9 个消费模块接受注入 hooks（llm_call_handler, tool_result_formatter, message_interceptor, prompt_builder, sub_agent_runner, context_manager, tool/manager, tool_utils, orchestrator）
- `load_project_hooks()` 支持可选 `hook_registry` 参数
- 多 `DeepResearch` 实例并行运行时 hooks/config_loader 完全隔离
- 全局单例保留为向后兼容 fallback

### Message Type System (`MT`)
- `constants.py` 新增 `MT` 类：20 种消息类型 + `PROTECTED_MESSAGE_TYPES` frozenset
- 24 个消息创建点统一打标 `_type` 字段
- `window_strategy` / `context_manager` / `message_utils` 优先检查 `_type`，keyword 为 fallback
- `_is_protected_message(msg)` 替代脆弱的 `_is_system_message(content)`

### Stability Fixes
- `SessionMemory`: 所有变更方法加 `threading.Lock`，`to_context_string()` 基于快照
- `orchestrator.py`: 修复 2 处 silent `except Exception: pass`，改为日志 fallback
- `tool/manager.py`: 5 处 `except Exception` 窄化为具体异常类型
- 修复 deadlock、context coupling、hooks 全局污染问题
- 修复空 text content block 和空 assistant content 导致的 API 错误
- 修正 `agent_deep_research.yaml` model_name 格式

### Integration Test Skeleton (491 tests)
- `test_agent_runtime.py`: AgentRuntime 隔离、全局 fallback、setup_hook_defaults、load_project_hooks
- `test_hook_injection.py`: 6 个消费模块 hooks 注入链验证
- `test_message_types.py`: MT 类型系统、保护消息判断、keyword fallback
- `test_session_memory.py`: SessionMemory 线程安全、去重、溢出截断
- `test_mainloop_tools.py`: 主循环 + 工具执行链（工具调用→结果回注→下轮 LLM）
- `test_compact_offload.py`: compact + offload + resume round-trip
- `test_long_term_memory.py`: LongTermMemory store/recall/forget 生命周期 + 持久化
- `test_sub_agent.py`: 子 Agent spawn + return + 上下文隔离

### Other
- Deep research config for Sonnet 4.6
- Pluggable offload backend via `on_result_offload` / `on_result_restore` hooks

## v1.1.1 (2026-04-07)

### New Features

#### Claude Code Skill 兼容
- **SkillCommand** (`skills/skill_command.py`): 统一 Skill 数据模型，兼容 Claude Code `SKILL.md` frontmatter + 遗留格式
  - `$ARGUMENTS/$0/$name` 参数替换（复刻 Claude Code `argumentSubstitution.ts`）
  - `${CLAUDE_SKILL_DIR}` 模板变量
  - `` !`cmd` `` 动态内容预处理
  - `paths:` glob 条件激活
  - `context: fork` 子 agent 隔离执行
  - `allowed-tools` 工具限制
  - Budget-aware catalog（1% context window, 250 char/skill）
- **SkillLoader** (`skills/skill_loader.py`): 多源扫描 `.claude/skills/` + `config/skills/definitions/`，后覆盖前去重
- **Meta message 注入**: `injection_mode: meta_message` 通过 hidden user message 注入 skill 内容（Claude Code 模式）

#### Auto 模式 + Adaptive Thinking
- **LLMRouter** (`core/llm_router.py`): 统一路由入口，结构信号 → hook → LLM 分类 → 默认
  - `on_route_classify` hook: 覆盖分类逻辑
  - `on_route_apply` hook: 覆盖应用逻辑（mode + reasoning_effort + thinking_params）
  - Hook 修改 effort 后自动重新生成 thinking_params
- **Claude Adaptive Thinking**: `thinking={"type": "adaptive"}` + `output_config={"effort": "low/medium/high"}`
  - auto 模式: quick→low, standard→medium, deep→high
  - fixed 模式: `budget_tokens`（兼容旧模型）
  - none 模式: 不注入 thinking
- **GPT-5 reasoning_effort**: 通过 `get_thinking_params()` 统一注入
- **Provider 抽象**: `supports_adaptive_thinking()` / `get_thinking_params()` 方法，子类覆盖

### Config (New Fields)
```yaml
main_agent:
  llm:
    thinking_mode: auto          # auto | adaptive | fixed | none
    reasoning_effort: medium     # low | medium | high
    router_model: null           # LLM 分类模型（可选）
  skill_selection:
    injection_mode: system_prompt  # system_prompt | meta_message
    catalog_budget_pct: 0.01
    description_max_chars: 250
    claude_code_skills_dirs: []
```

### Hooks (New)
- `on_route_classify`: 任务复杂度分类
- `on_route_apply`: 路由结果应用（mode + reasoning_effort）

## v1.1.0 (2026-04-01)

### New Modules
- **Deferred Tools** (`core/deferred_tools.py`): 工具数超阈值时延迟加载 schema，通过内置 `tool_search` 按需解析
- **Transcript** (`core/transcript.py`): 结构化 JSONL 事件日志，支持 replay 和调试
- **Input Compiler** (`core/input_compiler.py`): 查询预处理链（URL 提取、@file 展开、on_query_compile hook）
- **File State Cache** (`core/file_state_cache.py`): LRU 文件内容缓存，父/子 Agent 共享

### Context Management
- **Microcompact**: 每轮零成本清理旧 tool_result（LLM 调用前）
- **Session Memory Compaction (L1.5)**: 零 LLM 成本压缩，利用 SessionMemory findings
- **Compact Circuit Breaker**: LLMSummarize 连续失败 3 次后自动跳过
- **结构化卸载标记**: `[OFFLOADED:]` 标记防止二次压缩
- **内容恢复**: resume 场景自动恢复已卸载文件

### Tool Execution
- **并发工具执行**: search/scrape/fetch/read/calc 类工具并行运行
- **结果完整性检查**: 校验每个 tool_use 都有对应 tool_result
- **原生工具调用**: 未解析的工具名生成错误反馈而非静默跳过

### Cost Control
- **Token Budget Tracker**: 每任务 token 限额，80% 警告 + 100% 硬停
- **Provider Usage Tracking**: 真实 prompt/completion token 记录（`_record_usage()`）
- **Prompt Caching**: system prompt 在 `__DYNAMIC_BOUNDARY__` 处拆分，静态部分缓存
- **System Prompt Section Cache**: 静态段缓存，动态段重算

### Resilience
- **输出截断恢复**: `finish_reason=length` 时注入 continuation prompt
- **非标准 finish_reason 规范化**: OpenRouter `"error"` 等映射为 `"stop"`
- **子 Agent Prompt 复用**: spawn 子 Agent 复用父级渲染后的 system prompt

### Config (New Fields)
- `deferred_tools_threshold: 20` (0=禁用)
- `transcript_enabled: true`
- `task_token_budget: 0` (0=无限制)

## v1.0.5 (2026-03-31)

### Documentation Fixes (CRITICAL)
- 全文修正 `flash` → `quick`（执行模式名称）：docs/01-quick-start.md, docs/02-configuration.md, example_project/README.md
- 全文修正 `deep_research:` → `task_engine:`（配置字段名）：docs/01-quick-start.md, docs/02-configuration.md, docs/12-memory-and-todo.md
- `ResearchResult` → `TaskResult`（类名）：docs/01-quick-start.md，补充 v0.3 新增字段（turns, tool_calls, error_type, perf_metrics, checkpoints）
- CLI 示例修正 `--flash` → `--quick`

### Bug Fixes
- **memory.py**: `recall()` 的 `access_count` 更新和 `_save()` 移入 `threading.Lock` 内，修复竞态条件
- **transcript.py**: `record()` 兼容 enum 和字符串类型的 event_type；`load()` 捕获损坏 JSONL 行，跳过而非崩溃
- **input_compiler.py**: 文件引用正则 `\w{1,10}` → `\w+`，不再截断超长扩展名
- **agent_factory.py**: `initialize()` 开头校验 `cfg is not None`，防止空配置 AttributeError

## v1.0.4 (2026-03-30)

### Documentation
- **CLAUDE.md**: 补充 9 个未文档化的 core 模块（deferred_tools、input_compiler、transcript、memory、todo_tracker、pipeline、agent_factory、message_utils、tool_result_formatter）+ 4 个 utils 模块 + logging 模块
- **README.md**: Hook 表补充 `on_thinking_generate` 和 `on_message_intercept`
- **CHANGELOG.md**: 补充 v1.0.0 ~ v1.0.3 完整变更记录

### Fixes
- `__version__` 同步：pyproject.toml / `mem_deep_research/__init__.py` / `mem_deep_research_core/__init__.py` 统一版本号
- `mem_deep_research/__init__.py` docstring 模型名更新为 `claude-sonnet-4-20250514`

## v1.0.3 (2026-03-30)

### Bug Fixes
- **CRITICAL**: `provider_client_base` — `handle_max_turns_reached_summary_prompt` 不再 `pop()` message_history，修复 summary 重试时永久丢失原始 user 消息
- **CRITICAL**: `llm/util` — 流式响应异常不再被静默吞掉；无数据时 re-raise，有部分数据时标记 `finish_reason="error"`
- **HIGH**: `deepseek_openrouter_client` — `_pending_tool_list` / `_native_tool_name_map` 改为 `contextvars`，消除并发请求竞态
- **HIGH**: `sub_agent_runner` — `tool_manager` 为 None 时跳过 `ToolExecutor` 创建，避免 `AttributeError`
- **HIGH**: `provider_client_base` — `_filter_message_history` 裁剪时保留首条 user 消息（任务描述）
- **HIGH**: `secure_context` — 占位符正则从 `\w+` 放宽为 `[^\]]+`，支持连字符/点号字段名
- **MEDIUM**: `window_strategy` — `_collect_old_messages` 修复第一条 user 消息被跳过不参与压缩
- **MEDIUM**: `main_loop` — 移除并发子 Agent 的 `return_exceptions=True` 死代码分支
- **MEDIUM**: `sub_agent_runner` — `spawn()` 添加 `finally` 块记录 message_history
- **MEDIUM**: `orchestrator` — `expose_sub_agents_as_tools` 消除重复，修复预加载时子 Agent 工具缺失

## v1.0.2 (2026-03-30)

### Bug Fixes
- Native tool name resolution for hyphenated MCP server names (e.g., `tool-searching-serper`)

## v1.0.1 (2026-03-30)

### New Features
- **Native Tool Calling**: Sonnet 4.6 原生 tool_calls 支持（DeepSeek/OpenRouter client）
- Tool name 双向映射（MCP 名 ↔ LLM 原生名）

## v1.0.0 (2026-03-30)

### Stable Release
- 框架 API 稳定化，标记 v1.0.0
- CLI 工具：`mem-deep-research init / run / test`
- PyPI 发布：`pip install mem-deep-research`
- 完整文档：15 篇文档覆盖架构、配置、开发指南
- 264 个测试用例

## v0.3.0 (2026-03-26)

### New Features
- **Sub-Agent System**: spawn_agent built-in tool + pre-configured sub_agents (YAML), parallel execution via asyncio.gather
- **Execution Modes**: auto/flash/standard/deep — framework auto-selects based on task complexity
- **TodoTracker**: Built-in update_todo tool for task tracking, independent of message_history, survives context compression
- **Memory System**: SessionMemory (short-term findings/strategies) + LongTermMemory (cross-session persistence)
- **Skill Progressive Loading**: First turn loads catalog only, full content on demand via <next_skills>
- **Result Offloading**: Large tool results (>5000 chars) saved to filesystem, context gets summary reference
- **Context Compression Awareness**: [CONTEXT NOTE] injected when compression occurs
- **Concurrency Control**: max_concurrent_subagents (default 3) via asyncio.Semaphore

### Improvements
- Comprehensive monitoring logs at every decision point (task_failed transitions, execution mode, per-turn summary)
- Research preset prompt updated with spawn_agent and update_todo guidance
- Built-in tools rendered in MCP server format for correct system prompt display

### Bug Fixes
- SummaryHandler no longer falsely sets task_failed during context reduction retries
- Built-in tools (spawn_agent, update_todo) now correctly appear in system prompt tool list

## v0.2.0 (2026-03-26)

### Architecture Refactor
- **Constants**: All hardcoded values consolidated into core/constants.py (27 constants)
- **PromptBuilder**: Extracted from Orchestrator (system prompt, skills, hints, guidance)
- **MainLoopContext**: Replaced 25-parameter constructor with dataclass
- **SubAgentRunner**: Rewritten to reuse MainLoopRunner (479→317 lines), sub-agents get full capabilities

### Business Logic Decoupling
- UserContextBuilder/mirror mode/GAIA benchmark removed from core — injectable via hooks
- tool/manager.py context→env injection now data-driven (not hardcoded field list)

### Language Control
- New response_language config (auto/Chinese/English/Japanese/...)
- Auto-detection via detect_language_by_chars()
- Backwards compat: chinese_context=true → response_language="Chinese"

### Code Quality
- Consolidated generate_message_id (3→1), reasoning_tags (5→1)
- Fixed bugs: attribute name mismatch, task_guidance typo, variable shadowing
- 248 tests (11 integration tests added)
- Example project with multiple configs, hooks, skills, tools

## [0.1.0] - 2026-03-11

### Added
- MCP-native tool integration with stdio, SSE, and streamable-http transports
- Three-tier context management: Observation Masking, LLM summarization, emergency pruning
- Tool call deduplication with hit-count tracking and progressive escalation
- Execution monitoring with stall/loop detection and three-level escalation
- Skill system with rules, LLM, and inline selection modes
- Hook system for lifecycle events (agent, turn, tool)
- SecureContext for automatic sensitive data isolation
- Streaming output with structured tag extraction
- Deep research mode with reflection checkpoints and auto task planning
- CLI for project initialization, execution, and testing
- Pydantic-based configuration validation
