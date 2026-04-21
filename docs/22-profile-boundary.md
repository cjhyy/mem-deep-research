# Runtime vs Profile 边界盘点

> 状态基线：2026-04-21
> 文档定位：为 v1.4.0 "Deep research 从主链降级为 profile" 提供决策依据
> 配套阅读：`docs/20-roadmap.md`（版本路线图）、`docs/21-industry-framework-analysis.md`（定位转向依据）

## 盘点目标

把 `mem_deep_research_core/core/` 里每块可辨识的执行逻辑分到三类之一：

- **[Runtime]** 通用 Agent Runtime 能力 — 任何 profile（research / automation / coding / workflow）都会用到
- **[Profile: DeepResearch]** 研究场景专属 — 只有 deep research 任务用得上，其他场景要么无意义要么有冲突
- **[Shared/Gray]** 灰色地带 — 当前在研究场景用得多，但接口本身通用；或者未来可能被其他 profile 复用

盘点覆盖 9 个核心模块：`main_loop.py` / `orchestrator.py` / `context_manager.py` / `llm_router.py` / `llm_call_handler.py` / `task_planner.py` / `sub_agent_runner.py` / `memory.py` / `monitoring.py`。

行号是盘点时代码状态（post-v1.2.5），仅作定位参考；重构时以函数/类名为准。

## 判断规则

| 特征 | 归属 |
|------|------|
| 所有 mode 行为一致 | Runtime |
| Mode 分支只是性能/资源优化（如 quick 限制 turns） | Runtime |
| Mode 分支是功能差异（如 deep 才生成 summary、才抽 evidence） | Profile |
| `task_engine_cfg` / `is_deep_mode` 条件读取方 | Profile |
| 接口通用但目前唯一消费者是研究链路 | Shared/Gray |
| 未来计划被 workflow / automation / coding profile 复用 | Shared/Gray |

## 主循环核心：Turn Loop 骨架

| 位置 | 逻辑 | 归属 | 理由 |
|------|------|------|------|
| `main_loop.py` `MainLoopRunner.run()` | 轮次循环、max_turns 限制、break 条件聚合 | **Runtime** | 任何 agent 都需要 turn loop 骨架 |
| `main_loop.py` should_break 分支（line ~1155） | 基于 `stop_reason` 判退（v1.2.5 对齐 Claude Code） | **Runtime** | Provider 层通用信号 |
| `main_loop.py` 反思豁免分支 | `_reflection_pending` 轮豁免 | **Profile: DeepResearch** | 反思轮只在 deep 模式注入，豁免逻辑是 profile 专属行为 |
| `main_loop.py` `_has_tool_calls` 路径 | 无 tool call 时退出循环 | **Runtime** | 通用循环退出兜底 |
| `main_loop.py` `_execute_tools` & 并发分组 | 工具分批执行、concurrent-safe 分组 | **Runtime** | 所有 agent 通用 |
| `main_loop.py` tool result 配对完整性检查 | 缺失 tool_result 时注入 synthetic error | **Runtime** | Provider API 兼容要求，通用 |
| `main_loop.py` Grace turn（已删除） | — | — | 在 v1.2.5 移除，无需归属 |

## Execution Mode 路由

| 位置 | 逻辑 | 归属 | 理由 |
|------|------|------|------|
| `main_loop.py` `_resolve_execution_mode()` | Auto / simple_auto 模式解析 | **Runtime** | Mode dispatcher 本身通用 |
| `main_loop.py` `on_route_classify` hook | 业务层干预 mode 决策 | **Runtime** | Hook 系统通用扩展点 |
| `llm_router.py` `LLMRouter` | 结构信号路由 + LLM 分类 + adaptive 分类 | **Runtime** | 路由机制本身通用；但 adaptive 的阈值（`ADAPTIVE_DEEP_TOOL_THRESHOLD` 等）当前为 research 调校 |
| `llm_router.py` `adaptive_classify()` | 首轮后升级 mode | **Runtime** | 通用自适应能力 |
| `main_loop.py` `is_quick_mode` 分支 | Quick 模式 max_turns 下限、工具裁剪 | **Runtime** | 性能优化，不限于研究 |
| `main_loop.py` `is_deep_mode` 分支（多处） | 启用 reflection / verify / summary / planner | **Profile: DeepResearch** | 所有 `is_deep_mode` 的功能性分支 |
| `main_loop.py` `ADAPTIVE_DEEP_TOOL_THRESHOLD` | 首轮工具数 ≥ 阈值 → 升级 deep | **Shared/Gray** | 阈值本身是通用参数，但"tool_count 多→需要反思"是研究假设 |

## Context 管理

| 位置 | 逻辑 | 归属 | 理由 |
|------|------|------|------|
| `context_manager.py` `ContextManager` 骨架 | Token 估算、dedup cache、registry | **Runtime** | 长任务通用能力 |
| `context_manager.py` `filter_duplicate_calls()` | 跨轮工具去重 | **Runtime** | 所有 agent 有益 |
| `context_manager.py` `manage_context()` 三级策略 | ObservationMasking / LLMSummarize / BinaryReduction | **Runtime** | 通用压缩机制 |
| `window_strategy.py` `ObservationMasking` | 旧 tool result 替换为预览 | **Runtime** | 零 LLM 成本，通用 |
| `window_strategy.py` LLMSummarize | LLM 生成 context 摘要（**非最终 summary**） | **Runtime** | 通用压缩策略 |
| `context_manager.py` offload / restore / `_offload_registry` | 大结果卸载到文件 + 符号引用 + cleanup | **Runtime** | 长任务通用能力，已对接 resume |
| `context_manager.py` `SourceRegistry` / URL 提取 | 从工具结果提取来源 | **Shared/Gray** | URL 提取通用，但当前只有 research 流使用 |
| `context_manager.py` evidence binding | `finalize_offload_candidates` 绑定 evidence 到 offload record | **Profile: DeepResearch** | `EvidenceItem` 是研究向数据结构 |
| `window_strategy.py` ObservationMasking 用 session_memory findings 替换 | 研究产出替换原文 | **Shared/Gray** | 机制通用，数据源（findings / evidence）目前是 research 特性 |
| `main_loop.py` microcompact 调用点 | 每轮清理旧 tool_result | **Runtime** | 通用优化 |

## LLM 调用封装

| 位置 | 逻辑 | 归属 | 理由 |
|------|------|------|------|
| `llm_call_handler.py` `LLMCallHandler` | Provider 调用封装、重试、guardrail | **Runtime** | 所有场景通用 |
| `llm_call_handler.py` context_limit 重试 | Level 3 紧急裁剪 + 重试 | **Runtime** | 通用异常处理 |
| `llm_call_handler.py` `SummaryHandler` | **最终答案摘要**（非 context 摘要）生成 + 重试 | **Profile: DeepResearch** | Deep 模式 + 有工具调用才强制触发；是研究报告产出，非 agent 通用收尾 |
| `llm_call_handler.py` `generate_reflection_prompt()` | 反思 prompt 模板 | **Profile: DeepResearch** | 研究专属反思策略 |
| `llm_call_handler.py` `CONTEXT_REDUCTION_TARGET_RATIO` 使用 | 反压的 token 目标 | **Runtime** | 通用压缩辅助 |

## Summary 策略（v1.4.0 迁移重点）

| 位置 | 逻辑 | 归属 | 理由 |
|------|------|------|------|
| `main_loop.py` line ~1700 `generate_summary` 判定 | `deep + 用过工具 → 强制 summary；否则看配置` | **Profile: DeepResearch** | 三条判定分支都和 research 语义绑定 |
| `main_loop.py` `is_simple_response` 判定 | 决定"用最后一条 assistant 文本 vs 跑 summary" | **Profile: DeepResearch** | 分支依赖 `generate_summary`，属 profile 决策 |
| `main_loop.py` `_handle_summary` 调用 | 调用 `SummaryHandler` | **Profile: DeepResearch** | 研究场景收尾报告 |
| `answer_handler.py` final answer post-process | 最终答案标准化 | **Shared/Gray** | 通用 post-process 框架，但 `post_process_final_answer` hook 的默认实现偏研究 |

## Reflection / Verify / Plan（deep 专属三件套）

| 位置 | 逻辑 | 归属 | 理由 |
|------|------|------|------|
| `monitoring.py` `TurnCounter.should_inject_reflection()` | Deep 模式按间隔触发反思 | **Profile: DeepResearch** | `reflection_enabled` 只在 deep 模式 True |
| `main_loop.py` reflection 注入块 | 生成并注入反思 prompt | **Profile: DeepResearch** | 完全研究专属 |
| `main_loop.py` `_run_verify_checkpoint()` | Deep + task_engine.enable_verify 时执行 | **Profile: DeepResearch** | 证据覆盖率检测 |
| `task_planner.py` `TaskPlanner` | LLM 任务分解 | **Profile: DeepResearch** | 研究场景子问题拆分 |
| `main_loop.py` plan 注入 | `not is_quick_mode` 时注入 task plan | **Profile: DeepResearch** | Plan 语义是研究流 |
| `config_schema.py` `TaskEngineConfig` | Reflection / verify / planner 统一配置 | **Profile: DeepResearch** | 配置对象本身是 profile 载体 |

## Evidence / Tag 解析

| 位置 | 逻辑 | 归属 | 理由 |
|------|------|------|------|
| `main_loop.py` `_extract_evidence_tags()` | `<evidence>` tag 提炼 + 写入 SessionMemory | **Profile: DeepResearch** | 研究场景的结构化证据抽取 |
| `main_loop.py` `<evidence>` tag 清理 | 从 assistant_text 剥离 tag 后再对外输出 | **Profile: DeepResearch** | 研究格式要求 |
| `memory.py` `EvidenceItem` / `SessionMemory.evidence_items` | Evidence 数据结构 | **Profile: DeepResearch** | 研究向数据字段 |
| `main_loop.py` `<response_language>` tag 处理 | LLM 声明语言、框架解析 | **Shared/Gray** | 语言检测通用，但 prompt 引导是研究向 |
| `main_loop.py` `<next_skills>` tag 处理 | Inline skill 声明、下轮动态注入 | **Shared/Gray** | Skill 系统通用，当前主要研究链路使用 |

## Sub-Agent & Concurrency

| 位置 | 逻辑 | 归属 | 理由 |
|------|------|------|------|
| `sub_agent_runner.py` `SubAgentRunner` | 复用 MainLoopRunner 执行 sub-agent | **Runtime** | 通用子 agent 能力 |
| `main_loop.py` `BUILTIN_TOOL_SPAWN_AGENT` | 内置 spawn 工具 | **Runtime** | 通用并发能力 |
| `main_loop.py` `_execute_spawn_calls()` | spawn 并发 + semaphore 控制 | **Runtime** | 并发基础设施 |
| `sub_agent_runner.py` `_run_configured_sub_agent` | 命名 sub-agent 执行 | **Runtime** | 通用 |
| `sub_agent_runner.py` offload registry merge (v1.2.4 修复) | 子 agent offload 文件合并到主 registry | **Runtime** | 长任务通用能力 |
| `sub_agent_runner.py` inherit_with_override (v1.2.4) | Context manager 四层继承 | **Runtime** | 通用隔离/继承机制 |
| `sub_agent_runner.py` `_strip_language_section` | 子 agent 剥离父 prompt 的语言检测段 | **Runtime** | 通用 prompt 净化 |

## Monitoring / Loop Detection

| 位置 | 逻辑 | 归属 | 理由 |
|------|------|------|------|
| `monitoring.py` `ExecutionMonitor` | 循环检测 + 三级升级 + 温度提升 | **Runtime** | 所有长任务通用 |
| `monitoring.py` `record_progress()` | 响应 hash + 滑动窗口检测 | **Runtime** | 通用振荡检测 |
| `monitoring.py` `check_timeout()` | Soft / hard timeout | **Runtime** | 通用资源控制 |
| `monitoring.py` `get_loop_break_hint()` | 循环检测升级后的 hint 文本 | **Runtime** | 通用；研究场景无特殊性 |
| `monitoring.py` `TurnCounter.reflection_enabled` 字段 | 启用标志（False/True） | **Profile: DeepResearch** | 字段本身由 deep profile 决定 |

## Skill / Prompt

| 位置 | 逻辑 | 归属 | 理由 |
|------|------|------|------|
| `prompt_builder.py` system prompt 构建 | Skill 注入、user context、语言 section | **Runtime** | Prompt 构建骨架通用 |
| `prompt_builder.py` skill selection 触发 | Inline / LLM / rules 三策略 | **Shared/Gray** | 机制通用，消费者目前是研究 |
| `prompts/templates/` reflection / planning / verify | 研究专属模板 | **Profile: DeepResearch** | 模板文件本身是 research 内容 |
| `prompts/templates/` base / main / worker | 通用 agent prompt 骨架 | **Runtime** | 通用 |
| `skills/` 系统整体 | Skill 定义、匹配、注入 | **Runtime** | 扩展点系统通用 |

## 内置工具

| 名称 | 归属 | 理由 |
|------|------|------|
| `spawn_agent` | **Runtime** | 通用并发 |
| `read_result` | **Runtime** | 通用 offload 恢复 |
| `update_todo` | **Shared/Gray** | TodoTracker 接口通用；但"强制在 deep/task_engine 启用"是研究偏好 |
| `tool_search` (deferred tools) | **Runtime** | 工具动态发现，通用能力 |

## Hook 系统

| 位置 | 逻辑 | 归属 | 理由 |
|------|------|------|------|
| `hooks.py` `HookRegistry` | 注册、调用链、异常隔离 | **Runtime** | 扩展点系统 |
| `hooks.py` `on_agent_start` / `on_turn_end` / … 全部通用 hook | 生命周期钩子 | **Runtime** | 通用 |
| `hooks.py` `on_offload_evidence_prep` | 证据 offload 预处理 | **Profile: DeepResearch** | Hook 名称含 evidence，研究专属 |
| `hooks.py` `on_reflection_build` | 反思 prompt 构建 | **Profile: DeepResearch** | 研究专属 |

## Config Schema 归属（v1.4.0 迁移索引）

`MainAgentConfig` 中按字段归属分：

**Runtime**：`llm`、`tool_config`、`max_turns`、`max_tool_calls_per_turn`、`keep_tool_result`、`max_concurrent_subagents`、`parallel_spawn`、`context_manager`（大部分）、`monitoring`、`interceptor`、`input_process`、`prompt`

**Profile: DeepResearch**：`task_engine`、`generate_summary`、`todo_tracker`（在 deep 自动启用的部分）、`skill_selection` 的研究向默认值

**Shared/Gray**：`execution_mode`、`response_language`、`chinese_context`（遗留）、`add_message_id`

## 量化结论

- **Runtime 占比**：~55%（主循环骨架、工具调度、context 基础、监控、hook、sub-agent、LLM 封装）
- **Profile: DeepResearch 占比**：~25%（reflection / verify / planner / summary / evidence / 对应配置）
- **Shared/Gray 占比**：~20%（skill / tag 解析 / SourceRegistry / todo / adaptive 阈值 / 部分 prompt 模板）

## v1.4.0 拆分路径

基于上表，`DeepResearchProfile` 至少需要聚合这些能力：

1. **Reflection policy**：`should_inject_reflection()` + reflection prompt 生成 + 反思轮豁免
2. **Verify policy**：verify checkpoint 触发条件 + 证据覆盖/冲突检测
3. **Plan policy**：TaskPlanner 调用时机 + plan 注入
4. **Summary policy**：`generate_summary` 判定 + `SummaryHandler` 调用
5. **Evidence extraction**：`<evidence>` tag 解析 + SessionMemory 写入
6. **Research-flavored context strategies**：findings-based masking、evidence-aware offload binding

补充：`Profile` contract 不应只是一组 turn 级钩子，而应至少覆盖 5 层职责：

1. **Bootstrap / Assembly**：决定是否初始化 `TaskPlanner` / `SummaryHandler` / `TodoTracker` / research-specific monitor policy
2. **Route / Mode policy**：决定哪些 execution mode 对该 profile 有意义，以及 mode → feature set 的映射
3. **Prompt / Context policy**：扩展 system prompt、注入 research-specific guidance、控制哪些 tag/skill/profile hint 生效
4. **Turn policy**：在 turn 前后、LLM 后、tool batch 后插入 reflection / verify / evidence / research memory 行为
5. **Finalization policy**：决定最终答案是直接返回 assistant 文本、走 summary、还是做 profile-specific post-process

换句话说，`Profile` 至少要覆盖 `Orchestrator` 初始化期、`PromptBuilder` 构建期、`MainLoopRunner` 运行期、`AnswerHandler` 收尾期，而不只是 turn loop 中间的几个回调。

对应的主链改动：

- `Orchestrator` 不再直接初始化 `TaskPlanner` / `SummaryHandler` / `TodoTracker`；改为调用 `profile.bootstrap(...)`
- `PromptBuilder` / mode router 在构建 prompt 和解析 mode 时 consult `profile.prompt_policy(...)` / `profile.route_policy(...)`
- `MainLoopRunner` 去掉 `is_deep_mode` 所有功能性分支，改为调用 `profile` 的 turn/finalization hooks，而不是只塞几个 `before_turn()` 风格的薄回调
- `SummaryHandler` 从 `LLMCallHandler` 外移到 profile 层
- `TaskEngineConfig` 只被 profile 读取，主链不感知
- `EvidenceItem` 留在 `SessionMemory` 但是 profile 层数据（主链不操作）

一个更合理的最小接口形状类似：

```python
class AgentProfile(Protocol):
    def bootstrap(self, orchestrator_ctx) -> ProfileRuntime: ...
    def route_policy(self, route_ctx) -> RoutePolicyResult: ...
    def prompt_policy(self, prompt_ctx) -> PromptPolicyResult: ...
    async def before_turn(self, turn_ctx) -> None: ...
    async def after_llm(self, llm_ctx) -> None: ...
    async def after_tool_batch(self, tool_ctx) -> None: ...
    async def before_finalize(self, final_ctx) -> FinalizeDecision: ...
    async def finalize(self, final_ctx) -> str: ...
```

`StandardProfile` 应该是上述接口的近乎空实现；`DeepResearchProfile` 只覆写研究专属行为。这样才能真正验证 runtime 主链已经抽干净。

## Shared/Gray 的处理建议

- **Skill / `<next_skills>`**：保留在 runtime，workflow layer 和其他 profile 可自然复用
- **`<response_language>`**：语言检测本身通用，prompt 引导可能需要 profile 层定制
- **Todo tracker**：接口留 runtime，"deep 模式自动启用"归 profile
- **SourceRegistry / URL 提取**：接口通用留 runtime，"绑定 evidence" 归 profile
- **Adaptive 分类阈值**：阈值本身通用，默认值可由 profile 覆盖

## 下一步

1. 基于本文档，设计 `Profile` 接口（建议新建 `docs/24-profile-contract.md`，计划下周）
2. 设计时反向检验：每个 Profile 方法都能对应到本文档中至少一条研究专属逻辑
3. `StandardProfile`（默认）应该是一个"几乎全空实现"的 profile，证明 runtime 的通用性
4. v1.4.0 动手前，主循环的研究专属分支数可作为量化指标（现在约 15 处 `is_deep_mode` 功能性分支，目标 → 0）
