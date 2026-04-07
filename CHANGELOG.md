# Changelog

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
