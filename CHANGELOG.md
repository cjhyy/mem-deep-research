# Changelog

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
