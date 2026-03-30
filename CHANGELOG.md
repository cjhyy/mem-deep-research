# Changelog

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
