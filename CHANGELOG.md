# Changelog

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
