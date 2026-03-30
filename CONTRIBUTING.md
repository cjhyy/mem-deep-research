# Contributing to Mem Deep Research

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/cjhyy/mem-deep-research.git
cd mem-deep-research
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover: hooks, context manager, window strategy, monitoring, inline skill selector, secure context, task planner, interceptor config, and integration tests.

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check .
ruff format .
```

## Architecture Overview

See [docs/00-architecture.md](docs/00-architecture.md) for full architecture. Key principles:

- **All async** — based on asyncio
- **Stateless framework** — customization via project-level config + hooks.py
- **MCP protocol** — tools follow the Model Context Protocol
- **Pydantic validation** — all config validated via `config_schema.py`

## Development Standards

### Adding a Core Feature

1. Add configuration fields to `config_schema.py` with sensible defaults
2. Implement in `core/` or appropriate module
3. Add tests in `tests/`
4. Update `CLAUDE.md` and relevant docs under `docs/`

### Adding an LLM Provider

1. Create a class inheriting `OpenAICompatibleClient` or `LLMProviderClientBase`
2. Place in `llm/providers/`
3. Register in `llm/llm_client.py` factory
4. Add to `PROVIDER_REGISTRY` in `deep_research.py` (if using short names)

### Adding a Tool

1. Create YAML config in `config/tool/`
2. For Python tools: create MCP server using `fastmcp`
3. Reference in `agent.yaml` via `tool_config`

### Adding a Skill

1. Create Markdown file in `config/skills/definitions/`
2. Include YAML front matter with keywords, activation conditions
3. Test with `skill_selection.method: inline`

### Adding a Hook

1. Register in project's `hooks.py`
2. Use `hooks.register("hook_name", priority=N)` decorator
3. Always call `original_fn(ctx)` unless you intend to fully replace the behavior
4. Critical exceptions like `GuardrailError` are NOT swallowed by the hook chain

## Pull Request Process

1. Fork the repository and create your branch from `main`
2. Add tests for any new functionality
3. Ensure all tests pass: `python -m pytest tests/ -v`
4. Run linting: `ruff check .`
5. Update documentation if needed (CLAUDE.md + relevant docs/)
6. Keep PRs focused — one feature or fix per PR
7. Submit a pull request with a clear description

## Commit Messages

Use conventional format:

```
feat: add new tool integration for X
fix: resolve context limit handling in OpenAI client
docs: update hook system documentation
test: add tests for TodoTracker serialization
refactor: simplify window strategy pipeline
```

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include reproduction steps, expected behavior, and actual behavior
- Include relevant configuration (sanitize API keys)
- For security vulnerabilities, please email chenjunhong54321@163.com directly

## Project Structure

```
mem_deep_research_core/
├── deep_research.py          # Public API entry point
├── config_schema.py          # Pydantic config validation
├── exceptions.py             # Exception hierarchy
├── core/                     # Core execution engine
├── llm/                      # LLM provider clients
├── tool/                     # MCP tool management
├── prompts/                  # Prompt templates
├── skills/                   # Skill system
└── utils/                    # Utilities
config/                       # Default framework configs
tests/                        # Test suite
docs/                         # Documentation
```

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
