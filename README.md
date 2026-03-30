# Mem Deep Research

An extensible AI Agent framework focused on deep research tasks. Built on the MCP tool protocol with multi-LLM provider support.

## Features

- **MCP Tool System**: Local (stdio), remote HTTP (streamable-http), and SSE transport modes
- **Three-tier Context Management**: Observation Masking → LLM Summarization → Binary Reduction
- **Tool Call Deduplication**: Cross-turn dedup with hit-count tracking and progressive escalation
- **Execution Monitoring**: 3-tier escalation (WARN → INJECT_HINT → TERMINATE) with loop detection
- **Skill System**: Rules-based, LLM-based, and inline selection modes
- **Hook System**: Lifecycle hooks for custom logic injection
- **SecureContext**: Automatic sensitive data isolation with placeholder substitution
- **Prompt Templates**: Flexible template system with preset combinations
- **Multi-LLM Support**: Anthropic, OpenAI, OpenRouter, DeepSeek, etc.
- **Deep Research Mode**: Reflection checkpoints and automatic task planning

## Installation

```bash
pip install mem-deep-research
```

Or install from source:

```bash
git clone https://github.com/cjhyy/mem-deep-research.git
cd mem-deep-research
pip install -e .
```

## Quick Start

### Option 1: Python API

```python
from mem_deep_research import DeepResearch

# Load from a project directory
dr = DeepResearch.from_project("./my_project")
result = await dr.run("Research the latest developments in AI agents")
print(result.answer)

# Or use synchronous API
result = dr.run_sync("Your research task")
```

### Option 2: CLI

```bash
# Create a new project
mem-deep-research init my_project

# Run research (from your project directory)
python run.py "Your research task"
```

### Option 3: Programmatic Configuration

```python
from mem_deep_research import DeepResearch

dr = DeepResearch(
    llm_provider="anthropic",
    model="claude-sonnet-4-20250514",
    api_key="your-api-key",
)
result = await dr.run("Your task")
```

## Project Structure

A user project loaded via `DeepResearch.from_project()`:

```
my_project/
├── config/
│   ├── agent.yaml              # Agent configuration (LLM, tools, parameters)
│   ├── tool/                   # Tool configs (override framework defaults)
│   ├── skills/definitions/     # Custom skill definitions
│   └── prompts/                # Custom prompt templates
├── hooks.py                    # Lifecycle hooks (auto-loaded)
├── .env                        # API keys
└── run.py                      # Entry script
```

### Minimal `config/agent.yaml`

```yaml
main_agent:
  llm:
    provider_class: "ClaudeOpenRouterClient"
    model_name: "anthropic/claude-sonnet-4"
    temperature: 0.3
    max_tokens: 32000
    openrouter_api_key: "${oc.env:OPENROUTER_API_KEY}"

  tool_config:
    - tool-searching-serper

  max_turns: 20
```

### Full Configuration Reference

```yaml
main_agent:
  prompt:
    agent_type: main              # main | worker
    tool_format: xml              # xml | native
    presets: []                   # e.g. [research, time_sensitive]

  llm:
    provider_class: "ClaudeOpenRouterClient"
    model_name: "anthropic/claude-sonnet-4"
    temperature: 0.3
    max_tokens: 32000
    max_context_length: 128000    # -1 = unlimited
    keep_tool_result: 5           # -1 = keep all, N = keep last N

  tool_config: [tool-searching-serper]
  max_turns: 20
  max_tool_calls_per_turn: 10
  chinese_context: false

  skill_selection:
    enabled: true
    method: inline                # rules | llm | inline
    max_skills: 3

  context_manager:
    enable_dedup: true
    enable_compact: true
    compact_at_ratio: 0.6
    summarize_at_ratio: 0.8
    compact_keep_recent: 3

  monitoring:
    stall_detection_threshold: 120.0
    max_total_time: 600.0
    enable_loop_detection: true
    loop_escalation_terminate_threshold: 3

  deep_research:
    enabled: false
    reflection_interval: 5
    auto_planning: false
```

## Tool Configuration

### Local Tool (stdio)

```yaml
# config/tool/tool-my-custom.yaml
name: "tool-my-custom"
tool_command: "python"
args:
  - "tools/my_tool_server.py"
env:
  MY_API_KEY: "${oc.env:MY_API_KEY}"
```

### Remote Tool (streamable-http)

```yaml
# config/tool/tool-remote.yaml
name: "tool-remote"
url: "https://api.example.com/mcp"
transport: "streamable-http"
headers:
  Authorization: "Bearer ${oc.env:API_TOKEN}"
```

## Hook System

```python
# hooks.py (in your project directory, auto-loaded)
from mem_deep_research_core.core.hooks import hooks, HookContext

@hooks.register("on_env_inject", priority=10)
def inject_env(ctx: HookContext, original_fn):
    params = original_fn(ctx)
    params.env["MY_KEY"] = os.environ.get("MY_KEY", "")
    return params

@hooks.register("on_tool_result_format")
def format_result(ctx: HookContext, original_fn):
    if ctx.tool_name == "my_tool":
        return "Custom format"
    return original_fn(ctx)
```

| Hook | Timing | Modifiable |
|------|--------|------------|
| `on_agent_start` | Agent starts | — |
| `on_agent_end` | Agent completes | — |
| `on_turn_start` | Each turn starts | — |
| `on_turn_end` | Each turn ends | — |
| `on_tool_start` | Before tool call | arguments |
| `on_tool_end` | After tool call | tool_result |
| `on_tool_result_format` | Result formatting | return value |
| `on_thinking_generate` | Thinking description | return value |
| `on_env_inject` | MCP env vars | server_params |
| `on_message_intercept` | Message interception | — |

## SecureContext

Sensitive fields in the context dict are automatically masked in the system prompt and restored before tool execution:

```python
context = {
    "user_name": "Alice",             # Visible to LLM
    "_secure": {
        "user_id": "real-123",        # LLM sees [SECURE:user_id]
        "api_token": "secret-456",    # LLM sees [SECURE:api_token]
    }
}
# Tool calls with [SECURE:user_id] are auto-replaced with "real-123"
```

## LLM Providers

| Provider | Class | Base |
|----------|-------|------|
| Anthropic (native) | `ClaudeAnthropicClient` | `LLMProviderClientBase` |
| OpenAI (native) | `GPTOpenAIClient` | `LLMProviderClientBase` |
| OpenRouter Claude | `ClaudeOpenRouterClient` | `OpenAICompatibleClient` |
| OpenRouter GPT-5 | `GPT5OpenRouterClient` | `OpenAICompatibleClient` |
| OpenAI GPT-5 | `GPT5OpenAIClient` | `GPT5OpenRouterClient` |
| DeepSeek | `DeepSeekOpenRouterClient` | `OpenAICompatibleClient` |

## Environment Variables

```bash
# .env
OPENROUTER_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
SERPER_API_KEY=your_key
```

## Framework Directory Structure

```
mem_deep_research_core/              # Core framework code
├── deep_research.py                 # Main entry (DeepResearch API)
├── config_schema.py                 # Pydantic config validation
├── config/                          # Framework default configs
│   ├── tool/                        # Built-in tool configs (YAML)
│   └── skills/definitions/          # Built-in skill definitions (Markdown)
├── core/                            # Core modules
│   ├── orchestrator.py              # Agent orchestrator
│   ├── main_loop.py                 # Main loop runner
│   ├── monitoring.py                # Execution monitor + loop detection
│   ├── context_manager.py           # Context management (masking + dedup)
│   ├── secure_context.py            # Sensitive data isolation
│   ├── hooks.py                     # Hook system
│   ├── task_planner.py              # LLM task decomposition
│   └── ...
├── llm/                             # LLM clients
│   ├── provider_client_base.py      # Base class
│   └── providers/                   # Provider implementations
├── prompts/                         # Prompt system
│   ├── agent_prompt.py              # Unified AgentPrompt class
│   └── templates/                   # Markdown templates
├── tool/                            # MCP tool module
├── skills/                          # Skill selection system
└── utils/                           # Utilities
tests/                               # Unit tests
docs/                                # Documentation
```

## Architecture

See [`docs/00-architecture.md`](docs/00-architecture.md) for the full architecture overview, or browse [`docs/`](docs/) for detailed documentation on each subsystem.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
ruff check .
ruff format .
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
