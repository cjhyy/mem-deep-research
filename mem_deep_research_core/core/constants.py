"""
Framework constants — single source of truth for all configurable defaults.

Values that appear as config defaults should match config_schema.py.
Values that are internal implementation details live here.
"""

# ============================================================
# Token / Context Management
# ============================================================

# Result text preview length for logs and brief summaries
RESULT_BRIEF_LENGTH = 200

# Minimum chars in a message before it's worth compacting
COMPACT_MIN_CHARS = 200

# Default chars-per-token ratio (fallback when tiktoken unavailable)
DEFAULT_CHARS_PER_TOKEN = 3.5

# ============================================================
# Context Limit Recovery (SummaryHandler)
# ============================================================

# Max time for summary generation (seconds)
SUMMARY_GENERATION_TIMEOUT = 120

# Max retry attempts for context limit recovery
MAX_CONTEXT_LIMIT_RETRIES = 2

# Max context reduction retries in SummaryHandler
MAX_SUMMARY_CONTEXT_RETRIES = 5

# Target token ratio after context reduction
CONTEXT_REDUCTION_TARGET_RATIO = 0.6

# Minimum chars for an emergency summary to be considered valid
EMERGENCY_SUMMARY_MIN_CHARS = 50

# Task description preview length
TASK_PREVIEW_LENGTH = 500

# Network retry configuration for summary generation
SUMMARY_NETWORK_MAX_RETRIES = 3

# ============================================================
# Tool Management
# ============================================================

# Default tool call timeout (seconds)
DEFAULT_TOOL_CALL_TIMEOUT = 900.0

# Default tool definition cache TTL (seconds)
DEFAULT_CACHE_TTL = 300.0

# Default scrape max length
DEFAULT_SCRAPE_MAX_LENGTH = 20000

# Sub-agent tool name prefix
SUB_AGENT_PREFIX = "agent-"

# Default max concurrent sub-agents
DEFAULT_MAX_CONCURRENT_SUBAGENTS = 3

# Default threshold for offloading large tool results to filesystem
DEFAULT_RESULT_OFFLOAD_THRESHOLD = 5000

# Built-in tool names
BUILTIN_TOOL_UPDATE_TODO = "update_todo"
BUILTIN_TOOL_SPAWN_AGENT = "spawn_agent"

# Maximum nesting depth for spawned agents (prevents exponential resource consumption)
MAX_SPAWN_DEPTH = 2

# ============================================================
# Execution Modes
# ============================================================

EXECUTION_MODE_AUTO = "auto"
EXECUTION_MODE_QUICK = "quick"
EXECUTION_MODE_STANDARD = "standard"
EXECUTION_MODE_DEEP = "deep"

# Quick mode limits
QUICK_MODE_MAX_TURNS = 3

# Transient errors eligible for tool auto-retry
TOOL_TRANSIENT_ERRORS = (ConnectionError, TimeoutError, BrokenPipeError, EOFError, OSError)

# ============================================================
# Monitoring
# ============================================================

# Default temperature boost on loop detection
DEFAULT_TEMPERATURE_BOOST = 0.3

# Default temperature cap
DEFAULT_TEMPERATURE_BOOST_CAP = 1.0

# Number of recent messages to scan for tool names
RECENT_TOOL_LOOKBACK = 6

# ============================================================
# Message Detection Keywords
# ============================================================

# Keywords that identify system-injected messages (should not be compressed)
SYSTEM_MESSAGE_KEYWORDS = [
    "[REFLECTION",
    "REFLECTION CHECKPOINT",
    "MANDATORY DIRECTIVE",
    "[SYSTEM",
    "[RESEARCH CONTEXT SUMMARY",
    "[COLLECTED SOURCES",
    "[RESEARCH PLAN",
    "[TASK PROGRESS",
    "[CONTEXT NOTE",
    "[SESSION MEMORY",
    "[LONG-TERM MEMORY",
]

# ============================================================
# Interceptor Defaults
# ============================================================

# Default filter tags
DEFAULT_FILTER_TAGS = ["use_mcp_tool"]

# Default reasoning tags
DEFAULT_REASONING_TAGS = [
    "thinking",
    "think",
    "task_plan",
    "findings_update",
    "reflection_checkpoint",
]

# ============================================================
# Message Tags (injected into message_history)
# ============================================================

TAG_TASK_PLAN = "[TASK PLAN]"
TAG_COLLECTED_SOURCES = "[COLLECTED SOURCES]"
# NOTE: Intentionally without closing ']' — used as prefix for startswith() checks.
# Full format: "[CONTEXT SUMMARY — turns 1-N]"
TAG_CONTEXT_SUMMARY = "[CONTEXT SUMMARY"
TAG_CONTENT_REMOVED = "[Content removed to reduce context]"
TAG_TASK_PROGRESS = "[TASK PROGRESS]"

CONTEXT_COMPRESSION_NOTICE = (
    "[CONTEXT NOTE] Some earlier messages have been compressed to manage context size. "
    "Key findings are preserved in the task progress. "
    "If you need detailed information from compressed content, use available tools to re-fetch it."
)

# ============================================================
# Fallback Messages
# ============================================================

FALLBACK_EMERGENCY_SUMMARY = "[Emergency Summary - context limit reached]"
FALLBACK_NO_ANSWER = "No final answer generated."
FALLBACK_SUMMARY_ERROR = (
    "[ERROR] Unable to generate final summary due to context limit or network issues. "
    "You should try again."
)
FALLBACK_LOOP_TERMINATED = (
    "[SYSTEM] The research process was terminated due to a repeated response loop. "
    "Please synthesize a final answer based on all the information gathered so far. "
    "If insufficient information was collected, acknowledge the limitation and provide "
    "the best possible answer with what is available."
)

# ============================================================
# Config Utilities
# ============================================================


def generate_message_id() -> str:
    """Generate random message ID using common LLM format"""
    import uuid

    return f"msg_{uuid.uuid4().hex[:8]}"


def parse_bool_config(value, default=False) -> bool:
    """Parse config value to bool, handling string 'true'/'false'."""
    if isinstance(value, str):
        return value.lower().strip() == "true"
    return bool(value) if value is not None else default


def ensure_dict(value) -> dict:
    """Safely convert OmegaConf/config value to plain dict."""
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    return dict(value) if hasattr(value, "__iter__") else {}
