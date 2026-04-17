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

# Preview length retained in observation masking summaries (chars).
# Longer = better final summary quality, but slower context savings.
COMPACT_PREVIEW_LENGTH = 300

# Minimum chars in a message before it's worth compacting
COMPACT_MIN_CHARS = 200

# Minimum chars for microcompact to consider clearing (skip already-short messages)
MICROCOMPACT_MIN_CHARS = 300

# Default chars-per-token ratio (fallback when tiktoken unavailable)
DEFAULT_CHARS_PER_TOKEN = 3.5

# Evidence extraction: max chars per evidence item
EVIDENCE_MAX_CHARS = 500

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
BUILTIN_TOOL_SEARCH = "tool_search"
BUILTIN_TOOL_READ_RESULT = "read_result"

# Maximum nesting depth for spawned agents.
# 1 = only main agent can spawn; spawned agents cannot spawn further.
MAX_SPAWN_DEPTH = 1

# Tools that are safe to execute concurrently (read-only, no side effects).
# Matched by segment: tool_name is split on "_" and "-", then checked for exact
# segment matches. This avoids false positives like "research_and_write" matching "search".
CONCURRENT_SAFE_TOOL_SEGMENTS = frozenset({
    "search",
    "scrape",
    "fetch",
    "read",
    "get",
    "list",
    "query",
    "lookup",
    "wikipedia",
    "calculate",
})

# ============================================================
# Execution Modes
# ============================================================

EXECUTION_MODE_AUTO = "auto"
EXECUTION_MODE_SIMPLE_AUTO = "simple_auto"
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

# Token budget defaults
DEFAULT_TASK_TOKEN_BUDGET = 0  # 0 = unlimited (disabled by default)
TOKEN_BUDGET_WARNING_RATIO = 0.8  # 80% 时注入催促
TOKEN_BUDGET_HARD_RATIO = 1.0  # 100% 时强制终止

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
    "[EVIDENCE]",
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
    "[TOKEN BUDGET",
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

# Default strip tags — silently removed from stream output (not emitted as reasoning events).
# Supports attribute-bearing open tags like <offload_evidence ref="...">.
DEFAULT_STRIP_TAGS = [
    "evidence",
    "offload_evidence",
    "response_language",
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
TAG_OFFLOADED = "[OFFLOADED:"  # prefix for offloaded content markers
TAG_TASK_PROGRESS = "[TASK PROGRESS]"
TAG_EVIDENCE = "[EVIDENCE]"

_CONTEXT_COMPRESSION_BASE = (
    "[CONTEXT NOTE] Some earlier messages have been compressed to manage context size. "
    "Key findings and extracted evidence are preserved in session memory."
)

_CONTEXT_COMPRESSION_READ_RESULT = (
    " If you need the full original content from a compressed/offloaded tool result, "
    "use read_result(ref) with the file reference or read_result(ref='turn:N') for a specific turn."
)

def build_context_compression_notice(has_read_result: bool = True) -> str:
    """根据 read_result 工具是否可用动态生成压缩通知"""
    if has_read_result:
        return _CONTEXT_COMPRESSION_BASE + _CONTEXT_COMPRESSION_READ_RESULT
    return _CONTEXT_COMPRESSION_BASE

# ============================================================
# Message Types — 结构化消息分类
#
# 每条 message_history 中的消息可通过 `_type` 字段标明类型，
# 让 compact / offload / dedup 基于类型判断，而非关键词匹配。
# 无 `_type` 的旧消息仍通过 SYSTEM_MESSAGE_KEYWORDS 兜底。
# ============================================================


class MT:
    """Message Types — 简短常量集合，不做 enum 以避免序列化开销"""

    # === 用户输入 ===
    USER_INPUT = "user_input"

    # === LLM 输出 ===
    ASSISTANT = "assistant"

    # === 工具结果 ===
    TOOL_RESULT = "tool_result"

    # === 系统注入（受压缩保护） ===
    EVIDENCE = "evidence"
    SESSION_MEMORY = "session_memory"
    LONG_TERM_MEMORY = "long_term_memory"
    TASK_PROGRESS = "task_progress"
    PLAN = "plan"
    REFLECTION = "reflection"
    CITATION_SUMMARY = "citation_summary"
    TOKEN_WARNING = "token_warning"
    CONTEXT_COMPRESSION = "context_compression"
    RESUME_NOTICE = "resume_notice"

    # === 系统提示（不受压缩保护，可被清理） ===
    LOOP_HINT = "loop_hint"
    TRUNCATION_RECOVERY = "truncation_recovery"
    INLINE_SKILL = "inline_skill"

    # === 压缩产物 ===
    CONTEXT_SUMMARY = "context_summary"
    OFFLOADED = "offloaded"

    # === Offload 流程辅助 ===
    OFFLOAD_PREP = "offload_prep"  # sidecar prompt: 通知 LLM 即将 offload 的消息

    # === 内部 LLM 调用（不进入主 message_history） ===
    SUMMARY_PROMPT = "summary_prompt"
    ROUTING = "routing"
    TASK_PLANNING = "task_planning"


# 受压缩保护的消息类型集合（不会被 compact / masking / microcompact 清理）
PROTECTED_MESSAGE_TYPES = frozenset({
    MT.EVIDENCE,
    MT.SESSION_MEMORY,
    MT.LONG_TERM_MEMORY,
    MT.TASK_PROGRESS,
    MT.PLAN,
    MT.REFLECTION,
    MT.CITATION_SUMMARY,
    MT.TOKEN_WARNING,
    MT.CONTEXT_COMPRESSION,
    MT.RESUME_NOTICE,
    MT.CONTEXT_SUMMARY,
    MT.OFFLOADED,
})


def make_msg(role: str, text: str, _type: str | None = None, **extra) -> dict:
    """创建标准 message dict

    Args:
        role: "user" 或 "assistant"
        text: 消息文本
        _type: 消息类型 (MT.xxx)
        **extra: 额外字段 (如 _meta=True)

    Returns:
        {"role": ..., "content": [{"type": "text", "text": ...}], "_type": ...}
    """
    msg = {"role": role, "content": [{"type": "text", "text": text}]}
    if _type is not None:
        msg["_type"] = _type
    if extra:
        msg.update(extra)
    return msg


def make_tool_result_msg(text: str, **extra) -> dict:
    """创建 tool result 消息（统一入口，确保 _type=MT.TOOL_RESULT）

    所有 provider 的 update_message_history() 应使用此函数，
    而非手动构建 {"role": "user", "content": [...]}.

    Args:
        text: 工具结果文本
        **extra: 额外字段 (如 tool_call_id)

    Returns:
        {"role": "user", "_type": "tool_result", "content": [{"type": "text", "text": ...}]}
    """
    msg = {
        "role": "user",
        "_type": MT.TOOL_RESULT,
        "content": [{"type": "text", "text": text}],
    }
    if extra:
        msg.update(extra)
    return msg


def make_tool_result_msg_native(call_id: str, text: str) -> dict:
    """创建 OpenAI 原生格式的 tool result 消息

    用于 GPTOpenAIClient 等使用 role="tool" 的 provider。

    Args:
        call_id: tool_call_id
        text: 工具结果文本

    Returns:
        {"role": "tool", "_type": "tool_result", "tool_call_id": ..., "content": ...}
    """
    return {
        "role": "tool",
        "_type": MT.TOOL_RESULT,
        "tool_call_id": call_id,
        "content": text,
    }


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
