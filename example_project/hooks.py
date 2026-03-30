"""
Project hooks — auto-loaded by DeepResearch.from_project().

Active hooks:
  - Execution stats tracking (start/end/tool counts)
  - Turn progress logging
  - Context compression logging

Commented examples:
  - User identity injection
  - Custom guardrails
  - Language detection override
"""

import logging
import time

from mem_deep_research_core.core.hooks import HookContext, hooks

logger = logging.getLogger(__name__)

_stats = {"start_time": 0, "total_tool_calls": 0, "tool_errors": 0}


# ---------------------------------------------------------------------------
# Execution tracking
# ---------------------------------------------------------------------------


@hooks.register("on_agent_start", priority=10)
def on_agent_start(ctx: HookContext, original_fn):
    _stats["start_time"] = time.time()
    _stats["total_tool_calls"] = 0
    _stats["tool_errors"] = 0
    logger.info(f"[Agent] START | query={str(ctx.extra.get('query', ''))[:100]}")
    return original_fn(ctx)


@hooks.register("on_agent_end", priority=10)
def on_agent_end(ctx: HookContext, original_fn):
    elapsed = time.time() - _stats["start_time"] if _stats["start_time"] else 0
    logger.info(
        f"[Agent] END | {elapsed:.1f}s | "
        f"tools={_stats['total_tool_calls']} | errors={_stats['tool_errors']}"
    )
    return original_fn(ctx)


@hooks.register("on_tool_end", priority=10)
def log_tool_end(ctx: HookContext, original_fn):
    _stats["total_tool_calls"] += 1
    duration = f"{ctx.duration_ms}ms" if ctx.duration_ms else "?"
    error = (ctx.tool_result or {}).get("error")
    if error:
        _stats["tool_errors"] += 1
        logger.warning(f"[Tool] {ctx.tool_name} | {duration} | ERROR: {error}")
    else:
        logger.info(f"[Tool] {ctx.tool_name} | {duration} | OK")
    return original_fn(ctx)


@hooks.register("on_turn_end", priority=10)
def log_turn(ctx: HookContext, original_fn):
    total = ctx.extra.get("total_tool_calls", 0)
    msgs = ctx.extra.get("message_count", 0)
    logger.info(f"[Turn {ctx.turn_number}] tools={ctx.tool_calls_count} total={total} msgs={msgs}")
    return original_fn(ctx)


@hooks.register("on_context_compact", priority=10)
def log_compact(ctx: HookContext, original_fn):
    action = ctx.extra.get("compact_action", "unknown")
    logger.info(f"[Context] Compact: {action}")
    return original_fn(ctx)


# ---------------------------------------------------------------------------
# Examples (uncomment to enable)
# ---------------------------------------------------------------------------

# --- Guardrail: block dangerous patterns ---
# from mem_deep_research_core.exceptions import GuardrailError
#
# @hooks.register("on_before_llm_call", priority=10)
# def guardrail(ctx: HookContext, original_fn):
#     messages = ctx.extra.get("messages", [])
#     for msg in messages[-1:]:
#         if "DELETE FROM" in str(msg.get("content", "")).upper():
#             raise GuardrailError("Blocked: SQL DELETE detected")
#     return original_fn(ctx)

# --- User identity injection ---
# from mem_deep_research_core.core.user_context import UserContextBuilder
#
# @hooks.register("on_system_prompt_build", priority=50)
# def inject_user_context(ctx: HookContext, original_fn):
#     prompt = original_fn(ctx)
#     context = ctx.context or {}
#     if context.get("user_id"):
#         builder = UserContextBuilder(context)
#         identity = builder.build_user_identity_context()
#         if identity:
#             prompt += f"\n\n{identity}"
#     return prompt
