"""Example project hooks.

Demonstrates how to customize framework behavior via hooks.
This file is auto-loaded by DeepResearch.from_project() / AgentFactory.from_project_dir().

Available hooks:
    on_agent_start / on_agent_end    — Agent lifecycle
    on_turn_start / on_turn_end      — Per-turn lifecycle
    on_tool_start / on_tool_end      — Tool call before/after
    on_tool_filter                   — Filter tool calls after dedup
    on_system_prompt_build           — Modify system prompt
    on_summarize_prompt_build        — Modify summarize prompt
    on_tool_result_format            — Customize tool result display
    on_thinking_generate             — Modify thinking description
    on_env_inject                    — MCP environment variable injection
    on_before_llm_call               — Pre-LLM validation (guardrail)
    on_after_llm_call                — Post-LLM validation (guardrail)
    on_context_compact               — Context compression event
    on_reflection_build              — Modify reflection prompt
"""

import json
import logging
import time

from mem_deep_research_core.core.hooks import hooks, HookContext

logger = logging.getLogger(__name__)

# Track execution stats
_stats = {"start_time": 0, "total_tool_calls": 0, "tool_errors": 0}


# ---------------------------------------------------------------------------
# on_agent_start — Agent 开始时初始化统计
# ---------------------------------------------------------------------------
@hooks.register("on_agent_start", priority=10)
def on_agent_start(ctx: HookContext, original_fn):
    _stats["start_time"] = time.time()
    _stats["total_tool_calls"] = 0
    _stats["tool_errors"] = 0
    logger.info(f"[Agent] START | query={str(ctx.extra.get('query', ''))[:100]}")
    return original_fn(ctx)


# ---------------------------------------------------------------------------
# on_agent_end — Agent 结束时输出统计摘要
# ---------------------------------------------------------------------------
@hooks.register("on_agent_end", priority=10)
def on_agent_end(ctx: HookContext, original_fn):
    elapsed = time.time() - _stats["start_time"] if _stats["start_time"] else 0
    logger.info(
        f"[Agent] END | {elapsed:.1f}s | "
        f"tools={_stats['total_tool_calls']} | errors={_stats['tool_errors']}"
    )
    return original_fn(ctx)


# ---------------------------------------------------------------------------
# on_tool_start — 记录每次工具调用
# ---------------------------------------------------------------------------
@hooks.register("on_tool_start", priority=10)
def log_tool_start(ctx: HookContext, original_fn):
    args_str = json.dumps(ctx.arguments or {}, ensure_ascii=False, default=str)[:200]
    logger.info(f"[Tool] START {ctx.tool_name} | args={args_str}")
    return original_fn(ctx)


# ---------------------------------------------------------------------------
# on_tool_end — 记录工具结果，统计错误
# ---------------------------------------------------------------------------
@hooks.register("on_tool_end", priority=10)
def log_tool_end(ctx: HookContext, original_fn):
    _stats["total_tool_calls"] += 1
    duration = f"{ctx.duration_ms}ms" if ctx.duration_ms else "?"
    error = (ctx.tool_result or {}).get("error")
    if error:
        _stats["tool_errors"] += 1
        logger.warning(f"[Tool] END {ctx.tool_name} | {duration} | ERROR: {error}")
    else:
        logger.info(f"[Tool] END {ctx.tool_name} | {duration} | OK")
    return original_fn(ctx)


# ---------------------------------------------------------------------------
# on_turn_end — 记录每轮进度
# ---------------------------------------------------------------------------
@hooks.register("on_turn_end", priority=10)
def log_turn(ctx: HookContext, original_fn):
    total = ctx.extra.get("total_tool_calls", 0)
    msgs = ctx.extra.get("message_count", 0)
    logger.info(
        f"[Turn {ctx.turn_number}] {ctx.tool_calls_count} tool calls this turn, "
        f"{total} total, {msgs} messages"
    )
    return original_fn(ctx)


# ---------------------------------------------------------------------------
# on_tool_result_format — 精简 SSE 输出中的工具结果
# ---------------------------------------------------------------------------
@hooks.register("on_tool_result_format", priority=10)
def format_result(ctx: HookContext, original_fn):
    tool = ctx.tool_name or ""
    dur = f"{ctx.duration_ms}ms" if ctx.duration_ms else ""
    error = (ctx.tool_result or {}).get("error")

    if error:
        return f"[{tool}] Error: {str(error)[:80]} ({dur})"

    # Let framework handle it by default
    return original_fn(ctx)


# ---------------------------------------------------------------------------
# on_context_compact — 记录上下文压缩事件
# ---------------------------------------------------------------------------
@hooks.register("on_context_compact", priority=10)
def log_compact(ctx: HookContext, original_fn):
    action = ctx.extra.get("compact_action", "unknown")
    logger.info(f"[Context] Compact triggered: {action}")
    return original_fn(ctx)


# ---------------------------------------------------------------------------
# Example: on_system_prompt_build — 追加自定义指令（取消注释以启用）
# ---------------------------------------------------------------------------
# @hooks.register("on_system_prompt_build", priority=50)
# def customize_system_prompt(ctx: HookContext, original_fn):
#     prompt = original_fn(ctx)
#     extra_instruction = "\n\n## Additional Instructions\n- Always respond in Chinese.\n"
#     return prompt + extra_instruction


# ---------------------------------------------------------------------------
# Example: on_before_llm_call — Guardrail（取消注释以启用）
# ---------------------------------------------------------------------------
# from mem_deep_research_core.exceptions import GuardrailError
#
# @hooks.register("on_before_llm_call", priority=10)
# def content_guardrail(ctx: HookContext, original_fn):
#     """Block requests containing sensitive patterns."""
#     messages = ctx.extra.get("messages", [])
#     for msg in messages[-1:]:  # Check only the latest message
#         content = str(msg.get("content", ""))
#         if "DELETE FROM" in content.upper():
#             raise GuardrailError("Blocked: SQL DELETE detected in conversation")
#     return original_fn(ctx)


# ---------------------------------------------------------------------------
# Example: on_tool_filter — 过滤特定工具调用（取消注释以启用）
# ---------------------------------------------------------------------------
# @hooks.register("on_tool_filter", priority=10)
# def filter_tools(ctx: HookContext, original_fn):
#     """Skip tools matching a blacklist pattern."""
#     calls = ctx.extra.get("tool_calls_batch", [])
#     filtered = [c for c in calls if "dangerous" not in c.get("name", "")]
#     ctx.extra["tool_calls_batch"] = filtered
#     return original_fn(ctx)


# ---------------------------------------------------------------------------
# Example: Override language detection (uncomment to use)
# ---------------------------------------------------------------------------
# @hooks.register("on_agent_start", priority=50)
# def custom_language_detect(ctx: HookContext, original_fn):
#     """Use LLM-based language detection instead of char counting."""
#     # Set response_language in extra for downstream components
#     # ctx.extra["response_language"] = my_llm_detect(ctx.query)
#     return original_fn(ctx)


# ---------------------------------------------------------------------------
# Example: on_system_prompt_build — inject user identity (uncomment to use)
# ---------------------------------------------------------------------------
# from mem_deep_research_core.core.user_context import UserContextBuilder
#
# @hooks.register("on_system_prompt_build", priority=50)
# def inject_user_context(ctx: HookContext, original_fn):
#     """Inject user identity into system prompt via hook."""
#     prompt = original_fn(ctx)
#     context = ctx.context or {}
#     if context.get("user_id"):
#         builder = UserContextBuilder(context, chinese_context=False)
#         identity = builder.build_user_identity_context()
#         if identity:
#             prompt += f"\n\n{identity}"
#     return prompt
