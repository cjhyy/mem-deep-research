"""
窗口管理策略抽象

将 context 压缩算法与 ContextManager 解耦，支持可插拔的窗口策略。

内置策略：
  - ObservationMaskingStrategy: 遮蔽旧轮工具输出，保留推理链（零 LLM 成本）
  - LLMSummarizeStrategy: LLM 压缩旧历史为结构化摘要
  - BinaryReductionStrategy: 二分删除中间消息（紧急兜底）

自定义策略：
    from mem_deep_research_core.core.window_strategy import WindowStrategy, WindowContext

    class MyStrategy(WindowStrategy):
        def should_trigger(self, ctx: WindowContext) -> bool:
            return ctx.token_ratio > 0.7

        def apply(self, messages: list, ctx: WindowContext) -> CompressResult:
            # 自定义压缩逻辑
            ...

配置示例：
    context_manager:
      strategies:
        - type: observation_masking
          trigger_ratio: 0.6
          keep_recent: 3
        - type: llm_summarize
          trigger_ratio: 0.8
        - type: binary_reduction
          trigger_ratio: 0.95
"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mem_deep_research_core.core.constants import (
    COMPACT_MIN_CHARS,
    COMPACT_PREVIEW_LENGTH,
    MT,
    PROTECTED_MESSAGE_TYPES,
    SYSTEM_MESSAGE_KEYWORDS,
    TAG_CONTEXT_SUMMARY,
    TAG_OFFLOADED,
)

logger = logging.getLogger("mem_deep_research")


# ============================================================
# 数据结构
# ============================================================


@dataclass
class WindowContext:
    """传递给策略的运行时上下文"""

    current_turn: int
    max_turns: int
    token_count: int
    max_tokens: int
    token_ratio: float  # token_count / max_tokens
    message_count: int
    system_prompt: str = ""
    message_history: list = field(default_factory=list)  # 引用，策略可读不可替换

    # 策略可访问的 registry（只读）
    call_registry: list = field(default_factory=list)  # List[ToolCallRecord]
    compacted_turns: set = field(default_factory=set)

    # token 估算函数
    estimate_tokens_fn: Callable[[str], int] | None = None

    # Session memory（供 SessionMemoryCompactStrategy 使用）
    session_memory: Any = None  # Optional[SessionMemory]

    # Profile 引用（Phase 2a：供 LLMSummarize 后触发 extraction strategy 链）
    profile: Any = None


@dataclass
class CompressResult:
    """策略执行结果"""

    messages_affected: int = 0  # 被处理的消息数
    tokens_saved: int = 0  # 估算节省的 token 数
    summary_text: str = ""  # 如果生成了摘要
    action_label: str = ""  # 策略标识（用于日志）


# ============================================================
# 策略接口
# ============================================================


class WindowStrategy(ABC):
    """窗口压缩策略的抽象接口

    每个策略负责：
    1. should_trigger: 判断是否需要触发
    2. apply: 就地修改 message_history，返回结果
    """

    @abstractmethod
    def should_trigger(self, ctx: WindowContext) -> bool:
        """是否应该触发此策略"""
        ...

    @abstractmethod
    def apply(self, messages: list, ctx: WindowContext) -> CompressResult:
        """执行压缩（就地修改 messages）

        Args:
            messages: message_history 引用，直接修改
            ctx: 运行时上下文

        Returns:
            CompressResult 描述执行结果
        """
        ...

    # Attributes for polymorphic dispatch (override in subclasses)
    supports_async: bool = False
    strategy_type: str = ""


# ============================================================
# 内置策略
# ============================================================

# --- 结果摘要辅助函数 ---


def _get_message_char_count(msg: dict) -> int:
    """计算消息的字符数"""
    content = msg.get("content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict):
                total += len(item.get("text", ""))
            elif isinstance(item, str):
                total += len(item)
        return total
    return len(str(content))


def _is_protected_message(msg: dict) -> bool:
    """判断消息是否受压缩保护（不应被 compact / masking / microcompact 清理）

    优先检查 `_type` 字段（结构化），fallback 到关键词匹配（向后兼容）。
    """
    # Fast path: 结构化类型判断
    msg_type = msg.get("_type")
    if msg_type is not None:
        return msg_type in PROTECTED_MESSAGE_TYPES

    # Fallback: 关键词匹配（无 _type 的旧消息）
    return _content_matches_keywords(msg.get("content"))


def _is_system_message(content) -> bool:
    """判断是否是系统注入的消息（反思提示、hint 等），不应被压缩

    向后兼容接口 — 新代码请优先使用 _is_protected_message(msg)。
    """
    return _content_matches_keywords(content)


def _content_matches_keywords(content) -> bool:
    """通过关键词匹配检测系统注入消息"""
    if isinstance(content, str):
        return any(kw in content for kw in SYSTEM_MESSAGE_KEYWORDS)
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if any(kw in text for kw in SYSTEM_MESSAGE_KEYWORDS):
                    return True
    return False


def _is_offloaded(msg_or_content) -> bool:
    """判断消息是否是已卸载的内容（不应被二次压缩）

    接受完整 message dict 或 content 字段。
    """
    # Fast path: 结构化类型
    if isinstance(msg_or_content, dict) and msg_or_content.get("_type") == MT.OFFLOADED:
        return True

    content = msg_or_content.get("content") if isinstance(msg_or_content, dict) else msg_or_content
    if isinstance(content, str):
        return content.startswith(TAG_OFFLOADED)
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                if item.get("text", "").startswith(TAG_OFFLOADED):
                    return True
    return False


def _extract_key_argument(tool_name: str, arguments: dict) -> str | None:
    """提取工具的关键参数值（用于摘要展示）"""
    for key in ("query", "q", "search_query", "url", "keyword", "entity", "input", "text"):
        val = arguments.get(key)
        if val and isinstance(val, str):
            return val[:60]
    if len(arguments) == 1:
        val = list(arguments.values())[0]
        if isinstance(val, str) and len(val) < 80:
            return val
    return None


def _count_results(result_text: str) -> int | None:
    """尝试从结果中提取数量"""
    try:
        data = json.loads(result_text)
        for key in ("total", "count", "total_count", "num_results"):
            if key in data and isinstance(data[key], int):
                return data[key]
        if isinstance(data, list):
            return len(data)
        for key in ("items", "results", "data", "records", "organic_results"):
            if key in data and isinstance(data[key], list):
                return len(data[key])
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _estimate_ratio(
    messages: list,
    system_prompt: str,
    max_tokens: int,
    estimate_fn: Callable | None = None,
    chars_per_token: float = 3.5,
) -> float:
    """估算当前 context token 占比"""
    if max_tokens <= 0:
        return 0.0
    parts = [system_prompt]
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
    text = " ".join(parts)
    if estimate_fn:
        try:
            return estimate_fn(text) / max_tokens
        except Exception:
            pass
    return (len(text) / chars_per_token) / max_tokens


# ============================================================
# Strategy 1: Observation Masking（观察遮蔽）
# ============================================================


class ObservationMaskingStrategy(WindowStrategy):
    """Level 1: 遮蔽旧轮次的工具输出，用结构化一行摘要替换

    零 LLM 成本。保留推理链（assistant 消息不动），只压缩工具结果。
    参考: JetBrains NeurIPS 2025 "The Complexity Trap"

    Args:
        trigger_ratio: token 占比超过此值时触发
        keep_recent: 保留最近 N 轮完整结果
        chars_per_token: 无 tiktoken 时的估算比例
    """

    strategy_type = "observation_masking"

    def __init__(
        self,
        trigger_ratio: float = 0.6,
        keep_recent: int = 3,
        chars_per_token: float = 3.5,
        preview_length: int = COMPACT_PREVIEW_LENGTH,
    ):
        self.trigger_ratio = trigger_ratio
        self.keep_recent = keep_recent
        self.chars_per_token = chars_per_token
        self.preview_length = preview_length

    def should_trigger(self, ctx: WindowContext) -> bool:
        if ctx.max_tokens <= 0:
            # 无 token 限制时，按轮次判断
            return ctx.current_turn > self.keep_recent
        return ctx.token_ratio >= self.trigger_ratio

    def apply(self, messages: list, ctx: WindowContext) -> CompressResult:
        if ctx.max_tokens <= 0:
            return self._apply_by_turns(messages, ctx)
        return self._apply_by_tokens(messages, ctx)

    def _apply_by_tokens(self, messages: list, ctx: WindowContext) -> CompressResult:
        """Token-aware 压缩：从最旧的开始，直到降到目标比例"""
        target_ratio = self.trigger_ratio - 0.1
        candidates = self._collect_candidates(messages, ctx.current_turn)

        if not candidates:
            return CompressResult(action_label="observation_masking")

        compacted = 0
        for msg_idx, est_turn, _ in candidates:
            current_ratio = _estimate_ratio(
                messages,
                ctx.system_prompt,
                ctx.max_tokens,
                ctx.estimate_tokens_fn,
                self.chars_per_token,
            )
            if current_ratio <= target_ratio:
                break

            msg = messages[msg_idx]
            summary = self._generate_summary(msg, est_turn, ctx.call_registry)
            if summary:
                msg["content"] = [{"type": "text", "text": summary}]
                ctx.compacted_turns.add(est_turn)
                compacted += 1

        if compacted > 0:
            new_ratio = _estimate_ratio(
                messages,
                ctx.system_prompt,
                ctx.max_tokens,
                ctx.estimate_tokens_fn,
                self.chars_per_token,
            )
            logger.info(
                f"[WINDOW] ObservationMasking: compacted {compacted} messages "
                f"(ratio: {ctx.token_ratio:.1%} -> {new_ratio:.1%})"
            )

        return CompressResult(
            messages_affected=compacted,
            action_label="observation_masking",
        )

    def _apply_by_turns(self, messages: list, ctx: WindowContext) -> CompressResult:
        """回退策略：按轮次压缩"""
        cutoff_turn = ctx.current_turn - self.keep_recent
        if cutoff_turn <= 0:
            return CompressResult(action_label="observation_masking")

        compacted = 0
        estimated_turn = 0
        i = 1
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")

            if role == "assistant":
                estimated_turn += 1
            elif estimated_turn > 0 and estimated_turn <= cutoff_turn:
                # 白名单：只压缩显式标记为 TOOL_RESULT 的消息（role 可以是 user 或 tool）
                if msg.get("_type") == MT.TOOL_RESULT:
                    if _get_message_char_count(msg) > COMPACT_MIN_CHARS:
                        summary = self._generate_summary(msg, estimated_turn, ctx.call_registry)
                        if summary:
                            msg["content"] = [{"type": "text", "text": summary}]
                            ctx.compacted_turns.add(estimated_turn)
                            compacted += 1
            i += 1

        if compacted > 0:
            logger.info(
                f"[WINDOW] ObservationMasking: compacted {compacted} messages "
                f"(turns 1-{cutoff_turn})"
            )

        return CompressResult(messages_affected=compacted, action_label="observation_masking")

    def _collect_candidates(
        self,
        messages: list,
        current_turn: int,
    ) -> list[tuple[int, int, int]]:
        """收集可压缩的消息: [(msg_index, estimated_turn, char_count)]"""
        cutoff_turn = current_turn - self.keep_recent
        if cutoff_turn <= 0:
            return []

        candidates = []
        estimated_turn = 0
        i = 1
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")

            if role == "assistant":
                estimated_turn += 1
            elif estimated_turn > 0 and estimated_turn <= cutoff_turn:
                # 白名单：只收集显式标记为 TOOL_RESULT 的消息（role 可以是 user 或 tool）
                if msg.get("_type") == MT.TOOL_RESULT and not _is_offloaded(msg):
                    char_count = _get_message_char_count(msg)
                    if char_count > COMPACT_MIN_CHARS:
                        candidates.append((i, estimated_turn, char_count))
            i += 1

        return candidates

    def _generate_summary(self, msg: dict, turn: int, call_registry: list) -> str | None:
        """生成结构化摘要，保留足够信息供最终 summary 使用"""
        turn_records = [r for r in call_registry if r.turn == turn]
        preview_len = self.preview_length

        if turn_records:
            lines = []
            for record in turn_records:
                # Header line
                header_parts = [f"Turn {record.turn}"]
                key_arg = _extract_key_argument(record.tool_name, record.arguments)
                if key_arg:
                    header_parts.append(f'{record.tool_name}("{key_arg}")')
                else:
                    header_parts.append(record.tool_name)
                header_parts.append(f"{record.result_chars} chars")
                if "search" in record.tool_name.lower():
                    count = _count_results(record.result_full)
                    if count is not None:
                        header_parts.append(f"{count} results")
                lines.append(f"[{' | '.join(header_parts)}]")

                # Hint: offloaded content can be recalled via read_result
                expected_ref = f"turn{record.turn}_{record.tool_name}_{record.result_chars}chars.txt"
                if record.result_chars > 3000:
                    lines.append(f"  (use read_result(\"{expected_ref}\") for full content)")

                # Result preview — use full result_full for better extraction
                if preview_len > 0 and record.result_full:
                    preview = record.result_full[:preview_len].replace("\n", " ")
                    lines.append(f"  {preview}")
                elif record.result_brief:
                    preview = record.result_brief[:preview_len].replace("\n", " ")
                    lines.append(f"  {preview}")
            return "\n".join(lines)

        char_count = _get_message_char_count(msg)
        return f"[Turn {turn} | tool result compacted | {char_count} chars]"


# ============================================================
# Strategy 1.5: Session Memory Compact（零 LLM 成本压缩）
# ============================================================


class SessionMemoryCompactStrategy(WindowStrategy):
    """Level 1.5: 利用 SessionMemory 已有的 key_findings 生成零成本压缩摘要

    在 LLMSummarize 之前尝试：如果 session_memory 有足够的 findings，
    用它们替代旧消息，完全不需要 LLM 调用。

    参考 Claude Code 的 trySessionMemoryCompaction：
    先尝试零成本路径，失败了才回退到 LLM summarize。

    Args:
        trigger_ratio: token 占比超过此值时触发
        keep_recent: 保留最近 N 轮
        min_findings: 最少需要多少条 findings 才值得做 session memory compact
    """

    strategy_type = "session_memory_compact"

    def __init__(
        self,
        trigger_ratio: float = 0.7,
        keep_recent: int = 3,
        min_findings: int = 2,
    ):
        self.trigger_ratio = trigger_ratio
        self.keep_recent = keep_recent
        self.min_findings = min_findings

    def should_trigger(self, ctx: WindowContext) -> bool:
        if ctx.max_tokens <= 0:
            return False
        if ctx.token_ratio < self.trigger_ratio:
            return False
        # 只在有足够 session memory 时触发
        if ctx.session_memory is None:
            return False
        if hasattr(ctx.session_memory, "key_findings"):
            return len(ctx.session_memory.key_findings) >= self.min_findings
        return False

    def apply(self, messages: list, ctx: WindowContext) -> CompressResult:
        """用 session memory 的 findings 替换旧消息"""
        if ctx.session_memory is None:
            return CompressResult(action_label="session_memory_compact")

        cutoff_turn = ctx.current_turn - self.keep_recent
        if cutoff_turn <= 0:
            return CompressResult(action_label="session_memory_compact")

        # 生成基于 session memory 的摘要
        memory_summary = ctx.session_memory.to_context_string()
        if not memory_summary or len(memory_summary) < 50:
            return CompressResult(action_label="session_memory_compact")

        # 收集要替换的旧消息
        old_indices = []
        estimated_turn = 0
        for i in range(1, len(messages)):
            msg = messages[i]
            role = msg.get("role", "")

            if role == "assistant":
                estimated_turn += 1

            if estimated_turn > 0 and estimated_turn <= cutoff_turn:
                if not _is_protected_message(msg):
                    old_indices.append(i)
            elif estimated_turn > cutoff_turn:
                break

        if not old_indices:
            return CompressResult(action_label="session_memory_compact")

        # 删除旧消息（从后往前）
        for idx in sorted(old_indices, reverse=True):
            if idx < len(messages):
                messages.pop(idx)

        # 插入 session memory 摘要
        from mem_deep_research_core.core.constants import MT

        summary_msg = {
            "role": "user",
            "_type": MT.CONTEXT_SUMMARY,
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{TAG_CONTEXT_SUMMARY} — session memory compact, turns 1-{cutoff_turn}]\n\n"
                        f"{memory_summary}\n\n"
                        f"[End of session memory summary. The detailed conversation continues below.]"
                    ),
                }
            ],
        }
        messages.insert(1, summary_msg)

        logger.info(
            f"[WINDOW] SessionMemoryCompact: replaced {len(old_indices)} messages "
            f"with session memory summary (turns 1-{cutoff_turn})"
        )

        return CompressResult(
            messages_affected=len(old_indices),
            summary_text=memory_summary,
            action_label="session_memory_compact",
        )


# ============================================================
# Strategy 2: LLM Summarize（LLM 压缩）
# ============================================================


class LLMSummarizeStrategy(WindowStrategy):
    """Level 2: 用 LLM 压缩旧历史为一条结构化摘要

    把旧 turn 打包发给 LLM 做摘要，结果作为 [RESEARCH CONTEXT SUMMARY]
    插在 history 开头，原始旧 turn 全删。

    Args:
        trigger_ratio: token 占比超过此值时触发
        keep_recent: 保留最近 N 轮
        llm_call_fn: 异步 LLM 调用函数，签名: async (system_prompt, messages, purpose) -> str
    """

    supports_async = True
    strategy_type = "llm_summarize"

    def __init__(
        self,
        trigger_ratio: float = 0.8,
        keep_recent: int = 3,
    ):
        self.trigger_ratio = trigger_ratio
        self.keep_recent = keep_recent

        # 运行时状态
        self._summary_text: str | None = None
        self._summarized_up_to_turn: int = 0

    def should_trigger(self, ctx: WindowContext) -> bool:
        if ctx.max_tokens <= 0:
            return False
        return ctx.token_ratio >= self.trigger_ratio

    def apply(self, messages: list, ctx: WindowContext) -> CompressResult:
        # 同步版本只返回标记，实际压缩由 apply_async 完成
        return CompressResult(action_label="need_summarize")

    async def apply_async(
        self,
        messages: list,
        ctx: WindowContext,
        llm_call_fn: Callable,
    ) -> CompressResult:
        """异步执行 LLM 压缩"""
        cutoff_turn = ctx.current_turn - self.keep_recent

        if cutoff_turn <= self._summarized_up_to_turn:
            return CompressResult(action_label="llm_summarize")

        # 收集要压缩的消息
        old_messages, old_indices, start_idx = self._collect_old_messages(messages, cutoff_turn)

        if not old_messages:
            return CompressResult(action_label="llm_summarize")

        # 构建压缩 prompt
        summary_prompt = self._build_prompt(old_messages, cutoff_turn)

        try:
            summary = await llm_call_fn(
                "You are a concise research assistant. Summarize the research progress.",
                [{"role": "user", "content": [{"type": "text", "text": summary_prompt}]}],
                "Context summarization",
            )

            if not summary or not summary.strip():
                logger.warning("[WINDOW] LLMSummarize: LLM returned empty summary")
                return CompressResult(action_label="llm_summarize")

            # 删除旧消息（从后往前）
            for idx in sorted(old_indices, reverse=True):
                if idx < len(messages):
                    messages.pop(idx)

            # 删除旧 summary（如果存在）
            if self._summary_text and start_idx == 2 and len(messages) > 1:
                msg1 = messages[1]
                content = msg1.get("content", "")
                if isinstance(content, list) and content:
                    text = content[0].get("text", "") if isinstance(content[0], dict) else ""
                else:
                    text = str(content)
                if text.startswith(TAG_CONTEXT_SUMMARY):
                    messages.pop(1)

            # 插入新 summary
            from mem_deep_research_core.core.constants import MT

            summary_msg = {
                "role": "user",
                "_type": MT.CONTEXT_SUMMARY,
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{TAG_CONTEXT_SUMMARY} — turns 1-{cutoff_turn}]\n\n"
                            f"{summary}\n\n"
                            f"[End of summary. The detailed conversation continues below.]"
                        ),
                    }
                ],
            }
            messages.insert(1, summary_msg)

            self._summary_text = summary
            self._summarized_up_to_turn = cutoff_turn

            # 触发 extraction strategy 链的 on_compact（Phase 2a）
            # SummaryEvidenceStrategy 会从 summary 的 ## Evidence 段抽取细节存 session_memory
            # 兼容：ctx.profile 未注入时走老逻辑（直接调 _extract_evidence_from_summary）
            if ctx.profile is not None and ctx.session_memory is not None:
                from mem_deep_research_core.memory_extraction import ExtractionContext

                ext_ctx = ExtractionContext(
                    turn_number=cutoff_turn,
                    task_description="",
                    mode="",
                    session_memory=ctx.session_memory,
                    context_manager=None,  # on_compact 下 strategies 不应碰 context_manager
                    llm_client=None,
                )
                await ctx.profile.run_strategies_on_compact(
                    summary, cutoff_turn, ext_ctx,
                )
            elif ctx.session_memory is not None:
                # Fallback：没 profile 就保留老行为（测试/轻量使用场景）
                self._extract_evidence_from_summary(summary, cutoff_turn, ctx.session_memory)

            logger.info(
                f"[WINDOW] LLMSummarize: summarized turns 1-{cutoff_turn}, "
                f"removed {len(old_indices)} messages"
            )

            return CompressResult(
                messages_affected=len(old_indices),
                summary_text=summary,
                action_label="llm_summarize",
            )

        except Exception as e:
            logger.warning(f"[WINDOW] LLMSummarize failed: {e}")
            return CompressResult(action_label="llm_summarize")

    def _collect_old_messages(
        self,
        messages: list,
        cutoff_turn: int,
    ) -> tuple[list, list, int]:
        """收集要压缩的旧消息"""
        # 如果 history[1] 是之前的 summary，从 2 开始
        start_idx = 1
        if self._summary_text and len(messages) > 1:
            first_msg = messages[1]
            content = first_msg.get("content", "")
            if isinstance(content, list) and content:
                text = content[0].get("text", "") if isinstance(content[0], dict) else ""
            elif isinstance(content, str):
                text = content
            else:
                text = ""
            if text.startswith(TAG_CONTEXT_SUMMARY):
                start_idx = 2

        old_messages = []
        old_indices = []
        estimated_turn = 0
        i = start_idx

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")

            if role == "assistant":
                estimated_turn += 1

            if estimated_turn <= cutoff_turn:
                old_messages.append(msg)
                old_indices.append(i)
            elif estimated_turn > cutoff_turn:
                break

            i += 1

        return old_messages, old_indices, start_idx

    def _build_prompt(self, messages: list, up_to_turn: int) -> str:
        """构建给 LLM 的压缩 prompt"""
        lines = [
            f"Below is the research conversation from turns 1-{up_to_turn}.",
            "Summarize the KEY FINDINGS, FACTS DISCOVERED, and IMPORTANT DATA collected.",
            "",
            "Requirements:",
            "- Preserve ALL specific data: numbers, names, dates, URLs, IDs",
            "- Preserve which tools were used and what they found",
            "- Preserve any error or dead-end information (so the agent doesn't retry)",
            "- Be concise but DO NOT lose factual information",
            "- Output in the same language as the original content",
            "",
            "=== CONVERSATION TO SUMMARIZE ===",
            "",
        ]

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                text = "\n".join(text_parts)
            elif isinstance(content, str):
                text = content
            else:
                text = str(content)

            if len(text) > 3000:
                text = text[:2500] + f"\n... [truncated, total {len(text)} chars]"

            lines.append(f"[{role.upper()}]")
            lines.append(text)
            lines.append("")

        lines.append("=== END OF CONVERSATION ===")
        lines.append("")
        lines.append(
            "Now provide a concise summary. Your output MUST have two sections:\n"
            "\n"
            "## Summary\n"
            "A concise narrative preserving all key facts, specific data, and tool findings.\n"
            "\n"
            "## Evidence\n"
            "A bullet list of the most important verified facts extracted from tool results. "
            "Each bullet should be one specific, data-rich fact (include numbers, names, dates, URLs). "
            "This section will survive further context compression, so prioritize the facts "
            "that are most critical for answering the research question."
        )

        return "\n".join(lines)

    @staticmethod
    def _extract_evidence_from_summary(summary: str, up_to_turn: int, session_memory) -> None:
        """从 LLM 压缩摘要中提取 ## Evidence 部分，存入 session_memory。"""
        from mem_deep_research_core.core.memory import EvidenceItem

        # 查找 ## Evidence 部分
        evidence_start = summary.find("## Evidence")
        if evidence_start == -1:
            return

        evidence_text = summary[evidence_start + len("## Evidence") :]
        # 截取到下一个 ## 或末尾
        next_section = evidence_text.find("\n## ")
        if next_section != -1:
            evidence_text = evidence_text[:next_section]
        evidence_text = evidence_text.strip()

        if not evidence_text:
            return

        session_memory.add_evidence(
            EvidenceItem(
                tool_name="llm_summarize",
                turn=up_to_turn,
                summary=evidence_text[:1000],
            )
        )
        logger.info(
            f"[WINDOW] LLMSummarize: extracted evidence from summary "
            f"(turns 1-{up_to_turn}, {len(evidence_text)} chars)"
        )

    def reset(self):
        """重置状态"""
        self._summary_text = None
        self._summarized_up_to_turn = 0


# ============================================================
# Strategy 3: Binary Reduction（紧急裁剪）
# ============================================================


class BinaryReductionStrategy(WindowStrategy):
    """Level 3: 二分删除中间消息

    保留首条消息 + 最后 N 条，删除中间一半。最后手段。

    .. warning::
        ``apply()`` 会**原地修改**传入的 messages 列表。调用方如果需要
        保留原始列表，应先传入副本：``strategy.apply(list(messages), ctx)``。

    Args:
        trigger_ratio: token 占比超过此值时触发（通常 0.95）
        keep_tail: 保留最后 N 条消息
    """

    strategy_type = "binary_reduction"

    def __init__(
        self,
        trigger_ratio: float = 0.95,
        keep_tail: int = 2,
    ):
        self.trigger_ratio = trigger_ratio
        self.keep_tail = keep_tail

    def should_trigger(self, ctx: WindowContext) -> bool:
        # 通常由 emergency 场景触发，不走 ratio 判断
        if ctx.max_tokens <= 0:
            return False
        return ctx.token_ratio >= self.trigger_ratio

    def apply(self, messages: list, ctx: WindowContext) -> CompressResult:
        if len(messages) <= self.keep_tail + 2:
            return CompressResult(action_label="binary_reduction")

        middle = messages[1 : -self.keep_tail]
        keep_count = max(1, len(middle) // 2)
        kept = middle[-keep_count:]
        removed = len(middle) - len(kept)
        messages[1 : -self.keep_tail] = kept

        logger.info(
            f"[WINDOW] BinaryReduction: removed {removed} messages, "
            f"history now {len(messages)} messages"
        )

        return CompressResult(
            messages_affected=removed,
            action_label="binary_reduction",
        )


# ============================================================
# 策略管道
# ============================================================


@dataclass
class StrategyEntry:
    """策略管道条目"""

    strategy: WindowStrategy
    trigger_ratio: float = 0.0  # 覆盖策略内部的 trigger_ratio（可选）


class WindowStrategyPipeline:
    """策略管道：按 trigger_ratio 从低到高排序，依次检查和执行

    Usage:
        pipeline = WindowStrategyPipeline([
            ObservationMaskingStrategy(trigger_ratio=0.6),
            LLMSummarizeStrategy(trigger_ratio=0.8),
            BinaryReductionStrategy(trigger_ratio=0.95),
        ])

        # 同步管理（每轮调用）
        action = pipeline.manage(messages, ctx)

        # 异步 LLM 压缩（当 action == "need_summarize"）
        await pipeline.apply_summarize(messages, ctx, llm_call_fn)
    """

    # Circuit breaker: skip LLMSummarize after N consecutive failures
    SUMMARIZE_FUSE_THRESHOLD = 3

    def __init__(self, strategies: list[WindowStrategy] | None = None):
        self.strategies = strategies or self.default_strategies()
        self._summarize_consecutive_failures: int = 0
        self._summarize_fused: bool = False

    @staticmethod
    def default_strategies() -> list[WindowStrategy]:
        """默认四级策略（L1 → L1.5 → L2 → L3）"""
        return [
            ObservationMaskingStrategy(trigger_ratio=0.6, keep_recent=3),
            SessionMemoryCompactStrategy(trigger_ratio=0.7, keep_recent=3),
            LLMSummarizeStrategy(trigger_ratio=0.8, keep_recent=3),
            BinaryReductionStrategy(trigger_ratio=0.95),
        ]

    def manage(self, messages: list, ctx: WindowContext) -> str:
        """同步管理入口：检查并执行适用的策略

        Returns:
            执行的策略标识: "none" | "observation_masking" | "need_summarize" | ...
        """
        last_action = "none"

        for strategy in self.strategies:
            # Circuit breaker: skip LLMSummarize if fused
            if (
                strategy.strategy_type == "llm_summarize"
                and self._summarize_fused
            ):
                logger.info(
                    "[WINDOW] LLMSummarize fused (circuit breaker active), skipping"
                )
                continue

            if strategy.should_trigger(ctx):
                result = strategy.apply(messages, ctx)
                if result.action_label:
                    last_action = result.action_label

                # 如果需要异步 LLM 处理，提前返回
                if result.action_label == "need_summarize":
                    return "need_summarize"

                # 更新 ratio
                if ctx.max_tokens > 0 and result.messages_affected > 0:
                    ctx.token_ratio = _estimate_ratio(
                        messages,
                        ctx.system_prompt,
                        ctx.max_tokens,
                        ctx.estimate_tokens_fn,
                    )

        return last_action

    async def apply_summarize(
        self,
        messages: list,
        ctx: WindowContext,
        llm_call_fn: Callable,
    ) -> bool:
        """执行异步 LLM 压缩（找到 LLMSummarizeStrategy 并调用）

        Tracks consecutive failures and trips the circuit breaker after
        SUMMARIZE_FUSE_THRESHOLD consecutive failures.
        """
        if self._summarize_fused:
            logger.info("[WINDOW] LLMSummarize fused, skipping apply_summarize")
            return False

        for strategy in self.strategies:
            if hasattr(strategy, "apply_async") and strategy.supports_async:
                result = await strategy.apply_async(messages, ctx, llm_call_fn)
                if result.messages_affected > 0:
                    # Success: reset failure counter
                    self._summarize_consecutive_failures = 0
                    return True
                else:
                    # Failure: increment and check fuse
                    self._summarize_consecutive_failures += 1
                    if self._summarize_consecutive_failures >= self.SUMMARIZE_FUSE_THRESHOLD:
                        self._summarize_fused = True
                        logger.warning(
                            f"[WINDOW] LLMSummarize circuit breaker tripped after "
                            f"{self._summarize_consecutive_failures} consecutive failures. "
                            f"Skipping L2 for remainder of session."
                        )
                    return False
        return False

    def apply_emergency(
        self,
        messages: list,
        ctx: WindowContext,
    ) -> int:
        """紧急压缩：先用激进的 ObservationMasking，再用 BinaryReduction"""
        total_affected = 0

        # 先找 ObservationMasking 策略，用 keep_recent=1 执行
        for strategy in self.strategies:
            if strategy.strategy_type == "observation_masking":
                original_keep = strategy.keep_recent
                strategy.keep_recent = 1
                result = strategy.apply(messages, ctx)
                strategy.keep_recent = original_keep
                total_affected += result.messages_affected

                # 检查是否已降到安全水位
                if ctx.max_tokens > 0:
                    ratio = _estimate_ratio(
                        messages,
                        ctx.system_prompt,
                        ctx.max_tokens,
                        ctx.estimate_tokens_fn,
                    )
                    if ratio < 0.9:
                        logger.info(
                            f"[WINDOW] Emergency: ObservationMasking sufficient (ratio={ratio:.1%})"
                        )
                        return total_affected
                break

        # 再用 BinaryReduction
        for strategy in self.strategies:
            if strategy.strategy_type == "binary_reduction":
                result = strategy.apply(messages, ctx)
                total_affected += result.messages_affected
                break

        # 没配 BinaryReduction 的话，用默认的
        if total_affected == 0:
            fallback = BinaryReductionStrategy()
            result = fallback.apply(messages, ctx)
            total_affected += result.messages_affected

        return total_affected

    def reset(self):
        """重置所有策略状态"""
        self._summarize_consecutive_failures = 0
        self._summarize_fused = False
        for strategy in self.strategies:
            if hasattr(strategy, "reset"):
                strategy.reset()
