"""
主执行循环模块

从 Orchestrator 拆分，负责 Agent 主循环的执行逻辑：
- 轮次循环控制
- 工具调用执行与去重
- 监控检查与升级
- 反思检查点注入
- 最终摘要生成
"""

import asyncio
import json as _json
import logging
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mem_deep_research_core.core.constants import (
    BUILTIN_TOOL_READ_RESULT,
    BUILTIN_TOOL_SEARCH,
    BUILTIN_TOOL_SPAWN_AGENT,
    BUILTIN_TOOL_UPDATE_TODO,
    EVIDENCE_MAX_CHARS,
    CONCURRENT_SAFE_TOOL_SEGMENTS,
    build_context_compression_notice,
    DEFAULT_MAX_CONCURRENT_SUBAGENTS,
    DEFAULT_TASK_TOKEN_BUDGET,
    DEFAULT_TEMPERATURE_BOOST,
    DEFAULT_TEMPERATURE_BOOST_CAP,
    EXECUTION_MODE_AUTO,
    EXECUTION_MODE_DEEP,
    EXECUTION_MODE_QUICK,
    EXECUTION_MODE_SIMPLE_AUTO,
    EXECUTION_MODE_STANDARD,
    MAX_CONTEXT_LIMIT_RETRIES,
    MAX_SPAWN_DEPTH,
    QUICK_MODE_MAX_TURNS,
    SUB_AGENT_PREFIX,
    TAG_COLLECTED_SOURCES,
    TAG_OFFLOADED,
    TAG_TASK_PLAN,
    MT,
    TOKEN_BUDGET_HARD_RATIO,
    TOKEN_BUDGET_WARNING_RATIO,
    generate_message_id,
)
from mem_deep_research_core.core.hooks import HookContext, HookRegistry
from mem_deep_research_core.core.llm_call_handler import generate_reflection_prompt
from mem_deep_research_core.core.memory import EvidenceItem, SessionMemory
from mem_deep_research_core.core.monitoring import (
    EscalationAction,
    TurnCounter,
)

logger = logging.getLogger("mem_deep_research")


@dataclass
class MainLoopContext:
    """Bundles all dependencies for MainLoopRunner to avoid 25-param constructor."""

    cfg: Any
    monitor: Any
    context_manager: Any
    stream_handler: Any
    tool_executor: Any
    sub_agent_runner: Any
    llm_handler: Any
    summary_handler: Any
    task_planner: Any
    inline_skill_selector: Any
    llm_client: Any
    output_formatter: Any
    task_log: Any
    context: dict
    chinese_context: bool

    # Callbacks
    handle_llm_call: Callable
    handle_summary: Callable
    intercept_key_message: Callable
    streaming_final_message: Callable
    stream_tool_reasoning: Callable
    extract_recent_tool_names: Callable
    deduplicate_trailing_messages: Callable

    # Language (with default)
    response_language: str = "auto"

    # Execution mode
    execution_mode: str = "auto"

    # TodoTracker instance (optional)
    todo_tracker: Any = None

    # Agent name for stream events and hooks (default "main")
    agent_name: str = "main"

    # Long-term memory (optional, cross-session persistence)
    long_term_memory: Any = None

    # Current spawn nesting depth (0 = main agent, increments per spawn)
    spawn_depth: int = 0

    # HookRegistry instance (必传，由 Orchestrator 注入)
    hooks: Any = None

    # DeferredToolManager instance (optional, for lazy tool schema loading)
    deferred_tool_manager: Any = None

    # Transcript instance (optional, for event logging)
    transcript: Any = None

    # FileStateCache instance (optional, shared file content LRU cache)
    file_state_cache: Any = None

    # Skill commands registry (optional, unified SkillCommand dict)
    skill_commands: dict = field(default_factory=dict)

    # Router LLM client (optional, lightweight model for auto mode task classification)
    router_llm_client: Any = None

    # ConfigLoader instance (optional, needed for spawning sub-agents)
    config_loader: Any = None


def _get_spawn_agent_tool_definition() -> dict:
    """Built-in spawn_agent tool — MCP server format for system prompt rendering."""
    return {
        "name": "builtin-spawn-agent",
        "tools": [
            {
                "name": BUILTIN_TOOL_SPAWN_AGENT,
                "description": (
                    "Spawn a temporary sub-agent to handle ONE focused subtask independently. "
                    "The sub-agent has isolated context and the same tools as you.\n\n"
                    "WHEN TO USE:\n"
                    "- The overall task has multiple INDEPENDENT subtasks that can run in parallel "
                    "(spawn one agent per subtask, not one agent for everything).\n"
                    "- A subtask requires deep investigation that would consume too much of your context window.\n\n"
                    "WHEN NOT TO USE:\n"
                    "- The task is simple enough to do yourself in a few tool calls.\n"
                    "- You would put ALL the work into a single spawn — that just adds overhead with no benefit. "
                    "Either split into multiple spawns or do it yourself.\n\n"
                    "The sub-agent returns its final answer as the tool result."
                ),
                "schema": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "Detailed description of the subtask for the sub-agent to complete",
                        },
                        "max_turns": {
                            "type": "integer",
                            "description": "Maximum execution turns (1-10, default 5). "
                            "Each spawn should be a focused subtask — split large work into multiple spawns instead of raising this.",
                        },
                    },
                    "required": ["task_description"],
                },
            }
        ],
    }


def _get_read_result_tool_definition() -> dict:
    """Built-in read_result tool — allows LLM to recall offloaded/masked tool results."""
    return {
        "name": "builtin-read-result",
        "tools": [
            {
                "name": BUILTIN_TOOL_READ_RESULT,
                "description": (
                    "Read back the full content of a previously offloaded or compressed tool result. "
                    "Use this when you need detailed data from an earlier tool call whose result was "
                    "offloaded to a file or compressed. Pass the file reference from an "
                    "[OFFLOADED:ref|chars] marker (e.g. 'toolmsg_abcd1234.txt') "
                    "or 'turn:N' to retrieve all cached results from turn N."
                ),
                "schema": {
                    "type": "object",
                    "properties": {
                        "ref": {
                            "type": "string",
                            "description": (
                                "File reference from an [OFFLOADED:...] marker, "
                                "or 'turn:N' to get all cached results from turn N"
                            ),
                        },
                    },
                    "required": ["ref"],
                },
            }
        ],
    }


_RE_EVIDENCE = re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL)
_RE_SOURCE = re.compile(r"\(source:\s*(https?://[^\s)]+)\)", re.IGNORECASE)
_RE_CONFIDENCE = re.compile(r"\(confidence:\s*(high|medium|low)\)", re.IGNORECASE)
_RE_OFFLOAD_EVIDENCE = re.compile(
    r'<offload_evidence\s+ref="([^"]+)">(.*?)</offload_evidence>', re.DOTALL
)


def _extract_offload_evidence(assistant_text: str, context_manager) -> str:
    """从 LLM 回复中提取 <offload_evidence ref="..."> 块，绑定到 offload registry。

    Returns:
        清理掉 <offload_evidence> 标签后的 assistant_text
    """
    matches = _RE_OFFLOAD_EVIDENCE.findall(assistant_text)
    if not matches:
        return assistant_text

    for ref, block in matches:
        lines = [
            l.strip().lstrip("- ").strip()
            for l in block.strip().split("\n")
            if l.strip() and l.strip() != "-"
        ]
        if lines:
            context_manager.update_offload_evidence(ref, lines)

    return _RE_OFFLOAD_EVIDENCE.sub("", assistant_text).strip()


def _parse_evidence_line(line: str) -> tuple[str, str, str]:
    """从单行证据中提取 source_url 和 confidence，返回 (clean_text, source_url, confidence)。"""
    source_url = ""
    confidence = ""
    m = _RE_SOURCE.search(line)
    if m:
        source_url = m.group(1)
        line = line[: m.start()] + line[m.end() :]
    m = _RE_CONFIDENCE.search(line)
    if m:
        confidence = m.group(1).lower()
        line = line[: m.start()] + line[m.end() :]
    return line.strip().strip("-").strip(), source_url, confidence


def _extract_evidence_tags(
    assistant_text: str,
    turn: int,
    session_memory: SessionMemory,
) -> str:
    """从 LLM 回复中提取 <evidence> 标签内容，存入 session_memory。

    支持两种格式：
    1. 逐行结构化（推荐）：每行含 (source: URL) (confidence: high/medium/low)
    2. 整块文本（兼容旧格式）

    Returns:
        清理掉 <evidence> 标签后的 assistant_text
    """
    matches = _RE_EVIDENCE.findall(assistant_text)
    if not matches:
        return assistant_text

    count = 0
    for block in matches:
        block = block.strip()
        if not block:
            continue

        # 尝试逐行解析（每行以 - 开头）
        lines = [l.strip() for l in block.split("\n") if l.strip().startswith("-")]
        if lines:
            for line in lines:
                text, source_url, confidence = _parse_evidence_line(line)
                if not text:
                    continue
                if len(text) > EVIDENCE_MAX_CHARS:
                    text = text[:EVIDENCE_MAX_CHARS] + "..."
                session_memory.add_evidence(
                    EvidenceItem(
                        tool_name="llm_extraction",
                        turn=turn,
                        summary=text,
                        source_url=source_url,
                        confidence=confidence,
                    )
                )
                count += 1
        else:
            # 整块作为一个 EvidenceItem（兼容旧格式）
            if len(block) > EVIDENCE_MAX_CHARS:
                block = block[:EVIDENCE_MAX_CHARS] + "..."
            session_memory.add_evidence(
                EvidenceItem(
                    tool_name="llm_extraction",
                    turn=turn,
                    summary=block,
                )
            )
            count += 1

    logger.debug(f"[Evidence] Extracted {count} evidence items from turn {turn}")

    # 从 assistant 文本中移除 <evidence> 标签
    return _RE_EVIDENCE.sub("", assistant_text).strip()


def _strip_evidence_from_last_assistant(message_history: list) -> None:
    """清理 message_history 中最后一条 assistant 消息里的 evidence 相关标签。"""

    def _clean(text: str) -> str:
        text = _RE_EVIDENCE.sub("", text)
        text = _RE_OFFLOAD_EVIDENCE.sub("", text)
        return text.strip()

    for msg in reversed(message_history):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and ("<evidence>" in content or "<offload_evidence" in content):
            msg["content"] = _clean(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    t = item.get("text", "")
                    if "<evidence>" in t or "<offload_evidence" in t:
                        item["text"] = _clean(t)
        break


_RE_RESPONSE_LANGUAGE = re.compile(
    r"<response_language>\s*([\w]+)\s*</response_language>", re.IGNORECASE
)


def _extract_response_language(text: str) -> str | None:
    """Extract language from <response_language>X</response_language> tag."""
    m = _RE_RESPONSE_LANGUAGE.search(text)
    return m.group(1).strip() if m else None


def _strip_response_language_tag(text: str) -> str:
    """Remove <response_language> tag from text."""
    return _RE_RESPONSE_LANGUAGE.sub("", text).strip()


def _strip_tag_from_last_assistant(message_history: list) -> None:
    """Remove <response_language> tag from the last assistant message in history."""
    for msg in reversed(message_history):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and "<response_language>" in content:
            msg["content"] = _strip_response_language_tag(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "<response_language>" in item.get("text", ""):
                    item["text"] = _strip_response_language_tag(item["text"])
        break


class TokenBudgetTracker:
    """Token 预算追踪器

    跨轮次追踪总 token 消耗，接近预算时催促收尾，超预算时终止。
    适用于所有 LLM provider（按 token 计费的场景）。

    Args:
        budget: 总 token 预算（0 = 不限制）
        warning_ratio: 消耗达到此比例时注入催促（默认 0.8）
        hard_ratio: 消耗达到此比例时强制终止（默认 1.0）
    """

    def __init__(
        self,
        budget: int = 0,
        warning_ratio: float = TOKEN_BUDGET_WARNING_RATIO,
        hard_ratio: float = TOKEN_BUDGET_HARD_RATIO,
    ):
        self.budget = budget
        self.warning_ratio = warning_ratio
        self.hard_ratio = hard_ratio
        self._total_tokens_used: int = 0
        self._warned: bool = False

    @property
    def enabled(self) -> bool:
        return self.budget > 0

    @property
    def total_used(self) -> int:
        return self._total_tokens_used

    @property
    def remaining(self) -> int:
        if not self.enabled:
            return -1
        return max(0, self.budget - self._total_tokens_used)

    @property
    def usage_ratio(self) -> float:
        if not self.enabled:
            return 0.0
        return self._total_tokens_used / self.budget

    def record_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        """记录一次 LLM 调用的 token 消耗"""
        self._total_tokens_used += prompt_tokens + completion_tokens

    def check(self) -> str | None:
        """检查预算状态

        Returns:
            None — 正常
            "warning" — 接近预算，应催促收尾
            "exceeded" — 超预算，应终止
        """
        if not self.enabled:
            return None

        ratio = self.usage_ratio
        if ratio >= self.hard_ratio:
            return "exceeded"
        if ratio >= self.warning_ratio and not self._warned:
            self._warned = True
            return "warning"
        return None

    def reset(self):
        self._total_tokens_used = 0
        self._warned = False


class MainLoopRunner:
    """主执行循环运行器

    封装 Agent 主循环的完整执行逻辑，从 Orchestrator 拆分出来以降低复杂度。
    通过依赖注入获取所需组件，不直接引用 Orchestrator。
    """

    def __init__(self, ctx: MainLoopContext):
        self.cfg = ctx.cfg
        self.monitor = ctx.monitor
        self.context_manager = ctx.context_manager
        self.stream_handler = ctx.stream_handler
        self.tool_executor = ctx.tool_executor
        self.sub_agent_runner = ctx.sub_agent_runner
        self.llm_handler = ctx.llm_handler
        self.summary_handler = ctx.summary_handler
        self.task_planner = ctx.task_planner
        self.inline_skill_selector = ctx.inline_skill_selector
        self.llm_client = ctx.llm_client
        self.output_formatter = ctx.output_formatter
        self.task_log = ctx.task_log
        self.context = ctx.context
        self.chinese_context = ctx.chinese_context
        self.response_language = ctx.response_language
        self.execution_mode = ctx.execution_mode
        self.todo_tracker = ctx.todo_tracker
        self.agent_name = ctx.agent_name

        # Callbacks (injected from Orchestrator)
        self._handle_llm_call = ctx.handle_llm_call
        self._handle_summary = ctx.handle_summary
        self._intercept_key_message = ctx.intercept_key_message
        self._streaming_final_message = ctx.streaming_final_message
        self._stream_tool_reasoning = ctx.stream_tool_reasoning
        self._extract_recent_tool_names = ctx.extract_recent_tool_names
        self._deduplicate_trailing_messages = ctx.deduplicate_trailing_messages

        self.long_term_memory = ctx.long_term_memory
        self.spawn_depth = ctx.spawn_depth
        if ctx.hooks is None:
            raise ValueError("MainLoopContext.hooks is required — pass a HookRegistry instance")
        self.hooks = ctx.hooks
        self.deferred_tool_manager = ctx.deferred_tool_manager
        self.transcript = ctx.transcript
        self.file_state_cache = ctx.file_state_cache
        self.skill_commands = ctx.skill_commands
        self.router_llm_client = ctx.router_llm_client
        self.config_loader = ctx.config_loader

        # Session memory (within-run structured memory, survives context compression)
        self.session_memory = SessionMemory()

        # Sub-agent concurrency semaphore (instance-level, shared across turns)
        max_concurrent = getattr(
            self.cfg.main_agent,
            "max_concurrent_subagents",
            DEFAULT_MAX_CONCURRENT_SUBAGENTS,
        )
        self._sub_agent_semaphore = asyncio.Semaphore(max_concurrent)

        # 当前 Agent ID
        self.current_agent_id: str | None = None

        # 运行时上下文引用（在 run() 中设置）
        self._current_system_prompt: str = ""

    def _record_event(self, event_type, data=None, turn=0, ref_event_id=None, duration_ms=None):
        """Record a transcript event (no-op if transcript not configured)."""
        if self.transcript is None:
            return None
        from mem_deep_research_core.core.transcript import EventType as ET

        # Convert string to EventType if needed
        if isinstance(event_type, str):
            event_type = ET(event_type)
        return self.transcript.record(
            event_type=event_type,
            data=data or {},
            turn=turn,
            agent_name=self.agent_name,
            ref_event_id=ref_event_id,
            duration_ms=duration_ms,
        )

    async def run(
        self,
        system_prompt: str,
        message_history: list,
        tool_definitions: list,
        main_agent_prompt_instance,
        task_engine_cfg: dict | None,
        task_description: str,
        task_guidance: str,
        keep_tool_result: int,
        resume_from: dict | None = None,
    ) -> tuple[str, bool]:
        """运行主执行循环

        Args:
            resume_from: 恢复状态 (来自 TaskTracer.get_resumable_state())，
                         包含 message_history, last_turn 等。为 None 表示正常启动。

        Returns:
            (最终答案文本, is_simple_response)
        """
        max_turns = self.cfg.main_agent.max_turns
        if max_turns < 0:
            max_turns = sys.maxsize
        max_tool_calls = self.cfg.main_agent.max_tool_calls_per_turn

        # 初始化监控和计数器
        self.monitor.reset()
        self.context_manager.reset()
        self.session_memory = SessionMemory()  # Reset session memory between runs
        self.context_manager.set_session_memory(self.session_memory)

        # Token budget tracker
        task_token_budget = self.cfg.main_agent.get("task_token_budget", DEFAULT_TASK_TOKEN_BUDGET)
        token_budget = TokenBudgetTracker(budget=task_token_budget)
        if token_budget.enabled:
            logger.info(f"[{self.agent_name}] Token budget: {task_token_budget} tokens")
        if self.inline_skill_selector:
            self.inline_skill_selector.reset()

        # Resume: restore state from previous run
        _resume_turn_offset = 0
        if resume_from:
            prev_history = resume_from.get("message_history", [])
            if prev_history:
                message_history.clear()
                message_history.extend(prev_history)
            _resume_turn_offset = resume_from.get("last_turn", 0)
            # Restore session memory snapshot
            snapshot = resume_from.get("session_memory_snapshot", "")
            if snapshot:
                for line in snapshot.split("\n"):
                    line = line.strip().lstrip("- ")
                    if line:
                        self.session_memory.add_finding(line)
            # Restore todo tracker state
            todo_state = resume_from.get("todo_state")
            if todo_state and self.todo_tracker is not None:
                from mem_deep_research_core.core.todo_tracker import TodoTracker

                self.todo_tracker = TodoTracker.from_dict(
                    todo_state, enabled=self.todo_tracker.enabled
                )
                logger.info(
                    f"[Resume] Restored todo tracker: {len(self.todo_tracker._items)} items"
                )

            # Restore offloaded content (replace previews with full text)
            restored = self.context_manager.restore_offloaded_content(message_history)
            if restored > 0:
                logger.info(f"[Resume] Restored {restored} offloaded results")

            # Inject resume notice into conversation
            message_history.append(
                {
                    "role": "user",
                    "_type": MT.RESUME_NOTICE,
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"[RESUME NOTICE] This task was previously interrupted at turn {_resume_turn_offset}. "
                                f"The conversation history above contains all prior work. "
                                f"Please continue from where you left off and complete the remaining work."
                            ),
                        }
                    ],
                }
            )
            logger.info(
                f"[Resume] Restored state from turn {_resume_turn_offset}, "
                f"message_history={len(message_history)} messages"
            )

        # TurnCounter 延迟到路由后创建（reflection 由 effective_mode 驱动）
        turn_counter = None  # initialized after mode resolution

        self._current_system_prompt = system_prompt

        task_failed = False
        self.current_agent_id = await self.stream_handler.stream_start_agent(self.agent_name)
        await self.stream_handler.stream_start_llm(self.agent_name)

        # Hook: on_agent_start
        self.hooks.call(
            "on_agent_start",
            HookContext(
                hook_name="on_agent_start",
                query=task_description,
                context=self.context,
                extra={"agent_type": self.agent_name},
            ),
        )

        # Transcript: agent start
        self._record_event(
            "agent_start",
            {
                "agent_name": self.agent_name,
                "execution_mode": self.execution_mode,
                "task": task_description[:200],
            },
        )

        # Language detection is deferred to first LLM reply (see turn_count == 1 block)
        # when response_language == "auto". The LLM emits <response_language>X</response_language>
        # and we parse it. Fallback: char-based detection from query text.

        # Recall long-term memory if available
        if self.long_term_memory:
            memories = self.long_term_memory.recall(task_description, top_k=5)
            if memories:
                memory_text = "\n".join(f"- {m.value}" for m in memories)
                message_history.append(
                    {
                        "role": "user",
                        "_type": MT.LONG_TERM_MEMORY,
                        "content": [
                            {
                                "type": "text",
                                "text": f"[LONG-TERM MEMORY]\nRelevant past knowledge:\n{memory_text}",
                            }
                        ],
                    }
                )

        # Resolve execution mode
        # auto / simple_auto 使用 adaptive 策略：结构信号（零成本）→ hook → 第一轮后根据 LLM 行为定模式
        # simple_auto 与 auto 的区别：adaptive 阶段不会升级到 deep（clamp 到 standard）
        # 非 auto 模式直接使用配置值
        effective_mode = self.execution_mode
        _adaptive_pending = False  # True = auto 模式待第一轮后定模式
        _adaptive_allow_deep = True  # simple_auto 时为 False

        if effective_mode in (EXECUTION_MODE_AUTO, EXECUTION_MODE_SIMPLE_AUTO):
            _adaptive_allow_deep = effective_mode == EXECUTION_MODE_AUTO

            from mem_deep_research_core.core.llm_router import LLMRouter

            router = LLMRouter(
                hooks=self.hooks,
                llm_client=self.llm_client,
            )

            # 1. Hook: on_route_classify（用户完全自定义）
            _deterministic_result = None
            if self.hooks.has_hooks("on_route_classify"):
                from mem_deep_research_core.core.hooks import HookContext as _HC

                _tool_count = sum(
                    len(s.get("tools", [])) for s in tool_definitions if isinstance(s, dict)
                )
                hook_ctx = _HC(
                    hook_name="on_route_classify",
                    query=task_description,
                    context=self.context or {},
                    extra={
                        "tool_count": _tool_count,
                        "has_sub_agents": bool(getattr(self.cfg, "sub_agents", None)),
                        "task_engine_enabled": bool(
                            task_engine_cfg and task_engine_cfg.get("enabled")
                        ),
                    },
                )
                hook_result = self.hooks.call("on_route_classify", hook_ctx)
                _deterministic_result = router._parse_hook_result(hook_result)

            # 2. 结构信号路由（零成本）
            if _deterministic_result is None:
                _deterministic_result = router._structural_route(
                    tool_count=sum(
                        len(s.get("tools", [])) for s in tool_definitions if isinstance(s, dict)
                    ),
                    has_sub_agents=bool(getattr(self.cfg, "sub_agents", None)),
                )

            if _deterministic_result is not None:
                # simple_auto: 结构信号判定 deep 时，clamp 到 standard
                if not _adaptive_allow_deep and _deterministic_result.mode == EXECUTION_MODE_DEEP:
                    _deterministic_result.mode = EXECUTION_MODE_STANDARD
                    _deterministic_result.reasoning_effort = "medium"
                # finalize（注入 thinking_params 等）
                route_result = router._finalize(
                    _deterministic_result, task_description, self.context
                )
                effective_mode = route_result.mode
                logger.info(
                    f"[{self.agent_name}] Auto route (deterministic): {effective_mode} "
                    f"(source={route_result.source})"
                )
            else:
                # 无确定性信号 → adaptive：先用 standard 跑第一轮，之后根据行为定模式
                effective_mode = EXECUTION_MODE_STANDARD
                _adaptive_pending = True
                logger.info(
                    f"[{self.agent_name}] Auto route: adaptive pending, "
                    f"starting as standard (will finalize after turn 1)"
                )

        logger.info(
            f"[{self.agent_name}] Execution mode: {effective_mode} (config={self.execution_mode})"
        )
        self.task_log.record_perf("config_mode", self.execution_mode, unit="")

        # Deep 能力由 effective_mode 驱动（auto 路由到 deep 时也激活）
        is_deep_mode = effective_mode == EXECUTION_MODE_DEEP
        reflection_interval = (
            task_engine_cfg.get("reflection_interval", 5) if task_engine_cfg else 5
        )
        turn_counter = TurnCounter(
            max_turns=max_turns,
            reflection_enabled=is_deep_mode,
            reflection_interval=reflection_interval,
        )

        # Quick mode: limited turns, tools enabled, skip heavy features (reflection/skills/hints)
        is_quick_mode = effective_mode == EXECUTION_MODE_QUICK
        if is_quick_mode:
            max_turns = min(max_turns, QUICK_MODE_MAX_TURNS)
            turn_counter.max_turns = max_turns
            # 裁剪 quick 模式不需要的内置工具（spawn_agent / update_todo / read_result）
            _quick_strip = {BUILTIN_TOOL_SPAWN_AGENT, BUILTIN_TOOL_UPDATE_TODO, BUILTIN_TOOL_READ_RESULT}
            before_count = len(tool_definitions)
            tool_definitions = [
                td for td in tool_definitions
                if not any(
                    t.get("name") in _quick_strip
                    for t in td.get("tools", [td]) if isinstance(t, dict)
                )
            ]
            stripped = before_count - len(tool_definitions)
            # 注入 quick preset 到 system prompt（路由后动态追加）
            from mem_deep_research_core.prompts.template_loader import template_loader as _tpl_loader

            try:
                quick_preset = _tpl_loader.load_template("presets/quick")
                system_prompt = system_prompt + "\n\n" + quick_preset
            except FileNotFoundError:
                pass
            logger.info(
                f"[{self.agent_name}] QUICK mode: max_turns={max_turns}, "
                f"stripped {stripped} heavy tools, no reflection/skills"
            )

        # 自动任务分解（仅深度研究模式 + auto_planning 启用时，quick 模式跳过）
        if self.task_planner.enabled and not is_quick_mode:
            plan = await self.task_planner.create_plan(
                task_description=task_description,
                llm_client=self.llm_client,
            )
            if plan:
                message_history.append(
                    {
                        "role": "user",
                        "_type": MT.PLAN,
                        "content": [
                            {"type": "text", "text": f"{TAG_TASK_PLAN}\n{plan.to_context_string()}"}
                        ],
                    }
                )
                self.task_log.log_step(
                    "auto_planning",
                    f"Generated research plan with {len(plan.sub_questions)} sub-questions",
                )
                logger.info("[TaskPlanner] Plan injected into message history")

        # Built-in tools (spawn_agent, update_todo) are already injected by Orchestrator._get_tool_definitions
        self._current_tool_definitions = list(tool_definitions)
        self._current_system_prompt = system_prompt  # For sub-agent prompt sharing

        total_tool_calls_executed = 0
        last_assistant_text = ""
        _context_limit_retries = 0
        _reflection_pending = False  # 上一轮末尾注入了反思 prompt，下轮允许无工具回复
        self._spawn_executed = False  # spawn_agent 执行标记，由 _execute_tools 设置
        _perf_main_loop_start = time.perf_counter()
        _perf_total_llm_time = 0.0
        _perf_total_tool_time = 0.0

        while not turn_counter.is_max_reached():
            turn_count = turn_counter.increment()
            self.context_manager.set_turn(turn_count)
            self._record_event("turn_start", {"turn": turn_count}, turn=turn_count)
            logger.debug(f"\n--- Main Agent Turn {turn_count} ---")

            # Hook: on_turn_start
            self.hooks.call(
                "on_turn_start",
                HookContext(
                    hook_name="on_turn_start",
                    turn_number=turn_count,
                    query=task_description,
                    context=self.context,
                ),
            )
            self.task_log.save()

            # Inject todo state (survives context compression)
            if self.todo_tracker and not self.todo_tracker.is_empty:
                todo_msg = self.todo_tracker.build_injection_message(turn=turn_count)
                if todo_msg:
                    message_history.append(todo_msg)

            # Inject session memory context (survives context compression)
            if not self.session_memory.is_empty():
                memory_msg = {
                    "role": "user",
                    "_type": MT.SESSION_MEMORY,
                    "content": [{"type": "text", "text": self.session_memory.to_context_string()}],
                }
                # Replace previous session memory message (avoid duplicates)
                for i in range(len(message_history) - 1, -1, -1):
                    if message_history[i].get("_type") == MT.SESSION_MEMORY:
                        message_history.pop(i)
                        break
                    # Fallback: keyword match for untyped messages
                    content = message_history[i].get("content", "")
                    if isinstance(content, list) and content:
                        text = content[0].get("text", "") if isinstance(content[0], dict) else ""
                    elif isinstance(content, str):
                        text = content
                    else:
                        text = ""
                    if "[SESSION MEMORY]" in text:
                        message_history.pop(i)
                        break
                message_history.append(memory_msg)

            # 监控前检查
            terminate_reason = await self.monitor.pre_turn_check()
            if self._spawn_executed and terminate_reason:
                # Grace turn after spawn_agent: override hard termination so LLM can
                # process spawn results and produce a final answer. Inject urgency hint
                # but do NOT break — the LLM must run at least once.
                logger.info(
                    f"[{self.agent_name}] Grace turn after spawn_agent — "
                    f"overriding {terminate_reason}, allowing one more LLM call"
                )
                message_history.append(
                    {
                        "role": "user",
                        "_type": MT.TOKEN_WARNING,
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "[TIME WARNING] 子任务已全部完成，结果已就绪。"
                                    "请立即基于以上子任务结果输出最终汇总答案，不要再调用工具。"
                                ),
                            }
                        ],
                    }
                )
                self._spawn_executed = False
            elif terminate_reason == "soft_timeout":
                # 软超时：注入催促 hint，继续执行
                remaining = int(
                    self.monitor.config.max_total_time - self.monitor.get_elapsed_time()
                )
                message_history.append(
                    {
                        "role": "user",
                        "_type": MT.TOKEN_WARNING,
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"[TIME WARNING] 剩余时间约 {remaining}s。请尽快总结当前已有的发现和结论，"
                                    "如果核心任务已完成请直接给出最终答案。"
                                ),
                            }
                        ],
                    }
                )
            elif terminate_reason:
                # 硬超时或其他终止原因：标记失败，跳出循环走摘要流程
                task_failed = True
                break

            # Microcompact: 每轮 LLM 调用前清理旧 tool_result（零成本）
            keep_recent = self.context_manager.config.compact_keep_recent
            self.context_manager.microcompact(message_history, turn_count, keep_recent=keep_recent)

            # Phase 2: 标记即将滑出窗口的旧大结果，注入 sidecar prompt 要求产 evidence
            offload_candidates = self.context_manager.prepare_offload_candidates(
                message_history, turn_count, keep_recent=keep_recent
            )
            _offload_prep_injected = False
            if offload_candidates:
                candidate_lines = []
                for c in offload_candidates:
                    tools = ", ".join(c.get("tool_names", [])) or "unknown"
                    candidate_lines.append(
                        f'- ref="{c["ref"]}" tool={tools} (turn {c["turn"]}, {c["chars"]} chars)'
                    )
                sidecar = (
                    "[OFFLOAD PREP]\n\n"
                    "The following tool-result messages will be offloaded after this turn. "
                    "While completing your normal reasoning, emit evidence blocks for any candidate "
                    "you used.\n\n"
                    "Output format:\n"
                    '<offload_evidence ref="REF_ID">\n'
                    "- key fact 1\n"
                    "- key fact 2\n"
                    "</offload_evidence>\n\n"
                    "Candidates:\n" + "\n".join(candidate_lines)
                )
                # Hook: on_offload_evidence_prep — append tool-specific guidance
                if self.hooks.has_hooks("on_offload_evidence_prep"):
                    hook_result = self.hooks.call(
                        "on_offload_evidence_prep",
                        HookContext(
                            hook_name="on_offload_evidence_prep",
                            turn_number=turn_count,
                            context=self.context,
                            extra={"candidates": offload_candidates, "sidecar": sidecar},
                        ),
                    )
                    if isinstance(hook_result, str):
                        sidecar = hook_result
                    elif hook_result is not None:
                        logger.warning(
                            f"[Hook] on_offload_evidence_prep returned {type(hook_result).__name__} "
                            f"instead of str, ignoring (return a string to override sidecar prompt)"
                        )
                message_history.append({
                    "role": "user",
                    "_type": MT.OFFLOAD_PREP,
                    "content": [{"type": "text", "text": sidecar}],
                })
                _offload_prep_injected = True

            # LLM 调用
            _perf_llm_start = time.perf_counter()
            _llm_event_id = self._record_event(
                "llm_call",
                {
                    "turn": turn_count,
                    "message_count": len(message_history),
                },
                turn=turn_count,
            )
            try:
                assistant_response_text, should_break, tool_calls = await self._handle_llm_call(
                    system_prompt,
                    message_history,
                    tool_definitions,
                    turn_count,
                    f"{self.agent_name} agent turn {turn_count}",
                    agent_type=self.agent_name,
                    stream_message_callback=self._intercept_key_message,
                )
            finally:
                # 确保 OFFLOAD_PREP sidecar 即使 LLM 调用异常也被清理，
                # 避免临时消息残留在 message_history 中
                if _offload_prep_injected:
                    message_history[:] = [
                        m for m in message_history if m.get("_type") != MT.OFFLOAD_PREP
                    ]
            _perf_llm_elapsed = time.perf_counter() - _perf_llm_start
            _perf_total_llm_time += _perf_llm_elapsed
            self.task_log.append_perf("llm_call_durations", _perf_llm_elapsed)

            # Token budget: 记录本轮消耗并检查
            if token_budget.enabled:
                usage = self.llm_client.get_usage()
                # 用增量：总消耗 - 之前记录的
                current_total = usage.get("total_tokens", 0)
                if current_total > token_budget.total_used:
                    increment = current_total - token_budget.total_used
                    token_budget._total_tokens_used = current_total
                budget_status = token_budget.check()
                if budget_status == "exceeded":
                    logger.warning(
                        f"[{self.agent_name}] Token budget exceeded: "
                        f"{token_budget.total_used}/{token_budget.budget} tokens"
                    )
                    task_failed = True
                    break
                elif budget_status == "warning":
                    remaining_tokens = token_budget.remaining
                    message_history.append(
                        {
                            "role": "user",
                            "_type": MT.TOKEN_WARNING,
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"[TOKEN BUDGET WARNING] You have used {token_budget.usage_ratio:.0%} of your token budget "
                                        f"({token_budget.total_used}/{token_budget.budget} tokens, ~{remaining_tokens} remaining). "
                                        f"Please wrap up your research and provide a final answer soon."
                                    ),
                                }
                            ],
                        }
                    )

            self._record_event(
                "llm_response",
                {
                    "turn": turn_count,
                    "has_text": assistant_response_text is not None,
                    "has_tool_calls": bool(tool_calls and tool_calls != "context_limit"),
                    "should_break": should_break,
                },
                turn=turn_count,
                ref_event_id=_llm_event_id,
                duration_ms=int(_perf_llm_elapsed * 1000),
            )

            last_assistant_text = assistant_response_text or ""
            if assistant_response_text is not None:
                _context_limit_retries = 0

            # Extract <response_language> tag from LLM reply (typically turn 1, but LLM may delay)
            if (
                self.response_language == "auto"
                and assistant_response_text
            ):
                detected = _extract_response_language(assistant_response_text)
                if detected:
                    self.response_language = detected
                    self.chinese_context = detected == "Chinese"
                    logger.info(f"[Language] LLM declared: {detected}")
                    # Strip the tag from assistant text and message history
                    assistant_response_text = _strip_response_language_tag(assistant_response_text)
                    last_assistant_text = assistant_response_text
                    _strip_tag_from_last_assistant(message_history)
                else:
                    # Fallback to char-based detection
                    from mem_deep_research_core.core.user_context import (
                        detect_language_by_chars,
                    )
                    lang_text = (self.context or {}).get("original_query") or task_description
                    self.response_language = detect_language_by_chars(lang_text)
                    self.chinese_context = self.response_language == "Chinese"
                    logger.info(f"[Language] Fallback char-detection: {self.response_language}")
                # Sync to context so hooks can read it
                if self.context is not None:
                    self.context["response_language"] = self.response_language
                    self.context["chinese_context"] = self.chinese_context
            elif assistant_response_text and "<response_language>" in assistant_response_text:
                # Language already resolved but LLM emitted tag again — strip it
                assistant_response_text = _strip_response_language_tag(assistant_response_text)
                last_assistant_text = assistant_response_text
                _strip_tag_from_last_assistant(message_history)

            # max_output_tokens 续写恢复：output 被截断时注入续写提示
            _output_truncated = getattr(self.llm_client, "_output_truncated_flag", None)
            if _output_truncated is True:
                self.llm_client._output_truncated_flag = False
            if _output_truncated is True and assistant_response_text and not tool_calls:
                logger.info(
                    f"[{self.agent_name}] Output truncated (finish_reason=length), "
                    f"injecting continuation prompt (turn {turn_count})"
                )
                message_history.append(
                    {
                        "role": "user",
                        "_type": MT.TRUNCATION_RECOVERY,
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "[OUTPUT TRUNCATED] Your previous response was cut off due to length limits. "
                                    "Please continue from where you left off. Do not repeat what you already said."
                                ),
                            }
                        ],
                    }
                )
                self._record_event(
                    "llm_response",
                    {
                        "turn": turn_count,
                        "truncated": True,
                        "recovery": "continuation_prompt",
                    },
                    turn=turn_count,
                )
                continue  # retry with continuation prompt

            # 清除温度覆盖（无论 LLM 调用成功与否）
            self.llm_client.clear_temperature_override()

            # 监控后检查
            terminate_reason = await self.monitor.post_turn_check(
                response_text=assistant_response_text,
                llm_call_failed=tool_calls == "context_limit" or assistant_response_text is None,
            )

            # 立即检查终止原因
            if terminate_reason:
                logger.warning(
                    f"[{self.agent_name}] TERMINATED: {terminate_reason} (turn {turn_count})"
                )
                task_failed = True
                break

            # Adaptive mode finalization: 第一轮结束后根据 LLM 行为定模式
            # 跳过 resume 场景（恢复的任务已有足够上下文，不需要重新判断模式）
            if _adaptive_pending and turn_count == 1 and not resume_from:
                _adaptive_pending = False
                from mem_deep_research_core.core.llm_router import LLMRouter

                _has_spawn = bool(
                    tool_calls
                    and isinstance(tool_calls, list)
                    and len(tool_calls) > 0
                    and any(
                        c.get("tool_name") == BUILTIN_TOOL_SPAWN_AGENT
                        for c in (tool_calls[0] if tool_calls else [])
                    )
                )
                adaptive_result = LLMRouter.adaptive_classify(
                    should_break=should_break,
                    tool_calls=tool_calls,
                    has_spawn_agent=_has_spawn,
                    allow_deep=_adaptive_allow_deep,
                )
                effective_mode = adaptive_result.mode
                is_quick_mode = effective_mode == EXECUTION_MODE_QUICK
                is_deep_mode = effective_mode == EXECUTION_MODE_DEEP

                # Apply mode-specific configuration
                if is_deep_mode:
                    turn_counter.reflection_enabled = True
                    # 同步 reasoning_effort 到 llm_client（后续轮次生效）
                    if hasattr(self.llm_client, "reasoning_effort"):
                        self.llm_client.reasoning_effort = adaptive_result.reasoning_effort

                logger.info(
                    f"[{self.agent_name}] Adaptive finalized: {effective_mode} "
                    f"(reason={adaptive_result.metadata.get('reason', '?')})"
                )
                self.task_log.record_perf("adaptive_source", adaptive_result.source, unit="")
                self.task_log.record_perf(
                    "adaptive_reason", adaptive_result.metadata.get("reason", ""), unit=""
                )

            # 响应循环升级到 INJECT_HINT 时注入强制策略指令 + 温度提升
            if self.monitor.last_loop_action == EscalationAction.INJECT_HINT:
                recent_tools = self._extract_recent_tool_names(message_history)
                hint_text = self.monitor.get_loop_break_hint(
                    recent_tool_names=recent_tools,
                    chinese=self.chinese_context,
                )
                message_history.append(
                    {"role": "user", "_type": MT.LOOP_HINT, "content": [{"type": "text", "text": hint_text}]}
                )
                temp_boost = getattr(
                    self.monitor.config, "temperature_boost", DEFAULT_TEMPERATURE_BOOST
                )
                temp_cap = getattr(
                    self.monitor.config, "temperature_boost_cap", DEFAULT_TEMPERATURE_BOOST_CAP
                )
                self.llm_client.set_temperature_boost(boost=temp_boost, cap=temp_cap)

            # Inline Skill: 从 LLM 回复中解析 <next_skills>，下一轮动态注入（quick 模式跳过）
            if not is_quick_mode and self.inline_skill_selector and assistant_response_text:
                next_skills = self.inline_skill_selector.update_pending_skills(
                    assistant_response_text
                )
                # Strip <next_skills> tag from assistant text and message history
                stripped = self.inline_skill_selector.strip_next_skills_tag(
                    assistant_response_text
                )
                if stripped != assistant_response_text:
                    assistant_response_text = stripped
                    # Sync to message_history
                    for msg in reversed(message_history):
                        if msg.get("role") != "assistant":
                            continue
                        content = msg.get("content", "")
                        if isinstance(content, str) and "<next_skills>" in content:
                            msg["content"] = self.inline_skill_selector.strip_next_skills_tag(content)
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and "<next_skills>" in item.get("text", ""):
                                    item["text"] = self.inline_skill_selector.strip_next_skills_tag(item["text"])
                        break
                if next_skills:
                    # 检查 injection mode: meta_message 或 system_prompt
                    injection_mode = self.cfg.main_agent.get("skill_selection", {}).get(
                        "injection_mode", "system_prompt"
                    )
                    if injection_mode == "meta_message" and self.skill_commands:
                        # Claude Code 模式: 分流 fork 和 inline
                        for skill_name in next_skills:
                            sc = self.skill_commands.get(skill_name)
                            if not sc:
                                continue
                            try:
                                if sc.context_mode == "fork":
                                    # Fork: spawn 子 agent 执行
                                    fork_result = await self._run_fork_skill(sc)
                                    message_history.append(
                                        {
                                            "role": "user",
                                            "_type": MT.INLINE_SKILL,
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": f"[Skill Result: {skill_name}]\n{fork_result}",
                                                }
                                            ],
                                        }
                                    )
                                    logger.info(f"[InlineSkill] Fork skill '{skill_name}' completed")
                                else:
                                    # Inline: 渲染 prompt，注入 meta message
                                    rendered = await sc.get_prompt()
                                    message_history.append(
                                        {
                                            "role": "user",
                                            "_type": MT.INLINE_SKILL,
                                            "content": [{"type": "text", "text": rendered}],
                                            "_meta": True,
                                        }
                                    )
                                    logger.info(
                                        f"[InlineSkill] Injected meta message for skill '{skill_name}'"
                                )
                            except Exception as e:
                                logger.error(f"[InlineSkill] Skill '{skill_name}' failed: {e}")
                    else:
                        # 传统模式: 修改 system prompt
                        system_prompt = self.inline_skill_selector.inject_pending_skills(
                            system_prompt
                        )
                    logger.info(f"[InlineSkill] Processed skills for next turn: {next_skills}")

            # 处理 offload evidence（在 should_break 之前，确保最终回合也能解析）
            if assistant_response_text and _offload_prep_injected:
                cleaned = _extract_offload_evidence(
                    assistant_response_text, self.context_manager
                )
                if cleaned != assistant_response_text:
                    assistant_response_text = cleaned
                    _strip_evidence_from_last_assistant(message_history)

            # Finalize offload: 替换旧消息为 OFFLOADED marker
            if offload_candidates:
                self.context_manager.finalize_offload_candidates(
                    message_history, turn_count, keep_recent=keep_recent
                )

            # 处理 LLM 响应
            if assistant_response_text is not None:
                if should_break:
                    logger.info(
                        f"[{self.agent_name}] LLM signaled completion (turn {turn_count}, task_failed={task_failed})"
                    )
                    # 保存最终 checkpoint（即使 1 轮就结束）
                    self.task_log.save_checkpoint(
                        turn=turn_count,
                        message_count=len(message_history),
                        tool_calls_executed=total_tool_calls_executed,
                        last_assistant_text=last_assistant_text,
                        task_failed=task_failed,
                        todo_state=self.todo_tracker.to_dict()
                        if self.todo_tracker and not self.todo_tracker.is_empty
                        else None,
                        session_memory_snapshot=self.session_memory.to_context_string()
                        if not self.session_memory.is_empty()
                        else "",
                    )
                    break
            else:
                if tool_calls == "context_limit":
                    _context_limit_retries += 1
                    if _context_limit_retries > MAX_CONTEXT_LIMIT_RETRIES:
                        self.task_log.log_step(
                            "context_limit_exhausted",
                            f"Context limit retry exhausted after {MAX_CONTEXT_LIMIT_RETRIES} attempts",
                            "failed",
                        )
                        logger.warning(
                            f"[{self.agent_name}] TERMINATED: context limit exhausted after {MAX_CONTEXT_LIMIT_RETRIES} retries (turn {turn_count})"
                        )
                        task_failed = True
                        break
                    # Level 3: 紧急裁剪
                    emergency_count = self.context_manager.apply_emergency(
                        message_history,
                        turn_count,
                        system_prompt,
                        self.llm_client.max_context_length,
                    )
                    if emergency_count > 0:
                        self.task_log.log_step(
                            "context_emergency",
                            f"[CONTEXT L3] Emergency: processed {emergency_count} messages, "
                            f"history now {len(message_history)} messages",
                            "info",
                        )
                        self.hooks.call(
                            "on_context_compact",
                            HookContext(
                                hook_name="on_context_compact",
                                turn_number=turn_count,
                                compact_action="emergency",
                                query=task_description,
                                context=self.context,
                                extra={"message_count": len(message_history)},
                            ),
                        )
                        continue  # retry LLM call
                    self.task_log.log_step(
                        "main_agent_context_limit_reached", "Context limit reached", "warning"
                    )
                else:
                    self.task_log.log_step("main_agent", "LLM call failed", "failed")
                logger.warning(
                    f"[{self.agent_name}] LLM call failed, task_failed=True (turn {turn_count})"
                )
                task_failed = True
                break

            # 检查是否有工具调用
            if (
                not tool_calls
                or len(tool_calls) < 2
                or (len(tool_calls[0]) == 0 and len(tool_calls[1]) == 0)
            ):
                # 反思轮允许无工具回复：LLM 可能只输出反思文字，下一轮再调工具
                if _reflection_pending:
                    _reflection_pending = False
                    logger.info(
                        f"[{self.agent_name}] No tool calls after reflection, continuing (turn {turn_count})"
                    )
                    continue
                logger.info(
                    f"[{self.agent_name}] No tool calls, ending (turn {turn_count}, task_failed={task_failed})"
                )
                break

            # 跨轮次去重过滤
            calls_to_execute = tool_calls[0][:max_tool_calls]
            to_execute, cached_results = self.context_manager.filter_duplicate_calls(
                calls_to_execute
            )

            # Hook: on_tool_filter — 去重后、执行前，可修改/重排/拦截工具调用列表
            if to_execute and self.hooks.has_hooks("on_tool_filter"):
                filtered = self.hooks.call(
                    "on_tool_filter",
                    HookContext(
                        hook_name="on_tool_filter",
                        turn_number=turn_count,
                        tool_calls_batch=to_execute,
                        query=task_description,
                        context=self.context,
                    ),
                )
                if filtered is not None and isinstance(filtered, list):
                    to_execute = filtered

            # 执行工具调用（仅非重复的）
            executed_tool_calls = (
                [tool_calls[0][:0], tool_calls[1]] if len(tool_calls) > 1 else [[], []]
            )
            executed_tool_calls[0] = to_execute
            all_tool_results_with_id = []

            if to_execute:
                # Transcript: record each tool_use
                _tool_event_ids = {}
                for call in to_execute:
                    eid = self._record_event(
                        "tool_use",
                        {
                            "tool_name": call.get("tool_name", ""),
                            "server_name": call.get("server_name", ""),
                            "arguments_preview": str(call.get("arguments", ""))[:200],
                        },
                        turn=turn_count,
                    )
                    if eid:
                        _tool_event_ids[call.get("id", "")] = eid

                # 将当前轮 LLM 回复文本传入 tool_executor，供 on_thinking_generate hook 使用
                self.tool_executor._current_assistant_text = assistant_response_text

                _perf_tool_start = time.perf_counter()
                modified_tool_calls = [to_execute, tool_calls[1] if len(tool_calls) > 1 else []]
                try:
                    all_tool_results_with_id = await self._execute_tools(
                        modified_tool_calls, max_tool_calls, keep_tool_result
                    )
                finally:
                    self.tool_executor._current_assistant_text = None  # 清理，避免跨轮泄漏
                _perf_tool_elapsed = time.perf_counter() - _perf_tool_start
                _perf_total_tool_time += _perf_tool_elapsed
                self.task_log.append_perf("tool_batch_durations", _perf_tool_elapsed)
                total_tool_calls_executed += len(to_execute)

                # Transcript: record tool_results
                for call_id, result in all_tool_results_with_id:
                    result_text = (
                        result.get("text", "")[:100]
                        if isinstance(result, dict)
                        else str(result)[:100]
                    )
                    self._record_event(
                        "tool_result",
                        {
                            "call_id": call_id,
                            "result_preview": result_text,
                        },
                        turn=turn_count,
                        ref_event_id=_tool_event_ids.get(call_id),
                    )

                # Deferred tools: 如果 tool_search 发现了新工具，更新 tool_definitions
                # 注意：XML tool format 下工具描述在 system prompt 中，mid-run 发现新工具
                # 不会自动更新 system prompt。tool_search 返回的完整 schema 已在 tool_result
                # 消息中，LLM 可据此调用。Native tool format 无此限制。
                if self.deferred_tool_manager and self.deferred_tool_manager.is_active:
                    tool_definitions, _ = self.deferred_tool_manager.apply(
                        self._current_tool_definitions
                    )

            # 添加缓存结果（dedup 命中的）
            for call_id, cached_content in cached_results:
                all_tool_results_with_id.append((call_id, cached_content))

            # Tool result 配对完整性检查：确保每个 tool_use 都有对应 tool_result
            all_call_ids = {c.get("id") for c in calls_to_execute if c.get("id")}
            returned_ids = {rid for rid, _ in all_tool_results_with_id}
            missing_ids = all_call_ids - returned_ids
            if missing_ids:
                logger.warning(
                    f"[{self.agent_name}] Tool result integrity: {len(missing_ids)} missing results, "
                    f"injecting synthetic errors for: {missing_ids}"
                )
                for mid in missing_ids:
                    all_tool_results_with_id.append(
                        (
                            mid,
                            {"type": "text", "text": "[Tool result missing due to internal error]"},
                        )
                    )

            # 注册 tool results
            if to_execute and all_tool_results_with_id:
                executed_results = all_tool_results_with_id[: len(to_execute)]
                self.context_manager.register_tool_results(to_execute, executed_results, turn_count)

            # 提炼证据：从 LLM 回复中提取 <evidence> 标签内容（零额外 LLM 调用）
            # LLM 在 prompt 指令下会在回复中输出 <evidence>...</evidence> 标签
            # 返回清理后的文本（标签已移除，不会泄露到用户输出）
            if assistant_response_text and self.context_manager.config.enable_evidence_extraction:
                cleaned = _extract_evidence_tags(
                    assistant_response_text, turn_count, self.session_memory
                )
                if cleaned != assistant_response_text:
                    assistant_response_text = cleaned
                    # 同步清理 message_history 中已有的 assistant 消息
                    _strip_evidence_from_last_assistant(message_history)

            # Update session memory with findings from this turn (keyword-based fallback)
            if assistant_response_text:
                for line in assistant_response_text.split("\n"):
                    line = line.strip()
                    if len(line) > 30 and any(
                        kw in line.lower()
                        for kw in [
                            "found",
                            "result",
                            "answer",
                            "conclusion",
                            "shows",
                            "indicates",
                            "发现",
                            "结果",
                            "结论",
                            "表明",
                        ]
                    ):
                        self.session_memory.add_finding(line[:200])

            # Record tool strategies
            for call in to_execute:
                strategy = f"{call.get('tool_name', '?')}({str(call.get('arguments', ''))[:80]})"
                self.session_memory.add_strategy(strategy)

            # 记录策略摘要
            for call in to_execute:
                self.monitor.record_strategy_summary(
                    f"{call.get('tool_name', '?')}({str(call.get('arguments', ''))[:100]})"
                )

            # 更新消息历史
            tool_calls_exceeded = len(tool_calls[0]) > max_tool_calls
            message_history = self.llm_client.update_message_history(
                message_history, all_tool_results_with_id, tool_calls_exceeded
            )

            # 将 offload metadata 挂到刚加入的 TOOL_RESULT 消息上
            offload_refs = [
                r[1].get("_offload_ref")
                for r in all_tool_results_with_id
                if isinstance(r[1], dict) and r[1].get("_offload_ref")
            ]
            if offload_refs:
                # 找到刚加入的最后一条 TOOL_RESULT 消息
                for msg in reversed(message_history):
                    if msg.get("_type") == MT.TOOL_RESULT:
                        # 多个工具结果可能合并到一条消息中
                        msg["_offload_refs"] = offload_refs
                        total_chars = sum(
                            r[1].get("_offload_chars", 0)
                            for r in all_tool_results_with_id
                            if isinstance(r[1], dict) and r[1].get("_offload_ref")
                        )
                        msg["_offload_chars"] = total_chars
                        break

            # 三级 Context 管理
            action = self.context_manager.manage_context(
                message_history,
                turn_count,
                system_prompt,
                self.llm_client.max_context_length,
            )
            if action == "need_summarize":
                await self.context_manager.apply_summarize(
                    message_history,
                    turn_count,
                    system_prompt,
                    self.llm_client.max_context_length,
                    llm_call_fn=self._context_summarize_call,
                )

            # Transcript: record compression
            if action and action != "none":
                self._record_event(
                    "compact",
                    {
                        "action": action,
                        "message_count": len(message_history),
                    },
                    turn=turn_count,
                )

            # Hook: on_context_compact — 压缩发生后通知业务层
            if action is not None:
                self.hooks.call(
                    "on_context_compact",
                    HookContext(
                        hook_name="on_context_compact",
                        turn_number=turn_count,
                        compact_action="summarize" if action == "need_summarize" else "masking",
                        query=task_description,
                        context=self.context,
                        extra={"message_count": len(message_history)},
                    ),
                )

            # Context compression awareness — notify LLM
            if action and action != "none":
                recent_texts = [str(m.get("content", "")) for m in message_history[-3:]]
                if not any("[CONTEXT NOTE]" in t for t in recent_texts):
                    cm_cfg = self.cfg.main_agent.get("context_manager", {})
                    has_read_result = cm_cfg.get("result_offload_threshold", 5000) > 0
                    notice = build_context_compression_notice(has_read_result=has_read_result)
                    message_history.append(
                        {
                            "role": "user",
                            "_type": MT.CONTEXT_COMPRESSION,
                            "content": [{"type": "text", "text": notice}],
                        }
                    )

            # Sync last_assistant_text after all cleaning passes (evidence, offload,
            # response_language) so the simple-response path returns clean text.
            if assistant_response_text is not None:
                last_assistant_text = assistant_response_text

            # Hook: on_turn_end
            tool_calls_count = len(tool_calls[0]) if tool_calls and len(tool_calls) > 0 else 0
            self.hooks.call(
                "on_turn_end",
                HookContext(
                    hook_name="on_turn_end",
                    turn_number=turn_count,
                    tool_calls_count=tool_calls_count,
                    query=task_description,
                    context=self.context,
                    extra={
                        "assistant_text": last_assistant_text,
                        "message_count": len(message_history),
                        "total_tool_calls": total_tool_calls_executed,
                    },
                ),
            )

            logger.info(
                f"[{self.agent_name}] Turn {turn_count} complete: "
                f"tools={len(to_execute) if to_execute else 0}, "
                f"cached={len(cached_results) if cached_results else 0}, "
                f"msgs={len(message_history)}, "
                f"task_failed={task_failed}"
            )

            # Turn checkpoint — save progress for debugging and resume
            self.task_log.save_checkpoint(
                turn=turn_count,
                message_count=len(message_history),
                tool_calls_executed=total_tool_calls_executed,
                last_assistant_text=last_assistant_text,
                task_failed=task_failed,
                todo_state=self.todo_tracker.to_dict()
                if self.todo_tracker and not self.todo_tracker.is_empty
                else None,
                session_memory_snapshot=self.session_memory.to_context_string()
                if not self.session_memory.is_empty()
                else "",
            )

            # 反思检查点（quick 模式跳过）
            if not is_quick_mode and turn_counter.should_inject_reflection():
                reflection_prompt = generate_reflection_prompt(
                    turn_count, task_description, self.chinese_context
                )

                # Hook: on_reflection_build — 可修改反思 prompt
                if self.hooks.has_hooks("on_reflection_build"):
                    modified_prompt = self.hooks.call(
                        "on_reflection_build",
                        HookContext(
                            hook_name="on_reflection_build",
                            turn_number=turn_count,
                            result=reflection_prompt,
                            query=task_description,
                            context=self.context,
                        ),
                    )
                    if modified_prompt is not None and isinstance(modified_prompt, str):
                        reflection_prompt = modified_prompt

                message_history.append(
                    {"role": "user", "_type": MT.REFLECTION, "content": [{"type": "text", "text": reflection_prompt}]}
                )
                _reflection_pending = True
                self.task_log.log_step(
                    "reflection_checkpoint", f"Injected at turn {turn_count}", "info"
                )
                logger.debug(f"[Deep Research] Reflection checkpoint injected at turn {turn_count}")

        # Transcript: agent end
        self._record_event(
            "agent_end",
            {
                "task_failed": task_failed,
                "turns_used": turn_counter.current_turn,
                "total_tool_calls": total_tool_calls_executed,
            },
        )

        # Hook: on_agent_end
        self.hooks.call(
            "on_agent_end",
            HookContext(
                hook_name="on_agent_end",
                query=task_description,
                turn_number=turn_counter.current_turn,
                result=task_failed,
                context=self.context,
                extra={
                    "agent_type": self.agent_name,
                    "task_failed": task_failed,
                    "turns_used": turn_counter.current_turn,
                    "total_tool_calls": total_tool_calls_executed,
                    "duration_seconds": time.perf_counter() - _perf_main_loop_start,
                    "message_count": len(message_history),
                },
            ),
        )

        # Store session findings to long-term memory
        if self.long_term_memory and not self.session_memory.is_empty():
            for finding in self.session_memory.key_findings[:5]:
                self.long_term_memory.store(
                    key=finding[:50],
                    value=finding,
                    metadata={"task": task_description[:100], "type": "finding"},
                )

        # 退出 LLM/Agent
        await self.stream_handler.stream_end_llm(self.agent_name)
        await self.stream_handler.stream_end_agent(self.agent_name, self.current_agent_id)

        # 记录循环结束
        if turn_counter.is_max_reached():
            if not task_failed:
                task_failed = True
            logger.warning(
                f"[{self.agent_name}] MAX TURNS REACHED: {turn_counter.current_turn}/{max_turns}, task_failed=True"
            )
            self.task_log.log_step(
                "max_turns_reached", f"Reached maximum turns ({max_turns})", "warning"
            )
        else:
            self.task_log.log_step(
                "main_loop_completed", f"Completed after {turn_counter.current_turn} turns"
            )

        # 循环终止时清理重复的 assistant 响应
        if task_failed:
            self._deduplicate_trailing_messages(message_history)

        # 注入引用信息到消息历史
        citation_summary = self.context_manager.source_registry.get_citation_summary()
        if citation_summary:
            message_history.append(
                {
                    "role": "user",
                    "_type": MT.CITATION_SUMMARY,
                    "content": [
                        {
                            "type": "text",
                            "text": f"{TAG_COLLECTED_SOURCES}\n{citation_summary}\n\nPlease include these sources in your final summary where relevant.",
                        }
                    ],
                }
            )
            self.task_log.log_step(
                "citation_injection",
                f"Injected {len(self.context_manager.source_registry.get_all_sources())} sources into message history",
            )

        # Deep verify 检查点：summary 前验证证据质量
        if (
            is_deep_mode
            and not task_failed
            and total_tool_calls_executed > 0
            and task_engine_cfg
            and task_engine_cfg.get("enable_verify", True)
            and not self.session_memory.is_empty()
        ):
            await self._run_verify_checkpoint(
                system_prompt, message_history, task_description
            )

        # 是否跳过 summary：
        # 1. deep 模式且调用过工具 → 强制生成 summary（中间轮文本不能当最终答案）
        # 2. generate_summary=true → 仅无工具调用时跳过
        # 3. generate_summary=false（默认）→ 有文本就跳过
        generate_summary = self.cfg.main_agent.get("generate_summary", False)
        # deep 模式下如果使用了工具，强制生成 summary
        if effective_mode == EXECUTION_MODE_DEEP and total_tool_calls_executed > 0:
            generate_summary = True
        is_simple_response = (
            not task_failed
            and last_assistant_text
            and last_assistant_text.strip()
            and (not generate_summary or total_tool_calls_executed == 0)
        )

        # Record main loop timing
        _perf_main_loop_elapsed = time.perf_counter() - _perf_main_loop_start
        self.task_log.record_perf("main_loop_duration", _perf_main_loop_elapsed)
        self.task_log.record_perf("main_loop_total_llm_time", _perf_total_llm_time)
        self.task_log.record_perf("main_loop_total_tool_time", _perf_total_tool_time)
        self.task_log.record_perf("main_loop_turns", turn_counter.current_turn, unit="")
        self.task_log.record_perf("main_loop_tool_calls", total_tool_calls_executed, unit="")
        self.task_log.record_perf("effective_mode", effective_mode, unit="")
        if token_budget.enabled:
            self.task_log.record_perf("token_budget_total", token_budget.budget, unit="tokens")
            self.task_log.record_perf("token_budget_used", token_budget.total_used, unit="tokens")
            self.task_log.record_perf(
                "token_budget_ratio", round(token_budget.usage_ratio, 3), unit=""
            )

        if is_simple_response:
            # 简单响应：跳过 summary LLM 调用，直接使用最后的 assistant 文本
            self.task_log.log_step(
                "final_summary", "Simple response detected, skipping summary LLM call"
            )
            self.task_log.record_perf("summary_skipped", 1, unit="bool")
            self.task_log.record_perf("summary_duration", 0.0)

            self.current_agent_id = await self.stream_handler.stream_start_agent("reporter")
            await self.stream_handler.stream_start_llm("reporter")

            await self._streaming_final_message(generate_message_id(), last_assistant_text, True)
            final_answer_text = last_assistant_text
        else:
            # 生成最终摘要
            logger.info(
                f"[{self.agent_name}] Generating summary: task_failed={task_failed}, turns={turn_counter.current_turn}"
            )
            self.task_log.log_step("final_summary", "Generating final summary")
            self.task_log.record_perf("summary_skipped", 0, unit="bool")

            self.current_agent_id = await self.stream_handler.stream_start_agent("reporter")
            await self.stream_handler.stream_start_llm("reporter")

            _perf_summary_start = time.perf_counter()
            final_answer_text = await self._handle_summary(
                system_prompt,
                main_agent_prompt_instance,
                message_history,
                tool_definitions,
                "Final summary generation",
                task_description,
                task_failed,
                agent_type=self.agent_name,
                task_guidance=task_guidance,
                stream_message_callback=self._streaming_final_message,
            )
            self.task_log.record_perf("summary_duration", time.perf_counter() - _perf_summary_start)

        return final_answer_text, is_simple_response

    async def _execute_tools(
        self,
        tool_calls: list,
        max_tool_calls: int,
        keep_tool_result: int,
    ) -> list:
        """执行工具调用 — 普通工具串行，子 Agent 并行"""
        all_tool_results_with_id = []

        tool_calls_exceeded = len(tool_calls[0]) > max_tool_calls
        if tool_calls_exceeded:
            logger.warning(
                f"[ERROR] Tool call count too high ({len(tool_calls[0])}), "
                f"only processing first {max_tool_calls}"
            )

        calls_to_process = tool_calls[0][:max_tool_calls]

        # Handle built-in tools (update_todo, spawn_agent, read_result)
        builtin_results = []
        spawn_calls = []
        remaining_calls = []
        for call in calls_to_process:
            if call["tool_name"] == BUILTIN_TOOL_SEARCH and self.deferred_tool_manager:
                query = call["arguments"].get("query", "")
                max_results = call["arguments"].get("max_results", 5)
                results = self.deferred_tool_manager.resolve_tool_search(query, max_results)
                if results:
                    result_lines = []
                    for r in results:
                        result_lines.append(
                            f"**{r['tool_name']}** (server: {r['server_name']})\n"
                            f"  Description: {r['description']}\n"
                            f"  Schema: {_json.dumps(r['schema'], ensure_ascii=False)}"
                        )
                    result_text = (
                        f"Found {len(results)} matching tools. "
                        f"You can now call these tools directly:\n\n" + "\n\n".join(result_lines)
                    )
                else:
                    result_text = f"No tools found matching '{query}'. Try different keywords."
                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(
                    {
                        "server_name": "builtin",
                        "tool_name": BUILTIN_TOOL_SEARCH,
                        "result": result_text,
                    }
                )
                builtin_results.append((call["id"], tool_result_for_llm))
            elif call["tool_name"] == "update_todo":
                if self.todo_tracker:
                    logger.info(
                        f"[{self.agent_name}] Builtin: update_todo action={call['arguments'].get('action', '?')}"
                    )
                    result_text = self.todo_tracker.update_from_tool_call(call["arguments"])
                else:
                    logger.warning(
                        f"[{self.agent_name}] update_todo called but todo_tracker is not available "
                        f"(this agent does not have a todo tracker enabled)"
                    )
                    result_text = (
                        "Error: update_todo tool does not exist in this agent. "
                        "Do NOT call it again. Focus on your task using the tools available to you."
                    )
                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(
                    {
                        "server_name": "builtin",
                        "tool_name": "update_todo",
                        "result": result_text,
                    }
                )
                builtin_results.append((call["id"], tool_result_for_llm))
            elif call["tool_name"] == BUILTIN_TOOL_SPAWN_AGENT:
                spawn_calls.append(call)
            elif call["tool_name"] == BUILTIN_TOOL_READ_RESULT:
                ref = call["arguments"].get("ref", "")
                logger.info(f"[{self.agent_name}] Builtin: read_result ref={ref}")
                result_text = self._handle_read_result(ref)
                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(
                    {
                        "server_name": "builtin",
                        "tool_name": BUILTIN_TOOL_READ_RESULT,
                        "result": result_text,
                    }
                )
                builtin_results.append((call["id"], tool_result_for_llm))
            else:
                remaining_calls.append(call)

        # Execute spawn_agent calls (parallel by default, serial if configured)
        if spawn_calls:
            spawn_results = await self._execute_spawn_calls(spawn_calls, keep_tool_result)
            builtin_results.extend(spawn_results)

        all_tool_results_with_id.extend(builtin_results)

        # Separate regular tools and agent calls
        regular_calls = []
        agent_calls = []
        for call in remaining_calls:
            if call["server_name"].startswith(SUB_AGENT_PREFIX):
                agent_calls.append(call)
            else:
                regular_calls.append(call)

        # Execute regular tools with concurrency awareness
        # Concurrent-safe tools (search, scrape, read, etc.) run in parallel
        # Non-concurrent tools run sequentially, waiting for previous batch to finish
        regular_results = await self._execute_regular_tools_concurrent(regular_calls)
        all_tool_results_with_id.extend(regular_results)

        # Execute agent calls in parallel
        if agent_calls:
            if self.sub_agent_runner is None:
                for call in agent_calls:
                    tool_result_for_llm = {
                        "type": "text",
                        "text": "[Error] Sub-agent spawning is not available in this context.",
                    }
                    all_tool_results_with_id.append((call["id"], tool_result_for_llm))
            else:
                # Pause main agent stream
                await self.stream_handler.stream_end_llm(self.agent_name)
                await self.stream_handler.stream_end_agent(self.agent_name, self.current_agent_id)

                # Run sub-agents with instance-level concurrency limit
                async def _run_one_agent(call):
                    async with self._sub_agent_semaphore:
                        try:
                            result = await self.sub_agent_runner.run(
                                call["server_name"], call["arguments"], keep_tool_result
                            )
                            return call["id"], call["server_name"], call["tool_name"], result
                        except Exception as e:
                            logger.error(f"Sub-agent '{call['server_name']}' failed: {e}")
                            return (
                                call["id"],
                                call["server_name"],
                                call["tool_name"],
                                f"[Sub-agent Error] {str(e)[:500]}",
                            )

                agent_results = await asyncio.gather(
                    *[_run_one_agent(c) for c in agent_calls],
                )

                for item in agent_results:
                    call_id, server_name, tool_name, sub_result = item
                    self.session_memory.add_sub_agent_result(
                        server_name,
                        sub_result[:500] if isinstance(sub_result, str) else str(sub_result)[:500],
                    )
                    tool_result = {
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "result": sub_result,
                    }
                    tool_result_for_llm = self.output_formatter.format_tool_result_for_user(
                        tool_result
                    )
                    # Backup large sub-agent results for read_result recovery
                    sa_text = (
                        tool_result_for_llm.get("text", "")
                        if isinstance(tool_result_for_llm, dict)
                        else ""
                    )
                    if isinstance(sa_text, str):
                        sa_ref = self.context_manager.backup_large_result(
                            sa_text,
                            tool_name=tool_name,
                            turn=self.context_manager._current_turn,
                        )
                        if sa_ref:
                            tool_result_for_llm["_offload_ref"] = sa_ref
                            tool_result_for_llm["_offload_chars"] = len(sa_text)
                    all_tool_results_with_id.append((call_id, tool_result_for_llm))

                # Resume main agent stream
                self.current_agent_id = await self.stream_handler.stream_start_agent(
                    self.agent_name, display_name="Summarizing"
                )
                await self.stream_handler.stream_start_llm(
                    self.agent_name, display_name="Summarizing"
                )

        # Handle failed tool calls
        if len(tool_calls) > 1 and len(tool_calls[1]) > 0:
            _, failed_results = self.tool_executor.handle_failed_tool_calls(tool_calls[1])
            all_tool_results_with_id.extend(failed_results)

        return all_tool_results_with_id

    async def _run_verify_checkpoint(
        self,
        system_prompt: str,
        message_history: list,
        task_description: str,
    ) -> None:
        """Deep 模式 verify 检查点：summary 前验证证据质量。

        用一次 LLM 调用检查：
        - 证据覆盖率（关键结论是否有来源支撑）
        - 冲突证据
        - 未回答的子问题
        结果注入 message_history，供 summary LLM 参考。
        """
        _perf_start = time.perf_counter()

        # 构建 verify prompt
        from mem_deep_research_core.prompts.template_loader import template_loader as _tpl_loader

        evidence_summary = self.session_memory.to_evidence_string()
        if not evidence_summary:
            evidence_summary = self.session_memory.to_context_string()

        try:
            verify_prompt = _tpl_loader.load_and_render(
                "reflection/verify",
                task_preview=task_description[:300],
                evidence_summary=evidence_summary,
            )
        except FileNotFoundError:
            logger.debug("[Verify] verify template not found, skipping")
            return

        # 注入 verify prompt 到 message_history
        message_history.append(
            {
                "role": "user",
                "_type": MT.REFLECTION,
                "content": [{"type": "text", "text": verify_prompt}],
            }
        )

        # 调用 LLM 做 verify
        try:
            verify_text, _, _ = await self._handle_llm_call(
                system_prompt,
                message_history,
                [],  # verify 不需要工具
                999,
                "Deep verify checkpoint",
                agent_type=self.agent_name,
            )
            if verify_text:
                logger.info(
                    f"[Verify] Checkpoint completed ({len(verify_text)} chars, "
                    f"{time.perf_counter() - _perf_start:.1f}s)"
                )
                self.task_log.log_step(
                    "deep_verify",
                    f"Verify checkpoint: {verify_text[:200]}...",
                )
            else:
                logger.warning("[Verify] LLM returned empty verify response")
        except Exception as e:
            logger.warning(f"[Verify] Checkpoint failed: {e}")
            # 非致命错误，不影响后续 summary

        self.task_log.record_perf("verify_duration", time.perf_counter() - _perf_start)

    def _handle_read_result(self, ref: str) -> str:
        """处理 read_result 工具调用：从 offload 文件或 dedup 缓存中回捞完整结果。

        支持两种引用格式：
        - 文件引用: "turn3_search_15000chars.txt"
        - 轮次引用: "turn:3" — 返回该轮次所有缓存的工具结果
        """
        # 1. 轮次引用: "turn:N"
        if ref.startswith("turn:"):
            try:
                target_turn = int(ref.split(":")[1])
            except (ValueError, IndexError):
                return f"Invalid turn reference: {ref}. Use format 'turn:N' where N is a number."
            records = [
                r for r in self.context_manager._call_registry if r.turn == target_turn
            ]
            if not records:
                return f"No tool results found for turn {target_turn}."
            parts = []
            for r in records:
                content = r.result_full
                # 优先用 offload_ref 回捞
                if r.offload_ref:
                    restored = self.context_manager.restore_single_file(r.offload_ref)
                    if restored:
                        content = restored
                # 回退: 检查 content 中的 OFFLOADED marker
                elif content and TAG_OFFLOADED in content:
                    offload_ref = self._extract_offload_ref(content)
                    if offload_ref:
                        restored = self.context_manager.restore_single_file(offload_ref)
                        if restored:
                            content = restored
                if content and TAG_OFFLOADED not in content:
                    parts.append(
                        f"=== {r.tool_name} (turn {r.turn}, {r.result_chars} chars) ===\n"
                        f"{content}"
                    )
                else:
                    parts.append(
                        f"=== {r.tool_name} (turn {r.turn}) ===\n"
                        f"[Result not in cache, brief: {r.result_brief}]"
                    )
            return "\n\n".join(parts)

        # 2. 文件引用: 通过 context_manager 恢复（支持 on_result_restore hook + 路径遍历防护）
        content = self.context_manager.restore_single_file(ref)
        if content is not None:
            return content

        # 3. Fallback: 在 dedup 缓存中按文件名模式搜索
        # 文件名格式: turn{N}_{tool_name}_{chars}chars.txt
        for record in self.context_manager._call_registry:
            expected_name = f"turn{record.turn}_{record.tool_name}_{record.result_chars}chars.txt"
            if expected_name == ref and record.result_full:
                return record.result_full

        return (
            f"Could not find content for '{ref}'. "
            f"Available turns: {sorted({r.turn for r in self.context_manager._call_registry})}. "
            f"Try 'turn:N' format to get results from a specific turn."
        )

    @staticmethod
    def _extract_offload_ref(text: str) -> str | None:
        """Extract filename from an [OFFLOADED:filename|size] marker."""
        start = text.find(TAG_OFFLOADED)
        if start < 0:
            return None
        start += len(TAG_OFFLOADED)
        end = text.find("]", start)
        if end < 0:
            return None
        # Format: "filename|size" — take the filename part
        ref_part = text[start:end]
        return ref_part.split("|")[0].strip() or None

    @staticmethod
    def _is_concurrent_safe(tool_name: str) -> bool:
        """判断工具是否可以安全并发执行（只读、无副作用）。

        使用分段匹配：将工具名按 _ 和 - 拆分为段，检查是否有任一段
        命中安全集合。避免子串误判（如 "research_and_write" 不会匹配 "search"）。
        """
        import re
        segments = set(re.split(r"[_\-]+", tool_name.lower()))
        return bool(segments & CONCURRENT_SAFE_TOOL_SEGMENTS)

    async def _execute_one_regular_tool(self, call: dict) -> tuple[str, dict]:
        """执行单个普通工具并返回 (call_id, formatted_result)"""
        tool_result, _ = await self.tool_executor.execute_single_tool(
            server_name=call["server_name"],
            tool_name=call["tool_name"],
            arguments=call["arguments"],
            call_id=call["id"],
            agent_name=self.agent_name,
        )
        tool_result_for_llm = self.output_formatter.format_tool_result_for_user(tool_result)

        # Backup large results to file (不替换 history，由滑动窗口统一处理)
        result_text = (
            tool_result_for_llm.get("text", "") if isinstance(tool_result_for_llm, dict) else ""
        )
        if isinstance(result_text, str):
            ref = self.context_manager.backup_large_result(
                result_text,
                tool_name=call["tool_name"],
                turn=self.context_manager._current_turn,
            )
            if ref:
                # 挂 metadata 到结果上，供后续 message_history 中的消息携带
                tool_result_for_llm["_offload_ref"] = ref
                tool_result_for_llm["_offload_chars"] = len(result_text)

        return call["id"], tool_result_for_llm

    async def _execute_regular_tools_concurrent(
        self,
        regular_calls: list[dict],
    ) -> list[tuple[str, dict]]:
        """执行普通工具，concurrent-safe 工具并行，其他串行。

        参考 Claude Code 的 StreamingToolExecutor 模式：
        - 将工具分成连续的 batch：相邻的 concurrent-safe 工具合为一个并行 batch
        - 遇到 non-concurrent 工具时，等前面的 batch 完成，再独占执行
        - 结果按原始顺序返回
        """
        if not regular_calls:
            return []

        # 按并发安全性分 batch
        batches: list[tuple[bool, list[dict]]] = []  # (is_concurrent, calls)
        for call in regular_calls:
            is_safe = self._is_concurrent_safe(call.get("tool_name", ""))
            if batches and batches[-1][0] == is_safe:
                batches[-1][1].append(call)
            else:
                batches.append((is_safe, [call]))

        all_results: list[tuple[str, dict]] = []

        for is_concurrent, batch_calls in batches:
            if is_concurrent and len(batch_calls) > 1:
                # 并行执行 concurrent-safe 工具
                logger.info(
                    f"[{self.agent_name}] Executing {len(batch_calls)} concurrent-safe tools in parallel: "
                    f"{[c.get('tool_name', '?') for c in batch_calls]}"
                )
                tasks = [self._execute_one_regular_tool(c) for c in batch_calls]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        call = batch_calls[i]
                        logger.error(f"Concurrent tool {call.get('tool_name')} failed: {result}")
                        all_results.append(
                            (
                                call["id"],
                                {"type": "text", "text": f"[Tool Error] {str(result)[:500]}"},
                            )
                        )
                    else:
                        all_results.append(result)
            else:
                # 串行执行（non-concurrent 或单个 concurrent）
                for call in batch_calls:
                    result = await self._execute_one_regular_tool(call)
                    all_results.append(result)

        return all_results

    async def _execute_spawn_calls(
        self, spawn_calls: list[dict], keep_tool_result: int
    ) -> list[tuple[str, dict]]:
        """Execute spawn_agent calls — parallel (with semaphore) or serial based on config."""

        async def _run_one_spawn(call):
            task_desc = call["arguments"].get("task_description", str(call["arguments"]))
            spawn_max_turns = call["arguments"].get("max_turns")
            logger.info(f"[{self.agent_name}] spawn_agent task={task_desc[:80]}")
            try:
                async with self._sub_agent_semaphore:
                    spawn_result = await self._run_spawned_agent(
                        task_desc, keep_tool_result, max_turns=spawn_max_turns
                    )
            except Exception as e:
                logger.error(f"spawn_agent failed: {e}")
                spawn_result = f"[Spawn Error] {str(e)[:500]}"

            self.monitor.state.last_progress_time = time.time()
            self.monitor.state.stall_warned = False
            self._spawn_executed = True
            self.session_memory.add_sub_agent_result(
                BUILTIN_TOOL_SPAWN_AGENT, spawn_result[:500]
            )

            tool_result_for_llm = self.output_formatter.format_tool_result_for_user(
                {
                    "server_name": "builtin",
                    "tool_name": BUILTIN_TOOL_SPAWN_AGENT,
                    "result": spawn_result,
                }
            )
            result_text = (
                tool_result_for_llm.get("text", "")
                if isinstance(tool_result_for_llm, dict)
                else ""
            )
            if isinstance(result_text, str):
                ref = self.context_manager.backup_large_result(
                    result_text,
                    tool_name=BUILTIN_TOOL_SPAWN_AGENT,
                    turn=self.context_manager._current_turn,
                )
                if ref:
                    tool_result_for_llm["_offload_ref"] = ref
                    tool_result_for_llm["_offload_chars"] = len(result_text)
            return (call["id"], tool_result_for_llm)

        parallel = getattr(self.cfg.main_agent, "parallel_spawn", True)

        if parallel and len(spawn_calls) > 1:
            logger.info(
                f"[{self.agent_name}] Executing {len(spawn_calls)} spawn_agent calls in parallel "
                f"(semaphore={self._sub_agent_semaphore._value})"
            )
            results = await asyncio.gather(
                *[_run_one_spawn(c) for c in spawn_calls], return_exceptions=True
            )
            out = []
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    logger.error(f"spawn_agent parallel task failed: {r}")
                    out.append((
                        spawn_calls[i]["id"],
                        {"type": "text", "text": f"[Spawn Error] {str(r)[:500]}"},
                    ))
                else:
                    out.append(r)
            return out
        else:
            return [await _run_one_spawn(c) for c in spawn_calls]

    async def _run_spawned_agent(
        self, task_description: str, keep_tool_result: int, *, max_turns: int | None = None
    ) -> str:
        """Run a temporary spawned agent — delegates to SubAgentRunner.spawn()."""
        if self.spawn_depth >= MAX_SPAWN_DEPTH:
            return (
                f"[Spawn Rejected] Maximum nesting depth ({MAX_SPAWN_DEPTH}) reached. "
                "Cannot spawn further sub-agents. Please complete this task directly."
            )

        from mem_deep_research_core.core.sub_agent_runner import SubAgentRunner

        # Build a lightweight SubAgentRunner for spawning
        spawner = SubAgentRunner(
            sub_agent_tool_managers={},
            sub_agent_llm_client=self.llm_client,
            output_formatter=self.output_formatter,
            cfg=self.cfg,
            task_log=self.task_log,
            context=self.context,
            chinese_context=self.chinese_context,
            response_language=self.response_language,
            stream_handler=self.stream_handler,
            stream_tool_reasoning=self._stream_tool_reasoning,
            handle_llm_call=self._handle_llm_call,
            handle_summary=self._handle_summary,
            intercept_key_message=self._intercept_key_message,
            streaming_final_message=self._streaming_final_message,
            hooks=self.hooks,
            config_loader=self.config_loader,
        )

        # Tool definitions: remove builtin tools that spawned agents cannot handle
        next_depth = self.spawn_depth + 1
        # Always remove update_todo (spawned agents have no todo_tracker)
        # Remove spawn_agent if child can't spawn further
        tools_to_remove = {"update_todo"}
        if next_depth >= MAX_SPAWN_DEPTH:
            tools_to_remove.add(BUILTIN_TOOL_SPAWN_AGENT)
        spawn_tool_defs = [
            td
            for td in (self._current_tool_definitions or [])
            if not any(
                t.get("name") in tools_to_remove
                for t in td.get("tools", [td]) if isinstance(t, dict)
            )
        ]

        return await spawner.spawn(
            task_description,
            parent_llm_client=self.llm_client,
            parent_tool_executor=self.tool_executor,
            parent_tool_definitions=spawn_tool_defs,
            parent_callbacks={
                "handle_llm_call": self._handle_llm_call,
                "handle_summary": self._handle_summary,
                "intercept_key_message": self._intercept_key_message,
                "streaming_final_message": self._streaming_final_message,
                "stream_tool_reasoning": self._stream_tool_reasoning,
            },
            keep_tool_result=keep_tool_result,
            spawn_depth=next_depth,
            hooks_instance=self.hooks,
            parent_system_prompt=getattr(self, "_current_system_prompt", None),
            max_turns=max_turns,
            parent_context_manager=self.context_manager,
        )

    async def _run_fork_skill(self, skill_command) -> str:
        """Run a fork-mode skill as a spawned sub-agent with filtered tools."""
        # Render the skill prompt
        rendered_prompt = await skill_command.get_prompt()

        return await self._run_spawned_agent(rendered_prompt, -1)

    async def _context_summarize_call(
        self,
        summarize_system_prompt: str,
        summarize_messages: list,
        purpose: str,
    ) -> str:
        """Level 2 context 压缩的 LLM 调用"""
        response_text, _, _ = await self.llm_handler.handle_llm_call(
            summarize_system_prompt,
            summarize_messages,
            [],
            999,
            purpose,
            agent_type=self.agent_name,
        )
        return response_text or ""
