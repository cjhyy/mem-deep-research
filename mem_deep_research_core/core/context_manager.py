"""
Context Manager 模块

窗口压缩策略通过 WindowStrategyPipeline 可插拔管理：

  Token 用量:  0% -------- 60% -------- 80% -------- 95% ---> 爆了
  策略:        [完整保留]   [摘要替换]    [LLM压缩]    [紧急裁剪]
                             Level 1      Level 2      Level 3

窗口策略（可自定义，参见 window_strategy.py）：
  - ObservationMaskingStrategy: 遮蔽旧轮工具输出（零 LLM 成本）
  - LLMSummarizeStrategy: LLM 压缩旧历史为结构化摘要
  - BinaryReductionStrategy: 二分删除中间消息（紧急兜底）

其他功能：
- Tool Call Dedup：跨轮次去重，渐进升级提示
- Source Registry：引用来源追踪

参考：JetBrains NeurIPS 2025 Observation Masking + Anthropic Tool Result Clearing
"""

import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from mem_deep_research_core.core.constants import (
    MT,
    MICROCOMPACT_MIN_CHARS,
    PROTECTED_MESSAGE_TYPES,
    RESULT_BRIEF_LENGTH,
    SYSTEM_MESSAGE_KEYWORDS,
    TAG_OFFLOADED,
)
from mem_deep_research_core.core.window_strategy import (
    BinaryReductionStrategy,
    LLMSummarizeStrategy,
    ObservationMaskingStrategy,
    WindowContext,
    WindowStrategyPipeline,
)

logger = logging.getLogger("mem_deep_research")


@dataclass
class ToolCallRecord:
    """工具调用记录"""

    tool_name: str
    arguments_hash: str  # MD5(tool_name + sorted_json(arguments))
    arguments: dict  # 原始参数
    turn: int
    result_hash: str  # MD5(result_text)
    result_brief: str  # result 前 200 字符（日志用）
    result_full: str = ""  # 完整结果（dedup 缓存命中时返回给模型）
    result_chars: int = 0  # 结果字符数
    hit_count: int = 0  # dedup 命中次数（用于渐进升级）
    was_duplicate: bool = False
    source_url: str | None = None
    source_title: str | None = None
    source_date: str | None = None
    offload_ref: str = ""  # 逻辑 ref（如 toolmsg_abcd1234.txt）


@dataclass
class OffloadRecord:
    """Offload 记录 — 跟踪一条大工具结果的备份与替换状态"""

    ref: str  # 逻辑引用（toolmsg_{uuid8}.txt）
    turn: int
    char_count: int
    tool_names: list[str] = field(default_factory=list)
    state: str = "backed_up"  # backed_up | pending_evidence | offloaded
    evidence: list[str] = field(default_factory=list)


@dataclass
class SourceRecord:
    """来源记录"""

    url: str
    title: str = ""
    tool_name: str = ""
    turn: int = 0


class SourceRegistry:
    """来源注册表 -- 自动从工具调用中提取引用来源"""

    SOURCE_RULES: dict[str, dict[str, str]] = {
        "search": {"url": "url", "title": "title"},
        "google_search": {"url": "link", "title": "title"},
        "scrape_website": {"url": "url"},
        "scrape": {"url": "url"},
        "get_wikipedia_info": {"title": "entity"},
        "fetch": {"url": "url"},
        "web_search": {"url": "url", "title": "title"},
    }

    def __init__(self):
        import threading

        self._lock = threading.Lock()
        self._sources: list[SourceRecord] = []
        self._seen_urls: set = set()

    def extract_and_register(
        self,
        tool_name: str,
        arguments: dict,
        result_text: str,
        turn: int,
    ) -> list[SourceRecord]:
        """从工具调用中提取并注册来源（线程安全）"""
        new_sources = []

        # 1. 从 arguments 提取
        rule = self.SOURCE_RULES.get(tool_name)
        if rule:
            url = arguments.get(rule.get("url", ""), "") if "url" in rule else ""
            title = arguments.get(rule.get("title", ""), "") if "title" in rule else ""
            if url and url not in self._seen_urls:
                source = SourceRecord(url=url, title=title, tool_name=tool_name, turn=turn)
                new_sources.append(source)

        # 2. 从 JSON 结果中提取 URL 列表
        try:
            parsed = json.loads(result_text)
            items = []
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                for key in ("results", "organic_results", "items", "data"):
                    if key in parsed and isinstance(parsed[key], list):
                        items = parsed[key]
                        break

            for item in items:
                if not isinstance(item, dict):
                    continue
                url = ""
                for url_key in ("url", "link", "href"):
                    if url_key in item and item[url_key]:
                        url = str(item[url_key])
                        break
                if not url or url in self._seen_urls:
                    continue
                title = ""
                for title_key in ("title", "name", "snippet"):
                    if title_key in item and item[title_key]:
                        title = str(item[title_key])
                        break
                source = SourceRecord(url=url, title=title, tool_name=tool_name, turn=turn)
                new_sources.append(source)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Commit under lock to avoid race on _sources/_seen_urls
        with self._lock:
            for src in new_sources:
                if src.url not in self._seen_urls:
                    self._sources.append(src)
                    self._seen_urls.add(src.url)

        return new_sources

    def get_all_sources(self) -> list[SourceRecord]:
        with self._lock:
            return list(self._sources)

    def get_citation_summary(self) -> str:
        with self._lock:
            sources = list(self._sources)
        if not sources:
            return ""
        lines = ["## Sources"]
        for i, source in enumerate(sources, 1):
            if source.title:
                lines.append(f"{i}. [{source.title}]({source.url})")
            else:
                lines.append(f"{i}. {source.url}")
        return "\n".join(lines)

    def reset(self):
        with self._lock:
            self._sources.clear()
            self._seen_urls.clear()


# ============================================================
# 配置
# ============================================================


@dataclass
class ContextManagerConfig:
    """ContextManager 配置"""

    enable_dedup: bool = True  # 是否启用跨轮次去重
    enable_compact: bool = True  # 是否启用 Level 1 摘要替换

    # Level 1: 摘要替换
    compact_at_ratio: float = 0.6  # token 占比超过此值时触发 compact
    compact_keep_recent: int = 3  # 至少保留最近 N 轮完整结果
    compact_preview_length: int = 300  # masking 摘要中保留的结果预览字符数

    # Level 2: LLM 压缩
    summarize_at_ratio: float = 0.8  # token 占比超过此值时触发 LLM 压缩

    # Dedup cache
    max_dedup_cache_size: int = 200  # Maximum entries in dedup cache

    # Token 估算
    chars_per_token: float = 3.5  # 无 tiktoken 时的 fallback 估算

    # Result offloading
    result_offload_threshold: int = 5000  # 0 = disabled
    result_offload_dir: str = ""

    # Evidence extraction
    enable_evidence_extraction: bool = True


# ============================================================
# ContextManager
# ============================================================


class ContextManager:
    """Context 管理器

    窗口压缩策略通过 WindowStrategyPipeline 管理，支持可插拔自定义策略。
    默认内置三级策略：ObservationMasking → LLMSummarize → BinaryReduction

    其他功能：Dedup、SourceRegistry
    """

    def __init__(
        self,
        config: ContextManagerConfig | None = None,
        pipeline: WindowStrategyPipeline | None = None,
        hooks=None,
    ):
        self.config = config or ContextManagerConfig()
        self._hooks = hooks
        self._max_dedup_cache_size = self.config.max_dedup_cache_size
        self._current_turn: int = 0

        # Dedup cache: arguments_hash -> ToolCallRecord
        self._dedup_cache: dict[str, ToolCallRecord] = {}

        # Call registry: 所有 tool call 记录（按轮次）
        self._call_registry: list[ToolCallRecord] = []

        # Source Registry
        self.source_registry = SourceRegistry()

        # Token 估算函数（由 orchestrator 注入，可选）
        self._token_estimator: Callable[[str], int] | None = None

        # 已 compact 的轮次集合
        self._compacted_turns: set = set()

        # 窗口策略管道（可外部注入或从 config 自动构建）
        self._pipeline = pipeline or self._build_pipeline_from_config()

        # Session memory 引用（由 MainLoopRunner 注入，供 SessionMemoryCompactStrategy 使用）
        self._session_memory = None

        # Offload directory for large results
        self._offload_dir: str = ""

        # Offload registry: ref -> OffloadRecord
        self._offload_registry: dict[str, OffloadRecord] = {}

    # ---- 初始化 ----

    def _build_pipeline_from_config(self) -> WindowStrategyPipeline:
        """从 ContextManagerConfig 构建默认策略管道"""
        strategies = []
        if self.config.enable_compact:
            strategies.append(
                ObservationMaskingStrategy(
                    trigger_ratio=self.config.compact_at_ratio,
                    keep_recent=self.config.compact_keep_recent,
                    chars_per_token=self.config.chars_per_token,
                    preview_length=self.config.compact_preview_length,
                )
            )
        strategies.append(
            LLMSummarizeStrategy(
                trigger_ratio=self.config.summarize_at_ratio,
                keep_recent=self.config.compact_keep_recent,
            )
        )
        strategies.append(BinaryReductionStrategy(trigger_ratio=0.95))
        return WindowStrategyPipeline(strategies)

    @property
    def pipeline(self) -> WindowStrategyPipeline:
        """暴露策略管道，允许外部访问/替换"""
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value: WindowStrategyPipeline) -> None:
        self._pipeline = value

    def set_token_estimator(self, fn: Callable[[str], int]) -> None:
        """注入 token 估算函数（如 tiktoken）"""
        self._token_estimator = fn

    def set_turn(self, turn: int) -> None:
        self._current_turn = turn

    def set_session_memory(self, session_memory) -> None:
        """注入 SessionMemory 引用（供 SessionMemoryCompactStrategy 使用）"""
        self._session_memory = session_memory

    def set_offload_dir(self, path: str) -> None:
        """Set the directory for offloading large results."""
        self._offload_dir = path

    def _generate_offload_ref(self) -> str:
        """生成稳定的逻辑引用（短 UUID），含碰撞重试"""
        for _ in range(10):
            ref = f"toolmsg_{uuid.uuid4().hex[:8]}.txt"
            if ref not in self._offload_registry:
                return ref
        return f"toolmsg_{uuid.uuid4().hex[:12]}.txt"

    def backup_large_result(
        self,
        result_text: str,
        tool_name: str,
        turn: int,
    ) -> str | None:
        """Phase 1: 备份大工具结果到文件，返回逻辑 ref。

        只做备份，不替换 message_history 中的内容。
        真正的替换在 finalize_offload_candidates() 中统一处理。

        Hook 签名:
            on_result_offload(ctx, original_fn) -> dict | None
            ctx.extra = {"result_text": str, "tool_name": str, "turn": int, "file_name": str}
            返回 {"ref": str} 覆盖默认行为，返回 None 走默认。

        Returns:
            逻辑 ref（如 toolmsg_abcd1234.txt），备份失败或未触发返回 None
        """
        threshold = self.config.result_offload_threshold
        if threshold <= 0 or len(result_text) <= threshold:
            return None

        ref = self._generate_offload_ref()

        # Hook: on_result_offload — 用户可覆盖存储后端（S3/Redis/内存等）
        from mem_deep_research_core.core.hooks import HookContext

        if self._hooks is not None and self._hooks.has_hooks("on_result_offload"):
            hook_result = self._hooks.call(
                "on_result_offload",
                HookContext(
                    hook_name="on_result_offload",
                    extra={
                        "result_text": result_text,
                        "tool_name": tool_name,
                        "turn": turn,
                        "file_name": ref,
                    },
                ),
            )
            if isinstance(hook_result, dict) and hook_result.get("ref"):
                ref = hook_result["ref"]
                self._offload_registry[ref] = OffloadRecord(
                    ref=ref,
                    turn=turn,
                    char_count=len(result_text),
                    tool_names=[tool_name],
                    state="backed_up",
                )
                logger.info(
                    f"[Context] Backed up {len(result_text)} chars via hook, ref={ref}"
                )
                return ref

        # 默认：写本地文件系统
        offload_dir = self._offload_dir or self.config.result_offload_dir
        if not offload_dir:
            return None

        import os

        os.makedirs(offload_dir, exist_ok=True)

        file_path = os.path.join(offload_dir, ref)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result_text)
        except Exception as e:
            logger.warning(f"[Context] Failed to backup result: {e}")
            return None

        self._offload_registry[ref] = OffloadRecord(
            ref=ref,
            turn=turn,
            char_count=len(result_text),
            tool_names=[tool_name],
            state="backed_up",
        )
        logger.info(
            f"[Context] Backed up {len(result_text)} chars to {file_path}, ref={ref}"
        )
        return ref

    def restore_offloaded_content(self, message_history: list) -> int:
        """恢复 offloaded 内容（resume 场景使用）。

        默认从本地文件读取。可通过 on_result_restore hook 覆盖读取后端。

        Hook 签名:
            on_result_restore(ctx, original_fn) -> str | None
            ctx.extra = {"file_name": str, "marker_text": str}
            返回原始内容字符串覆盖默认行为，返回 None 走默认文件读取。

        Returns:
            恢复的消息数
        """
        import os

        from mem_deep_research_core.core.hooks import HookContext

        offload_dir = self._offload_dir or self.config.result_offload_dir

        restored = 0
        for msg in message_history:
            content = msg.get("content")
            if not isinstance(content, list) or not content:
                continue
            first = content[0] if isinstance(content[0], dict) else {}
            text = first.get("text", "")
            if not text.startswith(TAG_OFFLOADED):
                continue

            # Parse [OFFLOADED:filename|chars]
            try:
                marker_end = text.index("]")
                marker_body = text[len(TAG_OFFLOADED) : marker_end]
                file_name = marker_body.split("|")[0]
            except (ValueError, IndexError):
                continue

            # Hook: on_result_restore — 用户可覆盖读取后端
            original_content = None
            if self._hooks is not None and self._hooks.has_hooks("on_result_restore"):
                hook_result = self._hooks.call(
                    "on_result_restore",
                    HookContext(
                        hook_name="on_result_restore",
                        extra={"file_name": file_name, "marker_text": text},
                    ),
                )
                if isinstance(hook_result, str):
                    original_content = hook_result

            # 默认：从本地文件读取（含路径遍历防护）
            if original_content is None and offload_dir:
                real_offload = os.path.realpath(offload_dir)
                file_path = os.path.realpath(os.path.join(offload_dir, file_name))
                if not file_path.startswith(real_offload + os.sep) and file_path != real_offload:
                    logger.warning(
                        f"[Context] Path traversal blocked in restore: {file_name!r}"
                    )
                    continue
                if not os.path.isfile(file_path):
                    logger.debug(f"[Context] Offloaded file not found: {file_path}")
                    continue
                try:
                    with open(file_path, encoding="utf-8") as f:
                        original_content = f.read()
                except Exception as e:
                    logger.warning(f"[Context] Failed to restore offloaded content: {e}")
                    continue

            if original_content is not None:
                msg["content"] = [{"type": "text", "text": original_content}]
                restored += 1

        if restored > 0:
            logger.info(f"[Context] Restored {restored} offloaded messages from {offload_dir}")

        # Rebuild _offload_registry from message_history so that read_result,
        # finalize_offload_candidates, and microcompact work correctly after resume.
        self._rebuild_registry_from_history(message_history)

        return restored

    def _rebuild_registry_from_history(self, message_history: list) -> None:
        """Scan message_history to rebuild _offload_registry entries lost during resume.

        Handles two cases:
        1. Messages with _offload_refs whose content was restored (state="backed_up")
        2. Messages still carrying OFFLOADED markers (state="offloaded")
        """
        import os
        import re

        offload_dir = self._offload_dir or self.config.result_offload_dir

        for msg in message_history:
            # Case 1: messages with _offload_refs (content may have been restored)
            offload_refs = msg.get("_offload_refs")
            if offload_refs:
                for ref in offload_refs:
                    if ref in self._offload_registry:
                        continue
                    # Determine char_count from file on disk if available
                    char_count = 0
                    if offload_dir:
                        real_offload = os.path.realpath(offload_dir)
                        file_path = os.path.realpath(os.path.join(offload_dir, ref))
                        if file_path.startswith(real_offload + os.sep) and os.path.isfile(file_path):
                            try:
                                char_count = os.path.getsize(file_path)
                            except OSError:
                                pass
                    self._offload_registry[ref] = OffloadRecord(
                        ref=ref,
                        turn=0,
                        char_count=char_count,
                        state="backed_up",
                    )

            # Case 2: messages still carrying OFFLOADED markers
            content = msg.get("content")
            if not isinstance(content, list) or not content:
                continue
            first = content[0] if isinstance(content[0], dict) else {}
            text = first.get("text", "")
            if not text.startswith(TAG_OFFLOADED):
                continue

            # Parse all [OFFLOADED:ref|chars] markers in the text
            for match in re.finditer(
                re.escape(TAG_OFFLOADED) + r"([^|]+)\|(\d+)\]", text
            ):
                ref = match.group(1)
                chars = int(match.group(2))
                if ref not in self._offload_registry:
                    self._offload_registry[ref] = OffloadRecord(
                        ref=ref,
                        turn=0,
                        char_count=chars,
                        state="offloaded",
                    )

        if self._offload_registry:
            logger.info(
                f"[Context] Rebuilt offload registry: {len(self._offload_registry)} entries"
            )

    def restore_single_file(self, file_name: str) -> str | None:
        """根据文件名恢复单个 offloaded 结果。

        与 restore_offloaded_content 共享 on_result_restore hook 逻辑，
        确保自定义存储后端（S3/Redis）在 read_result 工具调用时也能正常工作。

        Returns:
            文件内容字符串，找不到返回 None
        """
        import os

        from mem_deep_research_core.core.hooks import HookContext

        # Hook: on_result_restore
        if self._hooks is not None and self._hooks.has_hooks("on_result_restore"):
            hook_result = self._hooks.call(
                "on_result_restore",
                HookContext(
                    hook_name="on_result_restore",
                    extra={"file_name": file_name, "marker_text": ""},
                ),
            )
            if isinstance(hook_result, str):
                return hook_result

        # 默认：从本地文件读取（含路径遍历防护）
        offload_dir = self._offload_dir or self.config.result_offload_dir
        if not offload_dir:
            return None

        real_offload = os.path.realpath(offload_dir)
        file_path = os.path.realpath(os.path.join(offload_dir, file_name))
        if not file_path.startswith(real_offload + os.sep) and file_path != real_offload:
            logger.warning(f"[Context] Path traversal blocked in restore: {file_name!r}")
            return None

        if not os.path.isfile(file_path):
            return None
        try:
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"[Context] Failed to restore file {file_name}: {e}")
            return None

    # ============================================================
    # Microcompact: 每轮自动清理旧 tool_result（零 LLM 成本）
    # ============================================================

    def microcompact(
        self,
        message_history: list,
        current_turn: int,
        keep_recent: int = 3,
    ) -> int:
        """每轮 LLM 调用前自动清理旧 tool_result 内容。

        与 ObservationMasking 不同，microcompact 不等阈值触发，
        每轮无条件清理 N 轮前的 tool_result 文本，只留简短占位符。
        参考 Claude Code 的 microcompact 机制。

        Args:
            message_history: 消息历史（就地修改）
            current_turn: 当前轮次
            keep_recent: 保留最近 N 轮完整结果

        Returns:
            被清理的消息数
        """
        cutoff_turn = current_turn - keep_recent
        if cutoff_turn <= 0:
            return 0

        cleaned = 0
        estimated_turn = 0
        for i in range(1, len(message_history)):
            msg = message_history[i]
            role = msg.get("role", "")

            if role == "assistant":
                estimated_turn += 1
                continue

            # 只清理旧轮次、显式标记为 TOOL_RESULT 的消息
            # role 可以是 "user"（Anthropic/OpenRouter）或 "tool"（GPT native）
            if estimated_turn == 0 or estimated_turn > cutoff_turn:
                continue
            if msg.get("_type") != MT.TOOL_RESULT:
                continue

            content = msg.get("content", "")

            # 计算字符数
            if isinstance(content, str):
                char_count = len(content)
            elif isinstance(content, list):
                char_count = sum(
                    len(item.get("text", "") if isinstance(item, dict) else str(item))
                    for item in content
                )
            else:
                continue

            # 跳过已经很短的消息（已被 compact 过或本身就短）
            if char_count <= MICROCOMPACT_MIN_CHARS:
                continue

            # 跳过已卸载的内容（OFFLOADED 标记可能在 content 中）
            if isinstance(content, list) and content:
                first_text = content[0].get("text", "") if isinstance(content[0], dict) else ""
                if first_text.startswith(TAG_OFFLOADED):
                    continue

            # 有 offload 备份的消息 → OFFLOADED marker（可通过 read_result 回捞）
            offload_refs = msg.get("_offload_refs")
            if offload_refs:
                placeholder = self._build_offload_marker(offload_refs, fallback_chars=char_count)
                msg["content"] = [{"type": "text", "text": placeholder}]
                msg["_type"] = MT.OFFLOADED
                cleaned += 1
                continue

            # 无 offload 备份 → 通用 placeholder
            turn_records = [r for r in self._call_registry if r.turn == estimated_turn]
            if turn_records:
                placeholders = []
                for r in turn_records:
                    brief = r.result_brief[:60].replace("\n", " ") if r.result_brief else ""
                    placeholders.append(
                        f"[microcompact] {r.tool_name}: {r.result_chars} chars — {brief}"
                    )
                placeholder = "\n".join(placeholders)
            else:
                placeholder = f"[microcompact] turn {estimated_turn} tool result cleared ({char_count} chars)"

            msg["content"] = [{"type": "text", "text": placeholder}]
            cleaned += 1

        if cleaned > 0:
            logger.debug(
                f"[CONTEXT] Microcompact: cleared {cleaned} old tool results "
                f"(turns 1-{cutoff_turn})"
            )

        return cleaned

    # ============================================================
    # Offload 滑动窗口: prepare + finalize
    # ============================================================

    def prepare_offload_candidates(
        self,
        message_history: list,
        current_turn: int,
        keep_recent: int = 0,
    ) -> list[dict]:
        """Phase 2: 标记即将滑出 keep_recent 窗口的旧 TOOL_RESULT 消息。

        在每轮 LLM 调用前执行。只做标记，不清扫消息。
        返回候选消息列表（含 ref），供 sidecar prompt 使用。

        Args:
            message_history: 消息历史
            current_turn: 当前轮次
            keep_recent: 保留最近 N 轮（0 = 使用 config 默认值）

        Returns:
            候选列表: [{"ref": str, "turn": int, "chars": int, "msg_index": int}]
        """
        if keep_recent <= 0:
            keep_recent = self.config.compact_keep_recent

        # 有意设计：prepare 比 finalize/microcompact 多看一轮。
        #
        # prepare cutoff  = N - K + 1  (标记即将滑出的消息)
        # finalize cutoff = N - K      (替换已滑出的消息)
        #
        # 这确保 prepare 在 turn N 标记的消息在当前轮仍有完整内容可见，
        # LLM 能在本轮 sidecar prompt 中提取 evidence。
        # 下一轮 (N+1) 的 finalize/microcompact 再执行实际替换。
        cutoff_turn = current_turn - keep_recent + 1
        if cutoff_turn <= 0:
            return []

        candidates = []
        estimated_turn = 0
        for i in range(1, len(message_history)):
            msg = message_history[i]

            if msg.get("role") == "assistant":
                estimated_turn += 1
                continue

            if estimated_turn == 0 or estimated_turn > cutoff_turn:
                continue
            if msg.get("_type") != MT.TOOL_RESULT:
                continue

            # 只标记有 offload 备份的消息
            offload_refs = msg.get("_offload_refs")
            if not offload_refs:
                continue

            for ref in offload_refs:
                record = self._offload_registry.get(ref)
                if not record:
                    continue
                # 重入兜底：pending_evidence 状态的记录也重新纳入候选
                # （上一轮 LLM 调用可能失败，导致 state 滞留）
                if record.state in ("backed_up", "pending_evidence"):
                    record.state = "pending_evidence"
                    candidates.append({
                        "ref": ref,
                        "turn": estimated_turn,
                        "chars": record.char_count,
                        "msg_index": i,
                    })

        if candidates:
            logger.debug(
                f"[CONTEXT] Offload candidates: {len(candidates)} messages "
                f"(turns <= {cutoff_turn}) marked for evidence extraction"
            )
        return candidates

    def finalize_offload_candidates(
        self,
        message_history: list,
        current_turn: int,
        keep_recent: int = 0,
    ) -> int:
        """Phase 4: 替换已准备好的候选消息为 OFFLOADED marker。

        在本轮 assistant 响应产生后执行。把 pending_evidence 状态的消息
        替换为 OFFLOADED marker（内联 evidence）。

        Args:
            message_history: 消息历史（就地修改）
            current_turn: 当前轮次
            keep_recent: 保留最近 N 轮（0 = 使用 config 默认值）

        Returns:
            被替换的消息数
        """
        if keep_recent <= 0:
            keep_recent = self.config.compact_keep_recent

        cutoff_turn = current_turn - keep_recent
        if cutoff_turn <= 0:
            return 0

        replaced = 0
        estimated_turn = 0
        for i in range(1, len(message_history)):
            msg = message_history[i]

            if msg.get("role") == "assistant":
                estimated_turn += 1
                continue

            if estimated_turn == 0 or estimated_turn > cutoff_turn:
                continue
            if msg.get("_type") not in (MT.TOOL_RESULT,):
                continue

            offload_refs = msg.get("_offload_refs")
            if not offload_refs:
                continue

            # 过滤掉 registry 中不存在的 ref（避免生成空 marker）
            valid_refs = [r for r in offload_refs if r in self._offload_registry]
            if not valid_refs:
                continue

            marker = self._build_offload_marker(valid_refs)
            msg["content"] = [{"type": "text", "text": marker}]
            msg["_type"] = MT.OFFLOADED
            replaced += 1

        if replaced > 0:
            logger.info(
                f"[CONTEXT] Finalized {replaced} offload candidates "
                f"(turns <= {cutoff_turn})"
            )
        return replaced

    def _build_offload_marker(self, refs: list[str], fallback_chars: int = 0) -> str:
        """构建 OFFLOADED marker 文本（microcompact 和 finalize 共用）。"""
        parts = []
        for ref in refs:
            record = self._offload_registry.get(ref)
            evidence_lines = ""
            if record and record.evidence:
                evidence_lines = (
                    "\n\nEvidence:\n"
                    + "\n".join(f"- {e}" for e in record.evidence)
                )
            ref_chars = record.char_count if record else fallback_chars
            parts.append(
                f"{TAG_OFFLOADED}{ref}|{ref_chars}]"
                f"{evidence_lines}\n"
                f"Full content: read_result(\"{ref}\")"
            )
            if record:
                record.state = "offloaded"
        return "\n\n".join(parts)

    def update_offload_evidence(self, ref: str, evidence: list[str]) -> None:
        """Phase 3: 将解析到的 evidence 绑定到 offload record。"""
        record = self._offload_registry.get(ref)
        if record:
            record.evidence = evidence
            logger.debug(
                f"[Context] Bound {len(evidence)} evidence items to ref={ref}"
            )

    # ============================================================
    # Token 估算
    # ============================================================

    def estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数"""
        if self._token_estimator:
            try:
                return self._token_estimator(text)
            except Exception:
                pass
        return int(len(text) / self.config.chars_per_token)

    def estimate_context_tokens(
        self,
        system_prompt: str,
        message_history: list,
    ) -> int:
        """估算当前 context 总 token 数"""
        parts = [system_prompt]
        for msg in message_history:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)
        return self.estimate_tokens(" ".join(parts))

    def get_context_ratio(
        self,
        system_prompt: str,
        message_history: list,
        max_context_length: int,
    ) -> float:
        """计算当前 context 占比"""
        if max_context_length <= 0:
            return 0.0
        tokens = self.estimate_context_tokens(system_prompt, message_history)
        return tokens / max_context_length

    # ============================================================
    # Dedup
    # ============================================================

    def filter_duplicate_calls(
        self, calls: list[dict]
    ) -> tuple[list[dict], list[tuple[str, dict]]]:
        """执行前检查：跨轮次去重，渐进升级提示

        第 1 次命中：返回完整缓存 + 轻提示
        第 2+ 次命中：返回完整缓存 + 强制警告要求换策略
        """
        if not self.config.enable_dedup:
            return calls, []

        to_execute = []
        cached_results = []

        for call in calls:
            args_hash = self._compute_arguments_hash(
                call.get("tool_name", ""), call.get("arguments", {})
            )

            if args_hash in self._dedup_cache:
                cached = self._dedup_cache[args_hash]
                cached.hit_count += 1

                if cached.hit_count <= 1:
                    prefix = (
                        f"[This tool was already called with identical arguments in turn {cached.turn}. "
                        f"The cached result is returned below.]\n\n"
                    )
                else:
                    prefix = (
                        f"[WARNING: This is attempt #{cached.hit_count + 1} calling {cached.tool_name} "
                        f"with the same arguments. The result will NOT change. "
                        f"You MUST use a different approach: try different parameters, "
                        f"a different tool, or synthesize an answer from what you already have.]\n\n"
                    )

                cached_results.append(
                    (
                        call.get("id", ""),
                        {"type": "text", "text": prefix + cached.result_full},
                    )
                )
                logger.info(
                    f"[CONTEXT] Duplicate tool call skipped: {cached.tool_name} "
                    f"(args_hash={args_hash[:8]}..., original turn={cached.turn}, "
                    f"hit_count={cached.hit_count})"
                )
            else:
                to_execute.append(call)

        # Evict oldest entries (by turn) if cache exceeds limit
        if len(self._dedup_cache) > self._max_dedup_cache_size:
            excess = len(self._dedup_cache) - self._max_dedup_cache_size
            evict_keys = sorted(self._dedup_cache, key=lambda k: self._dedup_cache[k].turn)[:excess]
            for k in evict_keys:
                del self._dedup_cache[k]

        return to_execute, cached_results

    # ============================================================
    # 注册工具结果
    # ============================================================

    def register_tool_results(
        self,
        tool_calls: list[dict],
        tool_results_with_id: list[tuple[str, dict]],
        turn: int,
    ) -> None:
        """执行后注册：存入 dedup cache、记录 call registry"""
        call_map = {}
        for call in tool_calls:
            call_map[call.get("id", "")] = call

        for call_id, result_content in tool_results_with_id:
            call = call_map.get(call_id)
            if not call:
                continue

            tool_name = call.get("tool_name", "")
            arguments = call.get("arguments", {})
            result_text = self._extract_result_text(result_content)

            args_hash = self._compute_arguments_hash(tool_name, arguments)
            result_hash = hashlib.md5(result_text.encode("utf-8", errors="replace")).hexdigest()
            result_brief = result_text[:RESULT_BRIEF_LENGTH].strip()

            # 从工具结果 metadata 中提取 offload ref
            offload_ref = ""
            if isinstance(result_content, dict):
                offload_ref = result_content.get("_offload_ref", "")

            record = ToolCallRecord(
                tool_name=tool_name,
                arguments_hash=args_hash,
                arguments=arguments,
                turn=turn,
                result_hash=result_hash,
                result_brief=result_brief,
                result_full=result_text,
                result_chars=len(result_text),
                offload_ref=offload_ref,
            )

            if self.config.enable_dedup and args_hash not in self._dedup_cache:
                self._dedup_cache[args_hash] = record

            self._call_registry.append(record)

            self.source_registry.extract_and_register(
                tool_name=tool_name,
                arguments=arguments,
                result_text=result_text,
                turn=turn,
            )

        logger.info(f"[CONTEXT] Registered {len(tool_results_with_id)} tool results")

    # ============================================================
    # Level 1: Compact（委托给 ObservationMaskingStrategy）
    # ============================================================

    def apply_compact(
        self,
        message_history: list,
        current_turn: int,
        system_prompt: str = "",
        max_context_length: int = 0,
    ) -> int:
        """Level 1: token-aware 摘要替换（委托给管道中的 ObservationMaskingStrategy）

        Returns:
            被 compact 的消息数
        """
        if not self.config.enable_compact:
            return 0

        ctx = self._build_window_context(
            message_history,
            current_turn,
            system_prompt,
            max_context_length,
        )

        for strategy in self._pipeline.strategies:
            if strategy.strategy_type == "observation_masking":
                if strategy.should_trigger(ctx):
                    result = strategy.apply(message_history, ctx)
                    return result.messages_affected
        return 0

    # ============================================================
    # Level 2: Summarize（委托给 LLMSummarizeStrategy）
    # ============================================================

    async def apply_summarize(
        self,
        message_history: list,
        current_turn: int,
        system_prompt: str,
        max_context_length: int,
        llm_call_fn: Callable,
    ) -> bool:
        """Level 2: LLM 压缩旧历史（委托给管道中的 LLMSummarizeStrategy）

        Args:
            message_history: 消息历史（就地修改）
            current_turn: 当前轮次
            system_prompt: 系统提示词
            max_context_length: 最大 context 长度
            llm_call_fn: LLM 调用函数，签名: async (system_prompt, messages, purpose) -> str

        Returns:
            是否成功执行了压缩
        """
        ctx = self._build_window_context(
            message_history,
            current_turn,
            system_prompt,
            max_context_length,
        )
        return await self._pipeline.apply_summarize(message_history, ctx, llm_call_fn)

    # ============================================================
    # Level 3: Emergency（委托给管道紧急模式）
    # ============================================================

    def apply_emergency(
        self,
        message_history: list,
        current_turn: int,
        system_prompt: str = "",
        max_context_length: int = 0,
    ) -> int:
        """Level 3: 紧急裁剪（委托给管道的 apply_emergency）

        Returns:
            被处理的消息数
        """
        ctx = self._build_window_context(
            message_history,
            current_turn,
            system_prompt,
            max_context_length,
        )
        return self._pipeline.apply_emergency(message_history, ctx)

    # ============================================================
    # 统一入口：manage_context
    # ============================================================

    def _build_window_context(
        self,
        message_history: list,
        current_turn: int,
        system_prompt: str = "",
        max_context_length: int = 0,
    ) -> WindowContext:
        """构建 WindowContext 供策略管道使用"""
        ratio = (
            self.get_context_ratio(system_prompt, message_history, max_context_length)
            if max_context_length > 0
            else 0.0
        )
        return WindowContext(
            current_turn=current_turn,
            max_turns=0,  # 由 orchestrator 管理，此处不需要
            token_count=int(ratio * max_context_length) if max_context_length > 0 else 0,
            max_tokens=max_context_length,
            token_ratio=ratio,
            message_count=len(message_history),
            system_prompt=system_prompt,
            message_history=message_history,
            call_registry=self._call_registry,
            compacted_turns=self._compacted_turns,
            estimate_tokens_fn=self._token_estimator,
            session_memory=self._session_memory,
        )

    def manage_context(
        self,
        message_history: list,
        current_turn: int,
        system_prompt: str = "",
        max_context_length: int = 0,
    ) -> str:
        """统一的同步 context 管理入口（每轮结束时调用）

        委托给 WindowStrategyPipeline，根据 token 占比自动决定执行哪一级策略。
        Level 2 需要异步 LLM 调用，这里只返回 action 标记。

        Returns:
            执行的策略: "none" | "compact" | "observation_masking" | "need_summarize"
        """
        ctx = self._build_window_context(
            message_history,
            current_turn,
            system_prompt,
            max_context_length,
        )
        return self._pipeline.manage(message_history, ctx)

    # ============================================================
    # 工具方法
    # ============================================================

    def reset(self) -> None:
        """重置所有状态"""
        self._current_turn = 0
        self._dedup_cache.clear()
        self._call_registry.clear()
        self.source_registry.reset()
        self._compacted_turns.clear()
        self._pipeline.reset()
        self._offload_registry.clear()

    @property
    def dedup_cache_size(self) -> int:
        return len(self._dedup_cache)

    # ---- 内部方法 ----

    @staticmethod
    def _compute_arguments_hash(tool_name: str, arguments: dict) -> str:
        try:
            args_str = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            args_str = str(arguments)
        raw = f"{tool_name}:{args_str}"
        return hashlib.md5(raw.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _extract_result_text(result_content) -> str:
        """从 tool result content 中提取纯文本"""
        if isinstance(result_content, str):
            return result_content
        if isinstance(result_content, dict):
            if result_content.get("type") == "text":
                return result_content.get("text", "")
            inner = result_content.get("content", "")
            if isinstance(inner, str):
                return inner
            if isinstance(inner, list):
                return "\n".join(
                    item.get("text", "")
                    for item in inner
                    if isinstance(item, dict) and item.get("type") == "text"
                )
        if isinstance(result_content, list):
            parts = []
            for item in result_content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif item.get("type") == "tool_result":
                        inner = item.get("content", "")
                        if isinstance(inner, str):
                            parts.append(inner)
                        elif isinstance(inner, list):
                            for sub in inner:
                                if isinstance(sub, dict) and sub.get("type") == "text":
                                    parts.append(sub.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(result_content)
