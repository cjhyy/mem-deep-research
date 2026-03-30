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
from collections.abc import Callable
from dataclasses import dataclass

from mem_deep_research_core.core.constants import RESULT_BRIEF_LENGTH
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
        self._sources: list[SourceRecord] = []
        self._seen_urls: set = set()

    def extract_and_register(
        self,
        tool_name: str,
        arguments: dict,
        result_text: str,
        turn: int,
    ) -> list[SourceRecord]:
        """从工具调用中提取并注册来源"""
        new_sources = []

        # 1. 从 arguments 提取
        rule = self.SOURCE_RULES.get(tool_name)
        if rule:
            url = arguments.get(rule.get("url", ""), "") if "url" in rule else ""
            title = arguments.get(rule.get("title", ""), "") if "title" in rule else ""
            if url and url not in self._seen_urls:
                source = SourceRecord(url=url, title=title, tool_name=tool_name, turn=turn)
                new_sources.append(source)
                self._seen_urls.add(url)

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
                self._seen_urls.add(url)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        self._sources.extend(new_sources)
        return new_sources

    def get_all_sources(self) -> list[SourceRecord]:
        return list(self._sources)

    def get_citation_summary(self) -> str:
        if not self._sources:
            return ""
        lines = ["## Sources"]
        for i, source in enumerate(self._sources, 1):
            if source.title:
                lines.append(f"{i}. [{source.title}]({source.url})")
            else:
                lines.append(f"{i}. {source.url}")
        return "\n".join(lines)

    def reset(self):
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

    # Level 2: LLM 压缩
    summarize_at_ratio: float = 0.8  # token 占比超过此值时触发 LLM 压缩

    # Dedup cache
    max_dedup_cache_size: int = 200  # Maximum entries in dedup cache

    # Token 估算
    chars_per_token: float = 3.5  # 无 tiktoken 时的 fallback 估算


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
    ):
        self.config = config or ContextManagerConfig()
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

        # Evict oldest entries if cache exceeds limit
        if len(self._dedup_cache) > self._max_dedup_cache_size:
            excess = len(self._dedup_cache) - self._max_dedup_cache_size
            for k in list(self._dedup_cache.keys())[:excess]:
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

            record = ToolCallRecord(
                tool_name=tool_name,
                arguments_hash=args_hash,
                arguments=arguments,
                turn=turn,
                result_hash=result_hash,
                result_brief=result_brief,
                result_full=result_text,
                result_chars=len(result_text),
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

    @property
    def dedup_cache_size(self) -> int:
        return len(self._dedup_cache)

    # ---- 内部方法 ----

    @staticmethod
    def _compute_arguments_hash(tool_name: str, arguments: dict) -> str:
        args_str = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
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
