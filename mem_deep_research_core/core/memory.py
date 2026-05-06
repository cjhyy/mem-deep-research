"""
记忆系统

- SessionMemory: 单次运行内的结构化记忆（关键发现、已用策略、来源）
- LongTermMemory: 跨 session 持久化记忆（文件/SQLite 存储）
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("mem_deep_research")


# ============================================================
# Session Memory (Short-term, within a single run)
# ============================================================


@dataclass
class SourceRecord:
    """引用来源记录（统一定义，供 memory + context_manager 共用）"""

    url: str = ""
    title: str = ""
    snippet: str = ""
    tool_name: str = ""
    turn: int = 0


@dataclass
class EvidenceItem:
    """从工具结果中提炼的高价值证据

    与 key_findings（从 LLM 回复文本提取）不同，evidence 直接从工具原始结果中
    提炼结构化事实，质量更高、信息密度更大。
    """

    tool_name: str  # 来源工具
    turn: int  # 产生的轮次
    summary: str  # 提炼后的证据摘要
    key_arg: str = ""  # 关键参数（如搜索词、URL）
    offload_ref: str = ""  # 如果原始结果被 offload，文件名引用
    source_url: str = ""  # 证据来源 URL
    confidence: str = ""  # 置信度: high / medium / low


@dataclass
class SessionMemory:
    """Session 内的结构化记忆

    在 MainLoopRunner 每轮结束时更新，context 压缩时保留。
    用于追踪关键发现、已尝试策略和引用来源。

    线程安全：所有读写操作通过 threading.Lock 保护，
    支持主 Agent 和并发子 Agent 安全访问。
    """

    key_findings: list[str] = field(default_factory=list)
    attempted_strategies: list[str] = field(default_factory=list)
    sources: list[SourceRecord] = field(default_factory=list)
    sub_agent_results: list[tuple[str, str]] = field(default_factory=list)
    evidence_items: list[EvidenceItem] = field(default_factory=list)

    # Limits
    max_findings: int = 20
    max_strategies: int = 15
    max_sources: int = 30
    max_evidence: int = 30

    # Threading lock (not included in repr/eq/hash)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def add_finding(self, finding: str):
        """添加关键发现（去重，线程安全）"""
        with self._lock:
            if finding and finding not in self.key_findings:
                self.key_findings.append(finding)
                if len(self.key_findings) > self.max_findings:
                    self.key_findings = self.key_findings[-self.max_findings :]

    def add_strategy(self, strategy: str):
        """记录已尝试的策略（避免重复尝试，线程安全）"""
        with self._lock:
            if strategy and strategy not in self.attempted_strategies:
                self.attempted_strategies.append(strategy)
                if len(self.attempted_strategies) > self.max_strategies:
                    self.attempted_strategies = self.attempted_strategies[-self.max_strategies :]

    def add_source(self, url: str = "", title: str = "", snippet: str = "", tool_name: str = ""):
        """添加引用来源（按 URL 去重，线程安全）"""
        with self._lock:
            if url and not any(s.url == url for s in self.sources):
                self.sources.append(
                    SourceRecord(url=url, title=title, snippet=snippet, tool_name=tool_name)
                )
                if len(self.sources) > self.max_sources:
                    self.sources = self.sources[-self.max_sources :]

    def add_evidence(self, item: EvidenceItem):
        """添加证据项（按 summary 去重，线程安全）"""
        with self._lock:
            if item.summary and not any(
                e.summary == item.summary for e in self.evidence_items
            ):
                self.evidence_items.append(item)
                if len(self.evidence_items) > self.max_evidence:
                    self.evidence_items = self.evidence_items[-self.max_evidence :]

    def add_sub_agent_result(self, agent_name: str, result: str):
        """记录子 Agent 结果（追加，线程安全）"""
        with self._lock:
            self.sub_agent_results.append((agent_name, result))

    def to_context_string(self) -> str:
        """生成可注入到消息历史的记忆摘要（线程安全快照）"""
        with self._lock:
            # 在锁内取快照，锁外构建字符串
            findings = list(self.key_findings)
            strategies = list(self.attempted_strategies)
            sources = list(self.sources)
            sub_results = list(self.sub_agent_results)
            evidence = list(self.evidence_items)

        sections = []

        if evidence:
            lines = []
            for i, e in enumerate(evidence, 1):
                header = f"[E{i}] Turn {e.turn} | {e.tool_name}"
                if e.key_arg:
                    header += f'("{e.key_arg}")'
                if e.confidence:
                    header += f" | {e.confidence}"
                if e.source_url:
                    lines.append(f"{header}\n  Source: {e.source_url}\n  {e.summary}")
                else:
                    lines.append(f"{header}\n  {e.summary}")
            sections.append("## Evidence Ledger\n" + "\n".join(lines))

        if findings:
            text = "\n".join(f"- {f}" for f in findings)
            sections.append(f"## Key Findings So Far\n{text}")

        if strategies:
            text = "\n".join(f"- {s}" for s in strategies)
            sections.append(f"## Attempted Strategies\n{text}")

        if sources:
            text = "\n".join(
                f"- [{s.title or s.url}]({s.url})" + (f" — {s.snippet[:100]}" if s.snippet else "")
                for s in sources
            )
            sections.append(f"## Sources Collected\n{text}")

        if sub_results:
            lines = []
            for i, (name, result) in enumerate(sub_results, 1):
                header = f"### {name} #{i}" if len(sub_results) > 1 else f"### {name}"
                body = f"{result[:200]}..." if len(result) > 200 else result
                lines.append(f"{header}\n{body}")
            sections.append(f"## Sub-Agent Results\n" + "\n".join(lines))

        if not sections:
            return ""

        return "[SESSION MEMORY]\n\n" + "\n\n".join(sections) + "\n"

    def to_evidence_string(self) -> str:
        """生成仅含证据的摘要（用于独立注入 message_history，受压缩保护）"""
        with self._lock:
            evidence = list(self.evidence_items)
        if not evidence:
            return ""
        lines = []
        for i, e in enumerate(evidence, 1):
            header = f"[E{i}] Turn {e.turn} | {e.tool_name}"
            if e.key_arg:
                header += f'("{e.key_arg}")'
            if e.confidence:
                header += f" | confidence: {e.confidence}"
            if e.source_url:
                header += f"\n  Source: {e.source_url}"
            if e.offload_ref:
                header += f"\n  Full content: read_result(\"{e.offload_ref}\")"
            lines.append(f"{header}\n  {e.summary}")
        return "[EVIDENCE LEDGER]\n\n" + "\n\n".join(lines) + "\n"

    def is_empty(self) -> bool:
        with self._lock:
            return (
                not self.key_findings
                and not self.attempted_strategies
                and not self.sources
                and not self.sub_agent_results
                and not self.evidence_items
            )

    def extract_from_tool_result(self, tool_name: str, tool_result: dict | str):
        """从工具结果中自动提取来源信息"""
        if isinstance(tool_result, dict):
            url = tool_result.get("url", "")
            title = tool_result.get("title", "")
            snippet = tool_result.get("snippet", "")
            if url:
                self.add_source(url=url, title=title, snippet=snippet, tool_name=tool_name)

    # ------------------------------------------------------------------
    # Snapshot contract (HITL / durable execution)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize session memory for RuntimeSnapshot (thread-safe)."""
        with self._lock:
            return {
                "key_findings": list(self.key_findings),
                "attempted_strategies": list(self.attempted_strategies),
                "sources": [
                    {
                        "url": s.url,
                        "title": s.title,
                        "snippet": s.snippet,
                        "tool_name": s.tool_name,
                        "turn": s.turn,
                    }
                    for s in self.sources
                ],
                "sub_agent_results": [list(pair) for pair in self.sub_agent_results],
                "evidence_items": [
                    {
                        "tool_name": e.tool_name,
                        "turn": e.turn,
                        "summary": e.summary,
                        "key_arg": e.key_arg,
                        "offload_ref": e.offload_ref,
                        "source_url": e.source_url,
                        "confidence": e.confidence,
                    }
                    for e in self.evidence_items
                ],
            }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMemory":
        """Inverse of :meth:`to_dict`."""
        inst = cls()
        with inst._lock:
            inst.key_findings = list(data.get("key_findings", []))
            inst.attempted_strategies = list(data.get("attempted_strategies", []))
            inst.sources = [SourceRecord(**s) for s in data.get("sources", [])]
            inst.sub_agent_results = [
                tuple(pair) for pair in data.get("sub_agent_results", [])
            ]
            inst.evidence_items = [
                EvidenceItem(**e) for e in data.get("evidence_items", [])
            ]
        return inst


# ============================================================
# Long-term Memory (Cross-session, persistent)
# ============================================================


@dataclass
class MemoryEntry:
    """单条记忆条目"""

    key: str
    value: str
    metadata: dict = field(default_factory=dict)
    timestamp: float = 0.0
    access_count: int = 0

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(**data)


class LongTermMemory:
    """跨 session 持久化记忆

    支持 file 存储（JSON），通过 recall/store API 操作。
    通过 hook 集成：on_agent_start 时 recall，on_agent_end 时 store。

    Usage:
        memory = LongTermMemory(storage_path="memory/")

        # Store
        memory.store("user_prefers_chinese", "用户偏好中文回答", {"source": "conversation"})

        # Recall
        entries = memory.recall("用户偏好")

        # In hooks:
        @hooks.register("on_agent_start", priority=5)
        def inject_memory(ctx, original_fn):
            entries = memory.recall(ctx.query, top_k=5)
            if entries:
                ctx.extra["memory_context"] = "\\n".join(e.value for e in entries)
            return original_fn(ctx)
    """

    def __init__(self, storage_path: str = "memory/", max_entries: int = 1000):
        self.storage_path = Path(storage_path)
        self.max_entries = max_entries
        self._entries: list[MemoryEntry] = []
        self._loaded = False
        self._lock = threading.RLock()  # RLock: store()/recall() 内部调用 _ensure_loaded() 需要可重入

    def _ensure_loaded(self):
        """Lazy load from disk (thread-safe via _lock)."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:  # double-check after acquiring lock
                return
            memory_file = self.storage_path / "memory.json"
            if memory_file.exists():
                try:
                    with open(memory_file, encoding="utf-8") as f:
                        data = json.load(f)
                    self._entries = [MemoryEntry.from_dict(e) for e in data]
                    logger.debug(
                        f"[LongTermMemory] Loaded {len(self._entries)} entries from {memory_file}"
                    )
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"[LongTermMemory] Failed to load: {e}")
                    self._entries = []
                except Exception as e:
                    logger.error(f"[LongTermMemory] Unexpected error loading: {e}")
                    self._entries = []
            self._loaded = True  # Set AFTER loading completes

    def _save(self):
        """Save to disk."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        memory_file = self.storage_path / "memory.json"
        try:
            data = [e.to_dict() for e in self._entries]
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[LongTermMemory] Failed to save: {e}")

    def store(self, key: str, value: str, metadata: dict | None = None):
        """存储一条记忆（按 key 去重，存在则更新）"""
        with self._lock:
            self._ensure_loaded()

            # Check for existing entry with same key
            for entry in self._entries:
                if entry.key == key:
                    entry.value = value
                    entry.metadata = metadata or {}
                    entry.timestamp = time.time()
                    self._save()
                    return

            # New entry
            entry = MemoryEntry(
                key=key,
                value=value,
                metadata=metadata or {},
                timestamp=time.time(),
            )
            self._entries.append(entry)

            # Enforce max_entries (remove oldest)
            if len(self._entries) > self.max_entries:
                self._entries.sort(key=lambda e: e.timestamp)
                self._entries = self._entries[-self.max_entries :]

            self._save()
            logger.debug(f"[LongTermMemory] Stored: {key}")

    def recall(
        self, query: str = "", top_k: int = 5, metadata_filter: dict | None = None
    ) -> list[MemoryEntry]:
        """召回相关记忆

        简单关键词匹配（可通过 hook 替换为向量检索）。

        Args:
            query: 查询关键词
            top_k: 返回最多 N 条
            metadata_filter: 按 metadata 过滤

        Returns:
            匹配的记忆条目列表
        """
        with self._lock:
            self._ensure_loaded()
            candidates = list(self._entries)  # Copy under lock to avoid race

        # Metadata filter
        if metadata_filter:
            candidates = [
                e
                for e in candidates
                if all(e.metadata.get(k) == v for k, v in metadata_filter.items())
            ]

        # Keyword matching (simple scoring)
        if query:
            query_lower = query.lower()
            scored = []
            for entry in candidates:
                score = 0
                text = f"{entry.key} {entry.value}".lower()
                for word in query_lower.split():
                    if word in text:
                        score += 1
                if score > 0:
                    scored.append((score, entry))
            scored.sort(key=lambda x: (-x[0], -x[1].timestamp))
            results = [entry for _, entry in scored[:top_k]]
        else:
            # No query: return most recent
            candidates_sorted = sorted(candidates, key=lambda e: -e.timestamp)
            results = candidates_sorted[:top_k]

        # Update access count and persist (under lock to avoid race condition)
        if results:
            with self._lock:
                for entry in results:
                    entry.access_count += 1
                self._save()

        return results

    def forget(self, key: str) -> bool:
        """删除指定 key 的记忆"""
        with self._lock:
            self._ensure_loaded()
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.key != key]
            if len(self._entries) < before:
                self._save()
                return True
            return False

    def clear(self):
        """清空所有记忆"""
        with self._lock:
            self._entries = []
            self._loaded = True
            self._save()

    def list_all(self) -> list[MemoryEntry]:
        """列出所有记忆"""
        with self._lock:
            self._ensure_loaded()
            return list(self._entries)

    def deduplicate(self):
        """去重：合并 key 相同的条目，保留最新"""
        with self._lock:
            self._ensure_loaded()
            seen = {}
            for entry in self._entries:
                if entry.key not in seen or entry.timestamp > seen[entry.key].timestamp:
                    seen[entry.key] = entry
            self._entries = list(seen.values())
            self._save()
            logger.debug(f"[LongTermMemory] Dedup: {len(self._entries)} entries remaining")
