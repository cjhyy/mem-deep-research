"""
文件状态缓存

LRU 缓存工具读取的文件内容，parent/subagent 间共享以避免重复读取。
参考 Claude Code 的 readFileState LRU 缓存。

Usage:
    cache = FileStateCache(max_size=100)
    cache.put("path/to/file", "file content...")
    content = cache.get("path/to/file")  # hit
    child_cache = cache.clone()  # sub-agent gets a snapshot copy
"""

import logging
import threading
from collections import OrderedDict

logger = logging.getLogger("mem_deep_research")


class FileStateCache:
    """LRU 文件状态缓存

    Args:
        max_size: 最大缓存条目数（默认 100）
    """

    def __init__(self, max_size: int = 100):
        self._max_size = max_size
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._hits: int = 0
        self._misses: int = 0
        self._lock = threading.Lock()

    def get(self, path: str) -> str | None:
        """获取缓存的文件内容（LRU 更新）"""
        with self._lock:
            if path in self._cache:
                self._cache.move_to_end(path)
                self._hits += 1
                return self._cache[path]
            self._misses += 1
            return None

    def put(self, path: str, content: str) -> None:
        """缓存文件内容"""
        with self._lock:
            if path in self._cache:
                self._cache.move_to_end(path)
                self._cache[path] = content
            else:
                self._cache[path] = content
                if len(self._cache) > self._max_size:
                    evicted = self._cache.popitem(last=False)
                    logger.debug(f"[FileStateCache] Evicted: {evicted[0]}")

    def invalidate(self, path: str) -> None:
        """使指定路径的缓存失效（文件被写入后调用）"""
        with self._lock:
            self._cache.pop(path, None)

    def clone(self) -> "FileStateCache":
        """克隆缓存（用于 sub-agent 隔离，共享读取结果）"""
        new = FileStateCache(max_size=self._max_size)
        with self._lock:
            new._cache = OrderedDict(self._cache)
        return new

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 2) if total > 0 else 0.0,
            }

    def reset(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
