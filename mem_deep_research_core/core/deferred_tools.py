"""
Deferred Tools — 延迟加载工具 schema

当工具数量超过阈值时，只向 LLM 暴露工具名称 + 描述（不含完整 schema），
并注入内置 `tool_search` 工具让 LLM 按需获取完整 schema。

参考 Claude Code 的 Deferred Tool / ToolSearch 机制。

Usage:
    manager = DeferredToolManager(threshold=20)
    tool_defs, was_deferred = manager.apply(full_tool_definitions)

    # LLM 调用 tool_search 后：
    resolved = manager.resolve_tool_search(query)
    # 下一轮把 resolved tools 的完整 schema 放回 tool_definitions
"""

import logging
from difflib import SequenceMatcher

from mem_deep_research_core.core.constants import BUILTIN_TOOL_SEARCH

logger = logging.getLogger("mem_deep_research")


def _get_tool_search_definition() -> dict:
    """Built-in tool_search tool definition in MCP server format."""
    return {
        "name": "builtin-tool-search",
        "tools": [
            {
                "name": BUILTIN_TOOL_SEARCH,
                "description": (
                    "Search for available tools by keyword. Use this when you need to find "
                    "the right tool for a task. Returns the full schema of matching tools "
                    "so you can call them in subsequent turns."
                ),
                "schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query — tool name, keyword, or description fragment",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of tools to return (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            }
        ],
    }


class DeferredToolManager:
    """管理工具的延迟加载。

    当总工具数 > threshold 时自动启用 deferred 模式：
    - 将所有 MCP 工具 schema 替换为 name + description 摘要
    - 注入 tool_search 内置工具
    - LLM 通过 tool_search 按需获取完整 schema
    - 已发现的工具在后续轮次自动包含完整 schema

    Args:
        threshold: 工具总数超过此值时启用 deferred 模式（0=禁用）
    """

    def __init__(self, threshold: int = 20):
        self.threshold = threshold

        # 完整的工具定义存储: {(server_name, tool_name): tool_dict}
        self._full_registry: dict[tuple[str, str], dict] = {}
        # 工具元数据: {(server_name, tool_name): {"name": ..., "description": ...}}
        self._tool_meta: dict[tuple[str, str], dict] = {}
        # 已被 LLM 发现（通过 tool_search）的工具集合
        self._discovered: set[tuple[str, str]] = set()
        # server name 映射
        self._tool_to_server: dict[str, str] = {}

        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def apply(
        self, tool_definitions: list[dict],
    ) -> tuple[list[dict], bool]:
        """对工具定义应用 deferred 策略。

        Args:
            tool_definitions: 原始工具定义列表（MCP server format）

        Returns:
            (处理后的工具定义, 是否启用了 deferred 模式)
        """
        if self.threshold <= 0:
            return tool_definitions, False

        # 计算总工具数
        total_tools = sum(
            len(server.get("tools", []))
            for server in tool_definitions
            if isinstance(server, dict)
        )

        if total_tools <= self.threshold:
            return tool_definitions, False

        # 启用 deferred 模式
        self._active = True
        self._full_registry.clear()
        self._tool_meta.clear()
        self._tool_to_server.clear()

        deferred_defs = []
        deferred_listing_lines = []

        for server in tool_definitions:
            if not isinstance(server, dict):
                deferred_defs.append(server)
                continue

            server_name = server.get("name", "")
            tools = server.get("tools", [])

            # 内置工具（builtin-*）保留完整 schema
            if server_name.startswith("builtin-"):
                deferred_defs.append(server)
                continue

            full_tools = []
            for tool in tools:
                tool_name = tool.get("name", "")
                key = (server_name, tool_name)

                # 存储完整定义
                self._full_registry[key] = tool
                self._tool_meta[key] = {
                    "name": tool_name,
                    "description": tool.get("description", "")[:200],
                    "server_name": server_name,
                }
                self._tool_to_server[tool_name] = server_name

                # 已发现的工具保留完整 schema
                if key in self._discovered:
                    full_tools.append(tool)
                else:
                    # 记录到 deferred listing
                    desc = tool.get("description", "")[:100]
                    deferred_listing_lines.append(f"- {tool_name}: {desc}")

            if full_tools:
                deferred_defs.append({"name": server_name, "tools": full_tools})

        # 注入 tool_search 内置工具
        deferred_defs.append(_get_tool_search_definition())

        # 将 deferred listing 注入到 tool_search 的描述中
        if deferred_listing_lines:
            listing = "\n".join(deferred_listing_lines)
            # 更新 tool_search description 以包含可用工具列表
            for server in deferred_defs:
                if server.get("name") == "builtin-tool-search":
                    for tool in server.get("tools", []):
                        if tool.get("name") == BUILTIN_TOOL_SEARCH:
                            tool["description"] = (
                                "Search for available tools by keyword to get their full schema. "
                                "You must call this before using any deferred tool.\n\n"
                                f"Available deferred tools ({len(deferred_listing_lines)} total):\n"
                                f"{listing}"
                            )

        logger.info(
            f"[DeferredTools] Activated: {total_tools} total tools, "
            f"{len(deferred_listing_lines)} deferred, "
            f"{len(self._discovered)} previously discovered"
        )

        return deferred_defs, True

    def resolve_tool_search(
        self, query: str, max_results: int = 5,
    ) -> list[dict]:
        """处理 tool_search 调用，返回匹配工具的完整 schema。

        Args:
            query: 搜索关键词
            max_results: 最大返回数

        Returns:
            匹配工具的完整定义列表
        """
        query_lower = query.lower()
        scored: list[tuple[float, tuple[str, str], dict]] = []

        for key, meta in self._tool_meta.items():
            name = meta["name"].lower()
            desc = meta.get("description", "").lower()

            # 精确名称匹配
            if query_lower == name:
                score = 1.0
            # 名称包含
            elif query_lower in name:
                score = 0.9
            # 描述包含
            elif query_lower in desc:
                score = 0.7
            # 模糊匹配
            else:
                score = max(
                    SequenceMatcher(None, query_lower, name).ratio(),
                    SequenceMatcher(None, query_lower, desc[:100]).ratio() * 0.8,
                )

            if score > 0.3:
                scored.append((score, key, self._full_registry[key]))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = scored[:max_results]

        # 标记为已发现
        for _, key, _ in results:
            self._discovered.add(key)

        tool_results = []
        for score, key, full_def in results:
            server_name = key[0]
            tool_results.append({
                "server_name": server_name,
                "tool_name": full_def["name"],
                "description": full_def.get("description", ""),
                "schema": full_def.get("schema", {}),
            })

        logger.info(
            f"[DeferredTools] tool_search('{query}'): "
            f"found {len(tool_results)} tools, "
            f"total discovered: {len(self._discovered)}"
        )

        return tool_results

    def get_discovered_tool_names(self) -> set[str]:
        """获取已发现的工具名称集合"""
        return {name for _, name in self._discovered}

    def reset(self):
        """重置状态（新 session 开始时调用）"""
        self._full_registry.clear()
        self._tool_meta.clear()
        self._discovered.clear()
        self._tool_to_server.clear()
        self._active = False
