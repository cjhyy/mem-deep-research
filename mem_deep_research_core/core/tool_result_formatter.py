"""
工具结果格式化模块

处理工具调用结果的格式化和摘要生成。
通过钩子系统支持项目自定义格式化逻辑。
"""

import json
import logging
from typing import Any

from mem_deep_research_core.core.hooks import HookContext, HookRegistry

logger = logging.getLogger("mem_deep_research")


def _default_thinking_generate(ctx: HookContext) -> str:
    """默认的 thinking 描述生成逻辑"""
    tool_name = ctx.tool_name or ""
    arguments = ctx.arguments or {}
    tool_name_lower = tool_name.lower()

    # 搜索类工具
    if "google" in tool_name_lower or "search" in tool_name_lower:
        query = arguments.get("query", "")
        if query:
            return f"🔍 Searching: {query[:50]}{'...' if len(query) > 50 else ''}"
        return "🔍 Searching the web..."

    # 网页抓取类工具
    if "scrape" in tool_name_lower or "fetch" in tool_name_lower or "browse" in tool_name_lower:
        url = arguments.get("url", "")
        if url:
            return f"📄 Fetching: {url[:50]}{'...' if len(url) > 50 else ''}"
        return "📄 Fetching webpage content..."

    # 语义搜索工具
    if "semantic" in tool_name_lower:
        query = arguments.get("query", "")
        if query:
            return f"🔎 Semantic search: {query[:50]}{'...' if len(query) > 50 else ''}"
        return "🔎 Searching knowledge base..."

    # 读取内容工具
    if "get_content" in tool_name_lower or "read" in tool_name_lower:
        return "📖 Reading content..."

    # 推理工具
    if "reasoning" in tool_name_lower or "think" in tool_name_lower:
        return "🧠 Deep reasoning..."

    # 默认
    return f"🔧 Using {tool_name}..."


def _default_tool_result_format(ctx: HookContext) -> str:
    """默认的工具结果格式化逻辑"""
    tool_name = ctx.tool_name or ""
    tool_result = ctx.tool_result or {}
    duration_ms = ctx.duration_ms or 0

    result = tool_result.get("result", "")
    error = tool_result.get("error")

    if error:
        return f"❌ Failed: {str(error)[:200]} ({duration_ms}ms)"

    if not result:
        return f"✅ Completed ({duration_ms}ms)"

    # 转换结果为字符串
    if isinstance(result, dict):
        result_str = json.dumps(result, ensure_ascii=False)
    else:
        result_str = str(result)

    # 搜索结果 - 显示找到的项目数
    if "search" in tool_name.lower():
        if isinstance(result, list):
            return f"✅ Found {len(result)} results ({duration_ms}ms)"
        count = _extract_result_count(result_str)
        if count is not None:
            return f"✅ Found {count} results ({duration_ms}ms)"

    # 网页抓取 - 显示内容长度
    if "scrape" in tool_name.lower() or "fetch" in tool_name.lower():
        content_len = len(result_str)
        return f"✅ Fetched {content_len} chars ({duration_ms}ms)"

    # 通用结果
    if len(result_str) > 300:
        return f"✅ Completed ({duration_ms}ms)\nResult: {result_str[:200]}..."
    else:
        return f"✅ Completed ({duration_ms}ms)\nResult: {result_str}"


def _extract_result_count(result_str: str) -> int | None:
    """尝试从结果中提取数量"""
    try:
        data = json.loads(result_str)
        for key in ["total", "count", "total_count", "num_results", "length"]:
            if key in data and isinstance(data[key], int):
                return data[key]
        if isinstance(data, list):
            return len(data)
        for key in ["items", "results", "data", "records"]:
            if key in data and isinstance(data[key], list):
                return len(data[key])
    except (json.JSONDecodeError, TypeError):
        pass
    return None


class ToolResultFormatter:
    """
    工具调用结果格式化器

    通过钩子系统支持自定义格式化逻辑。
    """

    def __init__(self, context: dict[str, Any] | None = None, *, hooks: HookRegistry):
        """
        初始化格式化器

        Args:
            context: 用户上下文
            hooks: HookRegistry 实例（必传）
        """
        self.context = context or {}
        self._hooks = hooks

    def get_tool_thinking_description(self, tool_name: str, arguments: dict = None) -> str:
        """
        根据工具名称和参数生成 THINKING 节点描述。

        通过 on_thinking_generate 钩子支持自定义。

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            用于 THINKING 节点的描述文本
        """
        ctx = HookContext(
            hook_name="on_thinking_generate",
            tool_name=tool_name,
            arguments=arguments or {},
            context=self.context,
        )
        return self._hooks.call("on_thinking_generate", ctx)

    def extract_tool_query_info(self, tool_name: str, arguments: dict) -> str | None:
        """
        从工具参数中提取查询信息用于显示。

        Returns:
            格式化的查询信息，如果无关则返回 None
        """
        # 搜索工具 - 显示搜索查询
        if "search" in tool_name.lower() or "google" in tool_name.lower():
            query = arguments.get("query") or arguments.get("q") or arguments.get("search_query")
            if query:
                return f"`{query}`"

        # 网页抓取工具 - 显示 URL
        if "scrape" in tool_name.lower() or "fetch" in tool_name.lower():
            url = arguments.get("url") or arguments.get("urls")
            if url:
                if isinstance(url, list):
                    return f"URLs: {', '.join(url[:3])}{'...' if len(url) > 3 else ''}"
                return f"URL: `{url[:80]}{'...' if len(str(url)) > 80 else ''}`"

        return None

    def summarize_tool_result(
        self, tool_name: str, tool_result: dict, duration_ms: int, arguments: dict = None
    ) -> str | None:
        """
        生成工具结果的简短摘要。

        通过 on_tool_result_format 钩子支持自定义。

        Args:
            tool_name: 工具名称
            tool_result: 工具执行结果
            duration_ms: 执行耗时（毫秒）
            arguments: 工具调用参数

        Returns:
            格式化的结果摘要
        """
        ctx = HookContext(
            hook_name="on_tool_result_format",
            tool_name=tool_name,
            tool_result=tool_result,
            duration_ms=duration_ms,
            arguments=arguments or {},
            context=self.context,
        )
        return self._hooks.call("on_tool_result_format", ctx)
