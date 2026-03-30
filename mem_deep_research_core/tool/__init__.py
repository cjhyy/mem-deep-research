"""
Tool 模块 - MCP 工具管理

主要组件:
- manager: 工具管理器，处理工具发现、执行
- mcp_servers: MCP 服务器实现
"""


def __getattr__(name):
    """Lazy import to avoid circular dependency when running MCP servers as subprocesses."""
    if name == "ToolManager":
        from mem_deep_research_core.tool.manager import ToolManager

        return ToolManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ToolManager",
]
