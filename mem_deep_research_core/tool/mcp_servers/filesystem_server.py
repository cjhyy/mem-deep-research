"""
文件系统 MCP 工具

提供安全的文件读写能力，通过 allowed_dirs 白名单限制访问范围。
默认不启用，需在 tool_config 中显式引用 tool-filesystem。

安全机制：
- 路径遍历检查（resolve 后必须在白名单目录内）
- 文件大小限制（读取最大 1MB）
- 写入目录必须已存在（不自动创建深层目录）
"""

import os
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("filesystem-server")

# 白名单目录（通过环境变量配置，逗号分隔）
_ALLOWED_DIRS: list[Path] = []
_MAX_READ_SIZE = 1024 * 1024  # 1MB
_MAX_WRITE_SIZE = 5 * 1024 * 1024  # 5MB


def _init_allowed_dirs():
    """从环境变量初始化白名单目录"""
    global _ALLOWED_DIRS
    raw = os.environ.get("FILESYSTEM_ALLOWED_DIRS", "")
    if raw:
        _ALLOWED_DIRS = [Path(d.strip()).resolve() for d in raw.split(",") if d.strip()]
    # 如果没配置白名单，默认允许当前工作目录
    if not _ALLOWED_DIRS:
        _ALLOWED_DIRS = [Path.cwd().resolve()]


def _check_path(path_str: str) -> Path:
    """校验路径安全性，返回解析后的绝对路径"""
    _init_allowed_dirs()
    resolved = Path(path_str).resolve()
    # 路径安全检查：必须在白名单目录下（使用路径层级比较，非字符串前缀）
    if not any(resolved == d or d in resolved.parents for d in _ALLOWED_DIRS):
        raise PermissionError(
            f"Access denied: '{path_str}' is outside allowed directories. "
            f"Allowed: {[str(d) for d in _ALLOWED_DIRS]}"
        )
    return resolved


@mcp.tool()
async def read_file(path: str, encoding: str = "utf-8") -> str:
    """Read a file and return its content.

    Args:
        path: File path (absolute or relative to working directory)
        encoding: File encoding (default: utf-8)

    Returns:
        File content as string. Large files are truncated to 1MB.
    """
    resolved = _check_path(path)
    if not resolved.exists():
        return f"[ERROR] File not found: {path}"
    if not resolved.is_file():
        return f"[ERROR] Not a file: {path}"
    size = resolved.stat().st_size
    if size > _MAX_READ_SIZE:
        content = resolved.read_text(encoding=encoding, errors="replace")[:_MAX_READ_SIZE]
        return f"[TRUNCATED to {_MAX_READ_SIZE} bytes, original size: {size}]\n{content}"
    return resolved.read_text(encoding=encoding, errors="replace")


@mcp.tool()
async def write_file(path: str, content: str, encoding: str = "utf-8") -> str:
    """Write content to a file. Creates the file if it doesn't exist.

    Args:
        path: File path (absolute or relative to working directory)
        content: Content to write
        encoding: File encoding (default: utf-8)

    Returns:
        Success message or error.
    """
    if len(content.encode(encoding, errors="replace")) > _MAX_WRITE_SIZE:
        return f"[ERROR] Content too large (max {_MAX_WRITE_SIZE} bytes)"
    resolved = _check_path(path)
    # 确保父目录存在（只创建一层）
    resolved.parent.mkdir(parents=False, exist_ok=True)
    resolved.write_text(content, encoding=encoding)
    return f"Successfully wrote {len(content)} chars to {resolved}"


@mcp.tool()
async def list_directory(path: str = ".") -> str:
    """List files and directories in a path.

    Args:
        path: Directory path (default: current directory)

    Returns:
        Formatted directory listing.
    """
    resolved = _check_path(path)
    if not resolved.exists():
        return f"[ERROR] Directory not found: {path}"
    if not resolved.is_dir():
        return f"[ERROR] Not a directory: {path}"
    entries = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines = []
    for entry in entries[:200]:  # Limit to 200 entries
        prefix = "📁 " if entry.is_dir() else "📄 "
        size = ""
        if entry.is_file():
            size = f" ({entry.stat().st_size:,} bytes)"
        lines.append(f"{prefix}{entry.name}{size}")
    total = len(list(resolved.iterdir()))
    if total > 200:
        lines.append(f"... and {total - 200} more entries")
    return "\n".join(lines) if lines else "(empty directory)"


@mcp.tool()
async def file_info(path: str) -> str:
    """Get file or directory metadata.

    Args:
        path: File or directory path

    Returns:
        File metadata (exists, type, size, modified time).
    """
    resolved = _check_path(path)
    if not resolved.exists():
        return f"Path does not exist: {path}"
    stat = resolved.stat()
    file_type = "directory" if resolved.is_dir() else "file"
    import datetime

    modified = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
    return (
        f"Path: {resolved}\nType: {file_type}\nSize: {stat.st_size:,} bytes\nModified: {modified}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
