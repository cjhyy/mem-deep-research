"""
代码执行 MCP 工具

提供 Python 和 Shell 命令执行能力。
默认不启用，需在 tool_config 中显式引用 tool-code-executor。

安全机制：
- subprocess 执行，非 eval/exec（隔离主进程）
- 硬超时（默认 30s）
- 输出大小限制（最大 100KB）
- 可选命令白名单（通过环境变量 CODE_EXECUTOR_SHELL_WHITELIST）
"""

import os
import subprocess
import tempfile
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("code-executor-server")

_DEFAULT_TIMEOUT = 30  # seconds
_MAX_OUTPUT_SIZE = 100 * 1024  # 100KB
_WORK_DIR = os.environ.get("CODE_EXECUTOR_WORK_DIR", "")


def _get_work_dir() -> Path:
    """获取代码执行的工作目录"""
    if _WORK_DIR:
        p = Path(_WORK_DIR).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    return Path.cwd()


def _truncate_output(output: str, max_size: int = _MAX_OUTPUT_SIZE) -> str:
    """截断过长的输出"""
    if len(output) > max_size:
        return output[:max_size] + f"\n... [TRUNCATED, total {len(output)} chars]"
    return output


@mcp.tool()
async def execute_python(code: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Execute Python code and return the output.

    The code runs in a subprocess, isolated from the main process.
    Both stdout and stderr are captured.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 30)

    Returns:
        Combined stdout and stderr output.
    """
    if timeout <= 0 or timeout > 300:
        timeout = min(max(timeout, 1), 300)

    work_dir = _get_work_dir()
    # 写入临时文件执行（避免命令行转义问题）
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=str(work_dir),
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(code)
            script_path = f.name

        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work_dir),
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output = ""
        if stdout:
            output += stdout
        if stderr:
            if output:
                output += "\n--- STDERR ---\n"
            output += stderr
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"

        return _truncate_output(output) if output else "(no output)"

    except subprocess.TimeoutExpired:
        return f"[ERROR] Execution timed out after {timeout}s"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"
    finally:
        if script_path:
            try:
                os.unlink(script_path)
            except OSError:
                pass


@mcp.tool()
async def execute_shell(command: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Execute a shell command and return the output.

    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds (default: 30)

    Returns:
        Combined stdout and stderr output.
    """
    if timeout <= 0 or timeout > 300:
        timeout = min(max(timeout, 1), 300)

    # 可选命令白名单
    whitelist_raw = os.environ.get("CODE_EXECUTOR_SHELL_WHITELIST", "")
    if whitelist_raw:
        allowed = {cmd.strip() for cmd in whitelist_raw.split(",") if cmd.strip()}
        # 提取命令名（第一个 token）
        cmd_name = command.strip().split()[0] if command.strip() else ""
        base_cmd = Path(cmd_name).name  # 去掉路径前缀
        if base_cmd not in allowed:
            return (
                f"[ERROR] Command '{base_cmd}' is not in the whitelist. "
                f"Allowed commands: {sorted(allowed)}"
            )

    work_dir = _get_work_dir()
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work_dir),
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output = ""
        if stdout:
            output += stdout
        if stderr:
            if output:
                output += "\n--- STDERR ---\n"
            output += stderr
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"

        return _truncate_output(output) if output else "(no output)"

    except subprocess.TimeoutExpired:
        return f"[ERROR] Command timed out after {timeout}s"
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
