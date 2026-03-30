"""
自定义异常类

定义框架中使用的所有自定义异常，避免使用裸 Exception 捕获。
"""

from typing import Any


class MemDeepResearchError(Exception):
    """框架基础异常类"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# ============================================================================
# Tool 相关异常
# ============================================================================


class ToolError(MemDeepResearchError):
    """工具执行相关的基础异常"""

    pass


class ToolNotFoundError(ToolError):
    """工具未找到"""

    def __init__(self, tool_name: str, server_name: str | None = None):
        message = f"Tool '{tool_name}' not found"
        if server_name:
            message += f" in server '{server_name}'"
        super().__init__(message, {"tool_name": tool_name, "server_name": server_name})
        self.tool_name = tool_name
        self.server_name = server_name


class ServerNotFoundError(ToolError):
    """MCP 服务器未找到"""

    def __init__(self, server_name: str, available_servers: list[str] | None = None):
        message = f"Server '{server_name}' not found"
        if available_servers:
            message += f". Available servers: {', '.join(available_servers)}"
        super().__init__(
            message, {"server_name": server_name, "available_servers": available_servers}
        )
        self.server_name = server_name
        self.available_servers = available_servers or []


class ToolExecutionError(ToolError):
    """工具执行失败"""

    def __init__(self, tool_name: str, server_name: str, cause: Exception | None = None):
        message = f"Tool '{tool_name}' execution failed on server '{server_name}'"
        if cause:
            message += f": {str(cause)}"
        super().__init__(message, {"tool_name": tool_name, "server_name": server_name})
        self.tool_name = tool_name
        self.server_name = server_name
        self.__cause__ = cause


class ToolConnectionError(ToolError):
    """无法连接到工具服务器"""

    def __init__(self, server_name: str, cause: Exception | None = None):
        message = f"Cannot connect to server '{server_name}'"
        if cause:
            message += f": {str(cause)}"
        super().__init__(message, {"server_name": server_name})
        self.server_name = server_name
        self.__cause__ = cause


class ToolTimeoutError(ToolError):
    """工具执行超时"""

    def __init__(self, tool_name: str, timeout_seconds: float):
        message = f"Tool '{tool_name}' execution timed out after {timeout_seconds}s"
        super().__init__(message, {"tool_name": tool_name, "timeout_seconds": timeout_seconds})
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds


# ============================================================================
# LLM 相关异常
# ============================================================================


class LLMError(MemDeepResearchError):
    """LLM 相关的基础异常"""

    pass


class LLMProviderNotFoundError(LLMError):
    """LLM Provider 未找到"""

    def __init__(self, provider_class: str):
        message = f"LLM provider class '{provider_class}' not found"
        super().__init__(message, {"provider_class": provider_class})
        self.provider_class = provider_class


class LLMAPIError(LLMError):
    """LLM API 调用错误"""

    def __init__(self, provider: str, status_code: int | None = None, response: str | None = None):
        message = f"LLM API error from provider '{provider}'"
        if status_code:
            message += f" (status: {status_code})"
        super().__init__(
            message, {"provider": provider, "status_code": status_code, "response": response}
        )
        self.provider = provider
        self.status_code = status_code
        self.response = response


class LLMRateLimitError(LLMError):
    """LLM 请求频率限制"""

    def __init__(self, provider: str, retry_after: float | None = None):
        message = f"Rate limit exceeded for provider '{provider}'"
        if retry_after:
            message += f". Retry after {retry_after}s"
        super().__init__(message, {"provider": provider, "retry_after": retry_after})
        self.provider = provider
        self.retry_after = retry_after


class LLMResponseParseError(LLMError):
    """LLM 响应解析错误"""

    def __init__(self, message: str, raw_response: str | None = None):
        super().__init__(message, {"raw_response": raw_response})
        self.raw_response = raw_response


class ContextLimitError(LLMError):
    """LLM 上下文长度超限"""

    def __init__(self, message: str = "Context limit exceeded"):
        super().__init__(message)


# ============================================================================
# 配置相关异常
# ============================================================================


class ConfigurationError(MemDeepResearchError):
    """配置相关的基础异常"""

    pass


class ConfigNotFoundError(ConfigurationError):
    """配置文件或配置项未找到"""

    def __init__(self, config_name: str, config_path: str | None = None):
        message = f"Configuration '{config_name}' not found"
        if config_path:
            message += f" at path '{config_path}'"
        super().__init__(message, {"config_name": config_name, "config_path": config_path})
        self.config_name = config_name
        self.config_path = config_path


class ConfigValidationError(ConfigurationError):
    """配置验证失败"""

    def __init__(self, message: str, field: str | None = None, value: Any = None):
        super().__init__(message, {"field": field, "value": value})
        self.field = field
        self.value = value


class MissingEnvVarError(ConfigurationError):
    """缺少必要的环境变量"""

    def __init__(self, var_name: str, description: str | None = None):
        message = f"Required environment variable '{var_name}' is not set"
        if description:
            message += f". {description}"
        super().__init__(message, {"var_name": var_name})
        self.var_name = var_name


# ============================================================================
# Pipeline/Orchestrator 相关异常
# ============================================================================


class PipelineError(MemDeepResearchError):
    """Pipeline 执行相关的基础异常"""

    pass


class MaxTurnsExceededError(PipelineError):
    """超过最大对话轮次"""

    def __init__(self, max_turns: int, task_id: str | None = None):
        message = f"Maximum turns ({max_turns}) exceeded"
        if task_id:
            message += f" for task '{task_id}'"
        super().__init__(message, {"max_turns": max_turns, "task_id": task_id})
        self.max_turns = max_turns
        self.task_id = task_id


class GuardrailError(PipelineError):
    """Guardrail validation failed — blocks LLM call or rejects output."""

    def __init__(self, guardrail_name: str, message: str):
        self.guardrail_name = guardrail_name
        super().__init__(f"[Guardrail:{guardrail_name}] {message}")


class TaskCancelledError(PipelineError):
    """任务被取消"""

    def __init__(self, task_id: str, reason: str | None = None):
        message = f"Task '{task_id}' was cancelled"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"task_id": task_id, "reason": reason})
        self.task_id = task_id
        self.reason = reason


# ============================================================================
# 解析相关异常
# ============================================================================


class ParseError(MemDeepResearchError):
    """解析相关的基础异常"""

    pass


class JSONParseError(ParseError):
    """JSON 解析错误"""

    def __init__(self, message: str, raw_content: str | None = None):
        super().__init__(message, {"raw_content": raw_content[:500] if raw_content else None})
        self.raw_content = raw_content


class ToolCallParseError(ParseError):
    """工具调用解析错误"""

    def __init__(self, message: str, raw_content: str | None = None):
        super().__init__(message, {"raw_content": raw_content[:500] if raw_content else None})
        self.raw_content = raw_content
