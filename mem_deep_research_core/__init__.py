"""
mem_deep_research - AI Agent 端到端任务完成框架

主要模块:
- core: Pipeline 和 Orchestrator 核心执行逻辑
- llm: LLM Provider 抽象层
- tool: MCP 工具管理
- utils: 工具函数
- exceptions: 自定义异常
- config_schema: 配置验证 Schema
"""

from mem_deep_research_core.exceptions import (
    ConfigNotFoundError,
    ConfigurationError,
    ConfigValidationError,
    ContextLimitError,
    GuardrailError,
    JSONParseError,
    LLMAPIError,
    LLMError,
    LLMProviderNotFoundError,
    LLMRateLimitError,
    LLMResponseParseError,
    MaxTurnsExceededError,
    MemDeepResearchError,
    MissingEnvVarError,
    ParseError,
    PipelineError,
    ServerNotFoundError,
    TaskCancelledError,
    ToolCallParseError,
    ToolConnectionError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolTimeoutError,
)

__version__ = "1.0.1"
__author__ = "maki maki"

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Base exceptions
    "MemDeepResearchError",
    # Tool exceptions
    "ToolError",
    "ToolNotFoundError",
    "ServerNotFoundError",
    "ToolExecutionError",
    "ToolConnectionError",
    "ToolTimeoutError",
    # LLM exceptions
    "LLMError",
    "LLMProviderNotFoundError",
    "LLMAPIError",
    "LLMRateLimitError",
    "LLMResponseParseError",
    "ContextLimitError",
    # Config exceptions
    "ConfigurationError",
    "ConfigNotFoundError",
    "ConfigValidationError",
    "MissingEnvVarError",
    # Pipeline exceptions
    "PipelineError",
    "MaxTurnsExceededError",
    "TaskCancelledError",
    "GuardrailError",
    # Parse exceptions
    "ParseError",
    "JSONParseError",
    "ToolCallParseError",
]
