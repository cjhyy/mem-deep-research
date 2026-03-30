"""异常层次结构单元测试"""

from mem_deep_research_core.exceptions import (
    ConfigNotFoundError,
    ConfigurationError,
    ConfigValidationError,
    ContextLimitError,
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


class TestExceptionHierarchy:
    """异常继承关系测试"""

    def test_tool_errors_inherit_from_tool_error(self):
        assert issubclass(ToolNotFoundError, ToolError)
        assert issubclass(ServerNotFoundError, ToolError)
        assert issubclass(ToolExecutionError, ToolError)
        assert issubclass(ToolConnectionError, ToolError)
        assert issubclass(ToolTimeoutError, ToolError)

    def test_tool_error_inherits_from_base(self):
        assert issubclass(ToolError, MemDeepResearchError)

    def test_llm_errors_inherit_from_llm_error(self):
        assert issubclass(LLMProviderNotFoundError, LLMError)
        assert issubclass(LLMAPIError, LLMError)
        assert issubclass(LLMRateLimitError, LLMError)
        assert issubclass(LLMResponseParseError, LLMError)
        assert issubclass(ContextLimitError, LLMError)

    def test_config_errors_inherit_from_config_error(self):
        assert issubclass(ConfigNotFoundError, ConfigurationError)
        assert issubclass(ConfigValidationError, ConfigurationError)
        assert issubclass(MissingEnvVarError, ConfigurationError)

    def test_pipeline_errors_inherit_from_pipeline_error(self):
        assert issubclass(MaxTurnsExceededError, PipelineError)
        assert issubclass(TaskCancelledError, PipelineError)

    def test_parse_errors_inherit_from_parse_error(self):
        assert issubclass(JSONParseError, ParseError)
        assert issubclass(ToolCallParseError, ParseError)

    def test_all_inherit_from_base(self):
        """所有异常最终继承 MemDeepResearchError"""
        for exc_class in [
            ToolError,
            LLMError,
            ConfigurationError,
            PipelineError,
            ParseError,
        ]:
            assert issubclass(exc_class, MemDeepResearchError)


class TestExceptionAttributes:
    """异常属性测试"""

    def test_tool_not_found(self):
        e = ToolNotFoundError("search", server_name="google")
        assert e.tool_name == "search"
        assert e.server_name == "google"
        assert "search" in str(e)
        assert "google" in str(e)

    def test_server_not_found(self):
        e = ServerNotFoundError("myserver", available_servers=["a", "b"])
        assert e.server_name == "myserver"
        assert e.available_servers == ["a", "b"]

    def test_tool_execution_error(self):
        cause = ValueError("bad input")
        e = ToolExecutionError("search", "google", cause=cause)
        assert e.tool_name == "search"
        assert e.__cause__ is cause

    def test_tool_timeout(self):
        e = ToolTimeoutError("search", 30.0)
        assert e.tool_name == "search"
        assert e.timeout_seconds == 30.0

    def test_llm_api_error(self):
        e = LLMAPIError("openai", status_code=429, response="rate limited")
        assert e.provider == "openai"
        assert e.status_code == 429

    def test_context_limit_error(self):
        e = ContextLimitError()
        assert "Context limit exceeded" in str(e)

    def test_max_turns_exceeded(self):
        e = MaxTurnsExceededError(20, task_id="t1")
        assert e.max_turns == 20
        assert e.task_id == "t1"

    def test_json_parse_error_truncates(self):
        long_content = "x" * 1000
        e = JSONParseError("parse failed", raw_content=long_content)
        assert len(e.details["raw_content"]) == 500

    def test_base_error_details(self):
        e = MemDeepResearchError("test error", details={"key": "value"})
        assert e.message == "test error"
        assert e.details == {"key": "value"}
