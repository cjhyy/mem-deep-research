"""Tests for LLMRouter, focusing on adaptive_classify (zero-cost mode classification)."""

import pytest

from mem_deep_research_core.core.llm_router import LLMRouter, RouteResult
from mem_deep_research_core.core.constants import (
    EXECUTION_MODE_QUICK,
    EXECUTION_MODE_STANDARD,
    EXECUTION_MODE_DEEP,
)


class TestAdaptiveClassify:
    """Test LLMRouter.adaptive_classify — zero-cost mode decision from first-turn behavior."""

    def test_direct_answer_no_tools_is_quick(self):
        """LLM answered directly without any tool calls → quick."""
        result = LLMRouter.adaptive_classify(
            should_break=True,
            tool_calls=None,
            has_spawn_agent=False,
        )
        assert result.mode == EXECUTION_MODE_QUICK
        assert result.source == "adaptive"

    def test_direct_answer_empty_tools_is_quick(self):
        """LLM answered with empty tool call lists → quick."""
        result = LLMRouter.adaptive_classify(
            should_break=True,
            tool_calls=[[], []],
            has_spawn_agent=False,
        )
        assert result.mode == EXECUTION_MODE_QUICK

    def test_spawn_agent_is_deep(self):
        """Any spawn_agent call → deep."""
        tool_calls = [
            [{"tool_name": "spawn_agent", "id": "1"}],
            [{"tool_name": "spawn_agent"}],
        ]
        result = LLMRouter.adaptive_classify(
            should_break=False,
            tool_calls=tool_calls,
            has_spawn_agent=True,
        )
        assert result.mode == EXECUTION_MODE_DEEP
        assert result.reasoning_effort == "high"

    def test_many_tools_is_deep(self):
        """3+ tool calls in first turn → deep."""
        calls = [{"tool_name": f"tool_{i}", "id": str(i)} for i in range(4)]
        tool_calls = [calls, calls]
        result = LLMRouter.adaptive_classify(
            should_break=False,
            tool_calls=tool_calls,
            has_spawn_agent=False,
        )
        assert result.mode == EXECUTION_MODE_DEEP

    def test_threshold_boundary_deep(self):
        """Exactly ADAPTIVE_DEEP_TOOL_THRESHOLD tools → deep."""
        threshold = LLMRouter.ADAPTIVE_DEEP_TOOL_THRESHOLD
        calls = [{"tool_name": f"tool_{i}", "id": str(i)} for i in range(threshold)]
        tool_calls = [calls, calls]
        result = LLMRouter.adaptive_classify(
            should_break=False,
            tool_calls=tool_calls,
            has_spawn_agent=False,
        )
        assert result.mode == EXECUTION_MODE_DEEP

    def test_threshold_boundary_standard(self):
        """Below threshold → standard."""
        threshold = LLMRouter.ADAPTIVE_DEEP_TOOL_THRESHOLD
        calls = [{"tool_name": f"tool_{i}", "id": str(i)} for i in range(threshold - 1)]
        tool_calls = [calls, calls]
        result = LLMRouter.adaptive_classify(
            should_break=False,
            tool_calls=tool_calls,
            has_spawn_agent=False,
        )
        assert result.mode == EXECUTION_MODE_STANDARD

    def test_moderate_tools_is_standard(self):
        """1-2 tool calls → standard."""
        calls = [{"tool_name": "search", "id": "1"}]
        tool_calls = [calls, calls]
        result = LLMRouter.adaptive_classify(
            should_break=False,
            tool_calls=tool_calls,
            has_spawn_agent=False,
        )
        assert result.mode == EXECUTION_MODE_STANDARD
        assert result.reasoning_effort == "medium"

    def test_should_break_with_tools_is_not_quick(self):
        """should_break=True but has tools → not quick (standard)."""
        calls = [{"tool_name": "search", "id": "1"}]
        tool_calls = [calls, calls]
        result = LLMRouter.adaptive_classify(
            should_break=True,
            tool_calls=tool_calls,
            has_spawn_agent=False,
        )
        # Has tools, so not quick even if should_break
        assert result.mode == EXECUTION_MODE_STANDARD

    def test_result_is_route_result(self):
        """Return type is RouteResult."""
        result = LLMRouter.adaptive_classify(
            should_break=True,
            tool_calls=None,
        )
        assert isinstance(result, RouteResult)
        assert result.source == "adaptive"
        assert "reason" in result.metadata


class TestAdaptiveClassifySimpleAuto:
    """Test adaptive_classify with allow_deep=False (simple_auto mode)."""

    def test_many_tools_clamped_to_standard(self):
        """With allow_deep=False, many tool calls → standard (not deep)."""
        calls = [{"tool_name": f"tool_{i}", "id": str(i)} for i in range(5)]
        tool_calls = [calls, calls]
        result = LLMRouter.adaptive_classify(
            should_break=False,
            tool_calls=tool_calls,
            has_spawn_agent=False,
            allow_deep=False,
        )
        assert result.mode == EXECUTION_MODE_STANDARD
        assert "clamped_from_deep" in result.metadata.get("reason", "")

    def test_spawn_agent_clamped_to_standard(self):
        """With allow_deep=False, spawn_agent → standard (not deep)."""
        calls = [{"tool_name": "spawn_agent", "id": "1"}]
        tool_calls = [calls, calls]
        result = LLMRouter.adaptive_classify(
            should_break=False,
            tool_calls=tool_calls,
            has_spawn_agent=True,
            allow_deep=False,
        )
        assert result.mode == EXECUTION_MODE_STANDARD

    def test_quick_still_works(self):
        """allow_deep=False doesn't affect quick classification."""
        result = LLMRouter.adaptive_classify(
            should_break=True,
            tool_calls=None,
            allow_deep=False,
        )
        assert result.mode == EXECUTION_MODE_QUICK

    def test_moderate_tools_still_standard(self):
        """Normal standard classification unaffected by allow_deep."""
        calls = [{"tool_name": "search", "id": "1"}]
        tool_calls = [calls, calls]
        result = LLMRouter.adaptive_classify(
            should_break=False,
            tool_calls=tool_calls,
            allow_deep=False,
        )
        assert result.mode == EXECUTION_MODE_STANDARD


class TestStructuralRoute:
    """Test deterministic structural routing (existing behavior, regression guard)."""

    def test_sub_agents_forces_deep(self):
        router = LLMRouter()
        result = router._structural_route(tool_count=5, has_sub_agents=True)
        assert result is not None
        assert result.mode == EXECUTION_MODE_DEEP

    def test_no_tools_forces_quick(self):
        router = LLMRouter()
        result = router._structural_route(tool_count=0, has_sub_agents=False)
        assert result is not None
        assert result.mode == EXECUTION_MODE_QUICK

    def test_normal_tools_returns_none(self):
        """Normal case: tools available, no sub-agents → None (defer to next layer)."""
        router = LLMRouter()
        result = router._structural_route(tool_count=5, has_sub_agents=False)
        assert result is None
