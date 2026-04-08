"""
Tests for MT message type system.

Covers:
- make_msg constructs correct structure with _type
- _is_protected_message: _type fast path + keyword fallback
- PROTECTED_MESSAGE_TYPES completeness
- Non-protected types are compressible
"""

import pytest

from mem_deep_research_core.core.constants import (
    MT,
    PROTECTED_MESSAGE_TYPES,
    SYSTEM_MESSAGE_KEYWORDS,
    make_msg,
    make_tool_result_msg,
    make_tool_result_msg_native,
)
from mem_deep_research_core.core.window_strategy import _is_protected_message


# ============================================================
# make_msg
# ============================================================


class TestMakeMsg:
    def test_basic_structure(self):
        msg = make_msg("user", "hello")
        assert msg["role"] == "user"
        assert msg["content"] == [{"type": "text", "text": "hello"}]
        assert "_type" not in msg

    def test_with_type(self):
        msg = make_msg("user", "hello", _type=MT.SESSION_MEMORY)
        assert msg["_type"] == MT.SESSION_MEMORY

    def test_extra_fields(self):
        msg = make_msg("user", "hello", _type=MT.PLAN, _meta=True, custom="val")
        assert msg["_meta"] is True
        assert msg["custom"] == "val"

    def test_none_type_omitted(self):
        msg = make_msg("assistant", "ok", _type=None)
        assert "_type" not in msg


# ============================================================
# _is_protected_message — _type fast path
# ============================================================


class TestIsProtectedByType:
    @pytest.mark.parametrize("msg_type", list(PROTECTED_MESSAGE_TYPES))
    def test_all_protected_types(self, msg_type):
        """Every type in PROTECTED_MESSAGE_TYPES should be protected."""
        msg = make_msg("user", "x", _type=msg_type)
        assert _is_protected_message(msg) is True

    @pytest.mark.parametrize("msg_type", [
        MT.LOOP_HINT,
        MT.TRUNCATION_RECOVERY,
        MT.INLINE_SKILL,
        MT.ASSISTANT,
        MT.TOOL_RESULT,
        MT.USER_INPUT,
        MT.SUMMARY_PROMPT,
        MT.ROUTING,
        MT.TASK_PLANNING,
    ])
    def test_non_protected_types(self, msg_type):
        """Types NOT in PROTECTED_MESSAGE_TYPES should be compressible."""
        msg = make_msg("user", "x", _type=msg_type)
        assert _is_protected_message(msg) is False


# ============================================================
# _is_protected_message — keyword fallback (no _type)
# ============================================================


class TestIsProtectedByKeyword:
    @pytest.mark.parametrize("keyword", SYSTEM_MESSAGE_KEYWORDS)
    def test_keyword_fallback_string_content(self, keyword):
        """Messages without _type fall back to keyword matching."""
        msg = {"role": "user", "content": f"{keyword} some details here"}
        assert _is_protected_message(msg) is True

    @pytest.mark.parametrize("keyword", SYSTEM_MESSAGE_KEYWORDS)
    def test_keyword_fallback_list_content(self, keyword):
        """Keyword fallback works with list-of-dict content format."""
        msg = {"role": "user", "content": [{"type": "text", "text": f"{keyword} data"}]}
        assert _is_protected_message(msg) is True

    def test_no_type_no_keyword(self):
        """Plain message without _type or keyword is not protected."""
        msg = {"role": "user", "content": "just a normal message"}
        assert _is_protected_message(msg) is False


# ============================================================
# PROTECTED_MESSAGE_TYPES completeness
# ============================================================


class TestProtectedTypesCompleteness:
    def test_session_memory_protected(self):
        assert MT.SESSION_MEMORY in PROTECTED_MESSAGE_TYPES

    def test_long_term_memory_protected(self):
        assert MT.LONG_TERM_MEMORY in PROTECTED_MESSAGE_TYPES

    def test_task_progress_protected(self):
        assert MT.TASK_PROGRESS in PROTECTED_MESSAGE_TYPES

    def test_plan_protected(self):
        assert MT.PLAN in PROTECTED_MESSAGE_TYPES

    def test_reflection_protected(self):
        assert MT.REFLECTION in PROTECTED_MESSAGE_TYPES

    def test_context_summary_protected(self):
        assert MT.CONTEXT_SUMMARY in PROTECTED_MESSAGE_TYPES

    def test_offloaded_protected(self):
        assert MT.OFFLOADED in PROTECTED_MESSAGE_TYPES

    def test_type_overrides_keyword(self):
        """_type takes precedence over keyword content."""
        # Non-protected type with keyword content → NOT protected (type wins)
        msg = make_msg("user", "[SESSION MEMORY] data", _type=MT.LOOP_HINT)
        assert _is_protected_message(msg) is False

        # Protected type with non-keyword content → protected (type wins)
        msg = make_msg("user", "plain text", _type=MT.SESSION_MEMORY)
        assert _is_protected_message(msg) is True


# ============================================================
# make_tool_result_msg helpers
# ============================================================


class TestMakeToolResultMsg:
    def test_has_correct_type(self):
        msg = make_tool_result_msg("search result text")
        assert msg["role"] == "user"
        assert msg["_type"] == MT.TOOL_RESULT
        assert msg["content"][0]["text"] == "search result text"

    def test_extra_fields(self):
        msg = make_tool_result_msg("result", custom_field="x")
        assert msg["custom_field"] == "x"
        assert msg["_type"] == MT.TOOL_RESULT

    def test_native_format(self):
        msg = make_tool_result_msg_native("call_123", "result text")
        assert msg["role"] == "tool"
        assert msg["_type"] == MT.TOOL_RESULT
        assert msg["tool_call_id"] == "call_123"
        assert msg["content"] == "result text"


# ============================================================
# Provider update_message_history tags _type
# ============================================================


class TestProviderToolResultTagging:
    """Verify that each provider's update_message_history() produces
    messages with _type=MT.TOOL_RESULT."""

    def _make_tool_call_info(self):
        """Simulate tool call results from ToolExecutor."""
        return [
            ("call_1", {"type": "text", "text": "Search result: found 3 items"}),
            ("call_2", {"type": "text", "text": "Scrape result: 500 chars"}),
        ]

    def test_claude_anthropic_tags_type(self):
        from mem_deep_research_core.llm.providers.claude_anthropic_client import (
            ClaudeAnthropicClient,
        )

        client = ClaudeAnthropicClient.__new__(ClaudeAnthropicClient)
        history = []
        client.update_message_history(history, self._make_tool_call_info())

        assert len(history) == 1
        assert history[0]["_type"] == MT.TOOL_RESULT
        assert history[0]["role"] == "user"

    def test_openai_compatible_tags_type(self):
        from mem_deep_research_core.llm.providers.openai_compatible_client import (
            OpenAICompatibleClient,
        )

        client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
        history = []
        client.update_message_history(history, self._make_tool_call_info())

        assert len(history) == 1
        assert history[0]["_type"] == MT.TOOL_RESULT
        assert history[0]["role"] == "user"

    def test_gpt_openai_tags_type(self):
        from mem_deep_research_core.llm.providers.gpt_openai_client import (
            GPTOpenAIClient,
        )

        client = GPTOpenAIClient.__new__(GPTOpenAIClient)
        history = []
        client.update_message_history(history, self._make_tool_call_info())

        assert len(history) == 2  # GPT appends one per call
        for msg in history:
            assert msg["_type"] == MT.TOOL_RESULT
            assert msg["role"] == "tool"


# ============================================================
# keep_tool_result only removes TOOL_RESULT, not user/memory/plan
# ============================================================


class TestKeepToolResultSelectivity:
    """_remove_tool_result_from_messages should only omit TOOL_RESULT,
    leaving USER_INPUT, SESSION_MEMORY, PLAN etc. intact."""

    def _make_client(self):
        from unittest.mock import MagicMock
        from mem_deep_research_core.llm.provider_client_base import LLMProviderClientBase

        client = MagicMock(spec=LLMProviderClientBase)
        client._remove_tool_result_from_messages = (
            LLMProviderClientBase._remove_tool_result_from_messages.__get__(client)
        )
        return client

    def test_only_removes_tool_result_typed(self):
        client = self._make_client()
        messages = [
            make_msg("user", "original query"),  # no _type — first user msg
            make_msg("assistant", "thinking..."),
            make_tool_result_msg("tool result 1"),  # _type=TOOL_RESULT
            make_msg("assistant", "more thinking"),
            make_msg("user", "user follow-up", _type=MT.USER_INPUT),  # must survive
            make_msg("assistant", "response"),
            make_tool_result_msg("tool result 2"),  # _type=TOOL_RESULT
            make_msg("user", "memory data", _type=MT.SESSION_MEMORY),  # must survive
            make_msg("assistant", "final"),
            make_tool_result_msg("tool result 3"),  # _type=TOOL_RESULT
        ]

        result = client._remove_tool_result_from_messages(messages, keep_tool_result=1)

        # Only keep the last 1 tool result — #3 survives, #1 and #2 get omitted
        assert result[2]["content"] == "Tool result is omitted to save tokens."  # TR #1
        assert result[6]["content"] == "Tool result is omitted to save tokens."  # TR #2
        assert result[9]["content"][0]["text"] == "tool result 3"  # TR #3 kept

        # Non-TOOL_RESULT messages must be intact
        assert result[0]["content"][0]["text"] == "original query"
        assert result[4]["content"][0]["text"] == "user follow-up"
        assert result[7]["content"][0]["text"] == "memory data"

    def test_keep_all_with_minus_one(self):
        client = self._make_client()
        messages = [
            make_msg("user", "query"),
            make_tool_result_msg("result 1"),
            make_tool_result_msg("result 2"),
        ]

        result = client._remove_tool_result_from_messages(messages, keep_tool_result=-1)
        # -1 means keep all
        assert result[1]["content"][0]["text"] == "result 1"
        assert result[2]["content"][0]["text"] == "result 2"


# ============================================================
# Microcompact conservatively preserves legacy (no _type) messages
# ============================================================


class TestMicrocompactLegacyPreservation:
    """Messages without _type should be conservatively preserved by microcompact."""

    def test_legacy_messages_not_cleaned(self):
        from mem_deep_research_core.core.context_manager import ContextManager, ContextManagerConfig

        cm = ContextManager(ContextManagerConfig())
        history = [
            {"role": "user", "content": [{"type": "text", "text": "initial task"}]},
            {"role": "assistant", "content": "response 1"},
            # Legacy message: no _type, long content — should NOT be cleaned
            {"role": "user", "content": [{"type": "text", "text": "A" * 1000}]},
            {"role": "assistant", "content": "response 2"},
            {"role": "user", "content": [{"type": "text", "text": "B" * 1000}]},
        ]

        cleaned = cm.microcompact(history, current_turn=5, keep_recent=1)
        assert cleaned == 0  # No _type=TOOL_RESULT → nothing cleaned
        assert history[2]["content"][0]["text"] == "A" * 1000
        assert history[4]["content"][0]["text"] == "B" * 1000

    def test_typed_tool_result_cleaned_but_legacy_preserved(self):
        from mem_deep_research_core.core.context_manager import ContextManager, ContextManagerConfig

        cm = ContextManager(ContextManagerConfig())
        history = [
            {"role": "user", "content": [{"type": "text", "text": "initial task"}]},
            {"role": "assistant", "content": "response 1"},
            # Typed TOOL_RESULT — should be cleaned
            make_tool_result_msg("C" * 1000),
            {"role": "assistant", "content": "response 2"},
            # Legacy (no _type) — should survive
            {"role": "user", "content": [{"type": "text", "text": "D" * 1000}]},
        ]

        cleaned = cm.microcompact(history, current_turn=5, keep_recent=1)
        assert cleaned == 1  # Only the typed TOOL_RESULT was cleaned
        assert "[microcompact]" in history[2]["content"][0]["text"]
        assert history[4]["content"][0]["text"] == "D" * 1000
