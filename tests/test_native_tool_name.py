"""Native tool name 拼接/解析 单元测试

验证 convert_tool_definition_to_tool_call 生成的 flat name
能被 parse_llm_response_for_tool_calls 正确还原为 (server_name, tool_name)。
"""

from unittest.mock import MagicMock

import pytest

from mem_deep_research_core.llm.provider_client_base import LLMProviderClientBase
from mem_deep_research_core.utils.parsing_utils import (
    _resolve_native_tool_name,
    parse_llm_response_for_tool_calls,
)

# ---------- fixtures ----------

TOOLS_DEFINITIONS = [
    {
        "name": "builtin-spawn-agent",
        "tools": [
            {
                "name": "spawn_agent",
                "description": "Spawn a sub-agent",
                "schema": {"type": "object", "properties": {}},
            }
        ],
    },
    {
        "name": "tool-searching-serper",
        "tools": [
            {
                "name": "web_search",
                "description": "Search the web",
                "schema": {"type": "object", "properties": {}},
            },
            {
                "name": "news-search",
                "description": "Search news",
                "schema": {"type": "object", "properties": {}},
            },
        ],
    },
    {
        "name": "simple",
        "tools": [
            {
                "name": "calc",
                "description": "Calculator",
                "schema": {"type": "object", "properties": {}},
            }
        ],
    },
]


# ---------- _resolve_native_tool_name ----------


class TestResolveNativeToolName:
    def test_lookup_hit(self):
        name_map = {"builtin-spawn-agent--spawn_agent": ("builtin-spawn-agent", "spawn_agent")}
        assert _resolve_native_tool_name("builtin-spawn-agent--spawn_agent", name_map) == (
            "builtin-spawn-agent",
            "spawn_agent",
        )

    def test_fallback_split(self):
        """name_map 为 None 时用 -- 分隔符 fallback"""
        assert _resolve_native_tool_name("server--tool", None) == ("server", "tool")

    def test_fallback_split_with_hyphens(self):
        """server name 含 - 时 fallback 也能工作（因为用 -- 分割）"""
        assert _resolve_native_tool_name("my-server--my-tool", None) == ("my-server", "my-tool")

    def test_unresolvable(self):
        """既没有 name_map 也没有 -- 分隔符时返回 None"""
        assert _resolve_native_tool_name("no-separator-here", None) is None

    def test_empty(self):
        assert _resolve_native_tool_name("", None) is None

    def test_name_map_priority_over_split(self):
        """name_map 命中时应优先于 fallback split"""
        name_map = {"a--b--c": ("a--b", "c")}
        assert _resolve_native_tool_name("a--b--c", name_map) == ("a--b", "c")
        # fallback split 会得到 ("a", "b--c")，但 name_map 优先
        assert _resolve_native_tool_name("a--b--c", None) == ("a", "b--c")


# ---------- convert_tool_definition_to_tool_call ----------


class TestConvertToolDefinition:
    @pytest.mark.asyncio
    async def test_returns_tool_list_and_name_map(self):
        tool_list, name_map = await LLMProviderClientBase.convert_tool_definition_to_tool_call(
            TOOLS_DEFINITIONS
        )
        assert len(tool_list) == 4
        assert len(name_map) == 4

    @pytest.mark.asyncio
    async def test_name_map_contains_all_tools(self):
        _, name_map = await LLMProviderClientBase.convert_tool_definition_to_tool_call(
            TOOLS_DEFINITIONS
        )
        assert name_map["builtin-spawn-agent--spawn_agent"] == (
            "builtin-spawn-agent",
            "spawn_agent",
        )
        assert name_map["tool-searching-serper--web_search"] == (
            "tool-searching-serper",
            "web_search",
        )
        assert name_map["tool-searching-serper--news-search"] == (
            "tool-searching-serper",
            "news-search",
        )
        assert name_map["simple--calc"] == ("simple", "calc")

    @pytest.mark.asyncio
    async def test_flat_names_in_tool_list(self):
        tool_list, _ = await LLMProviderClientBase.convert_tool_definition_to_tool_call(
            TOOLS_DEFINITIONS
        )
        names = [t["function"]["name"] for t in tool_list]
        assert "builtin-spawn-agent--spawn_agent" in names
        assert "tool-searching-serper--news-search" in names


# ---------- parse_llm_response_for_tool_calls (list path — OpenAI Completion API) ----------


def _make_tool_call_obj(name, arguments='{"key": "value"}', call_id="call_1"):
    """创建模拟的 OpenAI tool_call 对象"""
    func = MagicMock()
    func.name = name
    func.arguments = arguments
    tc = MagicMock()
    tc.function = func
    tc.id = call_id
    return tc


class TestParseNativeToolCallsList:
    def test_with_name_map(self):
        name_map = {
            "builtin-spawn-agent--spawn_agent": ("builtin-spawn-agent", "spawn_agent"),
        }
        tc = _make_tool_call_obj("builtin-spawn-agent--spawn_agent")
        tool_calls, bad = parse_llm_response_for_tool_calls([tc], name_map=name_map)
        assert len(tool_calls) == 1
        assert tool_calls[0]["server_name"] == "builtin-spawn-agent"
        assert tool_calls[0]["tool_name"] == "spawn_agent"

    def test_without_name_map_fallback(self):
        tc = _make_tool_call_obj("my-server--my_tool")
        tool_calls, bad = parse_llm_response_for_tool_calls([tc], name_map=None)
        assert len(tool_calls) == 1
        assert tool_calls[0]["server_name"] == "my-server"
        assert tool_calls[0]["tool_name"] == "my_tool"

    def test_invalid_name_skipped(self):
        tc = _make_tool_call_obj("no-double-dash-here")
        tool_calls, bad = parse_llm_response_for_tool_calls([tc], name_map=None)
        assert len(tool_calls) == 0

    def test_hyphenated_server_name_with_name_map(self):
        """核心 bug 场景：server name 含多个 - 时，name_map 保证正确解析"""
        name_map = {
            "builtin-spawn-agent--spawn_agent": ("builtin-spawn-agent", "spawn_agent"),
        }
        tc = _make_tool_call_obj("builtin-spawn-agent--spawn_agent")
        tool_calls, _ = parse_llm_response_for_tool_calls([tc], name_map=name_map)
        assert tool_calls[0]["server_name"] == "builtin-spawn-agent"
        assert tool_calls[0]["tool_name"] == "spawn_agent"


# ---------- parse_llm_response_for_tool_calls (dict path — OpenAI Response API) ----------


class TestParseNativeToolCallsDict:
    def test_with_name_map(self):
        name_map = {
            "tool-searching-serper--web_search": ("tool-searching-serper", "web_search"),
        }
        response = {
            "output": [
                {
                    "type": "function_call",
                    "name": "tool-searching-serper--web_search",
                    "arguments": '{"query": "test"}',
                    "call_id": "call_1",
                }
            ]
        }
        tool_calls, bad = parse_llm_response_for_tool_calls(response, name_map=name_map)
        assert len(tool_calls) == 1
        assert tool_calls[0]["server_name"] == "tool-searching-serper"
        assert tool_calls[0]["tool_name"] == "web_search"

    def test_invalid_name_skipped(self):
        response = {
            "output": [
                {
                    "type": "function_call",
                    "name": "bad-name",
                    "arguments": "{}",
                    "call_id": "call_1",
                }
            ]
        }
        tool_calls, _ = parse_llm_response_for_tool_calls(response, name_map=None)
        assert len(tool_calls) == 0


# ---------- 端到端：convert → parse roundtrip ----------


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_all_tools_roundtrip(self):
        """验证所有工具定义经过 convert → 模拟 LLM 返回 → parse 后完整还原"""
        tool_list, name_map = await LLMProviderClientBase.convert_tool_definition_to_tool_call(
            TOOLS_DEFINITIONS
        )

        expected = [
            ("builtin-spawn-agent", "spawn_agent"),
            ("tool-searching-serper", "web_search"),
            ("tool-searching-serper", "news-search"),
            ("simple", "calc"),
        ]

        for tool_def, (exp_server, exp_tool) in zip(tool_list, expected, strict=True):
            flat_name = tool_def["function"]["name"]
            tc = _make_tool_call_obj(flat_name)
            tool_calls, _ = parse_llm_response_for_tool_calls([tc], name_map=name_map)
            assert len(tool_calls) == 1
            assert tool_calls[0]["server_name"] == exp_server, f"Failed for {flat_name}"
            assert tool_calls[0]["tool_name"] == exp_tool, f"Failed for {flat_name}"
