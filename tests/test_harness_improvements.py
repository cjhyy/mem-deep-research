"""
Harness 改进功能测试

覆盖所有借鉴 Claude Code 架构新增的功能：
- DeferredToolManager: 工具延迟加载
- Transcript: 事件日志
- InputCompiler: 输入编译链

对应模块：
  mem_deep_research_core/core/deferred_tools.py
  mem_deep_research_core/core/transcript.py
  mem_deep_research_core/core/input_compiler.py
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mem_deep_research_core.core.constants import BUILTIN_TOOL_SEARCH
from mem_deep_research_core.core.deferred_tools import DeferredToolManager
from mem_deep_research_core.core.hooks import HookRegistry
from mem_deep_research_core.core.input_compiler import CompileResult, InputCompiler
from mem_deep_research_core.core.transcript import EventType, Transcript, TranscriptEvent


# ================================================================
# DeferredToolManager
# ================================================================


def _make_server(name: str, num_tools: int, prefix: str = "tool") -> dict:
    """生成带 N 个工具的 MCP server 定义"""
    tools = []
    for i in range(num_tools):
        tools.append({
            "name": f"{prefix}_{name}_{i}",
            "description": f"Description for {prefix}_{name}_{i}. Does useful work.",
            "schema": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string", "description": "first argument"},
                },
                "required": ["arg1"],
            },
        })
    return {"name": name, "tools": tools}


def _make_tool_defs(num_servers: int, tools_per_server: int) -> list[dict]:
    """生成多个 server 的工具定义"""
    return [_make_server(f"server_{i}", tools_per_server) for i in range(num_servers)]


class TestDeferredBelowThreshold:
    def test_no_deferral_when_under_threshold(self):
        """工具数不超过阈值时返回原始定义"""
        manager = DeferredToolManager(threshold=20)
        defs = _make_tool_defs(2, 5)  # 10 tools total
        result, was_deferred = manager.apply(defs)
        assert not was_deferred
        assert result is defs
        assert not manager.is_active

    def test_no_deferral_when_equal_to_threshold(self):
        """工具数恰好等于阈值时不延迟"""
        manager = DeferredToolManager(threshold=10)
        defs = _make_tool_defs(2, 5)  # 10 tools
        result, was_deferred = manager.apply(defs)
        assert not was_deferred

    def test_no_deferral_when_threshold_zero(self):
        """threshold=0 表示禁用"""
        manager = DeferredToolManager(threshold=0)
        defs = _make_tool_defs(5, 10)  # 50 tools
        result, was_deferred = manager.apply(defs)
        assert not was_deferred


class TestDeferredAboveThreshold:
    def test_deferral_when_above_threshold(self):
        """工具数超过阈值时启用延迟模式"""
        manager = DeferredToolManager(threshold=5)
        defs = _make_tool_defs(2, 5)  # 10 tools
        result, was_deferred = manager.apply(defs)
        assert was_deferred
        assert manager.is_active

    def test_deferred_tools_have_no_full_schema(self):
        """延迟模式下包含 tool_search"""
        manager = DeferredToolManager(threshold=5)
        defs = _make_tool_defs(2, 5)  # 10 tools
        result, _ = manager.apply(defs)

        tool_search_found = False
        for server in result:
            if server.get("name") == "builtin-tool-search":
                for tool in server.get("tools", []):
                    if tool.get("name") == BUILTIN_TOOL_SEARCH:
                        tool_search_found = True
                        assert "schema" in tool
        assert tool_search_found

    def test_tool_search_description_includes_listing(self):
        """tool_search 的描述应包含被延迟工具的列表"""
        manager = DeferredToolManager(threshold=5)
        defs = _make_tool_defs(2, 5)  # 10 tools
        result, _ = manager.apply(defs)

        for server in result:
            if server.get("name") == "builtin-tool-search":
                for tool in server.get("tools", []):
                    if tool.get("name") == BUILTIN_TOOL_SEARCH:
                        desc = tool["description"]
                        assert "10 total" in desc
                        assert "tool_server_0_0" in desc


class TestToolSearchResolution:
    @pytest.fixture
    def manager(self):
        m = DeferredToolManager(threshold=5)
        defs = [
            {
                "name": "search-server",
                "tools": [
                    {
                        "name": "web_search",
                        "description": "Search the web for information",
                        "schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                    },
                    {
                        "name": "image_search",
                        "description": "Search for images on the internet",
                        "schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                    },
                ],
            },
            {
                "name": "fetch-server",
                "tools": [
                    {
                        "name": "fetch_url",
                        "description": "Fetch content from a URL",
                        "schema": {"type": "object", "properties": {"url": {"type": "string"}}},
                    },
                    {
                        "name": "scrape_page",
                        "description": "Scrape and extract structured data from a page",
                        "schema": {"type": "object", "properties": {"url": {"type": "string"}}},
                    },
                    {
                        "name": "calculate_metrics",
                        "description": "Calculate statistical metrics from data",
                        "schema": {"type": "object", "properties": {"data": {"type": "array"}}},
                    },
                    {
                        "name": "generate_report",
                        "description": "Generate a formatted report from findings",
                        "schema": {"type": "object", "properties": {"findings": {"type": "array"}}},
                    },
                ],
            },
        ]
        m.apply(defs)
        return m

    def test_exact_match(self, manager):
        results = manager.resolve_tool_search("web_search")
        assert len(results) >= 1
        assert results[0]["tool_name"] == "web_search"
        assert "schema" in results[0]

    def test_partial_match(self, manager):
        results = manager.resolve_tool_search("search")
        tool_names = [r["tool_name"] for r in results]
        assert "web_search" in tool_names
        assert "image_search" in tool_names

    def test_description_match(self, manager):
        results = manager.resolve_tool_search("statistical")
        tool_names = [r["tool_name"] for r in results]
        assert "calculate_metrics" in tool_names

    def test_fuzzy_match(self, manager):
        results = manager.resolve_tool_search("web_serch")  # typo
        tool_names = [r["tool_name"] for r in results]
        assert "web_search" in tool_names

    def test_max_results_limit(self, manager):
        results = manager.resolve_tool_search("search", max_results=1)
        assert len(results) == 1

    def test_resolve_marks_discovered(self, manager):
        assert "web_search" not in manager.get_discovered_tool_names()
        manager.resolve_tool_search("web_search")
        assert "web_search" in manager.get_discovered_tool_names()

    def test_no_match_returns_empty(self, manager):
        results = manager.resolve_tool_search("zzzzzyyyy_nonexistent_xyzxyz")
        assert len(results) == 0


class TestDiscoveredToolsPersist:
    def test_discovered_tools_get_full_schema(self):
        """已发现的工具在后续 apply() 中获得完整 schema"""
        manager = DeferredToolManager(threshold=3)
        defs = [
            {
                "name": "myserver",
                "tools": [
                    {"name": f"tool_{c}", "description": f"Tool {c}", "schema": {"type": "object"}}
                    for c in "abcd"
                ],
            },
        ]

        result1, was_deferred = manager.apply(defs)
        assert was_deferred

        manager.resolve_tool_search("tool_a")
        result2, _ = manager.apply(defs)

        found_tool_a = False
        for server in result2:
            if server.get("name") == "myserver":
                for tool in server.get("tools", []):
                    if tool.get("name") == "tool_a":
                        assert "schema" in tool
                        found_tool_a = True
        assert found_tool_a


class TestDeferredReset:
    def test_reset_clears_state(self):
        manager = DeferredToolManager(threshold=3)
        defs = _make_tool_defs(1, 5)
        manager.apply(defs)
        manager.resolve_tool_search("tool_server_0_0")

        assert manager.is_active
        assert len(manager.get_discovered_tool_names()) > 0

        manager.reset()
        assert not manager.is_active
        assert len(manager.get_discovered_tool_names()) == 0


class TestBuiltinToolsNeverDeferred:
    def test_builtin_tools_preserved(self):
        """builtin-* server 的工具保留完整 schema"""
        manager = DeferredToolManager(threshold=3)
        builtin_server = {
            "name": "builtin-core",
            "tools": [
                {"name": "spawn_agent", "description": "Spawn a sub agent", "schema": {"type": "object"}},
            ],
        }
        regular_servers = _make_tool_defs(1, 5)
        defs = [builtin_server] + regular_servers

        result, was_deferred = manager.apply(defs)
        assert was_deferred

        found_builtin = False
        for server in result:
            if server.get("name") == "builtin-core":
                found_builtin = True
                tools = server.get("tools", [])
                assert len(tools) == 1
                assert "schema" in tools[0]
        assert found_builtin


# ================================================================
# Transcript
# ================================================================


class TestTranscriptRecord:
    def test_record_returns_event_id(self):
        t = Transcript()
        eid = t.record(EventType.AGENT_START, {"model": "claude-sonnet"})
        assert eid.startswith("evt_")

    def test_record_increments_count(self):
        t = Transcript()
        assert t.event_count == 0
        t.record(EventType.TURN_START, turn=1)
        t.record(EventType.TURN_END, turn=1)
        assert t.event_count == 2

    def test_record_stores_data(self):
        t = Transcript()
        t.record(EventType.LLM_CALL, {"model": "sonnet", "tokens": 1024}, turn=3)
        assert t.events[0].data == {"model": "sonnet", "tokens": 1024}
        assert t.events[0].turn == 3

    def test_record_with_ref_event_id(self):
        t = Transcript()
        eid1 = t.record(EventType.TOOL_USE, {"tool": "search"}, turn=1)
        eid2 = t.record(EventType.TOOL_RESULT, {"result": "ok"}, turn=1, ref_event_id=eid1)
        event = t.get_by_id(eid2)
        assert event.ref_event_id == eid1

    def test_record_with_duration(self):
        t = Transcript()
        t.record(EventType.LLM_RESPONSE, {"tokens": 500}, duration_ms=1200)
        assert t.events[0].duration_ms == 1200

    def test_record_with_custom_agent_name(self):
        t = Transcript(agent_name="main")
        t.record(EventType.SUBAGENT_START, agent_name="researcher")
        assert t.events[0].agent_name == "researcher"


class TestTranscriptFilter:
    @pytest.fixture
    def populated_transcript(self):
        t = Transcript()
        t.record(EventType.AGENT_START, turn=0)
        t.record(EventType.TURN_START, turn=1, agent_name="main")
        t.record(EventType.LLM_CALL, turn=1, agent_name="main")
        t.record(EventType.TOOL_USE, turn=1, agent_name="main")
        t.record(EventType.TOOL_RESULT, turn=1, agent_name="main")
        t.record(EventType.TURN_END, turn=1, agent_name="main")
        t.record(EventType.TURN_START, turn=2, agent_name="main")
        t.record(EventType.LLM_CALL, turn=2, agent_name="main")
        t.record(EventType.SUBAGENT_START, turn=2, agent_name="researcher")
        t.record(EventType.TOOL_USE, turn=2, agent_name="researcher")
        t.record(EventType.SUBAGENT_END, turn=2, agent_name="researcher")
        t.record(EventType.TURN_END, turn=2, agent_name="main")
        t.record(EventType.AGENT_END, turn=0)
        return t

    def test_filter_by_type(self, populated_transcript):
        results = populated_transcript.filter(event_type=EventType.TOOL_USE)
        assert len(results) == 2

    def test_filter_by_turn(self, populated_transcript):
        results = populated_transcript.filter(turn=1)
        assert len(results) == 5

    def test_filter_by_agent_name(self, populated_transcript):
        results = populated_transcript.filter(agent_name="researcher")
        assert len(results) == 3

    def test_filter_combined(self, populated_transcript):
        results = populated_transcript.filter(event_type=EventType.TOOL_USE, agent_name="researcher")
        assert len(results) == 1
        assert results[0].turn == 2

    def test_filter_no_match(self, populated_transcript):
        assert populated_transcript.filter(event_type=EventType.ERROR) == []

    def test_filter_none_returns_all(self, populated_transcript):
        assert len(populated_transcript.filter()) == populated_transcript.event_count


class TestTranscriptToolPairs:
    def test_matched_pair(self):
        t = Transcript()
        use_id = t.record(EventType.TOOL_USE, {"tool": "search"}, turn=1)
        t.record(EventType.TOOL_RESULT, {"result": "found"}, turn=1, ref_event_id=use_id)

        pairs = t.get_tool_pairs()
        assert len(pairs) == 1
        assert pairs[0][1] is not None
        assert pairs[0][1].ref_event_id == use_id

    def test_unmatched_use(self):
        t = Transcript()
        t.record(EventType.TOOL_USE, {"tool": "broken"}, turn=1)
        pairs = t.get_tool_pairs()
        assert len(pairs) == 1
        assert pairs[0][1] is None

    def test_pairs_filtered_by_turn(self):
        t = Transcript()
        use1 = t.record(EventType.TOOL_USE, {"tool": "a"}, turn=1)
        t.record(EventType.TOOL_RESULT, {"result": "a_r"}, turn=1, ref_event_id=use1)
        use2 = t.record(EventType.TOOL_USE, {"tool": "b"}, turn=2)
        t.record(EventType.TOOL_RESULT, {"result": "b_r"}, turn=2, ref_event_id=use2)

        pairs_t1 = t.get_tool_pairs(turn=1)
        assert len(pairs_t1) == 1
        assert pairs_t1[0][0].data["tool"] == "a"

    def test_multiple_pairs(self):
        t = Transcript()
        id1 = t.record(EventType.TOOL_USE, {"tool": "a"}, turn=1)
        id2 = t.record(EventType.TOOL_USE, {"tool": "b"}, turn=1)
        t.record(EventType.TOOL_RESULT, {"result": "r_a"}, turn=1, ref_event_id=id1)
        t.record(EventType.TOOL_RESULT, {"result": "r_b"}, turn=1, ref_event_id=id2)

        pairs = t.get_tool_pairs(turn=1)
        assert len(pairs) == 2
        assert all(p[1] is not None for p in pairs)


class TestTranscriptSummary:
    def test_summary_event_counts(self):
        t = Transcript()
        t.record(EventType.LLM_CALL)
        t.record(EventType.LLM_RESPONSE, duration_ms=500)
        t.record(EventType.TOOL_USE)
        t.record(EventType.TOOL_RESULT, duration_ms=200)
        t.record(EventType.TOOL_RESULT, duration_ms=300)

        s = t.summary()
        assert s["total_events"] == 5
        assert s["total_tool_duration_ms"] == 500
        assert s["total_llm_duration_ms"] == 500

    def test_summary_empty(self):
        s = Transcript().summary()
        assert s["total_events"] == 0


class TestTranscriptSaveLoad:
    def test_roundtrip(self):
        t = Transcript(agent_name="test-agent")
        id1 = t.record(EventType.AGENT_START, {"model": "sonnet"}, turn=0)
        id2 = t.record(EventType.TOOL_USE, {"tool": "search"}, turn=1)
        t.record(EventType.TOOL_RESULT, {"result": "found"}, turn=1, ref_event_id=id2, duration_ms=150)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "transcript.jsonl"
            t.save(path)

            with open(path) as f:
                lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 3

            loaded = Transcript.load(path)
            assert loaded.event_count == 3
            assert loaded.events[2].ref_event_id == id2
            assert loaded.events[2].duration_ms == 150

    def test_save_creates_parent_dirs(self):
        t = Transcript()
        t.record(EventType.AGENT_START)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "transcript.jsonl"
            t.save(path)
            assert path.exists()


class TestTranscriptReset:
    def test_reset_clears_events(self):
        t = Transcript()
        t.record(EventType.AGENT_START)
        t.record(EventType.TURN_START)
        t.reset()
        assert t.event_count == 0


class TestTranscriptEvent:
    def test_to_dict_omits_none(self):
        event = TranscriptEvent(
            event_id="evt_abc", event_type="tool_use", timestamp=1000.0,
            turn=1, agent_name="main", data={"tool": "search"},
        )
        d = event.to_dict()
        assert "ref_event_id" not in d
        assert "duration_ms" not in d

    def test_to_dict_includes_values(self):
        event = TranscriptEvent(
            event_id="evt_xyz", event_type="tool_result", timestamp=1001.0,
            turn=2, agent_name="main", data={},
            ref_event_id="evt_abc", duration_ms=250,
        )
        d = event.to_dict()
        assert d["ref_event_id"] == "evt_abc"
        assert d["duration_ms"] == 250


# ================================================================
# InputCompiler
# ================================================================


class TestInputCompilerURLExtraction:
    def test_single_url(self):
        result = InputCompiler(hooks=HookRegistry()).compile("Check https://example.com for info")
        assert result.extracted_urls == ["https://example.com"]

    def test_multiple_urls(self):
        result = InputCompiler(hooks=HookRegistry()).compile("Compare https://a.com and http://b.com/page?q=1")
        assert len(result.extracted_urls) == 2

    def test_duplicate_urls_deduped(self):
        result = InputCompiler(hooks=HookRegistry()).compile("Visit https://x.com then https://x.com again")
        assert len(result.extracted_urls) == 1

    def test_no_urls(self):
        result = InputCompiler(hooks=HookRegistry()).compile("Just a simple question")
        assert result.extracted_urls == []

    def test_url_extraction_disabled(self):
        result = InputCompiler(hooks=HookRegistry(), enable_url_extraction=False).compile("Check https://example.com")
        assert result.extracted_urls == []


class TestInputCompilerFileRef:
    def test_simple_file_ref(self):
        result = InputCompiler(hooks=HookRegistry()).compile("Read @data.csv and analyze")
        assert "data.csv" in result.file_refs

    def test_quoted_file_ref(self):
        result = InputCompiler(hooks=HookRegistry()).compile('Read @"my file.txt" for context')
        assert "my file.txt" in result.file_refs

    def test_no_file_refs(self):
        result = InputCompiler(hooks=HookRegistry()).compile("Just a simple question")
        assert result.file_refs == []

    def test_file_ref_disabled(self):
        result = InputCompiler(hooks=HookRegistry(), enable_file_refs=False).compile("Read @data.csv")
        assert result.file_refs == []

    def test_existing_file_creates_attachment(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("file content here")
            f.flush()
            filepath = f.name
        try:
            result = InputCompiler(hooks=HookRegistry()).compile(f"Analyze @{filepath}")
            assert len(result.attachments) == 1
            assert result.attachments[0]["content"] == "file content here"
            assert f"[attached: {filepath}]" in result.query
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_nonexistent_file_no_attachment(self):
        result = InputCompiler(hooks=HookRegistry()).compile("Read @/nonexistent/path/file.xyz")
        assert "/nonexistent/path/file.xyz" in result.file_refs
        assert len(result.attachments) == 0


class TestInputCompilerPassThrough:
    def test_no_urls_no_refs(self):
        query = "What is the meaning of life?"
        result = InputCompiler(hooks=HookRegistry()).compile(query)
        assert result.query == query
        assert result.original_query == query
        assert result.extracted_urls == []
        assert result.file_refs == []
        assert result.attachments == []


class TestInputCompilerHook:
    def test_hook_modifies_query_string(self):
        mock_hooks = MagicMock()
        mock_hooks.has_hooks.return_value = True
        mock_hooks.call_sync.return_value = "modified query"

        result = InputCompiler(hooks=mock_hooks).compile("original query")
        assert result.query == "modified query"

    def test_hook_modifies_query_dict(self):
        mock_hooks = MagicMock()
        mock_hooks.has_hooks.return_value = True
        mock_hooks.call_sync.return_value = {
            "query": "dict modified query",
            "attachments": [{"type": "custom", "data": "extra"}],
        }

        result = InputCompiler(hooks=mock_hooks).compile("original query")
        assert result.query == "dict modified query"
        assert len(result.attachments) == 1

    def test_no_hook_registered(self):
        mock_hooks = MagicMock()
        mock_hooks.has_hooks.return_value = False

        result = InputCompiler(hooks=mock_hooks).compile("just a query")
        assert result.query == "just a query"
        mock_hooks.call_sync.assert_not_called()

    def test_hook_returns_none_no_change(self):
        mock_hooks = MagicMock()
        mock_hooks.has_hooks.return_value = True
        mock_hooks.call_sync.return_value = None

        result = InputCompiler(hooks=mock_hooks).compile("unchanged query")
        assert result.query == "unchanged query"
