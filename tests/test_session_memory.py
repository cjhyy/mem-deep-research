"""
Tests for SessionMemory thread safety and correctness.

Covers:
- Concurrent add_finding / add_source deduplication
- to_context_string snapshot consistency
- Overflow truncation at max limits
- is_empty / add_sub_agent_result
"""

import threading

import pytest

from mem_deep_research_core.core.memory import SessionMemory


# ============================================================
# Basic operations
# ============================================================


class TestSessionMemoryBasic:
    def test_add_finding_dedup(self):
        mem = SessionMemory()
        mem.add_finding("fact A")
        mem.add_finding("fact A")
        mem.add_finding("fact B")
        assert mem.key_findings == ["fact A", "fact B"]

    def test_add_finding_ignores_empty(self):
        mem = SessionMemory()
        mem.add_finding("")
        mem.add_finding(None)
        assert mem.key_findings == []

    def test_add_strategy_dedup(self):
        mem = SessionMemory()
        mem.add_strategy("search Google")
        mem.add_strategy("search Google")
        assert mem.attempted_strategies == ["search Google"]

    def test_add_source_dedup_by_url(self):
        mem = SessionMemory()
        mem.add_source(url="https://a.com", title="A")
        mem.add_source(url="https://a.com", title="A duplicate")
        mem.add_source(url="https://b.com", title="B")
        assert len(mem.sources) == 2

    def test_add_source_ignores_empty_url(self):
        mem = SessionMemory()
        mem.add_source(url="", title="no url")
        assert len(mem.sources) == 0

    def test_add_sub_agent_result(self):
        mem = SessionMemory()
        mem.add_sub_agent_result("agent-1", "result 1")
        mem.add_sub_agent_result("agent-1", "result updated")
        # Both results are preserved (append, not overwrite)
        assert mem.sub_agent_results == [("agent-1", "result 1"), ("agent-1", "result updated")]

    def test_add_sub_agent_result_multiple_spawns(self):
        """Multiple spawn_agent calls should all be preserved."""
        mem = SessionMemory()
        mem.add_sub_agent_result("spawn_agent", "tanka core test result")
        mem.add_sub_agent_result("spawn_agent", "jira test result")
        mem.add_sub_agent_result("spawn_agent", "gitlab test result")
        assert len(mem.sub_agent_results) == 3
        assert mem.sub_agent_results[0] == ("spawn_agent", "tanka core test result")
        assert mem.sub_agent_results[2] == ("spawn_agent", "gitlab test result")

    def test_is_empty(self):
        mem = SessionMemory()
        assert mem.is_empty()
        mem.add_finding("x")
        assert not mem.is_empty()


# ============================================================
# Overflow truncation
# ============================================================


class TestSessionMemoryOverflow:
    def test_findings_truncation(self):
        mem = SessionMemory(max_findings=3)
        for i in range(5):
            mem.add_finding(f"fact {i}")
        assert len(mem.key_findings) == 3
        # Should keep the most recent
        assert mem.key_findings == ["fact 2", "fact 3", "fact 4"]

    def test_strategies_truncation(self):
        mem = SessionMemory(max_strategies=2)
        for i in range(4):
            mem.add_strategy(f"strat {i}")
        assert len(mem.attempted_strategies) == 2
        assert mem.attempted_strategies == ["strat 2", "strat 3"]

    def test_sources_truncation(self):
        mem = SessionMemory(max_sources=2)
        for i in range(4):
            mem.add_source(url=f"https://{i}.com", title=f"S{i}")
        assert len(mem.sources) == 2
        assert mem.sources[-1].url == "https://3.com"


# ============================================================
# to_context_string
# ============================================================


class TestSessionMemoryContextString:
    def test_empty_returns_empty(self):
        mem = SessionMemory()
        assert mem.to_context_string() == ""

    def test_contains_session_memory_tag(self):
        mem = SessionMemory()
        mem.add_finding("quantum entanglement")
        result = mem.to_context_string()
        assert result.startswith("[SESSION MEMORY]")
        assert "quantum entanglement" in result

    def test_all_sections_present(self):
        mem = SessionMemory()
        mem.add_finding("finding 1")
        mem.add_strategy("strategy 1")
        mem.add_source(url="https://x.com", title="Source X")
        mem.add_sub_agent_result("agent-r", "sub result")
        result = mem.to_context_string()
        assert "Key Findings" in result
        assert "Attempted Strategies" in result
        assert "Sources Collected" in result
        assert "Sub-Agent Results" in result


# ============================================================
# Thread safety — concurrent writes
# ============================================================


class TestSessionMemoryThreadSafety:
    def test_concurrent_add_finding(self):
        """Many threads adding findings concurrently — no duplicates, no crash."""
        mem = SessionMemory(max_findings=200)
        barrier = threading.Barrier(10)

        def writer(thread_id):
            barrier.wait()
            for i in range(20):
                mem.add_finding(f"t{thread_id}-f{i}")

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 10 threads × 20 unique findings = 200
        assert len(mem.key_findings) == 200

    def test_concurrent_add_source(self):
        """Concurrent source additions — dedup by URL, no crash."""
        mem = SessionMemory(max_sources=200)
        barrier = threading.Barrier(5)

        def writer(thread_id):
            barrier.wait()
            for i in range(20):
                mem.add_source(url=f"https://t{thread_id}-{i}.com", title=f"T{thread_id}")

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(mem.sources) == 100  # 5 × 20

    def test_concurrent_read_write(self):
        """to_context_string while another thread writes — no crash."""
        mem = SessionMemory(max_findings=100)
        errors = []
        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set():
                mem.add_finding(f"finding-{i}")
                i += 1

        def reader():
            for _ in range(50):
                try:
                    s = mem.to_context_string()
                    # Should always be well-formed (string, not None)
                    assert isinstance(s, str)
                except Exception as e:
                    errors.append(e)

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        w.start()
        r.start()
        r.join()
        stop.set()
        w.join()

        assert errors == [], f"Reader errors: {errors}"
