"""
Integration tests — LongTermMemory persistence.

Covers:
- store / recall / forget / clear lifecycle
- Keyword-based recall scoring
- Metadata filtering
- max_entries eviction
- Deduplication (same key updates in-place)
- File persistence (write + reload)
- Thread safety under concurrent writes
"""

import json
import threading

import pytest

from mem_deep_research_core.core.memory import LongTermMemory, MemoryEntry


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mem(tmp_path):
    """Fresh LongTermMemory writing to a temp directory."""
    return LongTermMemory(storage_path=str(tmp_path), max_entries=50)


@pytest.fixture
def mem_path(tmp_path):
    """Return both memory and its path for reload tests."""
    m = LongTermMemory(storage_path=str(tmp_path), max_entries=50)
    return m, tmp_path


# ============================================================
# Basic CRUD
# ============================================================


class TestLongTermMemoryBasic:
    def test_store_and_recall(self, mem):
        mem.store("pref_lang", "User prefers Chinese responses")
        results = mem.recall("Chinese")
        assert len(results) == 1
        assert results[0].key == "pref_lang"
        assert "Chinese" in results[0].value

    def test_store_updates_existing_key(self, mem):
        mem.store("k1", "value1")
        mem.store("k1", "value2_updated")
        all_entries = mem.list_all()
        assert len(all_entries) == 1
        assert all_entries[0].value == "value2_updated"

    def test_forget_removes_entry(self, mem):
        mem.store("temp", "temporary data")
        assert mem.forget("temp") is True
        assert mem.recall("temporary") == []

    def test_forget_nonexistent_returns_false(self, mem):
        assert mem.forget("nonexistent") is False

    def test_clear_removes_all(self, mem):
        mem.store("a", "1")
        mem.store("b", "2")
        mem.clear()
        assert mem.list_all() == []

    def test_list_all(self, mem):
        mem.store("x", "val_x")
        mem.store("y", "val_y")
        entries = mem.list_all()
        keys = {e.key for e in entries}
        assert keys == {"x", "y"}


# ============================================================
# Recall scoring
# ============================================================


class TestRecallScoring:
    def test_keyword_scoring_ranks_by_relevance(self, mem):
        mem.store("python_guide", "Python programming language tutorial")
        mem.store("rust_guide", "Rust systems programming")
        mem.store("python_web", "Python web framework Django Flask")

        results = mem.recall("Python programming", top_k=5)
        # "python_guide" matches both "python" and "programming" → highest score
        assert results[0].key == "python_guide"

    def test_no_match_returns_empty(self, mem):
        mem.store("topic", "unrelated content about cats")
        results = mem.recall("quantum physics")
        assert results == []

    def test_empty_query_returns_most_recent(self, mem):
        mem.store("old", "old entry")
        mem.store("new", "new entry")
        results = mem.recall("", top_k=1)
        assert len(results) == 1
        assert results[0].key == "new"

    def test_top_k_limits_results(self, mem):
        for i in range(10):
            mem.store(f"entry_{i}", f"common keyword topic {i}")
        results = mem.recall("common keyword", top_k=3)
        assert len(results) == 3


# ============================================================
# Metadata filtering
# ============================================================


class TestMetadataFilter:
    def test_filter_by_metadata(self, mem):
        mem.store("a", "val_a", metadata={"source": "conversation"})
        mem.store("b", "val_b", metadata={"source": "tool"})
        mem.store("c", "val_c", metadata={"source": "conversation"})

        results = mem.recall("", top_k=10, metadata_filter={"source": "conversation"})
        keys = {r.key for r in results}
        assert keys == {"a", "c"}

    def test_filter_no_match(self, mem):
        mem.store("x", "val", metadata={"env": "prod"})
        results = mem.recall("", metadata_filter={"env": "staging"})
        assert results == []


# ============================================================
# Max entries eviction
# ============================================================


class TestMaxEntries:
    def test_evicts_oldest_when_full(self, tmp_path):
        mem = LongTermMemory(storage_path=str(tmp_path), max_entries=3)
        mem.store("e1", "first")
        mem.store("e2", "second")
        mem.store("e3", "third")
        mem.store("e4", "fourth")  # Should evict e1

        all_keys = {e.key for e in mem.list_all()}
        assert "e1" not in all_keys
        assert len(all_keys) == 3


# ============================================================
# File persistence
# ============================================================


class TestFilePersistence:
    def test_persist_and_reload(self, mem_path):
        mem, path = mem_path
        mem.store("persistent_key", "persistent_value", metadata={"important": True})

        # Create a new instance pointing to the same path
        mem2 = LongTermMemory(storage_path=str(path), max_entries=50)
        results = mem2.recall("persistent")
        assert len(results) == 1
        assert results[0].key == "persistent_key"
        assert results[0].metadata == {"important": True}

    def test_handles_corrupted_file(self, tmp_path):
        """Corrupted memory.json should not crash — start with empty."""
        memory_file = tmp_path / "memory.json"
        memory_file.write_text("this is not valid json {{{}}")

        mem = LongTermMemory(storage_path=str(tmp_path))
        # Should load without error, starting empty
        assert mem.list_all() == []

        # Should be able to store normally
        mem.store("after_corruption", "works fine")
        assert len(mem.list_all()) == 1

    def test_handles_missing_directory(self, tmp_path):
        """Storage path that doesn't exist yet — created on first store."""
        deep_path = tmp_path / "sub" / "dir" / "memory"
        mem = LongTermMemory(storage_path=str(deep_path))
        mem.store("key", "value")

        assert (deep_path / "memory.json").exists()

    def test_access_count_persists(self, mem_path):
        mem, path = mem_path
        mem.store("counted", "data")
        mem.recall("data")  # access_count → 1
        mem.recall("data")  # access_count → 2

        mem2 = LongTermMemory(storage_path=str(path))
        entries = mem2.list_all()
        assert entries[0].access_count == 2


# ============================================================
# Deduplication
# ============================================================


class TestDeduplication:
    def test_deduplicate_keeps_newest(self, mem):
        # Manually inject duplicate keys
        import time

        mem._loaded = True
        mem._entries = [
            MemoryEntry(key="dup", value="old", timestamp=1.0),
            MemoryEntry(key="dup", value="new", timestamp=2.0),
            MemoryEntry(key="unique", value="solo", timestamp=1.5),
        ]
        mem.deduplicate()

        assert len(mem._entries) == 2
        dup_entry = next(e for e in mem._entries if e.key == "dup")
        assert dup_entry.value == "new"


# ============================================================
# Thread safety
# ============================================================


class TestLongTermMemoryThreadSafety:
    def test_concurrent_store(self, tmp_path):
        """Multiple threads storing concurrently — no crash, no data loss."""
        mem = LongTermMemory(storage_path=str(tmp_path), max_entries=500)
        barrier = threading.Barrier(5)

        def writer(thread_id):
            barrier.wait()
            for i in range(20):
                mem.store(f"t{thread_id}_k{i}", f"value_{thread_id}_{i}")

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 5 threads × 20 unique keys = 100
        assert len(mem.list_all()) == 100

    def test_concurrent_recall_during_store(self, tmp_path):
        """Recall while another thread stores — no crash."""
        mem = LongTermMemory(storage_path=str(tmp_path), max_entries=200)
        errors = []
        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set():
                mem.store(f"key_{i}", f"value for keyword common {i}")
                i += 1

        def reader():
            for _ in range(30):
                try:
                    results = mem.recall("common", top_k=5)
                    assert isinstance(results, list)
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
