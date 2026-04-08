"""Context Manager 单元测试

覆盖: 三级 context 管理, tool call dedup, source registry
"""

import pytest

from mem_deep_research_core.core.context_manager import (
    ContextManager,
    ContextManagerConfig,
    SourceRecord,
    SourceRegistry,
)


@pytest.fixture
def cm():
    """默认配置的 ContextManager"""
    return ContextManager(ContextManagerConfig(compact_keep_recent=3))


@pytest.fixture
def source_registry():
    """独立的 SourceRegistry"""
    return SourceRegistry()


# ========== Dedup 测试 ==========


class TestFilterDuplicateCalls:
    def test_first_call_passes(self, cm):
        """首次调用应该通过"""
        calls = [
            {"tool_name": "search", "arguments": {"q": "hello"}, "id": "c1", "server_name": "s1"},
        ]
        to_execute, cached = cm.filter_duplicate_calls(calls)
        assert len(to_execute) == 1
        assert len(cached) == 0

    def test_duplicate_returns_cached(self, cm):
        """注册后重复调用返回缓存（含完整结果）"""
        call = {"tool_name": "search", "arguments": {"q": "hello"}, "id": "c1", "server_name": "s1"}
        result = [("c1", {"type": "text", "text": "search result here"})]

        cm.register_tool_results([call], result, turn=1)

        # 第二次调用相同参数
        calls2 = [
            {"tool_name": "search", "arguments": {"q": "hello"}, "id": "c2", "server_name": "s1"},
        ]
        to_execute, cached = cm.filter_duplicate_calls(calls2)
        assert len(to_execute) == 0
        assert len(cached) == 1
        # 应包含完整缓存结果
        assert "search result here" in cached[0][1]["text"]
        # 应包含 dedup 提示
        assert "already called" in cached[0][1]["text"]

    def test_progressive_escalation(self, cm):
        """重复调用应渐进升级警告"""
        call = {"tool_name": "search", "arguments": {"q": "hello"}, "id": "c1", "server_name": "s1"}
        result = [("c1", {"type": "text", "text": "search result here"})]
        cm.register_tool_results([call], result, turn=1)

        # 第 2 次 -> 轻提示
        calls2 = [
            {"tool_name": "search", "arguments": {"q": "hello"}, "id": "c2", "server_name": "s1"}
        ]
        _, cached1 = cm.filter_duplicate_calls(calls2)
        assert "already called" in cached1[0][1]["text"]

        # 第 3 次 -> 强制警告
        calls3 = [
            {"tool_name": "search", "arguments": {"q": "hello"}, "id": "c3", "server_name": "s1"}
        ]
        _, cached2 = cm.filter_duplicate_calls(calls3)
        assert "WARNING" in cached2[0][1]["text"]
        assert "different approach" in cached2[0][1]["text"]

    def test_different_args_not_duplicate(self, cm):
        """不同参数不被视为重复"""
        call1 = {
            "tool_name": "search",
            "arguments": {"q": "hello"},
            "id": "c1",
            "server_name": "s1",
        }
        result1 = [("c1", {"type": "text", "text": "result1"})]
        cm.register_tool_results([call1], result1, turn=1)

        calls2 = [
            {"tool_name": "search", "arguments": {"q": "world"}, "id": "c2", "server_name": "s1"},
        ]
        to_execute, cached = cm.filter_duplicate_calls(calls2)
        assert len(to_execute) == 1
        assert len(cached) == 0

    def test_dedup_disabled(self):
        """禁用 dedup 时所有调用都通过"""
        cm = ContextManager(ContextManagerConfig(enable_dedup=False))
        call = {"tool_name": "search", "arguments": {"q": "hello"}, "id": "c1", "server_name": "s1"}
        cm.register_tool_results([call], [("c1", {"type": "text", "text": "result"})], turn=1)

        calls2 = [
            {"tool_name": "search", "arguments": {"q": "hello"}, "id": "c2", "server_name": "s1"}
        ]
        to_execute, cached = cm.filter_duplicate_calls(calls2)
        assert len(to_execute) == 1
        assert len(cached) == 0


# ========== Register 测试 ==========


class TestRegisterToolResults:
    def test_register_fills_cache(self, cm):
        """register_tool_results 填充 dedup 缓存"""
        call = {
            "tool_name": "fetch",
            "arguments": {"url": "http://x.com"},
            "id": "c1",
            "server_name": "s1",
        }
        result = [("c1", {"type": "text", "text": "page content"})]
        cm.register_tool_results([call], result, turn=1)
        assert cm.dedup_cache_size == 1

    def test_register_multiple(self, cm):
        """注册多个不同调用"""
        calls = [
            {"tool_name": "search", "arguments": {"q": "a"}, "id": "c1", "server_name": "s1"},
            {"tool_name": "search", "arguments": {"q": "b"}, "id": "c2", "server_name": "s1"},
        ]
        results = [
            ("c1", {"type": "text", "text": "result a"}),
            ("c2", {"type": "text", "text": "result b"}),
        ]
        cm.register_tool_results(calls, results, turn=1)
        assert cm.dedup_cache_size == 2

    def test_register_stores_full_result(self, cm):
        """注册后 dedup 缓存包含完整结果"""
        long_result = "x" * 5000
        call = {
            "tool_name": "fetch",
            "arguments": {"url": "http://x.com"},
            "id": "c1",
            "server_name": "s1",
        }
        result = [("c1", {"type": "text", "text": long_result})]
        cm.register_tool_results([call], result, turn=1)

        # 重复调用应返回完整内容
        calls2 = [
            {
                "tool_name": "fetch",
                "arguments": {"url": "http://x.com"},
                "id": "c2",
                "server_name": "s1",
            }
        ]
        _, cached = cm.filter_duplicate_calls(calls2)
        assert long_result in cached[0][1]["text"]


# ========== Level 1: Compact 测试 ==========


class TestCompact:
    def _build_history(self, num_turns):
        """构建模拟消息历史"""
        from mem_deep_research_core.core.constants import MT

        history = [
            {"role": "user", "content": [{"type": "text", "text": "initial task" * 50}]},
        ]
        for t in range(num_turns):
            history.append({"role": "assistant", "content": f"assistant response turn {t + 1}"})
            # 模拟 tool result（长文本，标记为 TOOL_RESULT）
            history.append(
                {
                    "role": "user",
                    "_type": MT.TOOL_RESULT,
                    "content": [{"type": "text", "text": "x" * 500}],
                }
            )
        return history

    def test_compact_old_turns(self, cm):
        """Level 1 compact 替换旧轮次工具结果为摘要"""
        history = self._build_history(6)
        compacted = cm.apply_compact(history, current_turn=6)
        # compact_keep_recent=3, current=6, 所以 turn 1-3 可被 compact
        assert compacted > 0
        # 被 compact 的消息应包含结构化摘要
        for msg in history:
            content = msg.get("content")
            if isinstance(content, list) and len(content) == 1:
                text = content[0].get("text", "")
                if "compacted" in text.lower() or text.startswith("[Turn"):
                    assert "chars" in text

    def test_compact_preserves_recent_turns(self, cm):
        """Level 1 compact 保留近轮次完整结果"""
        history = self._build_history(4)
        compacted = cm.apply_compact(history, current_turn=4)
        # compact_keep_recent=3, current=4, cutoff=1
        # 只有 turn 1 可能被 compact
        assert compacted <= 1

    def test_compact_disabled(self):
        """禁用 compact 时不做任何替换"""
        cm = ContextManager(ContextManagerConfig(enable_compact=False))
        history = [
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": [{"type": "text", "text": "x" * 500}]},
        ]
        compacted = cm.apply_compact(history, current_turn=10)
        assert compacted == 0

    def test_skips_system_messages(self, cm):
        """跳过系统注入的消息"""
        history = [
            {"role": "user", "content": [{"type": "text", "text": "initial task" * 50}]},
            {"role": "assistant", "content": "response"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "[REFLECTION CHECKPOINT] " + "x" * 500}],
            },
        ]
        compacted = cm.apply_compact(history, current_turn=10)
        assert compacted == 0  # 系统消息不会被 compact

    def test_token_aware_compact(self):
        """Token-aware compact 在达到目标 ratio 后停止"""
        from mem_deep_research_core.core.constants import MT

        cm = ContextManager(
            ContextManagerConfig(
                compact_at_ratio=0.5,
                compact_keep_recent=1,
            )
        )
        # 构建大量历史使 ratio 高
        history = [
            {"role": "user", "content": [{"type": "text", "text": "task" * 100}]},
        ]
        for t in range(10):
            history.append({"role": "assistant", "content": f"response {t + 1}" * 100})
            history.append({"role": "user", "_type": MT.TOOL_RESULT, "content": [{"type": "text", "text": "x" * 2000}]})

        compacted = cm.apply_compact(
            history,
            current_turn=10,
            system_prompt="sys" * 100,
            max_context_length=5000,  # 很小的 limit 确保触发
        )
        assert compacted > 0


# ========== Observation Masking 兼容测试 ==========


class TestObservationMaskingCompat:
    def _build_history(self, num_turns):
        """构建模拟消息历史"""
        from mem_deep_research_core.core.constants import MT

        history = [
            {"role": "user", "content": [{"type": "text", "text": "initial task" * 50}]},
        ]
        for t in range(num_turns):
            history.append({"role": "assistant", "content": f"assistant response turn {t + 1}"})
            history.append(
                {
                    "role": "user",
                    "_type": MT.TOOL_RESULT,
                    "content": [{"type": "text", "text": "x" * 500}],
                }
            )
        return history

    def test_apply_compact_masks_old_turns(self, cm):
        """apply_compact masks old turns"""
        history = self._build_history(6)
        masked = cm.apply_compact(history, current_turn=6)
        assert masked > 0


# ========== manage_context 统一入口测试 ==========


class TestManageContext:
    def test_returns_none_when_low_ratio(self):
        """低 ratio 时返回 none"""
        cm = ContextManager(ContextManagerConfig(compact_at_ratio=0.6))
        history = [
            {"role": "user", "content": [{"type": "text", "text": "short task"}]},
            {"role": "assistant", "content": "short response"},
        ]
        action = cm.manage_context(
            history,
            current_turn=1,
            system_prompt="sys",
            max_context_length=100000,
        )
        assert action == "none"

    def test_returns_compact_at_threshold(self):
        """达到 compact ratio 时返回 compact"""
        cm = ContextManager(
            ContextManagerConfig(
                compact_at_ratio=0.3,
                compact_keep_recent=1,
            )
        )
        history = [
            {"role": "user", "content": [{"type": "text", "text": "task" * 100}]},
        ]
        for t in range(5):
            history.append({"role": "assistant", "content": f"resp {t}" * 100})
            history.append({"role": "user", "content": [{"type": "text", "text": "x" * 2000}]})

        action = cm.manage_context(
            history,
            current_turn=5,
            system_prompt="sys" * 100,
            max_context_length=5000,
        )
        assert action in ("compact", "observation_masking", "need_summarize")

    def test_returns_need_summarize_at_high_ratio(self):
        """达到 summarize ratio 时返回 need_summarize"""
        cm = ContextManager(
            ContextManagerConfig(
                compact_at_ratio=0.3,
                summarize_at_ratio=0.5,
                compact_keep_recent=1,
            )
        )
        history = [
            {"role": "user", "content": [{"type": "text", "text": "task" * 200}]},
        ]
        for t in range(10):
            history.append({"role": "assistant", "content": f"resp {t}" * 200})
            history.append({"role": "user", "content": [{"type": "text", "text": "x" * 3000}]})

        action = cm.manage_context(
            history,
            current_turn=10,
            system_prompt="sys" * 200,
            max_context_length=5000,
        )
        assert action == "need_summarize"

    def test_fallback_without_max_context(self):
        """没有 max_context_length 时回退到 turn-based compact"""
        cm = ContextManager(ContextManagerConfig(compact_keep_recent=2))
        history = [
            {"role": "user", "content": [{"type": "text", "text": "task" * 50}]},
        ]
        for t in range(5):
            history.append({"role": "assistant", "content": f"resp {t}"})
            history.append({"role": "user", "content": [{"type": "text", "text": "x" * 500}]})

        action = cm.manage_context(history, current_turn=5)
        assert action in ("compact", "observation_masking")


# ========== Level 3: Emergency 测试 ==========


class TestEmergency:
    def test_emergency_reduces_history(self):
        """Emergency 减少消息历史"""
        cm = ContextManager(ContextManagerConfig(compact_keep_recent=1))
        history = [
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
        ]
        for t in range(10):
            history.append({"role": "assistant", "content": f"resp {t}"})
            history.append({"role": "user", "content": [{"type": "text", "text": "x" * 500}]})

        original_len = len(history)
        removed = cm.apply_emergency(history, current_turn=10)
        assert removed > 0
        assert len(history) < original_len


# ========== Reset 测试 ==========


class TestReset:
    def test_reset_clears_all(self, cm):
        """reset 清除所有状态"""
        call = {"tool_name": "search", "arguments": {"q": "hello"}, "id": "c1", "server_name": "s1"}
        cm.register_tool_results([call], [("c1", {"type": "text", "text": "result"})], turn=1)
        assert cm.dedup_cache_size > 0

        cm.reset()
        assert cm.dedup_cache_size == 0
        assert cm._current_turn == 0
        assert len(cm.source_registry.get_all_sources()) == 0


# ========== Token 估算测试 ==========


class TestTokenEstimation:
    def test_default_estimation(self, cm):
        """默认 chars_per_token 估算"""
        text = "hello world"  # 11 chars
        tokens = cm.estimate_tokens(text)
        assert tokens == int(11 / 3.5)

    def test_custom_estimator(self, cm):
        """注入自定义 token 估算函数"""
        cm.set_token_estimator(lambda text: len(text.split()))
        tokens = cm.estimate_tokens("hello world foo")
        assert tokens == 3

    def test_context_ratio(self, cm):
        """context ratio 计算"""
        ratio = cm.get_context_ratio(
            system_prompt="sys",
            message_history=[{"role": "user", "content": "hello"}],
            max_context_length=100,
        )
        assert 0 < ratio < 1

    def test_context_ratio_no_limit(self, cm):
        """无 context 限制时 ratio 为 0"""
        ratio = cm.get_context_ratio(
            system_prompt="sys",
            message_history=[{"role": "user", "content": "hello"}],
            max_context_length=0,
        )
        assert ratio == 0.0


# ========== Source Registry 测试 ==========


class TestSourceRegistry:
    def test_extract_from_arguments(self, source_registry):
        """从参数中提取来源"""
        sources = source_registry.extract_and_register(
            tool_name="scrape_website",
            arguments={"url": "https://example.com"},
            result_text="{}",
            turn=1,
        )
        assert len(sources) == 1
        assert sources[0].url == "https://example.com"

    def test_extract_from_json_results(self, source_registry):
        """从 JSON 结果中提取多个来源"""
        import json

        result = json.dumps(
            {
                "results": [
                    {"url": "https://a.com", "title": "Page A"},
                    {"url": "https://b.com", "title": "Page B"},
                ]
            }
        )
        sources = source_registry.extract_and_register(
            tool_name="web_search",
            arguments={},
            result_text=result,
            turn=1,
        )
        assert len(sources) == 2

    def test_dedup_urls(self, source_registry):
        """重复 URL 不会重复注册"""
        source_registry.extract_and_register(
            tool_name="scrape",
            arguments={"url": "https://same.com"},
            result_text="{}",
            turn=1,
        )
        source_registry.extract_and_register(
            tool_name="scrape",
            arguments={"url": "https://same.com"},
            result_text="{}",
            turn=2,
        )
        assert len(source_registry.get_all_sources()) == 1

    def test_get_citation_summary_empty(self, source_registry):
        """无来源时返回空字符串"""
        assert source_registry.get_citation_summary() == ""

    def test_get_citation_summary_format(self, source_registry):
        """引用摘要格式正确"""
        source_registry._sources = [
            SourceRecord(url="https://a.com", title="Page A"),
            SourceRecord(url="https://b.com", title=""),
        ]
        source_registry._seen_urls = {"https://a.com", "https://b.com"}
        summary = source_registry.get_citation_summary()
        assert "## Sources" in summary
        assert "[Page A](https://a.com)" in summary
        assert "https://b.com" in summary

    def test_extract_link_field(self, source_registry):
        """提取使用 link 字段名的结果"""
        import json

        result = json.dumps(
            [
                {"link": "https://c.com", "title": "Result C"},
            ]
        )
        sources = source_registry.extract_and_register(
            tool_name="google_search",
            arguments={},
            result_text=result,
            turn=1,
        )
        assert len(sources) >= 1
        urls = [s.url for s in source_registry.get_all_sources()]
        assert "https://c.com" in urls

    def test_reset(self, source_registry):
        """重置清空所有来源"""
        source_registry.extract_and_register(
            tool_name="scrape",
            arguments={"url": "https://x.com"},
            result_text="{}",
            turn=1,
        )
        source_registry.reset()
        assert len(source_registry.get_all_sources()) == 0
        assert source_registry.get_citation_summary() == ""


# ========== Microcompact 测试 ==========


class TestMicrocompact:
    @staticmethod
    def _build_history(num_turns, content_size=500):
        """构建模拟消息历史（assistant + user tool_result 交替）"""
        from mem_deep_research_core.core.constants import MT

        history = [
            {"role": "user", "content": [{"type": "text", "text": "initial task"}]},
        ]
        for t in range(num_turns):
            history.append({"role": "assistant", "content": f"assistant response turn {t + 1}"})
            history.append(
                {
                    "role": "user",
                    "_type": MT.TOOL_RESULT,
                    "content": [{"type": "text", "text": "x" * content_size}],
                }
            )
        return history

    def test_clears_old_tool_results(self):
        """microcompact 清理超过 keep_recent 的旧 tool_result"""
        cm = ContextManager(ContextManagerConfig(compact_keep_recent=2))
        history = self._build_history(5, content_size=500)

        cleaned = cm.microcompact(history, current_turn=5, keep_recent=2)
        assert cleaned > 0

        # 旧轮次消息应被替换为占位符
        for msg in history[1:]:
            content = msg.get("content", "")
            if isinstance(content, list) and content:
                text = content[0].get("text", "")
                if "[microcompact]" in text:
                    assert "chars" in text or "cleared" in text

    def test_preserves_recent_messages(self):
        """microcompact 保留最近 keep_recent 轮的消息"""
        cm = ContextManager(ContextManagerConfig())
        history = self._build_history(5, content_size=500)

        # 记录最后一轮的用户消息原文
        last_user_content = history[-1]["content"]
        if isinstance(last_user_content, list):
            original_text = last_user_content[0]["text"]
        else:
            original_text = last_user_content

        cm.microcompact(history, current_turn=5, keep_recent=3)

        # 最后一轮的消息应保持不变
        final_content = history[-1]["content"]
        if isinstance(final_content, list):
            final_text = final_content[0]["text"]
        else:
            final_text = final_content
        assert final_text == original_text

    def test_skips_non_tool_result_messages(self):
        """microcompact 只清理 TOOL_RESULT，跳过用户输入和系统注入"""
        from mem_deep_research_core.core.constants import MT

        cm = ContextManager(ContextManagerConfig())
        history = [
            {"role": "user", "content": [{"type": "text", "text": "initial task"}]},
            {"role": "assistant", "content": "response 1"},
            # 系统消息（无 _type，旧格式）
            {
                "role": "user",
                "content": [{"type": "text", "text": "[REFLECTION CHECKPOINT] " + "x" * 500}],
            },
            {"role": "assistant", "content": "response 2"},
            # 用户输入（有 _type=USER_INPUT）— 不应被清理
            {
                "role": "user",
                "_type": MT.USER_INPUT,
                "content": [{"type": "text", "text": "y" * 500}],
            },
            {"role": "assistant", "content": "response 3"},
            # 无 _type 的长消息 — 保守保留
            {"role": "user", "content": [{"type": "text", "text": "z" * 500}]},
        ]

        cleaned = cm.microcompact(history, current_turn=5, keep_recent=1)
        assert cleaned == 0  # 没有 TOOL_RESULT，不清理任何消息

        # 所有消息应保持不变
        assert "[REFLECTION CHECKPOINT]" in history[2]["content"][0]["text"]
        assert history[4]["content"][0]["text"] == "y" * 500
        assert history[6]["content"][0]["text"] == "z" * 500

    def test_skips_already_offloaded(self):
        """microcompact 跳过已卸载的 TOOL_RESULT"""
        from mem_deep_research_core.core.constants import MT

        cm = ContextManager(ContextManagerConfig())
        history = [
            {"role": "user", "content": [{"type": "text", "text": "initial task"}]},
            {"role": "assistant", "content": "response 1"},
            {
                "role": "user",
                "_type": MT.TOOL_RESULT,
                "content": [
                    {"type": "text", "text": "[OFFLOADED:/tmp/result.json|5000] summary here"}
                ],
            },
            {"role": "assistant", "content": "response 2"},
            {"role": "user", "_type": MT.TOOL_RESULT, "content": [{"type": "text", "text": "z" * 500}]},
        ]

        cleaned = cm.microcompact(history, current_turn=5, keep_recent=1)

        # 已卸载消息应保持不变
        offloaded_msg = history[2]
        text = offloaded_msg["content"][0]["text"]
        assert "[OFFLOADED:" in text
        assert "[microcompact]" not in text
        # 非卸载的 TOOL_RESULT 应被清理
        assert cleaned >= 1

    def test_no_cleanup_when_cutoff_zero(self):
        """current_turn <= keep_recent 时不清理"""
        cm = ContextManager(ContextManagerConfig())
        history = self._build_history(3, content_size=500)

        cleaned = cm.microcompact(history, current_turn=3, keep_recent=5)
        assert cleaned == 0

    def test_skips_short_messages(self):
        """短消息不被清理（即使是 TOOL_RESULT）"""
        from mem_deep_research_core.core.constants import MT

        cm = ContextManager(ContextManagerConfig())
        history = [
            {"role": "user", "content": [{"type": "text", "text": "initial task"}]},
            {"role": "assistant", "content": "resp 1"},
            {"role": "user", "_type": MT.TOOL_RESULT, "content": [{"type": "text", "text": "short"}]},  # too short
            {"role": "assistant", "content": "resp 2"},
            {"role": "user", "_type": MT.TOOL_RESULT, "content": [{"type": "text", "text": "z" * 500}]},
        ]

        cleaned = cm.microcompact(history, current_turn=5, keep_recent=1)

        # "short" message should not be touched (below MICROCOMPACT_MIN_CHARS)
        msg2_content = history[2]["content"]
        if isinstance(msg2_content, list):
            text = msg2_content[0]["text"]
        else:
            text = msg2_content
        assert text == "short"
