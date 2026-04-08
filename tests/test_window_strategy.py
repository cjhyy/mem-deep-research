"""
WindowStrategy 模块测试

覆盖：
- ObservationMaskingStrategy: 按轮次/token 两种模式
- LLMSummarizeStrategy: 异步 LLM 压缩
- BinaryReductionStrategy: 紧急裁剪
- WindowStrategyPipeline: 管道编排
- 自定义策略注入
"""

import pytest

from mem_deep_research_core.core.context_manager import ToolCallRecord
from mem_deep_research_core.core.window_strategy import (
    BinaryReductionStrategy,
    CompressResult,
    LLMSummarizeStrategy,
    ObservationMaskingStrategy,
    WindowContext,
    WindowStrategy,
    WindowStrategyPipeline,
    _estimate_ratio,
    _extract_key_argument,
    _get_message_char_count,
    _is_system_message,
)

# ========== 辅助函数 ==========


def _make_history(num_turns: int, content_size: int = 500) -> list:
    """生成测试用消息历史（tool result 类型）"""
    from mem_deep_research_core.core.constants import MT

    history = [
        {"role": "user", "content": [{"type": "text", "text": "initial task" * 10}]},
    ]
    for t in range(num_turns):
        history.append({"role": "assistant", "content": f"response {t}" * 20})
        history.append({"role": "user", "_type": MT.TOOL_RESULT, "content": [{"type": "text", "text": "x" * content_size}]})
    return history


def _make_ctx(
    current_turn: int = 5,
    max_tokens: int = 5000,
    history: list = None,
    system_prompt: str = "system",
    call_registry: list = None,
) -> WindowContext:
    """构建测试用 WindowContext"""
    if history is None:
        history = _make_history(current_turn)
    ratio = _estimate_ratio(history, system_prompt, max_tokens) if max_tokens > 0 else 0.0
    return WindowContext(
        current_turn=current_turn,
        max_turns=20,
        token_count=int(ratio * max_tokens),
        max_tokens=max_tokens,
        token_ratio=ratio,
        message_count=len(history),
        system_prompt=system_prompt,
        message_history=history,
        call_registry=call_registry or [],
        compacted_turns=set(),
    )


# ========== Helper 函数测试 ==========


class TestHelpers:
    def test_get_message_char_count_str(self):
        msg = {"content": "hello world"}
        assert _get_message_char_count(msg) == 11

    def test_get_message_char_count_list(self):
        msg = {"content": [{"type": "text", "text": "abc"}, {"type": "text", "text": "de"}]}
        assert _get_message_char_count(msg) == 5

    def test_is_system_message_true(self):
        assert _is_system_message("[REFLECTION CHECKPOINT] ...")
        assert _is_system_message([{"type": "text", "text": "[RESEARCH CONTEXT SUMMARY]"}])

    def test_is_system_message_false(self):
        assert not _is_system_message("regular tool result")
        assert not _is_system_message([{"type": "text", "text": "data"}])

    def test_extract_key_argument(self):
        assert _extract_key_argument("search", {"query": "hello"}) == "hello"
        assert _extract_key_argument("fetch", {"url": "http://example.com"}) == "http://example.com"
        assert _extract_key_argument("custom", {"only_param": "value"}) == "value"
        assert _extract_key_argument("custom", {"a": 1, "b": 2}) is None

    def test_estimate_ratio(self):
        msgs = [{"content": "x" * 1000}]
        ratio = _estimate_ratio(msgs, "sys", 1000, chars_per_token=1.0)
        # (1000 + 3) / 1.0 / 1000 ≈ 1.003
        assert ratio > 0.9

    def test_estimate_ratio_zero_max(self):
        assert _estimate_ratio([], "", 0) == 0.0


# ========== ObservationMaskingStrategy 测试 ==========


class TestObservationMasking:
    def test_should_trigger_by_ratio(self):
        strategy = ObservationMaskingStrategy(trigger_ratio=0.6)
        ctx = _make_ctx(current_turn=5, max_tokens=5000)
        # Large history → high ratio → should trigger
        if ctx.token_ratio >= 0.6:
            assert strategy.should_trigger(ctx)

    def test_should_trigger_by_turns_no_token_limit(self):
        strategy = ObservationMaskingStrategy(keep_recent=3)
        ctx = _make_ctx(current_turn=5, max_tokens=0)
        assert strategy.should_trigger(ctx)  # 5 > 3

    def test_no_trigger_when_few_turns(self):
        strategy = ObservationMaskingStrategy(keep_recent=3)
        ctx = _make_ctx(current_turn=2, max_tokens=0)
        assert not strategy.should_trigger(ctx)  # 2 <= 3

    def test_apply_by_turns(self):
        strategy = ObservationMaskingStrategy(keep_recent=2)
        history = _make_history(5, content_size=500)
        ctx = _make_ctx(current_turn=5, max_tokens=0, history=history)

        original_len = len(history)
        result = strategy.apply(history, ctx)

        assert result.action_label == "observation_masking"
        assert result.messages_affected > 0
        assert len(history) == original_len  # in-place, no removal

    def test_apply_with_call_registry(self):
        """有 call_registry 时生成详细摘要"""
        strategy = ObservationMaskingStrategy(keep_recent=2)
        history = _make_history(5, content_size=500)

        records = [
            ToolCallRecord(
                tool_name="web_search",
                arguments_hash="abc",
                arguments={"query": "test query"},
                turn=1,
                result_hash="def",
                result_brief="Some results...",
                result_full='{"results": [1, 2, 3]}',
                result_chars=100,
            ),
        ]

        ctx = _make_ctx(current_turn=5, max_tokens=0, history=history, call_registry=records)
        result = strategy.apply(history, ctx)
        assert result.messages_affected > 0

    def test_preserves_system_messages(self):
        """系统消息不被压缩"""
        strategy = ObservationMaskingStrategy(keep_recent=1)
        history = [
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {"role": "assistant", "content": "resp 1"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "[REFLECTION CHECKPOINT] Think carefully..."}],
            },
            {"role": "assistant", "content": "resp 2"},
            {"role": "user", "content": [{"type": "text", "text": "x" * 500}]},
        ]
        ctx = _make_ctx(current_turn=3, max_tokens=0, history=history)
        strategy.apply(history, ctx)

        # REFLECTION message should be preserved
        for msg in history:
            content = msg.get("content", "")
            if isinstance(content, list):
                text = content[0].get("text", "") if content else ""
            else:
                text = str(content)
            if "[REFLECTION" in text:
                assert len(text) > 30  # not compacted


# ========== LLMSummarizeStrategy 测试 ==========


class TestLLMSummarize:
    def test_should_trigger(self):
        strategy = LLMSummarizeStrategy(trigger_ratio=0.8)
        ctx = _make_ctx(current_turn=5, max_tokens=100)  # very small max → high ratio
        if ctx.token_ratio >= 0.8:
            assert strategy.should_trigger(ctx)

    def test_no_trigger_without_token_limit(self):
        strategy = LLMSummarizeStrategy(trigger_ratio=0.8)
        ctx = _make_ctx(current_turn=5, max_tokens=0)
        assert not strategy.should_trigger(ctx)

    def test_sync_apply_returns_marker(self):
        strategy = LLMSummarizeStrategy()
        ctx = _make_ctx()
        result = strategy.apply([], ctx)
        assert result.action_label == "need_summarize"

    @pytest.mark.asyncio
    async def test_async_apply(self):
        strategy = LLMSummarizeStrategy(trigger_ratio=0.5, keep_recent=2)
        history = _make_history(5, content_size=300)
        ctx = _make_ctx(current_turn=5, max_tokens=5000, history=history)

        async def mock_llm(system_prompt, messages, purpose):
            return "Summary: Found key data about topics A, B, C."

        original_count = len(history)
        result = await strategy.apply_async(history, ctx, mock_llm)

        assert result.action_label == "llm_summarize"
        assert result.messages_affected > 0
        assert result.summary_text != ""
        assert len(history) < original_count

        # Summary should be at index 1
        content = history[1]["content"]
        if isinstance(content, list):
            text = content[0]["text"]
        else:
            text = content
        assert "[CONTEXT SUMMARY" in text

    @pytest.mark.asyncio
    async def test_async_apply_empty_response(self):
        strategy = LLMSummarizeStrategy(trigger_ratio=0.5, keep_recent=2)
        history = _make_history(5, content_size=300)
        ctx = _make_ctx(current_turn=5, max_tokens=5000, history=history)

        async def mock_llm(system_prompt, messages, purpose):
            return ""

        original_count = len(history)
        result = await strategy.apply_async(history, ctx, mock_llm)
        assert result.messages_affected == 0
        assert len(history) == original_count

    @pytest.mark.asyncio
    async def test_async_apply_llm_error(self):
        strategy = LLMSummarizeStrategy(trigger_ratio=0.5, keep_recent=2)
        history = _make_history(5, content_size=300)
        ctx = _make_ctx(current_turn=5, max_tokens=5000, history=history)

        async def mock_llm(system_prompt, messages, purpose):
            raise RuntimeError("LLM timeout")

        original_count = len(history)
        result = await strategy.apply_async(history, ctx, mock_llm)
        assert result.messages_affected == 0
        assert len(history) == original_count

    def test_reset(self):
        strategy = LLMSummarizeStrategy()
        strategy._summary_text = "some summary"
        strategy._summarized_up_to_turn = 5
        strategy.reset()
        assert strategy._summary_text is None
        assert strategy._summarized_up_to_turn == 0


# ========== BinaryReductionStrategy 测试 ==========


class TestBinaryReduction:
    def test_should_trigger(self):
        strategy = BinaryReductionStrategy(trigger_ratio=0.95)
        ctx = _make_ctx(current_turn=5, max_tokens=100)
        if ctx.token_ratio >= 0.95:
            assert strategy.should_trigger(ctx)

    def test_apply_reduces_messages(self):
        strategy = BinaryReductionStrategy(keep_tail=2)
        history = _make_history(10, content_size=200)
        ctx = _make_ctx(current_turn=10, max_tokens=100, history=history)

        original_count = len(history)
        result = strategy.apply(history, ctx)

        assert result.action_label == "binary_reduction"
        assert result.messages_affected > 0
        assert len(history) < original_count

    def test_no_reduction_on_small_history(self):
        strategy = BinaryReductionStrategy(keep_tail=2)
        history = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "resp"},
            {"role": "user", "content": "q"},
        ]
        ctx = _make_ctx(current_turn=1, max_tokens=100, history=history)

        result = strategy.apply(history, ctx)
        assert result.messages_affected == 0
        assert len(history) == 3


# ========== WindowStrategyPipeline 测试 ==========


class TestPipeline:
    def test_default_strategies(self):
        pipeline = WindowStrategyPipeline()
        assert len(pipeline.strategies) == 4
        assert isinstance(pipeline.strategies[0], ObservationMaskingStrategy)
        assert pipeline.strategies[1].strategy_type == "session_memory_compact"
        assert isinstance(pipeline.strategies[2], LLMSummarizeStrategy)
        assert isinstance(pipeline.strategies[3], BinaryReductionStrategy)

    def test_manage_returns_none_when_low(self):
        pipeline = WindowStrategyPipeline()
        history = _make_history(2, content_size=50)
        ctx = _make_ctx(current_turn=2, max_tokens=100000, history=history)
        action = pipeline.manage(history, ctx)
        assert action == "none"

    def test_manage_triggers_masking(self):
        pipeline = WindowStrategyPipeline(
            [
                ObservationMaskingStrategy(trigger_ratio=0.3, keep_recent=1),
            ]
        )
        history = _make_history(5, content_size=500)
        ctx = _make_ctx(current_turn=5, max_tokens=3000, history=history)

        if ctx.token_ratio >= 0.3:
            action = pipeline.manage(history, ctx)
            assert action == "observation_masking"

    def test_manage_returns_need_summarize(self):
        """LLMSummarize 单独使用时，ratio 高于阈值返回 need_summarize"""
        pipeline = WindowStrategyPipeline(
            [
                LLMSummarizeStrategy(trigger_ratio=0.5),
            ]
        )
        history = _make_history(10, content_size=1000)
        ctx = _make_ctx(current_turn=10, max_tokens=3000, history=history)

        if ctx.token_ratio >= 0.5:
            action = pipeline.manage(history, ctx)
            assert action == "need_summarize"

    @pytest.mark.asyncio
    async def test_apply_summarize(self):
        pipeline = WindowStrategyPipeline()
        history = _make_history(5, content_size=300)
        ctx = _make_ctx(current_turn=5, max_tokens=5000, history=history)

        async def mock_llm(system_prompt, messages, purpose):
            return "Summary of research findings."

        result = await pipeline.apply_summarize(history, ctx, mock_llm)
        assert isinstance(result, bool)

    def test_apply_emergency(self):
        """紧急模式：masking 不足时 binary reduction 会删除消息"""
        pipeline = WindowStrategyPipeline(
            [
                BinaryReductionStrategy(trigger_ratio=0.95),
            ]
        )
        history = _make_history(10, content_size=500)
        ctx = _make_ctx(current_turn=10, max_tokens=1000, history=history)

        original_count = len(history)
        affected = pipeline.apply_emergency(history, ctx)
        # 没有 ObservationMasking 时直接走 BinaryReduction，会删除消息
        assert affected > 0
        assert len(history) < original_count

    def test_reset(self):
        pipeline = WindowStrategyPipeline()
        # Modify internal state of LLMSummarizeStrategy
        for s in pipeline.strategies:
            if isinstance(s, LLMSummarizeStrategy):
                s._summary_text = "test"
                s._summarized_up_to_turn = 5

        pipeline.reset()

        for s in pipeline.strategies:
            if isinstance(s, LLMSummarizeStrategy):
                assert s._summary_text is None
                assert s._summarized_up_to_turn == 0


# ========== 自定义策略测试 ==========


class TestCustomStrategy:
    def test_custom_strategy_in_pipeline(self):
        """验证自定义策略可以注入管道"""

        class AlwaysTriggerStrategy(WindowStrategy):
            def should_trigger(self, ctx: WindowContext) -> bool:
                return True

            def apply(self, messages: list, ctx: WindowContext) -> CompressResult:
                # 简单地截断所有消息为 10 字符
                affected = 0
                for msg in messages[1:]:
                    content = msg.get("content", "")
                    if isinstance(content, str) and len(content) > 10:
                        msg["content"] = content[:10]
                        affected += 1
                return CompressResult(messages_affected=affected, action_label="custom_truncate")

        pipeline = WindowStrategyPipeline([AlwaysTriggerStrategy()])
        history = _make_history(3, content_size=100)
        ctx = _make_ctx(current_turn=3, max_tokens=0, history=history)

        action = pipeline.manage(history, ctx)
        assert action == "custom_truncate"


# ========== ContextManager 集成测试（策略管道注入） ==========


# ========== Circuit Breaker 测试 ==========


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_trips_after_consecutive_failures(self):
        """连续 3 次 apply_summarize 失败后熔断"""
        pipeline = WindowStrategyPipeline()
        history = _make_history(5, content_size=300)
        ctx = _make_ctx(current_turn=5, max_tokens=5000, history=history)

        async def failing_llm(system_prompt, messages, purpose):
            return ""  # empty response = failure

        for _ in range(3):
            await pipeline.apply_summarize(list(history), ctx, failing_llm)

        assert pipeline._summarize_fused is True
        assert pipeline._summarize_consecutive_failures >= 3

    def test_circuit_breaker_skips_summarize_in_manage(self):
        """熔断后 manage() 跳过 LLMSummarize"""
        pipeline = WindowStrategyPipeline([
            LLMSummarizeStrategy(trigger_ratio=0.1),  # low threshold to ensure trigger
        ])
        pipeline._summarize_fused = True

        history = _make_history(5, content_size=300)
        ctx = _make_ctx(current_turn=5, max_tokens=500, history=history)

        action = pipeline.manage(history, ctx)
        # Should not return "need_summarize" because circuit breaker is active
        assert action != "need_summarize"

    @pytest.mark.asyncio
    async def test_successful_summarize_resets_counter(self):
        """成功的 summarize 重置失败计数器"""
        pipeline = WindowStrategyPipeline()
        pipeline._summarize_consecutive_failures = 2  # close to tripping

        history = _make_history(5, content_size=300)
        ctx = _make_ctx(current_turn=5, max_tokens=5000, history=history)

        async def success_llm(system_prompt, messages, purpose):
            return "Summary: key findings about A, B, C."

        result = await pipeline.apply_summarize(history, ctx, success_llm)
        assert result is True
        assert pipeline._summarize_consecutive_failures == 0
        assert pipeline._summarize_fused is False

    def test_reset_clears_circuit_breaker(self):
        """reset() 清除熔断状态"""
        pipeline = WindowStrategyPipeline()
        pipeline._summarize_consecutive_failures = 3
        pipeline._summarize_fused = True

        pipeline.reset()

        assert pipeline._summarize_consecutive_failures == 0
        assert pipeline._summarize_fused is False

    @pytest.mark.asyncio
    async def test_fused_apply_summarize_returns_false(self):
        """熔断后 apply_summarize 直接返回 False"""
        pipeline = WindowStrategyPipeline()
        pipeline._summarize_fused = True

        history = _make_history(3, content_size=100)
        ctx = _make_ctx(current_turn=3, max_tokens=5000, history=history)

        async def should_not_be_called(system_prompt, messages, purpose):
            raise AssertionError("LLM should not be called when fused")

        result = await pipeline.apply_summarize(history, ctx, should_not_be_called)
        assert result is False


# ========== ContextManager 集成测试（策略管道注入） ==========


class TestContextManagerPipelineIntegration:
    def test_custom_pipeline_injection(self):
        """验证 ContextManager 接受自定义 pipeline"""
        from mem_deep_research_core.core.context_manager import ContextManager

        custom_pipeline = WindowStrategyPipeline(
            [
                ObservationMaskingStrategy(trigger_ratio=0.3, keep_recent=1),
            ]
        )
        cm = ContextManager(pipeline=custom_pipeline)
        assert cm.pipeline is custom_pipeline
        assert len(cm.pipeline.strategies) == 1

    def test_pipeline_from_config(self):
        """验证从 config 自动构建 pipeline"""
        from mem_deep_research_core.core.context_manager import ContextManager, ContextManagerConfig

        config = ContextManagerConfig(
            compact_at_ratio=0.5,
            compact_keep_recent=2,
            summarize_at_ratio=0.7,
        )
        cm = ContextManager(config=config)

        # Should have 3 strategies
        assert len(cm.pipeline.strategies) == 3

        # First strategy should have custom ratio
        masking = cm.pipeline.strategies[0]
        assert isinstance(masking, ObservationMaskingStrategy)
        assert masking.trigger_ratio == 0.5
        assert masking.keep_recent == 2

        # Second strategy should have custom ratio
        summarize = cm.pipeline.strategies[1]
        assert isinstance(summarize, LLMSummarizeStrategy)
        assert summarize.trigger_ratio == 0.7

    def test_compact_disabled_no_masking_strategy(self):
        """enable_compact=False 时不应包含 ObservationMaskingStrategy"""
        from mem_deep_research_core.core.context_manager import ContextManager, ContextManagerConfig

        config = ContextManagerConfig(enable_compact=False)
        cm = ContextManager(config=config)

        has_masking = any(isinstance(s, ObservationMaskingStrategy) for s in cm.pipeline.strategies)
        assert not has_masking

    def test_pipeline_setter(self):
        """验证可以运行时替换 pipeline"""
        from mem_deep_research_core.core.context_manager import ContextManager

        cm = ContextManager()
        new_pipeline = WindowStrategyPipeline([BinaryReductionStrategy()])
        cm.pipeline = new_pipeline
        assert cm.pipeline is new_pipeline
        assert len(cm.pipeline.strategies) == 1
