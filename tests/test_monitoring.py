"""执行监控单元测试

覆盖: TurnCounter, ExecutionMonitor（纯逻辑，无需 mock LLM）
包括升级策略（escalation policy）测试
"""

import time

import pytest

from mem_deep_research_core.core.monitoring import (
    EscalationAction,
    ExecutionMonitor,
    MonitoringConfig,
    MonitoringState,
    TurnCounter,
)

# ========== TurnCounter 测试 ==========


class TestTurnCounter:
    def test_increment(self):
        """计数递增"""
        tc = TurnCounter(max_turns=10)
        assert tc.increment() == 1
        assert tc.increment() == 2
        assert tc.current_turn == 2

    def test_max_reached(self):
        """最大值检测"""
        tc = TurnCounter(max_turns=2)
        tc.increment()
        assert not tc.is_max_reached()
        tc.increment()
        assert tc.is_max_reached()

    def test_reflection_disabled(self):
        """反思禁用时不触发"""
        tc = TurnCounter(max_turns=20, reflection_enabled=False, reflection_interval=5)
        for _ in range(5):
            tc.increment()
        assert not tc.should_inject_reflection()

    def test_reflection_interval(self):
        """反思间隔触发"""
        tc = TurnCounter(max_turns=20, reflection_enabled=True, reflection_interval=3)
        tc.increment()  # 1
        assert not tc.should_inject_reflection()
        tc.increment()  # 2
        assert not tc.should_inject_reflection()
        tc.increment()  # 3
        assert tc.should_inject_reflection()
        tc.increment()  # 4
        assert not tc.should_inject_reflection()
        tc.increment()  # 5
        assert not tc.should_inject_reflection()
        tc.increment()  # 6
        assert tc.should_inject_reflection()

    def test_progress_percentage(self):
        """进度百分比"""
        tc = TurnCounter(max_turns=10)
        assert tc.get_progress_percentage() == 0.0
        tc.increment()  # 1/10
        assert tc.get_progress_percentage() == pytest.approx(10.0)
        for _ in range(9):
            tc.increment()  # 10/10
        assert tc.get_progress_percentage() == pytest.approx(100.0)

    def test_progress_zero_max(self):
        """max_turns=0 时进度为 0"""
        tc = TurnCounter(max_turns=0)
        assert tc.get_progress_percentage() == 0.0


# ========== ExecutionMonitor 测试 ==========


class TestExecutionMonitor:
    def test_record_empty_response_threshold(self):
        """连续空响应达到阈值"""
        monitor = ExecutionMonitor(config=MonitoringConfig(max_consecutive_empty_turns=3))
        assert not monitor.record_empty_response()  # 1
        assert not monitor.record_empty_response()  # 2
        assert monitor.record_empty_response()  # 3 → True

    def test_record_progress_resets_empty_count(self):
        """record_progress 重置空响应计数"""
        monitor = ExecutionMonitor(config=MonitoringConfig(max_consecutive_empty_turns=3))
        monitor.record_empty_response()  # 1
        monitor.record_empty_response()  # 2
        monitor.record_progress("some response")
        assert monitor.state.consecutive_empty_turns == 0

    def test_loop_detection(self):
        """重复响应检测"""
        monitor = ExecutionMonitor(config=MonitoringConfig(enable_loop_detection=True))
        # 第一次不算重复
        assert monitor.record_progress("same text response") == EscalationAction.NONE
        # 相同文本第二次 → 检测到循环 (WARN)
        assert monitor.record_progress("same text response") == EscalationAction.WARN

    def test_loop_detection_different_text(self):
        """不同文本不触发循环检测"""
        monitor = ExecutionMonitor(config=MonitoringConfig(enable_loop_detection=True))
        assert monitor.record_progress("response A") == EscalationAction.NONE
        assert monitor.record_progress("response B") == EscalationAction.NONE

    def test_loop_detection_disabled(self):
        """禁用循环检测"""
        monitor = ExecutionMonitor(config=MonitoringConfig(enable_loop_detection=False))
        assert monitor.record_progress("same") == EscalationAction.NONE
        assert monitor.record_progress("same") == EscalationAction.NONE

    def test_reset(self):
        """重置监控状态"""
        monitor = ExecutionMonitor()
        monitor.record_empty_response()
        monitor.record_progress("text")
        monitor.reset()
        assert monitor.state.consecutive_empty_turns == 0
        assert monitor.state.last_response_hash is None
        assert monitor.state.response_loop_escalation_count == 0
        assert monitor.state.response_hash_window == []
        assert monitor.state.tool_loop_retry_count == 0
        assert monitor.state.stall_warned is False
        assert monitor.last_loop_action == EscalationAction.NONE

    @pytest.mark.asyncio
    async def test_check_timeout(self):
        """超时检测"""
        monitor = ExecutionMonitor(
            config=MonitoringConfig(max_total_time=0.01)  # 10ms
        )
        time.sleep(0.02)
        assert await monitor.check_timeout()

    @pytest.mark.asyncio
    async def test_no_timeout(self):
        """未超时"""
        monitor = ExecutionMonitor(config=MonitoringConfig(max_total_time=60))
        assert not await monitor.check_timeout()

    def test_get_status_summary(self):
        """状态摘要包含必要字段"""
        monitor = ExecutionMonitor()
        summary = monitor.get_status_summary()
        assert "elapsed_time" in summary
        assert "time_since_progress" in summary
        assert "consecutive_empty_turns" in summary
        assert "timeout_threshold" in summary
        assert "stall_threshold" in summary


# ========== EscalationAction 测试 ==========


class TestEscalationAction:
    def test_enum_values(self):
        """枚举值正确"""
        assert EscalationAction.NONE == "none"
        assert EscalationAction.WARN == "warn"
        assert EscalationAction.INJECT_HINT == "inject_hint"
        assert EscalationAction.TERMINATE == "terminate"

    def test_enum_is_str(self):
        """枚举继承 str"""
        assert isinstance(EscalationAction.NONE, str)
        assert isinstance(EscalationAction.TERMINATE, str)


# ========== 响应循环升级策略测试 ==========


class TestResponseLoopEscalation:
    def test_first_repeat_warns(self):
        """第 1 次重复 → WARN"""
        monitor = ExecutionMonitor(config=MonitoringConfig(loop_escalation_terminate_threshold=3))
        monitor.record_progress("same text")
        action = monitor.record_progress("same text")
        assert action == EscalationAction.WARN

    def test_second_repeat_injects_hint(self):
        """第 2 次重复 → INJECT_HINT"""
        monitor = ExecutionMonitor(config=MonitoringConfig(loop_escalation_terminate_threshold=3))
        monitor.record_progress("same text")
        monitor.record_progress("same text")  # 1st → WARN
        action = monitor.record_progress("same text")  # 2nd → INJECT_HINT
        assert action == EscalationAction.INJECT_HINT

    def test_third_repeat_terminates(self):
        """第 3 次重复 → TERMINATE"""
        monitor = ExecutionMonitor(config=MonitoringConfig(loop_escalation_terminate_threshold=3))
        monitor.record_progress("same text")
        monitor.record_progress("same text")  # 1st
        monitor.record_progress("same text")  # 2nd
        action = monitor.record_progress("same text")  # 3rd → TERMINATE
        assert action == EscalationAction.TERMINATE

    def test_new_response_resets_escalation(self):
        """新响应重置升级计数"""
        monitor = ExecutionMonitor(config=MonitoringConfig(loop_escalation_terminate_threshold=3))
        monitor.record_progress("same text")
        monitor.record_progress("same text")  # escalation_count = 1
        assert monitor.state.response_loop_escalation_count == 1

        monitor.record_progress("different text")  # reset
        assert monitor.state.response_loop_escalation_count == 0

        # 新的重复从 WARN 重新开始
        monitor.record_progress("another text")
        action = monitor.record_progress("another text")
        assert action == EscalationAction.WARN

    def test_custom_terminate_threshold(self):
        """自定义终止阈值"""
        monitor = ExecutionMonitor(config=MonitoringConfig(loop_escalation_terminate_threshold=2))
        monitor.record_progress("same")
        monitor.record_progress("same")  # 1st → INJECT_HINT (threshold-1=1)
        action = monitor.record_progress("same")  # 2nd → TERMINATE
        assert action == EscalationAction.TERMINATE

    def test_last_loop_action_property(self):
        """last_loop_action 属性正确暴露"""
        monitor = ExecutionMonitor()
        assert monitor.last_loop_action == EscalationAction.NONE

        monitor.record_progress("text")
        monitor.record_progress("text")  # triggers WARN
        assert monitor.last_loop_action == EscalationAction.WARN

        monitor.record_progress("new text")  # reset
        assert monitor.last_loop_action == EscalationAction.NONE


# ========== 振荡检测测试 ==========


class TestOscillationDetection:
    def test_ab_ab_oscillation_detected(self):
        """A-B-A-B 交替模式检测"""
        monitor = ExecutionMonitor(
            config=MonitoringConfig(
                response_hash_window_size=8,
                response_hash_repeat_threshold=3,
            )
        )
        # A-B-A-B-A 模式: A 出现 3 次触发振荡检测
        monitor.record_progress("response A")  # A: window=[A]
        monitor.record_progress("response B")  # B: window=[A,B]
        monitor.record_progress("response A")  # A: window=[A,B,A]
        monitor.record_progress("response B")  # B: window=[A,B,A,B]
        action = monitor.record_progress("response A")  # A: window=[A,B,A,B,A], A出现3次
        assert action != EscalationAction.NONE

    def test_diverse_responses_no_false_positive(self):
        """多样响应不误报"""
        monitor = ExecutionMonitor(
            config=MonitoringConfig(
                response_hash_window_size=8,
                response_hash_repeat_threshold=3,
            )
        )
        for i in range(10):
            action = monitor.record_progress(f"unique response {i}")
            assert action == EscalationAction.NONE

    def test_window_slides(self):
        """滑动窗口正确滑动"""
        monitor = ExecutionMonitor(
            config=MonitoringConfig(
                response_hash_window_size=4,
                response_hash_repeat_threshold=3,
            )
        )
        # 先填满窗口让 A 出现 2 次
        monitor.record_progress("response A")
        monitor.record_progress("response B")
        monitor.record_progress("response A")
        monitor.record_progress("response C")
        # 窗口: [A, B, A, C], A 出现 2 次 < 3

        # 添加 D，窗口变为 [B, A, C, D]，A 只出现 1 次
        monitor.record_progress("response D")

        # 添加 E，窗口变为 [A, C, D, E]，A 只出现 1 次
        monitor.record_progress("response E")
        assert monitor.state.response_loop_escalation_count == 0


# ========== 工具循环升级策略测试 ==========


class TestToolLoopEscalation:
    def test_first_warning_returns_warn(self):
        """第 1 次工具循环 → WARN"""
        monitor = ExecutionMonitor(config=MonitoringConfig(max_tool_loop_retries=2))
        action = monitor.record_tool_loop_warning()
        assert action == EscalationAction.WARN

    def test_second_warning_returns_warn(self):
        """第 2 次工具循环 → WARN"""
        monitor = ExecutionMonitor(config=MonitoringConfig(max_tool_loop_retries=2))
        monitor.record_tool_loop_warning()  # 1
        action = monitor.record_tool_loop_warning()  # 2
        assert action == EscalationAction.WARN

    def test_exceeds_limit_terminates(self):
        """超过限制 → TERMINATE"""
        monitor = ExecutionMonitor(config=MonitoringConfig(max_tool_loop_retries=2))
        monitor.record_tool_loop_warning()  # 1
        monitor.record_tool_loop_warning()  # 2
        action = monitor.record_tool_loop_warning()  # 3 > 2 → TERMINATE
        assert action == EscalationAction.TERMINATE

    def test_reset_counter(self):
        """重置计数器"""
        monitor = ExecutionMonitor(config=MonitoringConfig(max_tool_loop_retries=2))
        monitor.record_tool_loop_warning()
        monitor.record_tool_loop_warning()
        monitor.reset_tool_loop_counter()
        assert monitor.state.tool_loop_retry_count == 0

        # 重置后重新计数
        action = monitor.record_tool_loop_warning()
        assert action == EscalationAction.WARN


# ========== 停滞升级策略测试 ==========


class TestStallEscalation:
    @pytest.mark.asyncio
    async def test_stall_warn(self):
        """超过阈值 → WARN"""
        monitor = ExecutionMonitor(
            config=MonitoringConfig(
                stall_detection_threshold=0.01,
                stall_terminate_multiplier=2.0,
            )
        )
        time.sleep(0.015)  # > 0.01 but < 0.02
        action = await monitor.check_stall()
        assert action == EscalationAction.WARN

    @pytest.mark.asyncio
    async def test_stall_terminate(self):
        """超过 threshold × multiplier → TERMINATE"""
        monitor = ExecutionMonitor(
            config=MonitoringConfig(
                stall_detection_threshold=0.01,
                stall_terminate_multiplier=2.0,
            )
        )
        time.sleep(0.025)  # > 0.02
        action = await monitor.check_stall()
        assert action == EscalationAction.TERMINATE

    @pytest.mark.asyncio
    async def test_stall_no_duplicate_warn(self):
        """不重复告警"""
        monitor = ExecutionMonitor(
            config=MonitoringConfig(
                stall_detection_threshold=0.01,
                stall_terminate_multiplier=100.0,  # 设高以防触发 TERMINATE
            )
        )
        time.sleep(0.015)
        action1 = await monitor.check_stall()
        assert action1 == EscalationAction.WARN
        assert monitor.state.stall_warned is True

        # 第二次调用不再触发回调（但仍返回 WARN）
        action2 = await monitor.check_stall()
        assert action2 == EscalationAction.WARN

    @pytest.mark.asyncio
    async def test_no_stall(self):
        """无停滞"""
        monitor = ExecutionMonitor(config=MonitoringConfig(stall_detection_threshold=60.0))
        action = await monitor.check_stall()
        assert action == EscalationAction.NONE

    @pytest.mark.asyncio
    async def test_pre_turn_check_stall_terminate(self):
        """pre_turn_check 在停滞终止时返回终止原因"""
        monitor = ExecutionMonitor(
            config=MonitoringConfig(
                max_total_time=600.0,
                stall_detection_threshold=0.01,
                stall_terminate_multiplier=2.0,
            )
        )
        time.sleep(0.025)
        result = await monitor.pre_turn_check()
        assert result == "stall_terminated"


# ========== post_turn_check 集成测试 ==========


class TestPostTurnCheckIntegration:
    @pytest.mark.asyncio
    async def test_three_repeats_terminates(self):
        """3 次重复后 post_turn_check 返回 response_loop_terminated"""
        monitor = ExecutionMonitor(config=MonitoringConfig(loop_escalation_terminate_threshold=3))
        # 第 1 次：正常
        result = await monitor.post_turn_check("same text", llm_call_failed=False)
        assert result is None

        # 第 2 次：重复 → WARN
        result = await monitor.post_turn_check("same text", llm_call_failed=False)
        assert result is None

        # 第 3 次：重复 → INJECT_HINT
        result = await monitor.post_turn_check("same text", llm_call_failed=False)
        assert result is None

        # 第 4 次：重复 → TERMINATE
        result = await monitor.post_turn_check("same text", llm_call_failed=False)
        assert result == "response_loop_terminated"

    @pytest.mark.asyncio
    async def test_no_terminate_on_different_responses(self):
        """不同响应不终止"""
        monitor = ExecutionMonitor()
        for i in range(10):
            result = await monitor.post_turn_check(f"response {i}", llm_call_failed=False)
            assert result is None

    @pytest.mark.asyncio
    async def test_empty_response_still_works(self):
        """空响应处理不受影响"""
        monitor = ExecutionMonitor(config=MonitoringConfig(max_consecutive_empty_turns=2))
        result = await monitor.post_turn_check(None, llm_call_failed=True)
        assert result is None
        result = await monitor.post_turn_check(None, llm_call_failed=True)
        assert result == "consecutive_empty_responses"

    @pytest.mark.asyncio
    async def test_inject_hint_action_exposed(self):
        """INJECT_HINT 动作通过 last_loop_action 暴露"""
        monitor = ExecutionMonitor(config=MonitoringConfig(loop_escalation_terminate_threshold=3))
        await monitor.post_turn_check("same", llm_call_failed=False)
        assert monitor.last_loop_action == EscalationAction.NONE

        await monitor.post_turn_check("same", llm_call_failed=False)  # WARN
        assert monitor.last_loop_action == EscalationAction.WARN

        await monitor.post_turn_check("same", llm_call_failed=False)  # INJECT_HINT
        assert monitor.last_loop_action == EscalationAction.INJECT_HINT


# ========== 向后兼容性测试 ==========


class TestBackwardCompatibility:
    def test_default_config_values(self):
        """默认配置值向后兼容"""
        config = MonitoringConfig()
        assert config.stall_detection_threshold == 120.0
        assert config.max_total_time == 600.0
        assert config.max_consecutive_empty_turns == 3
        assert config.enable_loop_detection is True
        assert config.loop_detection_text_length == 500
        # 新增字段有默认值
        assert config.loop_escalation_terminate_threshold == 3
        assert config.response_hash_window_size == 8
        assert config.response_hash_repeat_threshold == 3
        assert config.max_tool_loop_retries == 2
        assert config.stall_terminate_multiplier == 2.0

    def test_default_state_values(self):
        """默认状态值向后兼容"""
        state = MonitoringState()
        assert state.consecutive_empty_turns == 0
        assert state.last_response_hash is None
        # 新增字段有默认值
        assert state.response_loop_escalation_count == 0
        assert state.response_hash_window == []
        assert state.tool_loop_retry_count == 0
        assert state.stall_warned is False
        assert state.attempted_strategies == []

    def test_get_loop_break_hint_backward_compat(self):
        """get_loop_break_hint 无参数调用仍正常（向后兼容）"""
        monitor = ExecutionMonitor()
        hint = monitor.get_loop_break_hint()
        assert isinstance(hint, str)
        assert len(hint) > 0
        assert "MANDATORY" in hint


# ========== get_loop_break_hint 测试 ==========


class TestGetLoopBreakHint:
    def test_contains_mandatory_forbidden_required(self):
        """输出包含 MANDATORY/FORBIDDEN/REQUIRED 关键字"""
        monitor = ExecutionMonitor()
        hint = monitor.get_loop_break_hint()
        assert "MANDATORY" in hint
        assert "FORBIDDEN" in hint
        assert "REQUIRED" in hint

    def test_includes_attempted_strategies(self):
        """包含已尝试策略上下文"""
        monitor = ExecutionMonitor()
        monitor.record_strategy_summary("web_search(query='test')")
        monitor.record_strategy_summary("fetch_page(url='http://x.com')")
        hint = monitor.get_loop_break_hint()
        assert "web_search(query='test')" in hint
        assert "fetch_page(url='http://x.com')" in hint
        assert "ATTEMPTED STRATEGIES" in hint

    def test_includes_recent_tool_names(self):
        """包含最近工具名称"""
        monitor = ExecutionMonitor()
        hint = monitor.get_loop_break_hint(recent_tool_names=["web_search", "fetch_page"])
        assert "web_search" in hint
        assert "fetch_page" in hint
        assert "RECENT TOOLS" in hint

    def test_no_strategies_no_tools(self):
        """无策略无工具时仍有 MANDATORY/FORBIDDEN/REQUIRED"""
        monitor = ExecutionMonitor()
        hint = monitor.get_loop_break_hint()
        assert "MANDATORY" in hint
        assert "FORBIDDEN" in hint
        assert "REQUIRED" in hint
        assert "ATTEMPTED STRATEGIES" not in hint
        assert "RECENT TOOLS" not in hint

    def test_no_tools_section_when_empty(self):
        """工具为空列表时不包含工具段"""
        monitor = ExecutionMonitor()
        hint = monitor.get_loop_break_hint(recent_tool_names=[])
        assert "RECENT TOOLS" not in hint

    def test_options_abc(self):
        """包含 A/B/C 三个选项"""
        monitor = ExecutionMonitor()
        hint = monitor.get_loop_break_hint()
        assert "A)" in hint
        assert "B)" in hint
        assert "C)" in hint


# ========== record_strategy_summary 测试 ==========


class TestRecordStrategySummary:
    def test_record_and_retrieve(self):
        """记录策略并可见于 state"""
        monitor = ExecutionMonitor()
        monitor.record_strategy_summary("tool_a(x=1)")
        assert "tool_a(x=1)" in monitor.state.attempted_strategies

    def test_dedup(self):
        """相同策略不重复记录"""
        monitor = ExecutionMonitor()
        monitor.record_strategy_summary("tool_a(x=1)")
        monitor.record_strategy_summary("tool_a(x=1)")
        monitor.record_strategy_summary("tool_a(x=1)")
        assert monitor.state.attempted_strategies.count("tool_a(x=1)") == 1

    def test_cap_at_10(self):
        """上限 10 条"""
        monitor = ExecutionMonitor()
        for i in range(15):
            monitor.record_strategy_summary(f"strategy_{i}")
        assert len(monitor.state.attempted_strategies) == 10
        # 保留最后 10 条
        assert monitor.state.attempted_strategies[0] == "strategy_5"
        assert monitor.state.attempted_strategies[-1] == "strategy_14"

    def test_reset_clears_strategies(self):
        """reset 清除 attempted_strategies"""
        monitor = ExecutionMonitor()
        monitor.record_strategy_summary("tool_a(x=1)")
        monitor.record_strategy_summary("tool_b(y=2)")
        monitor.reset()
        assert monitor.state.attempted_strategies == []

    def test_strategies_appear_in_hint(self):
        """记录的策略出现在 get_loop_break_hint 输出中"""
        monitor = ExecutionMonitor()
        monitor.record_strategy_summary("web_search(q='AI')")
        hint = monitor.get_loop_break_hint()
        assert "web_search(q='AI')" in hint
