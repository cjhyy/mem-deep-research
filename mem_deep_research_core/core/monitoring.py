"""
执行监控模块

处理 Agent 执行过程中的进度监控、停滞检测、超时控制和循环检测。
支持三级升级策略：WARN → INJECT_HINT → TERMINATE
"""

import hashlib
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger("mem_deep_research")


class EscalationAction(StrEnum):
    """升级动作枚举"""

    NONE = "none"
    WARN = "warn"
    INJECT_HINT = "inject_hint"
    TERMINATE = "terminate"


@dataclass
class MonitoringConfig:
    """监控配置

    可通过 MonitoringConfig.from_schema(schema) 从 config_schema.MonitoringConfigSchema 创建。
    """

    # 停滞检测阈值（秒）
    stall_detection_threshold: float = 120.0
    # 最大总运行时间（秒），0 表示不限时
    max_total_time: float = 1800.0
    # 软超时比例（到达此比例时注入 hint 而非终止）
    soft_timeout_ratio: float = 0.8
    # 连续空响应导致终止的阈值
    max_consecutive_empty_turns: int = 3
    # 是否启用重复响应检测
    enable_loop_detection: bool = True
    # 重复响应检测的文本截取长度
    loop_detection_text_length: int = 500
    # 响应循环：连续检测到 N 次后终止
    loop_escalation_terminate_threshold: int = 3
    # 滑动窗口大小（存最近 N 个 hash）
    response_hash_window_size: int = 8
    # 窗口内同一 hash 出现 ≥N 次判定为振荡
    response_hash_repeat_threshold: int = 3
    # 工具循环：注入警告 N 次后终止
    max_tool_loop_retries: int = 2
    # 停滞终止：超过 stall_detection_threshold × 此倍数则终止
    stall_terminate_multiplier: float = 2.0
    # 温度提升（响应循环升级时使用）
    temperature_boost: float = 0.3
    temperature_boost_cap: float = 1.0

    @classmethod
    def from_schema(cls, schema) -> "MonitoringConfig":
        """从 MonitoringConfigSchema (Pydantic model) 创建 MonitoringConfig"""
        return cls(
            stall_detection_threshold=schema.stall_detection_threshold,
            max_total_time=schema.max_total_time,
            soft_timeout_ratio=getattr(schema, "soft_timeout_ratio", 0.8),
            max_consecutive_empty_turns=schema.max_consecutive_empty_turns,
            enable_loop_detection=schema.enable_loop_detection,
            loop_detection_text_length=schema.loop_detection_text_length,
            loop_escalation_terminate_threshold=schema.loop_escalation_terminate_threshold,
            response_hash_window_size=schema.response_hash_window_size,
            response_hash_repeat_threshold=schema.response_hash_repeat_threshold,
            max_tool_loop_retries=schema.max_tool_loop_retries,
            stall_terminate_multiplier=schema.stall_terminate_multiplier,
            temperature_boost=schema.temperature_boost,
            temperature_boost_cap=schema.temperature_boost_cap,
        )


@dataclass
class MonitoringState:
    """监控状态"""

    # 循环开始时间
    loop_start_time: float = field(default_factory=time.time)
    # 最后有进展的时间
    last_progress_time: float = field(default_factory=time.time)
    # 连续空轮次计数
    consecutive_empty_turns: int = 0
    # 上次响应的哈希值（用于检测循环）
    last_response_hash: str | None = None
    # 响应循环升级计数
    response_loop_escalation_count: int = 0
    # 滑动窗口（存最近 N 个 hash）
    response_hash_window: list = field(default_factory=list)
    # 工具循环重试计数
    tool_loop_retry_count: int = 0
    # 停滞已告警标记
    stall_warned: bool = False
    # 已尝试策略摘要（用于强制指令）
    attempted_strategies: list = field(default_factory=list)


class ExecutionMonitor:
    """执行监控器"""

    def __init__(
        self,
        config: MonitoringConfig | None = None,
        stream_reasoning_callback: Callable | None = None,
    ):
        """
        初始化执行监控器

        Args:
            config: 监控配置
            stream_reasoning_callback: 流式推理输出回调，签名: (tool_name, action, details) -> None
        """
        self.config = config or MonitoringConfig()
        self.stream_reasoning_callback = stream_reasoning_callback
        self.state = MonitoringState()
        self._last_loop_action = EscalationAction.NONE
        self._soft_timeout_fired = False

    def reset(self):
        """重置监控状态"""
        self.state = MonitoringState()
        self._last_loop_action = EscalationAction.NONE
        self._soft_timeout_fired = False

    @property
    def last_loop_action(self) -> EscalationAction:
        """暴露最后一次循环检测的升级动作"""
        return self._last_loop_action

    def get_elapsed_time(self) -> float:
        """获取已运行时间（秒）"""
        return time.time() - self.state.loop_start_time

    def get_time_since_progress(self) -> float:
        """获取距离上次进展的时间（秒）"""
        return time.time() - self.state.last_progress_time

    async def check_timeout(self) -> str | None:
        """
        检查是否超时

        Returns:
            bool: True 表示已超时，应该终止
        """
        if self.config.max_total_time <= 0:
            return None  # Unlimited time

        elapsed = self.get_elapsed_time()
        soft_limit = self.config.max_total_time * self.config.soft_timeout_ratio

        if elapsed > self.config.max_total_time:
            logger.warning(f"[MONITOR] 硬超时，已运行 {elapsed:.0f}s，生成中间结论")
            if self.stream_reasoning_callback:
                await self.stream_reasoning_callback(
                    "monitor",
                    "TIMEOUT",
                    f"已超过 {self.config.max_total_time:.0f}s 时间限制，将基于当前进展生成结论",
                )
            return "hard_timeout"
        elif elapsed > soft_limit and not getattr(self, "_soft_timeout_fired", False):
            self._soft_timeout_fired = True
            remaining = int(self.config.max_total_time - elapsed)
            logger.info(f"[MONITOR] 软超时，已运行 {elapsed:.0f}s，剩余 {remaining}s")
            if self.stream_reasoning_callback:
                await self.stream_reasoning_callback(
                    "monitor",
                    "SOFT_TIMEOUT",
                    f"已运行 {elapsed:.0f}s，剩余约 {remaining}s。请尽快总结当前进展。",
                )
            return "soft_timeout"
        return None

    async def check_stall(self) -> EscalationAction:
        """
        检查是否停滞

        Returns:
            EscalationAction: 升级动作
        """
        time_since_progress = self.get_time_since_progress()
        terminate_threshold = (
            self.config.stall_detection_threshold * self.config.stall_terminate_multiplier
        )

        if time_since_progress > terminate_threshold:
            logger.warning(
                f"[MONITOR] 停滞超过终止阈值，{time_since_progress:.0f}s 无进展，强制终止"
            )
            if self.stream_reasoning_callback:
                await self.stream_reasoning_callback(
                    "monitor",
                    "STALL_TERMINATE",
                    f"思考停滞 {time_since_progress:.0f}s 超过终止阈值 {terminate_threshold:.0f}s，强制终止",
                )
            return EscalationAction.TERMINATE

        if time_since_progress > self.config.stall_detection_threshold:
            if not self.state.stall_warned:
                self.state.stall_warned = True
                logger.warning(f"[MONITOR] 检测到停滞，{time_since_progress:.0f}s 无进展")
                if self.stream_reasoning_callback:
                    await self.stream_reasoning_callback(
                        "monitor",
                        "STALL_WARNING",
                        f"检测到思考停滞 {time_since_progress:.0f}s，正在尝试恢复",
                    )
            return EscalationAction.WARN

        return EscalationAction.NONE

    def _determine_escalation_action(self, count: int) -> EscalationAction:
        """根据升级计数确定动作"""
        threshold = self.config.loop_escalation_terminate_threshold
        if count >= threshold:
            return EscalationAction.TERMINATE
        elif count >= threshold - 1:
            return EscalationAction.INJECT_HINT
        else:
            return EscalationAction.WARN

    def record_progress(self, response_text: str | None = None) -> EscalationAction:
        """
        记录有进展

        Args:
            response_text: LLM 响应文本（用于循环检测）

        Returns:
            EscalationAction: 升级动作
        """
        self.state.last_progress_time = time.time()
        self.state.consecutive_empty_turns = 0

        if not response_text or not self.config.enable_loop_detection:
            self._last_loop_action = EscalationAction.NONE
            return EscalationAction.NONE

        current_hash = hashlib.sha256(response_text.encode()).hexdigest()

        # 维护滑动窗口
        self.state.response_hash_window.append(current_hash)
        if len(self.state.response_hash_window) > self.config.response_hash_window_size:
            self.state.response_hash_window = self.state.response_hash_window[
                -self.config.response_hash_window_size :
            ]

        loop_detected = False

        # 连续重复检测
        if current_hash == self.state.last_response_hash:
            loop_detected = True

        # 振荡检测（滑动窗口内同一 hash 出现次数 ≥ 阈值）
        if not loop_detected:
            counter = Counter(self.state.response_hash_window)
            if counter[current_hash] >= self.config.response_hash_repeat_threshold:
                loop_detected = True

        self.state.last_response_hash = current_hash

        if loop_detected:
            self.state.response_loop_escalation_count += 1
            action = self._determine_escalation_action(self.state.response_loop_escalation_count)
            logger.warning(
                f"[MONITOR] 检测到重复响应，升级计数={self.state.response_loop_escalation_count}，动作={action.value}"
            )
            self._last_loop_action = action
            return action
        else:
            # 新响应，重置升级计数
            self.state.response_loop_escalation_count = 0
            self._last_loop_action = EscalationAction.NONE
            return EscalationAction.NONE

    async def handle_loop_detected(self, action: EscalationAction):
        """处理检测到循环的情况"""
        if self.stream_reasoning_callback:
            if action == EscalationAction.TERMINATE:
                await self.stream_reasoning_callback(
                    "monitor", "LOOP_TERMINATE", "检测到持续重复响应模式，强制终止"
                )
            elif action == EscalationAction.INJECT_HINT:
                await self.stream_reasoning_callback(
                    "monitor", "LOOP_INJECT_HINT", "检测到重复响应模式，注入策略调整提示"
                )
            else:
                await self.stream_reasoning_callback(
                    "monitor", "LOOP_DETECTED", "检测到重复响应模式，正在调整策略"
                )

    def record_empty_response(self) -> bool:
        """
        记录空响应

        Returns:
            bool: True 表示连续空响应次数达到阈值，应该终止
        """
        self.state.consecutive_empty_turns += 1
        return self.state.consecutive_empty_turns >= self.config.max_consecutive_empty_turns

    async def handle_consecutive_empty_responses(self):
        """处理连续空响应的情况"""
        logger.warning(f"[MONITOR] 连续 {self.state.consecutive_empty_turns} 次空响应，强制结束")
        if self.stream_reasoning_callback:
            await self.stream_reasoning_callback(
                "monitor",
                "ERROR",
                f"连续 {self.state.consecutive_empty_turns} 次LLM响应失败，将生成当前结论",
            )

    def record_tool_loop_warning(self) -> EscalationAction:
        """
        记录工具循环警告，返回升级动作

        Returns:
            EscalationAction: WARN 或 TERMINATE
        """
        self.state.tool_loop_retry_count += 1
        if self.state.tool_loop_retry_count > self.config.max_tool_loop_retries:
            logger.warning(
                f"[MONITOR] 工具循环重试次数 {self.state.tool_loop_retry_count} 超过限制 "
                f"{self.config.max_tool_loop_retries}，强制终止"
            )
            return EscalationAction.TERMINATE
        logger.warning(
            f"[MONITOR] 工具循环重试 {self.state.tool_loop_retry_count}/{self.config.max_tool_loop_retries}"
        )
        return EscalationAction.WARN

    def reset_tool_loop_counter(self):
        """非重复工具调用时重置工具循环计数"""
        self.state.tool_loop_retry_count = 0

    def record_strategy_summary(self, summary: str):
        """记录工具调用策略摘要，去重，上限 10 条"""
        if summary not in self.state.attempted_strategies:
            self.state.attempted_strategies.append(summary)
            if len(self.state.attempted_strategies) > 10:
                self.state.attempted_strategies = self.state.attempted_strategies[-10:]

    def get_loop_break_hint(self, recent_tool_names: list = None, chinese: bool = False) -> str:
        """返回注入 message_history 的强制策略变更指令

        Args:
            recent_tool_names: 最近使用的工具名列表（可选，向后兼容）
            chinese: 是否返回中文版本
        """
        if chinese:
            lines = []
            lines.append("⚠️ [强制指令 — 系统监控]")
            lines.append("")
            lines.append("检测到重复响应模式。你必须立即改变策略。")
            lines.append("")

            if self.state.attempted_strategies:
                lines.append("== 已尝试的策略（禁止重复）==")
                for i, s in enumerate(self.state.attempted_strategies, 1):
                    lines.append(f"  {i}. {s}")
                lines.append("")

            if recent_tool_names:
                lines.append(f"== 最近使用的工具（请使用不同的）== {', '.join(recent_tool_names)}")
                lines.append("")

            lines.append("禁止：")
            lines.append("- 重复上述任何查询或工具调用")
            lines.append("- 使用相同或类似参数调用同一工具")
            lines.append("- 忽略此指令")
            lines.append("")
            lines.append("必须执行以下之一：")
            lines.append("A) 根据已收集的信息综合出最终答案")
            lines.append("B) 尝试全新的策略，使用不同的工具/参数")
            lines.append("C) 承认限制并提供当前最佳答案")

            return "\n".join(lines)

        lines = []
        lines.append("⚠️ [MANDATORY DIRECTIVE — SYSTEM MONITOR]")
        lines.append("")
        lines.append("Repeated response pattern detected. You MUST change your approach NOW.")
        lines.append("")

        # 已尝试策略
        if self.state.attempted_strategies:
            lines.append("== ATTEMPTED STRATEGIES (DO NOT REPEAT) ==")
            for i, s in enumerate(self.state.attempted_strategies, 1):
                lines.append(f"  {i}. {s}")
            lines.append("")

        # 最近工具
        if recent_tool_names:
            lines.append(f"== RECENT TOOLS (use DIFFERENT ones) == {', '.join(recent_tool_names)}")
            lines.append("")

        # FORBIDDEN
        lines.append("FORBIDDEN:")
        lines.append("- Repeating any query or tool call listed above")
        lines.append("- Using the same tool with the same or similar parameters")
        lines.append("- Ignoring this directive")
        lines.append("")

        # REQUIRED
        lines.append("REQUIRED — pick exactly ONE:")
        lines.append("A) Synthesize a final answer from information already gathered")
        lines.append("B) Try a fundamentally NEW strategy with DIFFERENT tools/parameters")
        lines.append("C) Acknowledge the limitation and provide your best current answer")

        return "\n".join(lines)

    async def pre_turn_check(self) -> str | None:
        """
        轮次前检查

        Returns:
            Optional[str]: 如果需要终止，返回终止原因；
                           "soft_timeout" 表示应注入催促但不终止；
                           None 表示正常继续。
        """
        timeout_result = await self.check_timeout()
        if timeout_result == "hard_timeout":
            return "timeout"
        elif timeout_result == "soft_timeout":
            return "soft_timeout"

        stall_action = await self.check_stall()
        if stall_action == EscalationAction.TERMINATE:
            return "stall_terminated"

        return None

    async def post_turn_check(
        self,
        response_text: str | None,
        llm_call_failed: bool,
    ) -> str | None:
        """
        轮次后检查

        Args:
            response_text: LLM 响应文本
            llm_call_failed: LLM 调用是否失败

        Returns:
            Optional[str]: 如果需要终止，返回终止原因；否则返回 None
        """
        if response_text:
            # 有响应，记录进展并检测循环
            action = self.record_progress(response_text)
            if action != EscalationAction.NONE:
                await self.handle_loop_detected(action)
                if action == EscalationAction.TERMINATE:
                    return "response_loop_terminated"
            return None
        else:
            # 空响应
            if llm_call_failed:
                should_terminate = self.record_empty_response()
                if should_terminate:
                    await self.handle_consecutive_empty_responses()
                    return "consecutive_empty_responses"
            return None

    def get_status_summary(self) -> dict:
        """
        获取监控状态摘要

        Returns:
            dict: 状态摘要
        """
        return {
            "elapsed_time": self.get_elapsed_time(),
            "time_since_progress": self.get_time_since_progress(),
            "consecutive_empty_turns": self.state.consecutive_empty_turns,
            "timeout_threshold": self.config.max_total_time,
            "stall_threshold": self.config.stall_detection_threshold,
        }


class TurnCounter:
    """轮次计数器，用于管理执行轮次和反思检查点"""

    def __init__(
        self,
        max_turns: int,
        reflection_enabled: bool = False,
        reflection_interval: int = 5,
    ):
        """
        初始化轮次计数器

        Args:
            max_turns: 最大轮次
            reflection_enabled: 是否启用反思检查点
            reflection_interval: 反思检查点间隔
        """
        self.max_turns = max_turns
        self.reflection_enabled = reflection_enabled
        self.reflection_interval = reflection_interval
        self.current_turn = 0

    def increment(self) -> int:
        """
        增加轮次计数

        Returns:
            int: 当前轮次
        """
        self.current_turn += 1
        return self.current_turn

    def is_max_reached(self) -> bool:
        """检查是否达到最大轮次"""
        return self.current_turn >= self.max_turns

    def should_inject_reflection(self) -> bool:
        """检查是否应该注入反思检查点"""
        if not self.reflection_enabled:
            return False
        return self.current_turn > 0 and self.current_turn % self.reflection_interval == 0

    def get_progress_percentage(self) -> float:
        """获取进度百分比"""
        if self.max_turns <= 0:
            return 0.0
        return min(100.0, (self.current_turn / self.max_turns) * 100)
