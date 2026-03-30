"""
消息拦截处理模块

处理流式消息的拦截、关键字过滤和 reasoning block 提取。
支持配置化的标签过滤和 reasoning 提取。
"""

import asyncio
import logging
from collections.abc import Callable

from mem_deep_research_core.core.hooks import HookContext, hooks
from mem_deep_research_core.core.interceptor_config import InterceptorConfig, InterceptorPresets
from mem_deep_research_core.utils.stream_parsing_utils import ReasoningBlock, TextInterceptor

logger = logging.getLogger("mem_deep_research")


# ========== 默认钩子实现 ==========


def _default_on_message_intercept(ctx: HookContext):
    """消息拦截 - 默认实现，返回原始拦截结果"""
    return ctx.extra.get("intercept_result")


hooks.set_default("on_message_intercept", _default_on_message_intercept)


class MessageInterceptorHandler:
    """
    消息拦截处理器

    支持配置化的消息拦截功能：
    - 过滤指定标签（如 <use_mcp_tool>）
    - 提取 reasoning 标签作为事件
    - 控制输出内容
    """

    def __init__(
        self,
        config: InterceptorConfig = None,
        keywords: list[str] = None,  # 兼容旧接口
        stream_reasoning_callback: Callable | None = None,
        stream_tool_call_callback: Callable | None = None,
        stream_message_callback: Callable | None = None,
        context: dict = None,
    ):
        """
        初始化消息拦截处理器

        Args:
            config: 拦截器配置，如果提供则忽略 keywords 参数
            keywords: 需要拦截的关键字列表（已废弃，建议使用 config）
            stream_reasoning_callback: 发送 reasoning 事件的回调
            stream_tool_call_callback: 发送 tool_call 事件的回调
            stream_message_callback: 发送 message 事件的回调
            context: 用户上下文，注入到所有 hook 调用中
        """
        # 配置处理
        if config is not None:
            self.config = config
        elif keywords is not None:
            # 兼容旧接口：从 keywords 创建配置
            self.config = InterceptorConfig(filter_tags=[k.strip("<>") for k in keywords])
        else:
            self.config = InterceptorConfig()

        # 用户上下文
        self.context = context or {}

        # 回调函数
        self.stream_reasoning_callback = stream_reasoning_callback
        self.stream_tool_call_callback = stream_tool_call_callback
        self.stream_message_callback = stream_message_callback

        # 创建拦截器
        self._create_interceptors()

        # 当前 agent ID，用于 reasoning 事件的 parent_uid
        self.current_agent_id: str | None = None

    def _create_interceptors(self) -> None:
        """根据配置创建拦截器实例"""
        filter_keywords = self.config.get_all_filter_keywords()
        reasoning_tags = self.config.reasoning_tags

        # 主拦截器（用于工具调用阶段）
        self.key_message_interceptor = TextInterceptor(
            filter_keywords, reasoning_tags=reasoning_tags
        )
        # 最终消息拦截器（用于最终输出）
        self._final_message_interceptor: TextInterceptor | None = None

    def update_config(self, config: InterceptorConfig) -> None:
        """
        更新拦截器配置

        Args:
            config: 新的配置
        """
        self.config = config
        self._create_interceptors()
        logger.info(
            f"[INTERCEPTOR] Config updated: filter_tags={config.filter_tags}, reasoning_tags={config.reasoning_tags}"
        )

    def set_preset(self, preset_name: str) -> None:
        """
        使用预设配置

        Args:
            preset_name: 预设名称 (default, verbose, minimal, debug)
        """
        self.config = InterceptorPresets.from_name(preset_name)
        self._create_interceptors()
        logger.info(f"[INTERCEPTOR] Using preset: {preset_name}")

    def set_current_agent_id(self, agent_id: str) -> None:
        """设置当前 agent ID"""
        self.current_agent_id = agent_id

    def reset_interceptor(self) -> None:
        """重置拦截器状态"""
        self._create_interceptors()

    async def _send_reasoning_blocks(self, reasoning_blocks: list[ReasoningBlock]) -> None:
        """发送 reasoning blocks"""
        # 检查配置是否允许发送 reasoning
        if not self.config.show_reasoning:
            return

        if not self.stream_reasoning_callback or not reasoning_blocks:
            return

        for block in reasoning_blocks:
            try:
                await self.stream_reasoning_callback(
                    reasoning_id=block.uid,
                    content=block.content,
                    parent_uid=self.current_agent_id,
                    status="SUCCESS",
                )
            except Exception as e:
                logger.warning(f"Failed to stream reasoning block: {e}")

    async def intercept_key_message(self, message_id: str, message: str, is_last: bool) -> bool:
        """
        拦截关键字消息（用于工具调用阶段）

        如果关键字在消息中，则不发送，返回 False；否则发送，返回 True。
        同时提取结构化标签并发送 REASONING 事件。

        Args:
            message_id: 消息 ID
            message: 消息内容
            is_last: 是否是最后一条消息

        Returns:
            bool: 是否成功处理（True 表示已发送或跳过，False 表示包含不可分割的关键字）
        """
        try:
            result, reasoning_blocks = self.key_message_interceptor.process(message, is_last)

            # Hook: on_message_intercept
            hook_result = hooks.call(
                "on_message_intercept",
                HookContext(
                    hook_name="on_message_intercept",
                    context=self.context,
                    extra={
                        "message": message,
                        "intercept_result": result,
                        "reasoning_blocks": reasoning_blocks,
                        "is_last": is_last,
                    },
                ),
            )
            if hook_result is not None:
                result = hook_result

            # Debug: Log extracted reasoning blocks
            if reasoning_blocks:
                logger.info(
                    f"[REASONING] Extracted {len(reasoning_blocks)} blocks: {[b.tag_name for b in reasoning_blocks]}"
                )

            # 发送 REASONING 事件
            await self._send_reasoning_blocks(reasoning_blocks)

            if result is not None:
                # 跳过空白内容
                if not result.strip():
                    return True

                if self.key_message_interceptor.is_unbreakable_string(result):
                    return False
                else:
                    # 检查配置是否允许显示文本输出
                    if self.config.show_text_output and self.stream_tool_call_callback:
                        await self.stream_tool_call_callback(
                            "show_text", {"text": result}, True, message_id
                        )
                    await asyncio.sleep(0)
                    return True
            return True

        except Exception as e:
            logger.error(f"Error in intercept_key_message: {e}")
            # 出错时尝试直接输出原始消息
            try:
                if message and message.strip() and self.stream_tool_call_callback:
                    await self.stream_tool_call_callback(
                        "show_text", {"text": message}, True, message_id
                    )
            except Exception:
                pass
            return True

    async def intercept_final_message(self, message_id: str, message: str, is_last: bool) -> bool:
        """
        拦截最终消息（用于最终输出阶段）

        使用独立的 interceptor 实例，避免工具调用阶段的 buffer 残留影响。

        Args:
            message_id: 消息 ID
            message: 消息内容
            is_last: 是否是最后一条消息

        Returns:
            bool: 是否成功处理
        """
        try:
            # 确保使用独立的 interceptor 实例
            if self._final_message_interceptor is None:
                self._final_message_interceptor = TextInterceptor(
                    self.config.get_all_filter_keywords(), reasoning_tags=self.config.reasoning_tags
                )

            result, reasoning_blocks = self._final_message_interceptor.process(message, is_last)

            # 发送 REASONING 事件
            await self._send_reasoning_blocks(reasoning_blocks)

            if result is not None:
                # 跳过空白内容
                if not result.strip():
                    return True

                if self._final_message_interceptor.is_unbreakable_string(result):
                    return False
                else:
                    # 检查配置是否允许显示文本输出
                    if self.config.show_text_output and self.stream_message_callback:
                        await self.stream_message_callback(
                            message_id=message_id, delta_content=result
                        )
                    await asyncio.sleep(0)
                    return True
            return True

        except Exception as e:
            logger.error(f"Error in intercept_final_message: {e}")
            # 出错时尝试直接输出原始消息
            try:
                if message and message.strip() and self.stream_message_callback:
                    await self.stream_message_callback(message_id=message_id, delta_content=message)
            except Exception:
                pass
            return True

    def create_key_message_callback(self) -> Callable:
        """创建用于工具调用阶段的消息回调"""

        async def callback(message_id: str, message: str, is_last: bool) -> bool:
            return await self.intercept_key_message(message_id, message, is_last)

        return callback

    def create_final_message_callback(self) -> Callable:
        """创建用于最终输出阶段的消息回调"""

        async def callback(message_id: str, message: str, is_last: bool) -> bool:
            return await self.intercept_final_message(message_id, message, is_last)

        return callback

    # ========== 配置查询方法 ==========

    def get_config(self) -> InterceptorConfig:
        """获取当前配置"""
        return self.config

    def get_filter_tags(self) -> list[str]:
        """获取当前过滤的标签列表"""
        return self.config.filter_tags

    def get_reasoning_tags(self) -> list[str]:
        """获取当前 reasoning 标签列表"""
        return self.config.reasoning_tags

    def is_showing_reasoning(self) -> bool:
        """是否显示 reasoning"""
        return self.config.show_reasoning

    def is_showing_tool_calls(self) -> bool:
        """是否显示工具调用"""
        return self.config.show_tool_calls

    def is_showing_text_output(self) -> bool:
        """是否显示文本输出"""
        return self.config.show_text_output
