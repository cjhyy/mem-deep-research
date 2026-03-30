"""
消息拦截器配置模块

提供配置化的消息拦截功能，允许用户自定义：
- 需要过滤的标签（不输出到用户）
- 需要提取为 reasoning 的标签
- 输出内容控制
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from mem_deep_research_core.core.constants import DEFAULT_FILTER_TAGS, DEFAULT_REASONING_TAGS

logger = logging.getLogger("mem_deep_research")


@dataclass
class InterceptorConfig:
    """
    消息拦截器配置

    Attributes:
        filter_tags: 需要过滤的标签列表（这些标签的内容不会输出到用户）
            默认: ["use_mcp_tool"]

        reasoning_tags: 需要提取为 reasoning block 的标签列表
            默认: ["thinking", "think", "task_plan", "findings_update", "reflection_checkpoint"]

        show_reasoning: 是否将 reasoning 内容作为事件发送
            默认: True

        show_tool_calls: 是否显示工具调用信息
            默认: True

        show_text_output: 是否显示文本输出
            默认: True

        strip_reasoning_from_output: 是否从最终输出中移除 reasoning 标签内容
            默认: True（reasoning 内容通过事件发送，不在主输出中显示）

        custom_tag_handlers: 自定义标签处理器映射
            格式: {"tag_name": "handler_type"}
            handler_type 可以是: "filter", "reasoning", "passthrough"
    """

    # 需要过滤的标签（这些标签及其内容不会输出）
    filter_tags: list[str] = field(default_factory=lambda: list(DEFAULT_FILTER_TAGS))

    # 需要提取为 reasoning 的标签
    reasoning_tags: list[str] = field(default_factory=lambda: list(DEFAULT_REASONING_TAGS))

    # 输出控制
    show_reasoning: bool = True
    show_tool_calls: bool = True
    show_text_output: bool = True
    strip_reasoning_from_output: bool = True

    # 自定义标签处理
    custom_tag_handlers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "InterceptorConfig":
        """从字典创建配置"""
        if not config_dict:
            return cls()

        return cls(
            filter_tags=config_dict.get("filter_tags", list(DEFAULT_FILTER_TAGS)),
            reasoning_tags=config_dict.get(
                "reasoning_tags",
                list(DEFAULT_REASONING_TAGS),
            ),
            show_reasoning=config_dict.get("show_reasoning", True),
            show_tool_calls=config_dict.get("show_tool_calls", True),
            show_text_output=config_dict.get("show_text_output", True),
            strip_reasoning_from_output=config_dict.get("strip_reasoning_from_output", True),
            custom_tag_handlers=config_dict.get("custom_tag_handlers", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "filter_tags": self.filter_tags,
            "reasoning_tags": self.reasoning_tags,
            "show_reasoning": self.show_reasoning,
            "show_tool_calls": self.show_tool_calls,
            "show_text_output": self.show_text_output,
            "strip_reasoning_from_output": self.strip_reasoning_from_output,
            "custom_tag_handlers": self.custom_tag_handlers,
        }

    def get_all_filter_keywords(self) -> list[str]:
        """获取所有需要过滤的完整标签（带尖括号）"""
        return [f"<{tag}>" for tag in self.filter_tags if tag and not tag.startswith("<")]

    def should_extract_as_reasoning(self, tag_name: str) -> bool:
        """判断标签是否应该提取为 reasoning"""
        # 先检查自定义处理器
        if tag_name in self.custom_tag_handlers:
            return self.custom_tag_handlers[tag_name] == "reasoning"
        return tag_name in self.reasoning_tags

    def should_filter(self, tag_name: str) -> bool:
        """判断标签是否应该被过滤"""
        if tag_name in self.custom_tag_handlers:
            return self.custom_tag_handlers[tag_name] == "filter"
        return tag_name in self.filter_tags


# 预定义配置模板
class InterceptorPresets:
    """预定义的拦截器配置模板"""

    @staticmethod
    def default() -> InterceptorConfig:
        """默认配置：过滤工具调用，提取 reasoning"""
        return InterceptorConfig()

    @staticmethod
    def verbose() -> InterceptorConfig:
        """详细模式：显示所有内容，包括 reasoning"""
        return InterceptorConfig(
            strip_reasoning_from_output=False,  # reasoning 也显示在主输出中
        )

    @staticmethod
    def minimal() -> InterceptorConfig:
        """精简模式：只显示最终文本，不显示 reasoning"""
        return InterceptorConfig(
            show_reasoning=False,  # 不发送 reasoning 事件
            show_tool_calls=False,  # 不显示工具调用
        )

    @staticmethod
    def debug() -> InterceptorConfig:
        """调试模式：显示所有内容，不过滤任何标签"""
        return InterceptorConfig(
            filter_tags=[],  # 不过滤任何标签
            strip_reasoning_from_output=False,
        )

    @staticmethod
    def from_name(name: str) -> InterceptorConfig:
        """根据名称获取预设配置"""
        presets = {
            "default": InterceptorPresets.default,
            "verbose": InterceptorPresets.verbose,
            "minimal": InterceptorPresets.minimal,
            "debug": InterceptorPresets.debug,
        }

        if name in presets:
            return presets[name]()

        logger.warning(f"Unknown interceptor preset: {name}, using default")
        return InterceptorPresets.default()
