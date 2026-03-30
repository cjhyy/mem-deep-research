import uuid
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ReasoningBlock:
    """Represents an extracted reasoning block from structured tags."""

    tag_name: str  # e.g., "research_plan", "findings_update", "tool_reasoning"
    content: str
    uid: str


class StructuredTagExtractor:
    """
    Extracts structured tags (research_plan, findings_update) from streaming text.

    关键设计：跨 chunk 累积标签内容，确保完整提取
    - 当检测到开始标签但没有结束标签时，缓存所有内容
    - 只有当检测到完整的标签对时，才提取为 ReasoningBlock
    """

    # Default tags that should be extracted as REASONING blocks
    DEFAULT_REASONING_TAGS = [
        "task_plan",
        "findings_update",
        "reflection_checkpoint",
        "thinking",
        "think",
    ]

    # Safety limit: flush buffer if it grows beyond this size without finding matching tags
    MAX_BUFFER_SIZE = 1_000_000  # 1MB

    def __init__(self, reasoning_tags: list[str] = None):
        """
        初始化标签提取器

        Args:
            reasoning_tags: 需要提取为 reasoning 的标签列表，如果为 None 则使用默认列表
        """
        self.reasoning_tags = (
            reasoning_tags if reasoning_tags is not None else self.DEFAULT_REASONING_TAGS
        )
        self.buffer = ""  # 累积所有输入
        self.pending_blocks: list[ReasoningBlock] = []
        self.in_tag: str | None = None  # 当前正在收集的标签名
        self.tag_start_pos: int = -1  # 当前标签的开始位置

    def set_reasoning_tags(self, tags: list[str]) -> None:
        """动态更新 reasoning 标签列表"""
        self.reasoning_tags = tags

    def process(self, text: str, is_last: bool = False) -> tuple[str, list[ReasoningBlock]]:
        """
        Process incoming text, extract structured tags.

        核心逻辑：
        1. 累积所有文本到 buffer
        2. 检测完整的标签对，提取内容
        3. 如果有未闭合的标签，保留在 buffer 中等待
        4. 只输出确定不在标签内的文本
        """
        self.buffer += text

        # Safety: if buffer grows too large without finding tags, flush it
        if len(self.buffer) > self.MAX_BUFFER_SIZE:
            flushed = self.buffer
            self.buffer = ""
            return flushed, []

        reasoning_blocks: list[ReasoningBlock] = []
        output_parts = []

        # 循环处理所有完整的标签
        while True:
            # 查找最近的开始标签
            earliest_open = -1
            earliest_tag = None
            for tag_name in self.reasoning_tags:
                open_tag = f"<{tag_name}>"
                pos = self.buffer.find(open_tag)
                if pos != -1 and (earliest_open == -1 or pos < earliest_open):
                    earliest_open = pos
                    earliest_tag = tag_name

            if earliest_tag is None:
                # 没有找到任何开始标签
                break

            open_tag = f"<{earliest_tag}>"
            close_tag = f"</{earliest_tag}>"
            close_pos = self.buffer.find(close_tag, earliest_open + len(open_tag))

            if close_pos == -1:
                # 找到开始标签但没有结束标签 - 需要等待更多数据
                # 输出开始标签之前的内容
                if earliest_open > 0:
                    output_parts.append(self.buffer[:earliest_open])
                    self.buffer = self.buffer[earliest_open:]
                break

            # 找到完整的标签对
            # 1. 输出标签之前的内容
            if earliest_open > 0:
                output_parts.append(self.buffer[:earliest_open])

            # 2. 提取标签内容
            content_start = earliest_open + len(open_tag)
            content_end = close_pos
            tag_content = self.buffer[content_start:content_end].strip()

            if tag_content:  # 只在有内容时创建 block
                reasoning_blocks.append(
                    ReasoningBlock(
                        tag_name=earliest_tag,
                        content=tag_content,
                        uid=f"reasoning_{uuid.uuid4().hex[:12]}",
                    )
                )

            # 3. 从 buffer 中移除已处理的部分
            self.buffer = self.buffer[close_pos + len(close_tag) :]

        # 处理剩余 buffer
        if is_last:
            # 最后一个 chunk，输出所有剩余内容
            if self.buffer:
                output_parts.append(self.buffer)
                self.buffer = ""
        else:
            # 检查是否有部分开始标签在末尾（需要保留等待）
            has_partial = False
            for tag_name in self.reasoning_tags:
                open_tag = f"<{tag_name}>"
                # 检查 buffer 末尾是否是开始标签的前缀
                for i in range(1, len(open_tag)):
                    if self.buffer.endswith(open_tag[:i]):
                        # 保留可能的部分标签
                        output_parts.append(self.buffer[:-i])
                        self.buffer = self.buffer[-i:]
                        has_partial = True
                        break
                if has_partial:
                    break

            if not has_partial and self.buffer:
                # 没有任何标签在处理中，检查是否安全输出
                # 如果 buffer 中没有 "<" 字符，则全部输出
                last_lt = self.buffer.rfind("<")
                if last_lt == -1:
                    output_parts.append(self.buffer)
                    self.buffer = ""
                else:
                    # 保留 "<" 及其后面的内容（可能是标签开始）
                    output_parts.append(self.buffer[:last_lt])
                    self.buffer = self.buffer[last_lt:]

        return "".join(output_parts), reasoning_blocks

    def reset(self):
        """Reset the extractor state."""
        self.buffer = ""
        self.pending_blocks = []
        self.in_tag = None
        self.tag_start_pos = -1


class TextInterceptor:
    def __init__(
        self,
        unbreakable_strings: list[str],
        reasoning_callback: Callable | None = None,
        reasoning_tags: list[str] = None,
    ):
        """
        初始化截流器

        Args:
            unbreakable_strings: 不可被分割的字符串列表（需要过滤的标签）
            reasoning_callback: Optional async callback for REASONING events
            reasoning_tags: 需要提取为 reasoning 的标签列表
        """
        self.unbreakable_strings = unbreakable_strings
        self.buffer = ""
        self.tag_extractor = StructuredTagExtractor(reasoning_tags=reasoning_tags)
        self.reasoning_callback = reasoning_callback

    def set_reasoning_tags(self, tags: list[str]) -> None:
        """动态更新 reasoning 标签列表"""
        self.tag_extractor.set_reasoning_tags(tags)

    def is_unbreakable_string(self, text):
        return any(unbreakable in text for unbreakable in self.unbreakable_strings)

    def get_reasoning_blocks(self) -> list[ReasoningBlock]:
        """Get any pending reasoning blocks extracted from the stream."""
        blocks = self.tag_extractor.pending_blocks
        self.tag_extractor.pending_blocks = []
        return blocks

    def process(self, text, is_last) -> tuple[str | None, list[ReasoningBlock]]:
        """
        处理输入的文本流，同时提取结构化标签作为REASONING事件

        Args:
            text (str): 输入的文本片段
            is_last (bool): 是否是最后一个片段

        Returns:
            Tuple[str or None, List[ReasoningBlock]]:
                - 可以安全输出的文本，如果需要继续缓存则返回None
                - 提取的REASONING块列表
        """
        # 首先通过tag extractor处理，提取结构化标签
        cleaned_text, reasoning_blocks = self.tag_extractor.process(text, is_last)

        # 如果提取到reasoning blocks，将其保存
        if reasoning_blocks:
            self.tag_extractor.pending_blocks.extend(reasoning_blocks)

        # 如果清理后的文本为空，继续等待（可能是标签内容正在缓存中）
        if not cleaned_text:
            return None, self.get_reasoning_blocks()

        # 将清理后的文本添加到缓冲区
        self.buffer += cleaned_text

        # 如果是最后一个片段，需要处理包含不可分割字符串的情况
        if is_last:
            result = self.buffer
            self.buffer = ""

            # 检查是否包含完整的不可分割字符串
            for unbreakable in self.unbreakable_strings:
                if unbreakable in result:
                    # 找到不可分割字符串的位置
                    unbreakable_pos = result.find(unbreakable)
                    if unbreakable_pos > 0:
                        # 如果不可分割字符串前面有内容，只输出前面的部分
                        return result[:unbreakable_pos], self.get_reasoning_blocks()
                    else:
                        # 如果不可分割字符串在开头，不输出任何内容
                        return None, self.get_reasoning_blocks()

            # 如果不包含任何不可分割字符串，直接输出
            return result, self.get_reasoning_blocks()

        # 检查缓冲区是否可能是某个不可分割字符串的前缀
        might_be_prefix = False
        for unbreakable in self.unbreakable_strings:
            if unbreakable.startswith(self.buffer) and len(self.buffer) < len(unbreakable):
                might_be_prefix = True
                break

        # 如果可能是前缀，继续缓存
        if might_be_prefix:
            return None, self.get_reasoning_blocks()

        # 检查是否包含完整的不可分割字符串
        for unbreakable in self.unbreakable_strings:
            if unbreakable in self.buffer:
                # 找到不可分割字符串的位置
                unbreakable_pos = self.buffer.find(unbreakable)
                if unbreakable_pos > 0:
                    # 如果不可分割字符串前面有内容，输出前面的部分
                    result = self.buffer[:unbreakable_pos]
                    # 保留不可分割字符串及其后面的内容在缓冲区中
                    self.buffer = self.buffer[unbreakable_pos:]
                    return result, self.get_reasoning_blocks()
                else:
                    # 如果不可分割字符串在开头，不输出任何内容，保持缓冲区不变
                    return None, self.get_reasoning_blocks()

        # 如果不包含完整的不可分割字符串，找到最后一个安全的输出位置
        safe_output_end = 0

        for i in range(1, len(self.buffer) + 1):
            current_suffix = self.buffer[safe_output_end:i]

            # 检查当前后缀是否是某个不可分割字符串的前缀
            is_dangerous_suffix = False
            for unbreakable in self.unbreakable_strings:
                if unbreakable.startswith(current_suffix) and len(current_suffix) < len(
                    unbreakable
                ):
                    is_dangerous_suffix = True
                    break

            # 如果不是危险后缀，更新安全输出位置
            if not is_dangerous_suffix:
                safe_output_end = i

        # 如果没有安全输出位置，继续缓存
        if safe_output_end == 0:
            return None, self.get_reasoning_blocks()

        # 输出安全部分，保留可能危险的后缀
        result = self.buffer[:safe_output_end]
        self.buffer = self.buffer[safe_output_end:]

        return (result if result else None), self.get_reasoning_blocks()
