"""MemoryExtractionStrategy 抽象基类

Strategy 是 "长任务细节保鲜" 的可插拔机制：从 LLM 响应 / 工具结果 / context 压缩
中抽取结构化细节存入 session_memory / context_manager。

设计依据：docs/26-memory-extraction-strategy.md

Strategy 的生命周期由 Profile 层管理（`profile.extraction_strategies`）。
Runtime 通过 `profile.run_strategies_on_*` 调用 strategy 链。
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any


@dataclass
class ExtractionContext:
    """Strategy 方法可访问的 runtime 状态视图。

    字段由 MainLoopRunner 在每个触发点填充。Strategy 通过 session_memory
    / context_manager 的写 API 持久化抽取的细节。
    """

    turn_number: int
    task_description: str
    mode: str
    session_memory: Any
    context_manager: Any  # 完整实例暴露，strategy 自律使用
    llm_client: Any       # strategy 需要跑 LLM 时用（如 fact extraction）


class MemoryExtractionStrategy(ABC):
    """从 LLM 响应 / 工具结果 / context 压缩中抽取长期细节的策略。

    所有方法都有默认 no-op 实现，子类按需覆盖。每个 strategy 只需关心自己的触发点。

    契约：
    - Strategy 原则上只读，修改 assistant_text 由 list 顺序决定，不建议滥用
    - Tag 清理由 Runtime 统一做，strategy 不负责输出卫生
    - Strategy 之间不共享状态，通过 session_memory / context_manager 间接交互
    - 异常不吞，抛给 runtime 处理
    """

    name: str = "base"

    async def on_llm_response(
        self,
        assistant_text: str,
        ctx: ExtractionContext,
    ) -> str:
        """LLM 响应后触发。返回可能被修改的 text。

        典型用途：EvidenceTagStrategy 抽 <evidence> tag 存 session_memory。
        原则：strategy 只读取并存储结构化数据，返回原 text 即可。
        """
        return assistant_text

    async def on_tool_result(
        self,
        tool_name: str,
        tool_result: Any,
        ctx: ExtractionContext,
    ) -> None:
        """工具结果回来后触发（执行后、注入 message_history 前）。

        并发工具场景：每个工具分别触发一次（不做 batch）。
        需要 batch 处理的 strategy 在内部维护 buffer。

        典型用途：FactExtractionStrategy 用轻量 LLM 抽 facts。
        """
        return None

    async def on_compact(
        self,
        summary: str,
        up_to_turn: int,
        ctx: ExtractionContext,
    ) -> None:
        """Context 压缩完成后触发（LLMSummarize 产出 summary 后）。

        典型用途：SummaryEvidenceStrategy 从 summary 的 ## Evidence 段抽细节。
        """
        return None

    async def on_offload(
        self,
        ref: str,
        tool_name: str,
        original_content: str,
        ctx: ExtractionContext,
    ) -> None:
        """工具结果被 offload 到文件时触发。

        典型用途：用户自定义 strategy 把内容同步入外部存储
        （vector store / 全文索引 / 外部 blob store 等）。
        """
        return None

    # ========== Snapshot（HITL resume 支持）==========

    def snapshot(self) -> dict:
        """返回 strategy 内部状态，用于 checkpoint。默认空（无状态）。"""
        return {}

    def restore(self, state: dict) -> None:
        """从 snapshot 恢复内部状态。默认 no-op。"""
        return None
