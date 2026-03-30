"""
Transcript 事件日志

将 Agent 执行过程记录为结构化事件流（非消息数组），
支持精确 replay、排障和可观测性分析。

参考 Claude Code 的 Transcript 设计：
- Transcript 是事件日志，toMessages() 是派生视图
- 每个事件带 UUID + 类型 + 时间戳
- 支持事件间引用关系（tool_use → tool_result 配对）

Usage:
    transcript = Transcript()
    transcript.record(EventType.LLM_CALL, {"turn": 1, "model": "claude-sonnet"})
    transcript.record(EventType.TOOL_USE, {"tool": "search", "args": {...}})
    transcript.record(EventType.TOOL_RESULT, {"tool": "search", "result": "..."})
    transcript.record(EventType.COMPACT, {"level": 1, "messages_affected": 5})

    # 导出为 JSONL
    transcript.save("transcript.jsonl")

    # 查询
    events = transcript.filter(EventType.TOOL_USE)
"""

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger("mem_deep_research")


class EventType(str, Enum):
    """Transcript 事件类型"""

    # Agent lifecycle
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"

    # LLM interaction
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"

    # Tool execution
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    TOOL_DEDUP = "tool_dedup"

    # Context management
    COMPACT = "compact"
    MICROCOMPACT = "microcompact"
    SUMMARIZE = "summarize"
    EMERGENCY = "emergency"
    OFFLOAD = "offload"

    # Monitoring
    LOOP_DETECTED = "loop_detected"
    TIMEOUT = "timeout"
    ESCALATION = "escalation"

    # Skills & planning
    SKILL_INJECT = "skill_inject"
    REFLECTION = "reflection"
    TASK_PLAN = "task_plan"

    # Sub-agent
    SUBAGENT_START = "subagent_start"
    SUBAGENT_END = "subagent_end"

    # System
    RESUME = "resume"
    ERROR = "error"
    CHECKPOINT = "checkpoint"


@dataclass
class TranscriptEvent:
    """单个 transcript 事件"""

    event_id: str
    event_type: str  # EventType value
    timestamp: float  # time.time()
    turn: int = 0
    agent_name: str = "main"
    data: dict = field(default_factory=dict)
    # 事件关联
    ref_event_id: str | None = None  # 引用另一个事件（如 tool_result → tool_use）
    duration_ms: int | None = None  # 事件耗时

    def to_dict(self) -> dict:
        """转为可序列化的字典"""
        d = asdict(self)
        # 去除 None 字段以节省空间
        return {k: v for k, v in d.items() if v is not None}


class Transcript:
    """Transcript 事件日志

    记录 Agent 执行过程的所有事件，支持查询和导出。
    """

    def __init__(self, agent_name: str = "main"):
        self.agent_name = agent_name
        self._events: list[TranscriptEvent] = []
        self._start_time: float = time.time()

    def record(
        self,
        event_type: EventType,
        data: dict | None = None,
        turn: int = 0,
        ref_event_id: str | None = None,
        duration_ms: int | None = None,
        agent_name: str | None = None,
    ) -> str:
        """记录一个事件

        Args:
            event_type: 事件类型
            data: 事件数据
            turn: 当前轮次
            ref_event_id: 关联事件 ID
            duration_ms: 耗时
            agent_name: Agent 名称（覆盖默认）

        Returns:
            事件 ID
        """
        event = TranscriptEvent(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            event_type=event_type.value,
            timestamp=time.time(),
            turn=turn,
            agent_name=agent_name or self.agent_name,
            data=data or {},
            ref_event_id=ref_event_id,
            duration_ms=duration_ms,
        )
        self._events.append(event)
        return event.event_id

    def filter(
        self,
        event_type: EventType | None = None,
        turn: int | None = None,
        agent_name: str | None = None,
    ) -> list[TranscriptEvent]:
        """按条件过滤事件"""
        results = self._events
        if event_type is not None:
            results = [e for e in results if e.event_type == event_type.value]
        if turn is not None:
            results = [e for e in results if e.turn == turn]
        if agent_name is not None:
            results = [e for e in results if e.agent_name == agent_name]
        return results

    def get_by_id(self, event_id: str) -> TranscriptEvent | None:
        """按 ID 获取事件"""
        for e in self._events:
            if e.event_id == event_id:
                return e
        return None

    def get_tool_pairs(self, turn: int | None = None) -> list[tuple[TranscriptEvent, TranscriptEvent | None]]:
        """获取 tool_use → tool_result 配对"""
        uses = self.filter(EventType.TOOL_USE, turn=turn)
        pairs = []
        for use in uses:
            result = None
            for e in self._events:
                if e.event_type == EventType.TOOL_RESULT.value and e.ref_event_id == use.event_id:
                    result = e
                    break
            pairs.append((use, result))
        return pairs

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def events(self) -> list[TranscriptEvent]:
        return list(self._events)

    def summary(self) -> dict:
        """生成 transcript 摘要统计"""
        type_counts: dict[str, int] = {}
        total_tool_ms = 0
        total_llm_ms = 0
        for e in self._events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
            if e.duration_ms:
                if e.event_type == EventType.TOOL_RESULT.value:
                    total_tool_ms += e.duration_ms
                elif e.event_type == EventType.LLM_RESPONSE.value:
                    total_llm_ms += e.duration_ms

        return {
            "total_events": len(self._events),
            "event_types": type_counts,
            "total_tool_duration_ms": total_tool_ms,
            "total_llm_duration_ms": total_llm_ms,
            "wall_time_seconds": round(time.time() - self._start_time, 2),
        }

    def save(self, path: str | Path) -> None:
        """保存为 JSONL 文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False, default=str) + "\n")
        logger.info(f"[Transcript] Saved {len(self._events)} events to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Transcript":
        """从 JSONL 文件加载"""
        transcript = cls()
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                event = TranscriptEvent(**data)
                transcript._events.append(event)
        logger.info(f"[Transcript] Loaded {len(transcript._events)} events from {path}")
        return transcript

    def reset(self):
        """重置"""
        self._events.clear()
        self._start_time = time.time()
