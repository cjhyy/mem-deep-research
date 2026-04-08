"""
TodoTracker — 任务追踪系统

独立于 message_history 的任务状态管理。
Context 截断时 todo 状态不会丢失，每轮自动重新注入。

设计参考：Deer-Flow todo_middleware

用法：
    tracker = TodoTracker()

    # LLM 通过内置工具更新任务
    tracker.update_from_tool_call({"action": "add", "task": "搜索量子计算论文", "priority": "high"})
    tracker.update_from_tool_call({"action": "complete", "task_id": 1})

    # 每轮开始时注入状态到消息
    todo_msg = tracker.build_injection_message()
    if todo_msg:
        message_history.insert(-1, todo_msg)  # 插入到最后一条 user 消息前
"""

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger("mem_deep_research")


class TodoStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


@dataclass
class TodoItem:
    """单个待办任务"""

    id: int
    task: str
    status: TodoStatus = TodoStatus.PENDING
    priority: str = "medium"  # high, medium, low
    result: str = ""  # 完成时的结果摘要

    def to_display(self) -> str:
        status_icon = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}
        icon = status_icon.get(self.status, "⬜")
        priority_marker = {"high": "!!!", "medium": "!!", "low": "!"}.get(self.priority, "!!")
        line = f"{icon} [{self.id}] [{priority_marker}] {self.task}"
        if self.status == TodoStatus.COMPLETED and self.result:
            line += f" → {self.result[:100]}"
        return line


class TodoTracker:
    """任务追踪器 — 独立于 message_history 的状态管理

    Context 被压缩/截断时，todo 状态不会丢失。
    每轮开始时自动注入当前状态到消息历史。
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._items: list[TodoItem] = []
        self._next_id: int = 1
        self._last_injected_turn: int = -1

    # ---- Tool interface (LLM calls these via update_todo tool) ----

    def update_from_tool_call(self, arguments: dict) -> str:
        """处理 LLM 的 update_todo 工具调用

        Args:
            arguments: 工具调用参数，格式：
                - {"action": "add", "task": "...", "priority": "high"}
                - {"action": "start", "task_id": 1}
                - {"action": "complete", "task_id": 1, "result": "..."}
                - {"action": "list"}

        Returns:
            操作结果描述
        """
        action = arguments.get("action", "list")

        if action == "add":
            return self._add(arguments.get("task", ""), arguments.get("priority", "medium"))
        elif action == "start":
            return self._set_status(arguments.get("task_id"), TodoStatus.IN_PROGRESS)
        elif action == "complete":
            return self._complete(arguments.get("task_id"), arguments.get("result", ""))
        elif action == "list":
            return self._list()
        else:
            return f"Unknown action: {action}. Use: add, start, complete, list"

    def _add(self, task: str, priority: str = "medium") -> str:
        if not task:
            return "Error: task description is required"
        item = TodoItem(id=self._next_id, task=task, priority=priority)
        self._items.append(item)
        self._next_id += 1
        logger.info(f"[Todo] Added #{item.id}: {task}")
        return f"Added task #{item.id}: {task}"

    def _set_status(self, task_id: int | None, status: TodoStatus) -> str:
        if task_id is None:
            return "Error: task_id is required"
        try:
            task_id = int(task_id)
        except (ValueError, TypeError):
            return f"Error: invalid task_id '{task_id}', must be a number"
        for item in self._items:
            if item.id == task_id:
                item.status = status
                logger.info(f"[Todo] #{task_id} → {status}")
                return f"Task #{task_id} is now {status}"
        return f"Error: task #{task_id} not found"

    def _complete(self, task_id: int | None, result: str = "") -> str:
        if task_id is None:
            return "Error: task_id is required"
        try:
            task_id = int(task_id)
        except (ValueError, TypeError):
            return f"Error: invalid task_id '{task_id}', must be a number"
        for item in self._items:
            if item.id == task_id:
                item.status = TodoStatus.COMPLETED
                item.result = result
                logger.info(f"[Todo] #{task_id} completed: {result[:80]}")
                return f"Task #{task_id} completed" + (f": {result[:100]}" if result else "")
        return f"Error: task #{task_id} not found"

    def _list(self) -> str:
        if not self._items:
            return "No tasks."
        return "\n".join(item.to_display() for item in self._items)

    # ---- State queries ----

    @property
    def has_pending_work(self) -> bool:
        return any(item.status != TodoStatus.COMPLETED for item in self._items)

    @property
    def progress(self) -> float:
        if not self._items:
            return 1.0
        completed = sum(1 for item in self._items if item.status == TodoStatus.COMPLETED)
        return completed / len(self._items)

    @property
    def is_empty(self) -> bool:
        return len(self._items) == 0

    # ---- Injection into message_history ----

    def build_injection_message(self, turn: int = 0) -> dict | None:
        """构建注入到 message_history 的 todo 状态消息

        如果没有任务或状态未变化，返回 None。
        """
        if not self.enabled or not self._items:
            return None

        lines = ["[TASK PROGRESS]", ""]
        lines.append(
            f"Progress: {self.progress:.0%} ({sum(1 for i in self._items if i.status == TodoStatus.COMPLETED)}/{len(self._items)} completed)"
        )
        lines.append("")
        for item in self._items:
            lines.append(item.to_display())
        lines.append("")

        pending = [i for i in self._items if i.status == TodoStatus.PENDING]
        in_progress = [i for i in self._items if i.status == TodoStatus.IN_PROGRESS]

        if in_progress:
            lines.append(f"Currently working on: #{in_progress[0].id} {in_progress[0].task}")
        elif pending:
            lines.append(f"Next: #{pending[0].id} {pending[0].task}")

        lines.append("")
        lines.append(
            "Use the update_todo tool to update task status when you start or complete a task."
        )

        self._last_injected_turn = turn

        from mem_deep_research_core.core.constants import MT

        return {
            "role": "user",
            "_type": MT.TASK_PROGRESS,
            "content": [{"type": "text", "text": "\n".join(lines)}],
        }

    # ---- Tool definition (exposed to LLM) ----

    @staticmethod
    def get_tool_definition() -> dict:
        """返回 update_todo 工具的 MCP server 格式定义"""
        return {
            "name": "builtin-todo-tracker",
            "tools": [
                {
                    "name": "update_todo",
                    "description": (
                        "Manage your task list. Use this to break down complex tasks, "
                        "track progress, and mark tasks as completed. "
                        "IMPORTANT: Always mark tasks as completed after finishing them."
                    ),
                    "schema": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["add", "start", "complete", "list"],
                                "description": "Action to perform: add a new task, start working on a task, complete a task, or list all tasks",
                            },
                            "task": {
                                "type": "string",
                                "description": "Task description (required for 'add' action)",
                            },
                            "task_id": {
                                "type": "integer",
                                "description": "Task ID (required for 'start' and 'complete' actions)",
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "Task priority (for 'add' action, default: medium)",
                            },
                            "result": {
                                "type": "string",
                                "description": "Result summary (optional, for 'complete' action)",
                            },
                        },
                        "required": ["action"],
                    },
                }
            ],
        }

    # ---- Serialization ----

    def to_dict(self) -> dict:
        return {
            "items": [
                {
                    "id": i.id,
                    "task": i.task,
                    "status": i.status,
                    "priority": i.priority,
                    "result": i.result,
                }
                for i in self._items
            ],
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, data: dict, enabled: bool = True) -> "TodoTracker":
        tracker = cls(enabled=enabled)
        for item_data in data.get("items", []):
            tracker._items.append(
                TodoItem(
                    id=item_data["id"],
                    task=item_data["task"],
                    status=TodoStatus(item_data.get("status", "pending")),
                    priority=item_data.get("priority", "medium"),
                    result=item_data.get("result", ""),
                )
            )
        tracker._next_id = data.get(
            "next_id", max((item.id for item in tracker._items), default=0) + 1
        )
        return tracker

    def reset(self):
        self._items.clear()
        self._next_id = 1
        self._last_injected_turn = -1
