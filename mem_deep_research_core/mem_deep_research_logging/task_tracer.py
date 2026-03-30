import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .logger import bootstrap_logger

LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "INFO")
logger = bootstrap_logger(level=LOGGER_LEVEL)


class StepRecord(BaseModel):
    """Record detailed information of task execution steps"""

    step_name: str
    message: str
    timestamp: datetime
    status: Literal["info", "warning", "failed", "success", "debug"] = "info"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskTracer(BaseModel):
    """Only use primitive types, datatime, Path etc."""

    status: Literal["pending", "running", "completed", "interrupted", "failed"] = "pending"

    # task info. hydrated BEFORE task execution.
    task_id: str = ""
    task_name: str = ""
    task_file_name: str | None = ""
    ground_truth: str | None
    input: Any = None

    # not task-related info. hydrated BEFORE task execution.
    log_path: Path

    # profile exeuction time. hydrated AFTER task execution.
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime = Field(default_factory=datetime.now)

    # record task result. hydrdrated AFTER task execution.
    final_boxed_answer: str = ""
    judge_result: str = ""
    error: str = ""

    # record task exection detail. hydrated DURING task_execution.
    current_main_turn_id: int = 0
    current_sub_agent_turn_id: int = 0
    sub_agent_counter: int = 0
    current_sub_agent_session_id: str | None = None
    main_agent_message_history: dict[str, Any] = Field(default_factory=dict)
    sub_agent_message_history_sessions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    step_logs: list[StepRecord] = Field(default_factory=list)

    # Performance metrics for optimization tracking
    perf_metrics: dict[str, Any] = Field(default_factory=dict)

    # Turn-level checkpoints for progress tracking and potential resume
    checkpoints: list[dict[str, Any]] = Field(default_factory=list)

    def start_sub_agent_session(self, sub_agent_name: str, subtask_description: str) -> str:
        """Start a new sub-agent session"""
        self.sub_agent_counter += 1
        session_id = f"{sub_agent_name}_{self.sub_agent_counter}"
        self.current_sub_agent_session_id = session_id

        # record sub-agent session start
        self.log_step(
            f"sub_{sub_agent_name}_session_start",
            f"Starting {session_id} for subtask: {subtask_description[:100]}{'...' if len(subtask_description) > 100 else ''}",
            "info",
            metadata={"session_id": session_id, "subtask": subtask_description},
        )

        return session_id

    def end_sub_agent_session(self, sub_agent_name: str):
        """End the current sub-agent session"""
        self.log_step(
            f"sub_{sub_agent_name}_session_end",
            f"Ending {self.current_sub_agent_session_id}",
            "success",
            metadata={"session_id": self.current_sub_agent_session_id},
        )
        self.current_sub_agent_session_id = None
        return None

    def log_step(
        self,
        step_name: str,
        message: str,
        status: Literal["info", "warning", "failed", "success", "debug"] = "info",
        metadata: dict[str, Any] | None = None,
    ):
        """Record execution step"""
        step_log = StepRecord(
            step_name=step_name,
            message=message,
            timestamp=datetime.now(),
            status=status,
            metadata=metadata or {},
        )
        self.step_logs.append(step_log)
        # Also print to console
        logger.debug(f"{step_name}: {message}")

    def record_perf(self, key: str, value: float, unit: str = "s") -> None:
        """Record a performance metric.

        Args:
            key: Metric name (e.g., "main_loop_duration", "tool_call.search.session_create")
            value: Metric value
            unit: Unit of measurement (default: "s" for seconds)
        """
        self.perf_metrics[key] = {"value": round(value, 4), "unit": unit}

    def append_perf(self, key: str, value: float, unit: str = "s") -> None:
        """Append a value to a list-type performance metric (e.g., per-tool-call timings).

        Args:
            key: Metric name
            value: Value to append
            unit: Unit of measurement
        """
        if key not in self.perf_metrics:
            self.perf_metrics[key] = {"values": [], "unit": unit}
        entry = self.perf_metrics[key]
        if "values" not in entry:
            # Convert scalar to list
            existing = entry.get("value")
            entry.pop("value", None)
            entry["values"] = [existing] if existing is not None else []
        entry["values"].append(round(value, 4))

    def get_perf_summary(self) -> str:
        """Return a human-readable summary of all recorded perf metrics."""
        if not self.perf_metrics:
            return "No performance metrics recorded."
        lines = []
        for key, data in sorted(self.perf_metrics.items()):
            if "values" in data:
                vals = data["values"]
                total = sum(vals)
                avg = total / len(vals) if vals else 0
                lines.append(
                    f"  {key}: count={len(vals)}, total={total:.3f}{data['unit']}, avg={avg:.3f}{data['unit']}"
                )
            else:
                lines.append(f"  {key}: {data['value']}{data['unit']}")
        return "Performance Metrics:\n" + "\n".join(lines)

    def save_checkpoint(
        self,
        turn: int,
        message_count: int,
        tool_calls_executed: int,
        last_assistant_text: str = "",
        task_failed: bool = False,
    ) -> None:
        """Save a turn-level checkpoint for progress tracking.

        Enables callers to see what was accomplished before a failure.
        Each checkpoint captures the state at the end of a turn.
        """
        self.checkpoints.append({
            "turn": turn,
            "timestamp": datetime.now().isoformat(),
            "message_count": message_count,
            "tool_calls_executed": tool_calls_executed,
            "last_assistant_preview": last_assistant_text[:200] if last_assistant_text else "",
            "task_failed": task_failed,
        })

    def save(self):
        """Persist TaskTracer to disk. used in a finally block, thus never raise Exception."""
        try:
            if not self.log_path.exists():
                self.log_path.parent.mkdir(exist_ok=True, parents=True)
            with open(self.log_path, mode="w") as dest:
                dest.write(self.model_dump_json(indent=2))
        except Exception as e:
            logger.error(e, stack_info=True, exc_info=True)

    def cleanup(self):
        """
        Clean up memory-heavy fields after task execution.
        Call this after save() to release memory.
        """
        try:
            # Clear message histories (can be large with many turns)
            self.main_agent_message_history.clear()
            self.sub_agent_message_history_sessions.clear()

            # Clear step logs (can accumulate many entries)
            self.step_logs.clear()

            # Clear checkpoints
            self.checkpoints.clear()

            # Clear input data
            self.input = None

            logger.debug(f"TaskTracer cleanup completed for task_id={self.task_id}")
        except Exception as e:
            logger.warning(f"Error during TaskTracer cleanup: {e}")
