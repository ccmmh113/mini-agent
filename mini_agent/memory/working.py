"""Working memory storage for active task execution state."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..schema import TaskMemoryItem, TaskMemoryState
from ..tools.base import ToolResult

MAX_COMPLETED_STEPS = 20
COMPLETED_STEP_ARCHIVE_THRESHOLD = 50
MAX_ARTIFACTS = 20
MAX_RESUME_EVENTS = 10


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def empty_task_memory() -> dict[str, Any]:
    return TaskMemoryState().model_dump(mode="json")


def normalize_task_memory(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return empty_task_memory()

    try:
        normalized = TaskMemoryState.model_validate(data)
    except Exception:
        return empty_task_memory()
    dumped = normalized.model_dump(mode="json")
    for task in dumped.get("tasks", []):
        _compact_task(task)
    return dumped


def _append_archived_summary(task: dict[str, Any], archived_steps: list[dict[str, Any]]) -> None:
    if not archived_steps:
        return

    archived_descriptions = [str(step.get("description", "")).strip() for step in archived_steps if step.get("description")]
    if not archived_descriptions:
        return

    summary_line = (
        f"[{_now()}] Archived {len(archived_descriptions)} earlier completed steps: "
        + "; ".join(archived_descriptions[:5])
    )
    if len(archived_descriptions) > 5:
        summary_line += "; ..."

    existing = str(task.get("archived_steps_summary", "")).strip()
    task["archived_steps_summary"] = summary_line if not existing else f"{existing}\n{summary_line}"


def _compact_task(task: dict[str, Any]) -> None:
    completed_steps = task.get("completed_steps", [])
    if isinstance(completed_steps, list) and len(completed_steps) > COMPLETED_STEP_ARCHIVE_THRESHOLD:
        keep_steps = completed_steps[-MAX_COMPLETED_STEPS:]
        archived_steps = completed_steps[:-MAX_COMPLETED_STEPS]
        task["completed_steps"] = keep_steps
        _append_archived_summary(task, archived_steps)

    artifacts = task.get("artifacts", [])
    if isinstance(artifacts, list) and len(artifacts) > MAX_ARTIFACTS:
        task["artifacts"] = artifacts[-MAX_ARTIFACTS:]

    resume_events = task.get("resume_events", [])
    if isinstance(resume_events, list) and len(resume_events) > MAX_RESUME_EVENTS:
        task["resume_events"] = resume_events[-MAX_RESUME_EVENTS:]


def new_task(goal: str, task_type: str) -> dict[str, Any]:
    timestamp = _now()
    task = TaskMemoryItem(
        task_id=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}",
        goal=goal,
        task_type=task_type,
        status="active",
        created_at=timestamp,
        updated_at=timestamp,
        archived_steps_summary="",
    )
    return task.model_dump(mode="json")


class TaskMemoryStore:
    """Small JSON-backed store for active task working memory."""

    def __init__(self, memory_file: str):
        self.memory_file = Path(memory_file)

    def load(self) -> dict[str, Any]:
        if not self.memory_file.exists():
            return empty_task_memory()

        try:
            data = json.loads(self.memory_file.read_text(encoding="utf-8"))
        except Exception:
            return empty_task_memory()
        return normalize_task_memory(data)

    def save(self, data: dict[str, Any]) -> None:
        normalized = normalize_task_memory(data)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(
            json.dumps(normalized, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_active_task(self, data: dict[str, Any]) -> dict[str, Any] | None:
        active_task_id = data.get("active_task_id")
        if not active_task_id:
            return None
        for task in data.get("tasks", []):
            if task.get("task_id") == active_task_id:
                return task
        return None

    def require_active_task(self, data: dict[str, Any]) -> tuple[dict[str, Any] | None, ToolResult | None]:
        task = self.get_active_task(data)
        if task is None:
            return None, ToolResult(
                success=False,
                content="",
                error="No active task. Call start_task first.",
            )
        return task, None

    def ensure_active_task(self, goal: str, task_type: str = "general", source: str = "auto") -> dict[str, Any]:
        data = self.load()
        task = self.get_active_task(data)
        if task is not None:
            task.setdefault("resume_events", []).append(
                {
                    "timestamp": _now(),
                    "source": source,
                    "goal": goal,
                }
            )
            task["updated_at"] = _now()
            self.save(data)
            return task

        task = new_task(goal=goal, task_type=task_type)
        task["source"] = source
        data["tasks"].append(task)
        data["active_task_id"] = task["task_id"]
        self.save(data)
        return task

    def append_step(self, description: str, source: str = "manual") -> None:
        data = self.load()
        task = self.get_active_task(data)
        if task is None:
            return
        task["completed_steps"].append(
            {
                "description": description,
                "source": source,
                "timestamp": _now(),
            }
        )
        task["updated_at"] = _now()
        self.save(data)

    def append_artifact(self, artifact: dict[str, Any]) -> None:
        data = self.load()
        task = self.get_active_task(data)
        if task is None:
            return
        artifact.setdefault("timestamp", _now())
        task["artifacts"].append(artifact)
        task["updated_at"] = _now()
        self.save(data)

    def finish_active_task(self, summary: str = "", source: str = "manual") -> dict[str, Any] | None:
        data = self.load()
        task = self.get_active_task(data)
        if task is None:
            return None
        task["status"] = "completed"
        task["summary"] = summary
        task["completed_by"] = source
        task["updated_at"] = _now()
        data["active_task_id"] = None
        self.save(data)
        return task

    def abandon_active_task(self, reason: str = "", source: str = "manual") -> dict[str, Any] | None:
        data = self.load()
        task = self.get_active_task(data)
        if task is None:
            return None
        task["status"] = "abandoned"
        task["abandoned_by"] = source
        task["abandoned_reason"] = reason
        task["updated_at"] = _now()
        data["active_task_id"] = None
        self.save(data)
        return task
