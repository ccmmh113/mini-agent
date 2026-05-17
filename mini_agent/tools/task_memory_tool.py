"""Task memory runtime utilities for tracking structured task progress."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..memory import (
    EpisodeMemoryStore,
    TaskMemoryStore,
    build_episode_from_task,
    new_task,
)
from .base import ToolResult


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _resolve_workspace_path(path: str, workspace_dir: Path) -> Path:
    file_path = Path(path).expanduser()
    if not file_path.is_absolute():
        file_path = workspace_dir / file_path
    return file_path


def file_artifact_metadata(path: str, workspace_dir: str | Path) -> dict[str, Any]:
    """Return verifiable metadata for a file artifact when it is safe to inspect."""

    workspace_path = Path(workspace_dir).resolve()
    file_path = _resolve_workspace_path(path, workspace_path).resolve()
    metadata: dict[str, Any] = {
        "path": str(file_path),
        "exists": file_path.exists(),
        "verified": False,
    }

    if not _is_subpath(file_path, workspace_path):
        metadata["verification_error"] = "Path is outside workspace"
        return metadata

    if not file_path.exists():
        metadata["verification_error"] = "File does not exist"
        return metadata

    if not file_path.is_file():
        metadata["verification_error"] = "Path is not a file"
        return metadata

    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    stat = file_path.stat()
    metadata.update(
        {
            "sha256": digest.hexdigest(),
            "size_bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "verified": True,
        }
    )
    return metadata


def verify_task_artifact(artifact: dict[str, Any], workspace_dir: str | Path) -> dict[str, Any]:
    """Compare the current file state with the artifact's recorded evidence.

    The original artifact verification remains immutable audit evidence. This
    function returns a transient report for resume/review display.
    """

    path = str(artifact.get("path", ""))
    original = artifact.get("verification") if isinstance(artifact.get("verification"), dict) else {}
    if not path:
        return {"path": path, "status": "not_verifiable", "reason": "Artifact path is empty"}

    if artifact.get("artifact_type") not in {"file_read", "file_written", "file_edited", "file"}:
        return {"path": path, "status": "not_verifiable", "reason": "Artifact is not a file"}

    if not original.get("sha256"):
        return {"path": path, "status": "not_verifiable", "reason": "Artifact has no recorded sha256"}

    current = file_artifact_metadata(path, workspace_dir)
    if current.get("verification_error") == "Path is outside workspace":
        status = "outside_workspace"
    elif not current.get("exists"):
        status = "missing"
    elif current.get("sha256") == original.get("sha256"):
        status = "matched"
    else:
        status = "changed"

    return {
        "path": path,
        "status": status,
        "recorded": original,
        "current": current,
    }


def verify_task_artifacts(data: dict[str, Any], workspace_dir: str | Path) -> dict[str, Any]:
    """Return transient verification status for active task file artifacts."""

    active_task_id = data.get("active_task_id")
    tasks = data.get("tasks", [])
    task = next((item for item in tasks if item.get("task_id") == active_task_id), None)
    if task is None:
        return {"total": 0, "matched": 0, "changed": 0, "missing": 0, "outside_workspace": 0, "not_verifiable": 0, "items": []}

    items = [verify_task_artifact(artifact, workspace_dir) for artifact in task.get("artifacts", []) if isinstance(artifact, dict)]
    counts = {
        "total": len(items),
        "matched": 0,
        "changed": 0,
        "missing": 0,
        "outside_workspace": 0,
        "not_verifiable": 0,
    }
    for item in items:
        status = str(item.get("status", "not_verifiable"))
        if status in counts:
            counts[status] += 1
    return {**counts, "items": items}


def format_task_memory(data: dict[str, Any], artifact_verification: dict[str, Any] | None = None) -> str:
    """Format task memory for CLI display."""

    tasks = data.get("tasks", [])
    active_task_id = data.get("active_task_id")
    if not tasks:
        return "No task memory recorded yet."

    active_task = None
    for task in tasks:
        if task.get("task_id") == active_task_id:
            active_task = task
            break

    if active_task is None:
        latest_task = tasks[-1]
        lines = [
            f"No active task. Stored tasks: {len(tasks)}",
            "",
            "Latest Task",
            f"  ID: {latest_task.get('task_id', '')}",
            f"  Goal: {latest_task.get('goal', '')}",
            f"  Type: {latest_task.get('task_type', '')}",
            f"  Status: {latest_task.get('status', '')}",
            f"  Updated: {latest_task.get('updated_at', '')}",
        ]
        return "\n".join(lines)

    lines = [
        "Current Task",
        f"  ID: {active_task.get('task_id', '')}",
        f"  Goal: {active_task.get('goal', '')}",
        f"  Type: {active_task.get('task_type', '')}",
        f"  Status: {active_task.get('status', '')}",
        f"  Updated: {active_task.get('updated_at', '')}",
    ]

    archived_summary = str(active_task.get("archived_steps_summary", "")).strip()
    if archived_summary:
        lines.extend(
            [
                "",
                "Archived Steps Summary:",
                f"  {archived_summary}",
            ]
        )

    sections = [
        ("Completed Steps", active_task.get("completed_steps", []), "description"),
        ("Decisions", active_task.get("decisions", []), "decision"),
        ("Artifacts", active_task.get("artifacts", []), "path"),
        ("Open Questions", active_task.get("open_questions", []), "question"),
        ("Next Steps", active_task.get("next_steps", []), "description"),
    ]

    verification_by_path: dict[str, str] = {}
    if artifact_verification:
        verification_by_path = {
            str(item.get("path", "")): str(item.get("status", "not_verifiable"))
            for item in artifact_verification.get("items", [])
            if isinstance(item, dict)
        }

    for title, items, main_key in sections:
        lines.append(f"\n{title}:")
        if not items:
            lines.append("  - none")
            continue
        for item in items:
            if isinstance(item, dict):
                value = item.get(main_key) or item.get("description") or item.get("content") or str(item)
                if title == "Artifacts":
                    status = verification_by_path.get(str(item.get("path", "")))
                    if status:
                        value = f"{value} [{status}]"
            else:
                value = str(item)
            lines.append(f"  - {value}")

    return "\n".join(lines)


class TaskMemoryHook:
    """System-level hook that records task progress without model tool calls.

    Wired into Agent.run() to:
    - Auto-start or resume a task at the beginning of each run
    - Record every tool execution as a step
    - Record file operations as verifiable artifacts (SHA256, size, mtime)
    - Auto-finish the task when the agent completes successfully
    """

    FILE_TOOLS = {"read_file", "write_file", "edit_file"}

    def __init__(
        self,
        memory_file: str,
        workspace_dir: str,
        episode_memory_file: str | None = None,
        auto_start: bool = True,
        auto_finish: bool = True,
    ):
        self.store = TaskMemoryStore(memory_file)
        self.workspace_dir = Path(workspace_dir).resolve()
        self.episode_store = (
            EpisodeMemoryStore(episode_memory_file) if episode_memory_file is not None else None
        )
        self.auto_start = auto_start
        self.auto_finish = auto_finish

    def get_resume_summary(self) -> str | None:
        """Return a human-readable summary of the active task, or None if no task is active.

        Used on CLI startup to inform the user that an incomplete task exists.
        """
        data = self.store.load()
        task = self.store.get_active_task(data)
        if task is None:
            return None
        steps = len(task.get("completed_steps", []))
        artifacts = len(task.get("artifacts", []))
        decisions = len(task.get("decisions", []))
        archived = "yes" if str(task.get("archived_steps_summary", "")).strip() else "no"
        created = task.get("created_at", "unknown")
        return (
            f"Task {task['task_id']} ({task.get('task_type', 'general')})\n"
            f"  Goal: {task.get('goal', 'unknown')}\n"
            f"  Steps completed: {steps}, Artifacts: {artifacts}, Decisions: {decisions}, Archived: {archived}\n"
            f"  Created: {created}"
        )

    def start_or_resume_task(self, goal: str, task_type: str = "general") -> None:
        if not self.auto_start:
            return
        self.store.ensure_active_task(goal=goal, task_type=task_type, source="auto")

    def record_tool_result(self, tool_name: str, arguments: dict[str, Any], result: ToolResult) -> None:
        status = "succeeded" if result.success else "failed"
        self.store.append_step(
            description=f"Tool {tool_name} {status}",
            source="auto",
        )

        if tool_name in self.FILE_TOOLS:
            path = str(arguments.get("path", ""))
            if not path:
                return
            metadata = file_artifact_metadata(path, self.workspace_dir)
            artifact_type = {
                "read_file": "file_read",
                "write_file": "file_written",
                "edit_file": "file_edited",
            }[tool_name]
            self.store.append_artifact(
                {
                    "path": metadata.get("path", path),
                    "artifact_type": artifact_type,
                    "description": f"{tool_name} {status}",
                    "tool": tool_name,
                    "success": result.success,
                    "verification": metadata,
                    "source": "auto",
                }
            )
            return

        if tool_name == "bash":
            self.store.append_artifact(
                {
                    "path": str(arguments.get("command", "")),
                    "artifact_type": "command",
                    "description": f"bash {status}",
                    "tool": tool_name,
                    "success": result.success,
                    "source": "auto",
                }
            )

    def finish_task(self, summary: str = "") -> None:
        if not self.auto_finish:
            return
        task = self.store.finish_active_task(summary=summary, source="auto")
        if task is None:
            return

        episode = build_episode_from_task(task, summary=summary, source="auto")
        if self.episode_store is not None:
            self.episode_store.append(episode.model_dump(mode="json"))

    def abandon_task(self, reason: str = "") -> None:
        self.store.abandon_active_task(reason=reason, source="user")

