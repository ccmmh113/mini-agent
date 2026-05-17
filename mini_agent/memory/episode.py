"""Episodic memory storage and task-to-episode conversion."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..schema import EpisodeMemoryState, EpisodeRecord


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def empty_episode_memory() -> dict[str, Any]:
    return EpisodeMemoryState().model_dump(mode="json")


def normalize_episode_memory(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return empty_episode_memory()

    try:
        normalized = EpisodeMemoryState.model_validate(data)
    except Exception:
        return empty_episode_memory()
    return normalized.model_dump(mode="json")


class EpisodeMemoryStore:
    """JSON/JSONL-backed store for episodic task summaries."""

    def __init__(self, memory_file: str):
        self.memory_file = Path(memory_file)

    def load(self) -> dict[str, Any]:
        if not self.memory_file.exists():
            return empty_episode_memory()

        if self.memory_file.suffix == ".jsonl":
            episodes: list[dict[str, Any]] = []
            try:
                lines = self.memory_file.read_text(encoding="utf-8").splitlines()
            except Exception:
                return empty_episode_memory()
            for line in lines:
                if not line.strip():
                    continue
                try:
                    episodes.append(EpisodeRecord.model_validate(json.loads(line)).model_dump(mode="json"))
                except Exception:
                    continue
            return {"episodes": episodes}

        try:
            data = json.loads(self.memory_file.read_text(encoding="utf-8"))
        except Exception:
            return empty_episode_memory()
        return normalize_episode_memory(data)

    def save(self, data: dict[str, Any]) -> None:
        normalized = normalize_episode_memory(data)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        if self.memory_file.suffix == ".jsonl":
            lines = [
                json.dumps(episode, ensure_ascii=False)
                for episode in normalized.get("episodes", [])
            ]
            self.memory_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            return
        self.memory_file.write_text(
            json.dumps(normalized, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def append(self, episode: dict[str, Any]) -> dict[str, Any]:
        normalized = EpisodeRecord.model_validate(episode).model_dump(mode="json")
        if self.memory_file.suffix == ".jsonl":
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.memory_file, "a", encoding="utf-8") as file:
                file.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            return self.load()

        data = self.load()
        data.setdefault("episodes", []).append(normalized)
        self.save(data)
        return data


def build_episode_from_task(task: dict[str, Any], summary: str = "", source: str = "auto") -> EpisodeRecord:
    completed_at = str(task.get("updated_at", _now()))
    final_summary = summary.strip() or str(task.get("summary", "")).strip()
    compact_artifacts: list[dict[str, Any]] = []
    for artifact in task.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        compact_artifacts.append(
            {
                "path": str(artifact.get("path", "")),
                "timestamp": str(artifact.get("timestamp", "")),
                "artifact_type": str(artifact.get("artifact_type", "artifact")),
                "description": str(artifact.get("description", "")),
                "tool": artifact.get("tool"),
                "success": artifact.get("success"),
                "source": artifact.get("source"),
            }
        )
    return EpisodeRecord(
        episode_id=f"episode-{task.get('task_id', uuid4().hex)}",
        task_id=str(task.get("task_id", "")),
        goal=str(task.get("goal", "")),
        task_type=str(task.get("task_type", "general")),
        status=str(task.get("status", "completed")),
        summary=final_summary,
        completed_steps=task.get("completed_steps", []),
        decisions=task.get("decisions", []),
        artifacts=compact_artifacts,
        archived_steps_summary=str(task.get("archived_steps_summary", "")),
        created_at=str(task.get("created_at", completed_at)),
        updated_at=completed_at,
        completed_at=completed_at,
        source=source,
        metadata={
            "completed_by": str(task.get("completed_by", source)),
        },
    )
