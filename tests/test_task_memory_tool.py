"""Tests for task memory runtime utilities."""

import json
from pathlib import Path

from mini_agent.schema import TaskMemoryState
from mini_agent.tools.task_memory_tool import (
    EpisodeMemoryStore,
    TaskMemoryHook,
    TaskMemoryStore,
    format_task_memory,
    verify_task_artifacts,
)


def test_task_memory_store_lifecycle(tmp_path: Path):
    memory_file = tmp_path / ".mini_agent" / "task_memory.json"
    store = TaskMemoryStore(str(memory_file))

    task = store.ensure_active_task(goal="Optimize BashTool safety", task_type="coding", source="test")
    assert task["goal"] == "Optimize BashTool safety"
    assert memory_file.exists()

    data = store.load()
    assert data["active_task_id"]
    assert len(data["tasks"]) == 1
    current = data["tasks"][0]
    assert current["goal"] == "Optimize BashTool safety"
    assert current["task_type"] == "coding"
    assert current["status"] == "active"

    store.append_step(description="Read bash_tool.py", source="test")

    data = store.load()
    active = store.get_active_task(data)
    assert active is not None
    active["decisions"].append(
        {
            "decision": "Use allowlist and denylist policy",
            "reason": "Keep command execution controllable",
            "timestamp": "2026-05-16T12:00:00",
        }
    )
    store.save(data)

    formatted = format_task_memory(store.load())
    assert "Current Task" in formatted
    assert "Read bash_tool.py" in formatted
    assert "Use allowlist and denylist policy" in formatted

    finished_task = store.finish_active_task(summary="Implemented safety policy", source="test")
    assert finished_task is not None
    data = store.load()
    assert data["active_task_id"] is None
    assert data["tasks"][0]["status"] == "completed"
    assert data["tasks"][0]["summary"] == "Implemented safety policy"


def test_task_memory_store_requires_active_task(tmp_path: Path):
    memory_file = tmp_path / ".mini_agent" / "task_memory.json"
    store = TaskMemoryStore(str(memory_file))

    task, error = store.require_active_task(store.load())

    assert task is None
    assert error is not None
    assert "No active task" in error.error
    assert not memory_file.exists()


def test_format_task_memory_without_file(tmp_path: Path):
    memory_file = tmp_path / ".mini_agent" / "task_memory.json"
    store = TaskMemoryStore(str(memory_file))

    assert format_task_memory(store.load()) == "No task memory recorded yet."


def test_task_memory_hook_manages_runtime_state_without_model_tools(tmp_path: Path):
    hook = TaskMemoryHook(
        memory_file=str(tmp_path / ".mini_agent" / "task_memory.json"),
        workspace_dir=str(tmp_path),
        episode_memory_file=str(tmp_path / ".mini_agent" / "episodes.jsonl"),
    )

    hook.start_or_resume_task(goal="Review runtime-managed task memory", task_type="coding")
    state = hook.store.load()
    assert state["active_task_id"]
    assert len(state["tasks"]) == 1

    hook.finish_task(summary="Completed via runtime hook")
    finished = hook.store.load()
    assert finished["active_task_id"] is None
    assert finished["tasks"][0]["summary"] == "Completed via runtime hook"


def test_task_memory_hook_can_abandon_active_task(tmp_path: Path):
    hook = TaskMemoryHook(
        memory_file=str(tmp_path / ".mini_agent" / "task_memory.json"),
        workspace_dir=str(tmp_path),
        episode_memory_file=str(tmp_path / ".mini_agent" / "episodes.jsonl"),
    )

    hook.start_or_resume_task(goal="Interrupted task", task_type="coding")
    hook.abandon_task(reason="User declined resume")

    state = hook.store.load()
    assert state["active_task_id"] is None
    assert state["tasks"][0]["status"] == "abandoned"


def test_task_memory_storage_conforms_to_schema(tmp_path: Path):
    memory_file = tmp_path / ".mini_agent" / "task_memory.json"
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    memory_file.write_text(
        json.dumps(
            {
                "active_task_id": "task-1",
                "tasks": [
                    {
                        "task_id": "task-1",
                        "goal": "Keep checkpoint compatibility",
                        "task_type": "coding",
                        "status": "active",
                        "completed_steps": [{"description": "Read task_memory_tool.py", "timestamp": "2026-05-15T12:00:00"}],
                        "decisions": [],
                        "artifacts": [],
                        "open_questions": [],
                        "next_steps": [],
                        "created_at": "2026-05-15T12:00:00",
                        "updated_at": "2026-05-15T12:01:00",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    state = TaskMemoryState.model_validate(json.loads(memory_file.read_text(encoding="utf-8")))
    assert state.active_task_id == "task-1"
    assert state.tasks[0].goal == "Keep checkpoint compatibility"


def test_verify_task_artifacts_reports_current_file_state(tmp_path: Path):
    tracked = tmp_path / "tracked.txt"
    missing = tmp_path / "missing.txt"
    tracked.write_text("original", encoding="utf-8")
    missing.write_text("temporary", encoding="utf-8")

    hook = TaskMemoryHook(
        memory_file=str(tmp_path / ".mini_agent" / "task_memory.json"),
        workspace_dir=str(tmp_path),
    )
    hook.start_or_resume_task(goal="Verify artifacts", task_type="coding")
    hook.record_tool_result("write_file", {"path": "tracked.txt"}, result=type("Result", (), {"success": True})())
    hook.record_tool_result("write_file", {"path": "missing.txt"}, result=type("Result", (), {"success": True})())

    data = hook.store.load()
    first_report = verify_task_artifacts(data, tmp_path)
    assert first_report["matched"] == 2
    assert first_report["changed"] == 0
    assert first_report["missing"] == 0

    tracked.write_text("changed", encoding="utf-8")
    missing.unlink()

    second_report = verify_task_artifacts(data, tmp_path)
    assert second_report["matched"] == 0
    assert second_report["changed"] == 1
    assert second_report["missing"] == 1

    persisted = TaskMemoryStore(str(tmp_path / ".mini_agent" / "task_memory.json")).load()
    assert persisted["tasks"][0]["artifacts"][0]["verification"]["sha256"] != second_report["items"][0]["current"]["sha256"]


def test_task_memory_compacts_long_histories(tmp_path: Path):
    memory_file = tmp_path / ".mini_agent" / "task_memory.json"
    store = TaskMemoryStore(str(memory_file))
    store.ensure_active_task(goal="Long-running task", task_type="coding", source="test")

    for index in range(55):
        store.append_step(description=f"step-{index}", source="test")
        store.append_artifact({"path": f"file-{index}.txt", "artifact_type": "file", "description": "artifact"})

    data = store.load()
    task = store.get_active_task(data)
    assert task is not None
    assert 20 <= len(task["completed_steps"]) <= 24
    assert task["completed_steps"][0]["description"] == "step-31"
    assert task["completed_steps"][-1]["description"] == "step-54"
    assert len(task["artifacts"]) == 20
    assert task["artifacts"][0]["path"].endswith("file-35.txt")
    assert "Archived" in task["archived_steps_summary"]

    formatted = format_task_memory(data)
    assert "Archived Steps Summary:" in formatted


def test_task_completion_records_episode_without_semantic_promotion(tmp_path: Path):
    task_memory_file = tmp_path / ".mini_agent" / "task_memory.json"
    episode_memory_file = tmp_path / ".mini_agent" / "episodes.jsonl"
    removed_memory_file = tmp_path / ".mini_agent" / "memories.json"

    hook = TaskMemoryHook(
        memory_file=str(task_memory_file),
        workspace_dir=str(tmp_path),
        episode_memory_file=str(episode_memory_file),
    )

    hook.start_or_resume_task(goal="Refactor memory pipeline", task_type="coding")
    hook.store.append_step("Read note_tool.py", source="auto")
    data = hook.store.load()
    task = hook.store.get_active_task(data)
    assert task is not None
    task["decisions"].append(
        {
            "decision": "Long-term memory should stay separate from episode review",
            "reason": "Reduce noise from runtime state",
            "timestamp": "2026-05-15T12:01:00",
        }
    )
    hook.store.save(data)

    hook.finish_task(summary="Recorded an episode without long-term memory promotion")

    episodes = EpisodeMemoryStore(str(episode_memory_file)).load()["episodes"]
    assert len(episodes) == 1
    assert episodes[0]["goal"] == "Refactor memory pipeline"
    assert episodes[0]["summary"] == "Recorded an episode without long-term memory promotion"
    assert len(episode_memory_file.read_text(encoding="utf-8").splitlines()) == 1
    assert episodes[0]["artifacts"] == []
    assert not removed_memory_file.exists()


def test_episode_compacts_artifact_verification_for_review(tmp_path: Path):
    task_memory_file = tmp_path / ".mini_agent" / "task_memory.json"
    episode_memory_file = tmp_path / ".mini_agent" / "episodes.jsonl"
    hook = TaskMemoryHook(
        memory_file=str(task_memory_file),
        workspace_dir=str(tmp_path),
        episode_memory_file=str(episode_memory_file),
    )

    tracked = tmp_path / "tracked.txt"
    tracked.write_text("hello", encoding="utf-8")
    hook.start_or_resume_task(goal="Review artifact payload", task_type="coding")
    hook.record_tool_result("write_file", {"path": "tracked.txt"}, result=type("Result", (), {"success": True})())
    hook.finish_task(summary="done")

    episode = EpisodeMemoryStore(str(episode_memory_file)).load()["episodes"][0]
    assert episode["artifacts"][0]["path"].endswith("tracked.txt")
    assert episode["artifacts"][0]["verification"] is None


def test_episode_memory_store_reads_legacy_json(tmp_path: Path):
    episode_memory_file = tmp_path / ".mini_agent" / "episodes.json"
    episode_memory_file.parent.mkdir(parents=True, exist_ok=True)
    episode_memory_file.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_id": "episode-task-1",
                        "task_id": "task-1",
                        "goal": "Legacy episode",
                        "task_type": "coding",
                        "status": "completed",
                        "summary": "Legacy JSON remains readable",
                        "completed_steps": [],
                        "decisions": [],
                        "artifacts": [],
                        "created_at": "2026-05-15T12:00:00",
                        "updated_at": "2026-05-15T12:01:00",
                        "completed_at": "2026-05-15T12:01:00",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    episodes = EpisodeMemoryStore(str(episode_memory_file)).load()["episodes"]

    assert len(episodes) == 1
    assert episodes[0]["goal"] == "Legacy episode"



