"""Tests for loading evaluation suites from YAML."""

from __future__ import annotations

from pathlib import Path

import pytest

from mini_agent.evals import load_eval_suite_yaml


def test_load_eval_suite_yaml_maps_tasks_and_expectations(tmp_path):
    suite_path = tmp_path / "agent-core.yaml"
    suite_path.write_text(
        """
suite_id: agent-core
name: Agent Core Capability
version: 2026-05-23
description: Core Agent task evaluation
metadata:
  owner: qa
tasks:
  - task_id: write-report
    prompt: "创建 report.md，总结项目能力"
    description: Writes a markdown report
    expected_output_contains:
      - "完成"
    expected_files:
      report.md:
        - "Agent"
        - "评测"
    expected_tool_evidence_contains:
      - "write_file"
    expected_status: completed
    scorers:
      - status
      - output_contains
      - file_contains
    metadata:
      difficulty: smoke
""".strip(),
        encoding="utf-8",
    )

    suite = load_eval_suite_yaml(suite_path)

    assert suite.suite_key == "agent-core@2026-05-23"
    assert suite.description == "Core Agent task evaluation"
    assert suite.metadata == {"owner": "qa"}
    task = suite.tasks[0]
    assert task.task_id == "write-report"
    assert task.prompt == "创建 report.md，总结项目能力"
    assert task.expected_output_contains == ["完成"]
    assert task.expected_files == {"report.md": ["Agent", "评测"]}
    assert task.expected_tool_evidence_contains == ["write_file"]
    assert task.expected_status == "completed"
    assert task.scorers == ["status", "output_contains", "file_contains"]
    assert task.metadata == {"difficulty": "smoke"}


def test_load_eval_suite_yaml_rejects_missing_tasks(tmp_path):
    suite_path = tmp_path / "empty.yaml"
    suite_path.write_text(
        """
suite_id: empty
name: Empty
version: 1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tasks"):
        load_eval_suite_yaml(suite_path)


def test_load_eval_suite_yaml_rejects_task_without_prompt(tmp_path):
    suite_path = tmp_path / "bad.yaml"
    suite_path.write_text(
        """
suite_id: bad
name: Bad
version: 1
tasks:
  - task_id: missing-prompt
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompt"):
        load_eval_suite_yaml(suite_path)


def test_repo_eval_suites_load_successfully():
    suite_paths = sorted(Path("eval_suites").glob("*.yaml"))

    assert suite_paths
    for suite_path in suite_paths:
        suite = load_eval_suite_yaml(suite_path)
        assert suite.suite_id
        assert suite.version
        assert suite.tasks


def test_context_governance_multi_step_state_task_scores_file_artifacts_over_final_reply():
    suite = load_eval_suite_yaml(Path("eval_suites/context_governance_suite.yaml"))
    task = next(task for task in suite.tasks if task.task_id == "multi_step_state_integrity_under_pressure")

    assert task.expected_output_contains == ["STATE_OK"]
    assert "output_contains" not in task.scorers
    assert "output_excludes" not in task.scorers
    assert "file_contains" in task.scorers
    assert "file_excludes" in task.scorers


def test_task_memory_suite_includes_memory_effectiveness_cases():
    suite = load_eval_suite_yaml(Path("eval_suites/task_memory_suite.yaml"))
    task_ids = {task.task_id for task in suite.tasks}

    assert {
        "task_memory_records_file_artifact",
        "task_memory_records_command_artifact",
        "long_term_memory_recalls_project_decision_under_noise",
        "task_memory_reuses_active_state_without_rereading_large_source",
        "task_memory_prefers_latest_instruction_over_stale_memory",
        "task_memory_episode_records_cross_task_summary",
        "memory_recall_reduces_redundant_reading",
        "task_memory_cross_task_continuation",
        "long_term_memory_selects_correct_record_under_similar_noise",
    }.issubset(task_ids)

    reuse_task = next(
        task
        for task in suite.tasks
        if task.task_id == "task_memory_reuses_active_state_without_rereading_large_source"
    )
    assert "tool_evidence_excludes" in reuse_task.scorers
    assert reuse_task.metadata["expected_tool_evidence_not_contains"] == ["DO_NOT_REREAD_HUGE_SOURCE"]

    recall_task = next(task for task in suite.tasks if task.task_id == "memory_recall_reduces_redundant_reading")
    assert "metadata_contains" in recall_task.scorers
    assert recall_task.metadata["memory_effectiveness"]["avoid_read_files"] == ["archive/product_spec_large.md"]
    assert recall_task.metadata["memory_effectiveness"]["allowed_read_calls_per_avoided_file"] == 1
