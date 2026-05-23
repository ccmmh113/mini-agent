"""Tests for loading evaluation suites from YAML."""

from __future__ import annotations

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
