"""Load evaluation suites from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .spec import DEFAULT_SCORERS, EvalSuite, EvalTask


def load_eval_suite_yaml(path: str | Path) -> EvalSuite:
    """Load an evaluation suite YAML file."""

    suite_path = Path(path)
    with open(suite_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("suite YAML must contain a mapping")

    tasks_data = data.get("tasks")
    if not isinstance(tasks_data, list) or not tasks_data:
        raise ValueError("suite YAML must define at least one task in tasks")

    tasks = [_task_from_yaml(index, task_data) for index, task_data in enumerate(tasks_data, start=1)]
    return EvalSuite(
        suite_id=_required_str(data, "suite_id"),
        name=_required_str(data, "name"),
        version=str(_required_value(data, "version")),
        description=str(data.get("description") or ""),
        tasks=tasks,
        metadata=_mapping_or_empty(data.get("metadata"), "metadata"),
    )


def _task_from_yaml(index: int, data: Any) -> EvalTask:
    if not isinstance(data, dict):
        raise ValueError(f"task {index} must be a mapping")
    return EvalTask(
        task_id=_required_str(data, "task_id"),
        prompt=_required_str(data, "prompt"),
        description=str(data.get("description") or ""),
        expected_output_contains=_string_list(data.get("expected_output_contains"), "expected_output_contains"),
        expected_files=_expected_files(data.get("expected_files")),
        expected_tool_evidence_contains=_string_list(
            data.get("expected_tool_evidence_contains"),
            "expected_tool_evidence_contains",
        ),
        expected_status=str(data.get("expected_status") or "completed"),
        scorers=_string_list(data.get("scorers"), "scorers") or list(DEFAULT_SCORERS),
        metadata=_mapping_or_empty(data.get("metadata"), "task metadata"),
    )


def _required_value(data: dict[str, Any], field: str) -> Any:
    if field not in data or data[field] in {None, ""}:
        raise ValueError(f"suite YAML missing required field: {field}")
    return data[field]


def _required_str(data: dict[str, Any], field: str) -> str:
    value = _required_value(data, field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} entries must be strings")
    return list(value)


def _expected_files(value: Any) -> dict[str, str | list[str]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("expected_files must be a mapping")
    expected: dict[str, str | list[str]] = {}
    for path, fragments in value.items():
        if not isinstance(path, str):
            raise ValueError("expected_files paths must be strings")
        if isinstance(fragments, str):
            expected[path] = fragments
        elif isinstance(fragments, list) and all(isinstance(item, str) for item in fragments):
            expected[path] = list(fragments)
        else:
            raise ValueError("expected_files values must be strings or string lists")
    return expected


def _mapping_or_empty(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)
