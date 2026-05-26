"""Deterministic scoring rules for Mini Agent evaluations."""

from __future__ import annotations

from .spec import EvalExecution, EvalScore, EvalTask


def score_task_result(task: EvalTask, execution: EvalExecution) -> EvalScore:
    """Score one task execution with the task's configured deterministic scorers."""

    breakdown: dict[str, bool] = {}
    failure_reasons: list[str] = []
    score = 0.0
    max_score = 0.0

    for scorer in task.scorers:
        if scorer == "status":
            passed, reasons = _score_status(task, execution)
        elif scorer == "output_contains":
            passed, reasons = _score_output_contains(task, execution)
        elif scorer == "file_contains":
            passed, reasons = _score_file_contains(task, execution)
        elif scorer == "tool_evidence_contains":
            passed, reasons = _score_tool_evidence(task, execution)
        elif scorer == "metadata_contains":
            passed, reasons = _score_metadata_contains(task, execution)
        elif scorer == "output_excludes":
            passed, reasons = _score_output_excludes(task, execution)
        elif scorer == "file_excludes":
            passed, reasons = _score_file_excludes(task, execution)
        elif scorer == "tool_evidence_excludes":
            passed, reasons = _score_tool_evidence_excludes(task, execution)
        else:
            passed, reasons = False, [f"unknown scorer: {scorer}"]

        max_score += 1
        if passed:
            score += 1
        breakdown[scorer] = passed
        failure_reasons.extend(reasons)

    return EvalScore(
        passed=not failure_reasons,
        score=score,
        max_score=max_score,
        breakdown=breakdown,
        failure_reasons=failure_reasons,
    )


def _score_status(task: EvalTask, execution: EvalExecution) -> tuple[bool, list[str]]:
    if execution.status == task.expected_status:
        return True, []
    return False, [f"expected status {task.expected_status}, got {execution.status}"]


def _score_output_contains(task: EvalTask, execution: EvalExecution) -> tuple[bool, list[str]]:
    missing = [fragment for fragment in task.expected_output_contains if fragment not in execution.output]
    return _pass_or_fail([f"output missing fragment: {fragment}" for fragment in missing])


def _score_file_contains(task: EvalTask, execution: EvalExecution) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for path, expected_value in task.expected_files.items():
        if path not in execution.workspace_files:
            reasons.append(f"expected file missing: {path}")
            continue
        expected_fragments = expected_value if isinstance(expected_value, list) else [expected_value]
        for expected_fragment in expected_fragments:
            if expected_fragment not in execution.workspace_files[path]:
                reasons.append(f"file {path} missing fragment: {expected_fragment}")
    return _pass_or_fail(reasons)


def _score_tool_evidence(task: EvalTask, execution: EvalExecution) -> tuple[bool, list[str]]:
    evidence_text = "\n".join(execution.tool_evidence)
    missing = [fragment for fragment in task.expected_tool_evidence_contains if fragment not in evidence_text]
    return _pass_or_fail([f"tool evidence missing fragment: {fragment}" for fragment in missing])


def _score_tool_evidence_excludes(task: EvalTask, execution: EvalExecution) -> tuple[bool, list[str]]:
    forbidden = task.metadata.get("expected_tool_evidence_not_contains")
    if forbidden is None:
        return True, []
    if not isinstance(forbidden, list) or not all(isinstance(item, str) for item in forbidden):
        return False, ["metadata expected_tool_evidence_not_contains must be a string list"]
    evidence_text = "\n".join(execution.tool_evidence)
    found = [fragment for fragment in forbidden if fragment in evidence_text]
    return _pass_or_fail([f"tool evidence contains forbidden fragment: {fragment}" for fragment in found])


def _score_metadata_contains(task: EvalTask, execution: EvalExecution) -> tuple[bool, list[str]]:
    expected = task.metadata.get("expected_metadata_contains")
    if expected is None:
        return True, []
    if not isinstance(expected, dict):
        return False, ["metadata expected_metadata_contains must be a mapping"]

    reasons: list[str] = []
    for path, expected_value in expected.items():
        if not isinstance(path, str):
            reasons.append("metadata expectation paths must be strings")
            continue
        found, actual_value = _metadata_path(execution.metadata, path)
        if not found:
            reasons.append(f"metadata missing path: {path}")
            continue
        reasons.extend(_metadata_value_reasons(path, actual_value, expected_value))
    return _pass_or_fail(reasons)


def _score_output_excludes(task: EvalTask, execution: EvalExecution) -> tuple[bool, list[str]]:
    forbidden = task.metadata.get("expected_output_not_contains")
    if forbidden is None:
        return True, []
    if not isinstance(forbidden, list) or not all(isinstance(item, str) for item in forbidden):
        return False, ["metadata expected_output_not_contains must be a string list"]
    found = [fragment for fragment in forbidden if fragment in execution.output]
    return _pass_or_fail([f"output contains forbidden fragment: {fragment}" for fragment in found])


def _score_file_excludes(task: EvalTask, execution: EvalExecution) -> tuple[bool, list[str]]:
    forbidden = task.metadata.get("expected_files_not_contains")
    if forbidden is None:
        return True, []
    if not isinstance(forbidden, dict):
        return False, ["metadata expected_files_not_contains must be a mapping"]

    reasons: list[str] = []
    for path, fragments in forbidden.items():
        if not isinstance(path, str):
            reasons.append("metadata expected_files_not_contains paths must be strings")
            continue
        if isinstance(fragments, str):
            expected_fragments = [fragments]
        elif isinstance(fragments, list) and all(isinstance(item, str) for item in fragments):
            expected_fragments = fragments
        else:
            reasons.append("metadata expected_files_not_contains values must be strings or string lists")
            continue
        content = execution.workspace_files.get(path, "")
        for fragment in expected_fragments:
            if fragment in content:
                reasons.append(f"file {path} contains forbidden fragment: {fragment}")
    return _pass_or_fail(reasons)


def _metadata_path(metadata: dict[str, object], path: str) -> tuple[bool, object]:
    current: object = metadata
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _metadata_value_reasons(path: str, actual_value: object, expected_value: object) -> list[str]:
    if isinstance(expected_value, str):
        if isinstance(actual_value, list):
            if expected_value in [str(item) for item in actual_value]:
                return []
        elif expected_value in str(actual_value):
            return []
        return [f"metadata {path} missing fragment: {expected_value}"]

    if isinstance(expected_value, list):
        actual_text = "\n".join(str(item) for item in actual_value) if isinstance(actual_value, list) else str(actual_value)
        missing = [str(fragment) for fragment in expected_value if str(fragment) not in actual_text]
        return [f"metadata {path} missing fragment: {fragment}" for fragment in missing]

    if actual_value == expected_value:
        return []
    return [f"metadata {path} expected {expected_value}, got {actual_value}"]


def _pass_or_fail(reasons: list[str]) -> tuple[bool, list[str]]:
    return not reasons, reasons
