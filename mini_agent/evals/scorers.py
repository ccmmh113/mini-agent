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


def _pass_or_fail(reasons: list[str]) -> tuple[bool, list[str]]:
    return not reasons, reasons
