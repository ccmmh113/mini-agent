"""Evaluation suite orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from .scorers import score_task_result
from .spec import EvalCandidate, EvalExecution, EvalResult, EvalRunReport, EvalScore, EvalSuite, EvalTask

EvalCandidateRunner = Callable[[EvalCandidate, EvalTask], Awaitable[EvalExecution]]


async def run_eval_suite(
    eval_run_id: str,
    suite: EvalSuite,
    candidates: list[EvalCandidate],
    candidate_runner: EvalCandidateRunner,
) -> EvalRunReport:
    """Run every candidate against every task in a suite."""

    results: list[EvalResult] = []
    for candidate in candidates:
        for task in suite.tasks:
            results.append(await _run_one(eval_run_id, suite, candidate, task, candidate_runner))

    return EvalRunReport(eval_run_id=eval_run_id, suite=suite, candidates=candidates, results=results)


async def _run_one(
    eval_run_id: str,
    suite: EvalSuite,
    candidate: EvalCandidate,
    task: EvalTask,
    candidate_runner: EvalCandidateRunner,
) -> EvalResult:
    try:
        execution = await candidate_runner(candidate, task)
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        return EvalResult(
            eval_run_id=eval_run_id,
            suite_id=suite.suite_id,
            suite_version=suite.version,
            candidate_id=candidate.candidate_id,
            task_id=task.task_id,
            agent_run_id=None,
            passed=False,
            score=EvalScore(
                passed=False,
                score=0.0,
                max_score=float(len(task.scorers)),
                failure_reasons=[failure_reason],
            ),
            status="failed",
            failure_reason=failure_reason,
        )

    score = score_task_result(task, execution)
    return EvalResult(
        eval_run_id=eval_run_id,
        suite_id=suite.suite_id,
        suite_version=suite.version,
        candidate_id=candidate.candidate_id,
        task_id=task.task_id,
        agent_run_id=execution.agent_run_id,
        passed=score.passed,
        score=score,
        output=execution.output,
        status=execution.status,
        duration_ms=execution.duration_ms,
        prompt_tokens=execution.prompt_tokens,
        completion_tokens=execution.completion_tokens,
        total_tokens=execution.total_tokens,
        total_cost=execution.total_cost,
        currency=execution.currency,
        failure_reason="; ".join(score.failure_reasons) or None,
        metadata=execution.metadata,
    )
