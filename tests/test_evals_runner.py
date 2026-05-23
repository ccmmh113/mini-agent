"""Tests for evaluation suite runner."""

from __future__ import annotations

import pytest

from mini_agent.evals import EvalCandidate, EvalExecution, EvalSuite, EvalTask
from mini_agent.evals.runner import run_eval_suite


@pytest.mark.asyncio
async def test_run_eval_suite_scores_each_candidate_task_pair():
    suite = EvalSuite(
        suite_id="smoke",
        name="Smoke Suite",
        version="1",
        tasks=[
            EvalTask(task_id="direct", prompt="Answer", expected_output_contains=["done"]),
            EvalTask(task_id="file", prompt="Write file", expected_files={"result.md": "passed"}),
        ],
    )
    candidates = [
        EvalCandidate(candidate_id="model-a", model="gpt-a"),
        EvalCandidate(candidate_id="model-b", model="gpt-b"),
    ]

    async def run_candidate(candidate: EvalCandidate, task: EvalTask) -> EvalExecution:
        return EvalExecution(
            output=f"{candidate.candidate_id} done",
            agent_run_id=f"run-{candidate.candidate_id}-{task.task_id}",
            workspace_files={"result.md": "passed"},
            duration_ms=100,
            total_tokens=10,
            total_cost=0.01,
        )

    report = await run_eval_suite("eval-1", suite, candidates, run_candidate)

    assert report.case_count == 4
    assert report.failed == 0
    assert report.pass_rate == 1.0
    assert report.total_duration_ms == 400
    assert report.total_tokens == 40
    assert report.total_cost == 0.04
    assert [result.agent_run_id for result in report.results] == [
        "run-model-a-direct",
        "run-model-a-file",
        "run-model-b-direct",
        "run-model-b-file",
    ]


@pytest.mark.asyncio
async def test_run_eval_suite_records_candidate_failures_as_failed_results():
    suite = EvalSuite(
        suite_id="smoke",
        name="Smoke Suite",
        version="1",
        tasks=[EvalTask(task_id="direct", prompt="Answer", expected_output_contains=["done"])],
    )
    candidates = [EvalCandidate(candidate_id="model-a", model="gpt-a")]

    async def failing_candidate(candidate: EvalCandidate, task: EvalTask) -> EvalExecution:
        del candidate, task
        raise RuntimeError("provider unavailable")

    report = await run_eval_suite("eval-2", suite, candidates, failing_candidate)
    result = report.results[0]

    assert report.failed == 1
    assert result.passed is False
    assert result.status == "failed"
    assert result.agent_run_id is None
    assert result.failure_reason == "RuntimeError: provider unavailable"
    assert result.score.failure_reasons == ["RuntimeError: provider unavailable"]
