"""Tests for evaluation runtime contracts."""

from __future__ import annotations

from mini_agent.evals import EvalCandidate, EvalRunReport, EvalScore, EvalSuite, EvalTask, EvalResult


def test_eval_task_defaults_to_deterministic_scorers():
    task = EvalTask(
        task_id="task-write",
        prompt="Create result.md",
        expected_output_contains=["created"],
    )

    assert task.expected_status == "completed"
    assert task.scorers == [
        "status",
        "output_contains",
        "file_contains",
        "tool_evidence_contains",
    ]


def test_eval_suite_keeps_stable_identity_and_tasks():
    suite = EvalSuite(
        suite_id="smoke",
        name="Smoke Suite",
        version="2026.05",
        tasks=[
            EvalTask(task_id="direct", prompt="Answer directly"),
            EvalTask(task_id="write", prompt="Create a file"),
        ],
    )

    assert suite.suite_key == "smoke@2026.05"
    assert [task.task_id for task in suite.tasks] == ["direct", "write"]


def test_eval_candidate_records_model_label():
    candidate = EvalCandidate(candidate_id="minimax-m2", model="MiniMax-M2.5", label="MiniMax baseline")

    assert candidate.candidate_id == "minimax-m2"
    assert candidate.model == "MiniMax-M2.5"
    assert candidate.label == "MiniMax baseline"


def test_eval_run_report_aggregates_pass_rate_tokens_and_cost():
    suite = EvalSuite(
        suite_id="smoke",
        name="Smoke Suite",
        version="1",
        tasks=[EvalTask(task_id="a", prompt="A"), EvalTask(task_id="b", prompt="B")],
    )
    candidate = EvalCandidate(candidate_id="model-a", model="gpt-test")
    results = [
        EvalResult(
            eval_run_id="eval-1",
            suite_id=suite.suite_id,
            suite_version=suite.version,
            candidate_id=candidate.candidate_id,
            task_id="a",
            agent_run_id="run-a",
            passed=True,
            score=EvalScore(passed=True, score=1, max_score=1),
            duration_ms=1200,
            total_tokens=30,
            total_cost=0.01,
        ),
        EvalResult(
            eval_run_id="eval-1",
            suite_id=suite.suite_id,
            suite_version=suite.version,
            candidate_id=candidate.candidate_id,
            task_id="b",
            agent_run_id="run-b",
            passed=False,
            score=EvalScore(passed=False, score=0, max_score=1, failure_reasons=["missing output"]),
            duration_ms=800,
            total_tokens=20,
            total_cost=0.02,
            failure_reason="missing output",
        ),
    ]

    report = EvalRunReport(eval_run_id="eval-1", suite=suite, candidates=[candidate], results=results)

    assert report.case_count == 2
    assert report.failed == 1
    assert report.pass_rate == 0.5
    assert report.total_duration_ms == 2000
    assert report.total_tokens == 50
    assert report.total_cost == 0.03
