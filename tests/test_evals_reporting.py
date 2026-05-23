"""Tests for evaluation Markdown reporting."""

from __future__ import annotations

from mini_agent.evals import EvalCandidate, EvalResult, EvalRunReport, EvalScore, EvalSuite, EvalTask
from mini_agent.evals.reporting import format_eval_report


def test_format_eval_report_includes_summary_candidate_table_and_trace_ids():
    suite = EvalSuite(
        suite_id="smoke",
        name="Smoke Suite",
        version="1",
        tasks=[
            EvalTask(task_id="direct", prompt="Answer directly"),
            EvalTask(task_id="write", prompt="Write a file"),
        ],
    )
    candidates = [
        EvalCandidate(candidate_id="model-a", model="gpt-a", label="Model A"),
        EvalCandidate(candidate_id="model-b", model="gpt-b", label="Model B"),
    ]
    report = EvalRunReport(
        eval_run_id="eval-1",
        suite=suite,
        candidates=candidates,
        results=[
            EvalResult(
                eval_run_id="eval-1",
                suite_id="smoke",
                suite_version="1",
                candidate_id="model-a",
                task_id="direct",
                agent_run_id="run-a-direct",
                passed=True,
                score=EvalScore(passed=True, score=4, max_score=4),
                duration_ms=100,
                total_tokens=10,
                total_cost=0.01,
            ),
            EvalResult(
                eval_run_id="eval-1",
                suite_id="smoke",
                suite_version="1",
                candidate_id="model-a",
                task_id="write",
                agent_run_id="run-a-write",
                passed=True,
                score=EvalScore(passed=True, score=4, max_score=4),
                duration_ms=200,
                total_tokens=20,
                total_cost=0.02,
            ),
            EvalResult(
                eval_run_id="eval-1",
                suite_id="smoke",
                suite_version="1",
                candidate_id="model-b",
                task_id="direct",
                agent_run_id="run-b-direct",
                passed=False,
                score=EvalScore(passed=False, score=3, max_score=4, failure_reasons=["missing output"]),
                status="failed",
                duration_ms=300,
                total_tokens=30,
                total_cost=0.03,
                failure_reason="missing output",
            ),
        ],
    )

    markdown = format_eval_report(report)

    assert markdown.startswith("# Evaluation Report: Smoke Suite")
    assert "**Suite:** `smoke@1`" in markdown
    assert "**Eval Run:** `eval-1`" in markdown
    assert "**Pass Rate:** 66.67%" in markdown
    assert "| Candidate | Model | Cases | Failed | Pass Rate | Tokens | Cost | Duration |" in markdown
    assert "| Model A | `gpt-a` | 2 | 0 | 100.00% | 30 | 0.0300 | 300ms |" in markdown
    assert "| Model B | `gpt-b` | 1 | 1 | 0.00% | 30 | 0.0300 | 300ms |" in markdown
    assert "| model-b | direct | FAIL | 3/4 | failed | 300ms | 30 | 0.0300 | `run-b-direct` | missing output |" in markdown
