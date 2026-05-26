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


def test_format_eval_report_includes_metrics_section_when_present():
    suite = EvalSuite(
        suite_id="metrics",
        name="Metrics Suite",
        version="1",
        tasks=[EvalTask(task_id="a", prompt="A")],
    )
    report = EvalRunReport(
        eval_run_id="eval-metrics",
        suite=suite,
        candidates=[EvalCandidate(candidate_id="gpt", model="gpt-4o")],
        results=[],
        metadata={
            "metrics": {
                "latency_ms": {"avg": 120.0, "p50": 100.0, "p95": 200.0},
                "tokens": {"avg": 42.0},
                "cost": {"per_passed": 0.05},
                "max_steps": {"count": 1, "rate": 0.25},
                "tool_evidence_failures": {"count": 2, "rate": 0.5},
                "scorer_failures": {"status": 1, "tool_evidence_contains": 2},
                "trace_linkage": {"count": 3, "rate": 0.75},
                "context_governance": {
                    "compression_triggered": {"count": 2, "rate": 0.5},
                    "avg_compression_ratio": 0.42,
                    "avg_tokens_before_compression": 1000.0,
                    "avg_tokens_after_compression": 580.0,
                },
                "observability": {"avg_llm_calls": 2.5, "avg_tool_calls": 1.25},
                "memory_effectiveness": {
                    "recall_notes_called": {"count": 3, "rate": 0.75},
                    "redundant_read_avoided": {"count": 2, "rate": 0.5},
                    "avg_recall_notes_calls": 1.5,
                    "avg_read_file_calls": 0.5,
                    "avg_record_note_calls": 0.25,
                    "avoided_read_token_estimate": 1200,
                },
            }
        },
    )

    markdown = format_eval_report(report)

    assert "## Metrics" in markdown
    assert "- Avg latency: 120ms" in markdown
    assert "- P95 latency: 200ms" in markdown
    assert "- Avg tokens: 42.00" in markdown
    assert "- Cost per passed task: 0.0500" in markdown
    assert "- Max-step rate: 25.00% (1)" in markdown
    assert "- Tool-evidence failure rate: 50.00% (2)" in markdown
    assert "- Scorer failures: status=1, tool_evidence_contains=2" in markdown
    assert "- Trace linkage rate: 75.00% (3)" in markdown
    assert "- Compression trigger rate: 50.00% (2)" in markdown
    assert "- Avg compression ratio: 42.00%" in markdown
    assert "- Avg compression tokens: 1000.00 -> 580.00" in markdown
    assert "- Avg LLM calls/task: 2.50" in markdown
    assert "- Avg tool calls/task: 1.25" in markdown
    assert "- Memory recall usage: 75.00% (3)" in markdown
    assert "- Redundant-read avoided rate: 50.00% (2)" in markdown
    assert "- Avg recall_notes calls/task: 1.50" in markdown
    assert "- Avg read_file calls/task: 0.50" in markdown
    assert "- Estimated avoided read tokens: 1200" in markdown
