"""Tests for evaluation aggregate metrics."""

from __future__ import annotations

from mini_agent.evals import EvalCandidate, EvalResult, EvalRunReport, EvalScore, EvalSuite, EvalTask
from mini_agent.evals.metrics import compute_eval_metrics, with_eval_metrics


def _metrics_report() -> EvalRunReport:
    suite = EvalSuite(
        suite_id="metrics",
        name="Metrics",
        version="1",
        tasks=[EvalTask(task_id="a", prompt="A"), EvalTask(task_id="b", prompt="B")],
    )
    candidates = [
        EvalCandidate(candidate_id="gpt", model="gpt-4o"),
        EvalCandidate(candidate_id="deepseek", model="deepseek-chat"),
    ]
    results = [
        EvalResult(
            eval_run_id="eval-1",
            suite_id=suite.suite_id,
            suite_version=suite.version,
            candidate_id="gpt",
            task_id="a",
            agent_run_id="run-1",
            passed=True,
            score=EvalScore(passed=True, score=4, max_score=4, breakdown={"status": True, "tool_evidence_contains": True}),
            status="completed",
            duration_ms=100,
            total_tokens=10,
            total_cost=0.01,
            metadata={
                "context_governance": {
                    "compression_triggered": True,
                    "before_tokens": 1000,
                    "after_tokens": 400,
                    "compression_ratio": 0.6,
                },
                "observability": {"llm_call_count": 2, "tool_call_count": 1},
                "memory_effectiveness": {
                    "recall_notes_calls": 1,
                    "read_file_calls": 0,
                    "record_note_calls": 1,
                    "redundant_read_avoided": True,
                    "avoided_read_token_estimate": 300,
                },
            },
        ),
        EvalResult(
            eval_run_id="eval-1",
            suite_id=suite.suite_id,
            suite_version=suite.version,
            candidate_id="gpt",
            task_id="b",
            agent_run_id="run-2",
            passed=False,
            score=EvalScore(
                passed=False,
                score=2,
                max_score=4,
                breakdown={"status": True, "tool_evidence_contains": False},
                failure_reasons=["tool evidence missing fragment: write_file"],
            ),
            status="completed",
            duration_ms=200,
            total_tokens=20,
            total_cost=0.02,
            failure_reason="tool evidence missing fragment: write_file",
            metadata={
                "context_governance": {
                    "compression_triggered": False,
                    "before_tokens": 500,
                    "after_tokens": 500,
                    "compression_ratio": 0.0,
                },
                "observability": {"llm_call_count": 1, "tool_call_count": 0},
                "memory_effectiveness": {
                    "recall_notes_calls": 0,
                    "read_file_calls": 2,
                    "record_note_calls": 0,
                    "redundant_read_avoided": False,
                    "avoided_read_token_estimate": 0,
                },
            },
        ),
        EvalResult(
            eval_run_id="eval-1",
            suite_id=suite.suite_id,
            suite_version=suite.version,
            candidate_id="deepseek",
            task_id="a",
            agent_run_id="run-3",
            passed=False,
            score=EvalScore(
                passed=False,
                score=1,
                max_score=4,
                breakdown={"status": False, "tool_evidence_contains": True},
                failure_reasons=["expected status completed, got max_steps"],
            ),
            status="max_steps",
            duration_ms=400,
            total_tokens=40,
            total_cost=0.04,
            failure_reason="expected status completed, got max_steps",
        ),
    ]
    return EvalRunReport(eval_run_id="eval-1", suite=suite, candidates=candidates, results=results)


def test_compute_eval_metrics_aggregates_latency_cost_tokens_and_failures():
    metrics = compute_eval_metrics(_metrics_report())

    assert metrics["case_count"] == 3
    assert metrics["failed"] == 2
    assert metrics["pass_rate"] == 1 / 3
    assert metrics["latency_ms"]["avg"] == 700 / 3
    assert metrics["latency_ms"]["p50"] == 200
    assert metrics["latency_ms"]["p95"] == 400
    assert metrics["tokens"]["total"] == 70
    assert metrics["tokens"]["avg"] == 70 / 3
    assert metrics["cost"]["total"] == 0.07
    assert metrics["cost"]["avg"] == 0.07 / 3
    assert metrics["cost"]["per_passed"] == 0.07
    assert metrics["max_steps"]["count"] == 1
    assert metrics["status_failures"]["count"] == 1
    assert metrics["tool_evidence_failures"]["count"] == 1
    assert metrics["scorer_failures"] == {"status": 1, "tool_evidence_contains": 1}
    assert metrics["trace_linkage"]["count"] == 3
    assert metrics["trace_linkage"]["rate"] == 1.0
    assert metrics["context_governance"]["compression_triggered"]["count"] == 1
    assert metrics["context_governance"]["compression_triggered"]["rate"] == 0.5
    assert metrics["context_governance"]["avg_compression_ratio"] == 0.3
    assert metrics["context_governance"]["avg_tokens_before_compression"] == 750
    assert metrics["context_governance"]["avg_tokens_after_compression"] == 450
    assert metrics["observability"]["avg_llm_calls"] == 1.5
    assert metrics["observability"]["avg_tool_calls"] == 0.5
    assert metrics["memory_effectiveness"]["recall_notes_called"]["count"] == 1
    assert metrics["memory_effectiveness"]["recall_notes_called"]["rate"] == 0.5
    assert metrics["memory_effectiveness"]["redundant_read_avoided"]["count"] == 1
    assert metrics["memory_effectiveness"]["redundant_read_avoided"]["rate"] == 0.5
    assert metrics["memory_effectiveness"]["avg_recall_notes_calls"] == 0.5
    assert metrics["memory_effectiveness"]["avg_read_file_calls"] == 1.0
    assert metrics["memory_effectiveness"]["avg_record_note_calls"] == 0.5
    assert metrics["memory_effectiveness"]["avoided_read_token_estimate"] == 300
    assert metrics["candidates"]["gpt"]["pass_rate"] == 0.5
    assert metrics["candidates"]["deepseek"]["max_steps"] == 1


def test_compute_eval_metrics_compares_memory_baseline_read_calls():
    suite = EvalSuite(
        suite_id="memory-baseline",
        name="Memory Baseline",
        version="1",
        tasks=[EvalTask(task_id="reuse-large-source", prompt="Reuse memory")],
    )
    report = EvalRunReport(
        eval_run_id="eval-1",
        suite=suite,
        candidates=[
            EvalCandidate(candidate_id="gpt", model="gpt-4o"),
            EvalCandidate(
                candidate_id="gpt-memory-off",
                model="gpt-4o",
                metadata={"baseline_for": "gpt", "memory_mode": "off"},
            ),
        ],
        results=[
            EvalResult(
                eval_run_id="eval-1",
                suite_id=suite.suite_id,
                suite_version=suite.version,
                candidate_id="gpt",
                task_id="reuse-large-source",
                agent_run_id="run-gpt",
                passed=True,
                score=EvalScore(passed=True, score=1, max_score=1),
                total_tokens=900,
                metadata={
                    "memory_effectiveness": {
                        "read_file_calls": 1,
                        "recall_notes_calls": 1,
                        "record_note_calls": 1,
                    }
                },
            ),
            EvalResult(
                eval_run_id="eval-1",
                suite_id=suite.suite_id,
                suite_version=suite.version,
                candidate_id="gpt-memory-off",
                task_id="reuse-large-source",
                agent_run_id="run-gpt-off",
                passed=True,
                score=EvalScore(passed=True, score=1, max_score=1),
                total_tokens=1500,
                metadata={
                    "memory_effectiveness": {
                        "read_file_calls": 3,
                        "recall_notes_calls": 0,
                        "record_note_calls": 0,
                    }
                },
            ),
        ],
    )

    metrics = compute_eval_metrics(report)

    baseline = metrics["memory_effectiveness"]["baseline_comparison"]
    assert baseline["pair_count"] == 1
    assert baseline["baseline_read_file_calls"] == 3
    assert baseline["memory_read_file_calls"] == 1
    assert baseline["read_file_call_delta"] == 2
    assert baseline["read_file_call_reduction_rate"] == 2 / 3
    assert baseline["baseline_total_tokens"] == 1500
    assert baseline["memory_total_tokens"] == 900
    assert baseline["total_token_delta"] == 600


def test_with_eval_metrics_returns_report_with_metrics_metadata():
    report = with_eval_metrics(_metrics_report())

    assert "metrics" in report.metadata
    assert report.metadata["metrics"]["case_count"] == 3
    assert report.results[0].agent_run_id == "run-1"
