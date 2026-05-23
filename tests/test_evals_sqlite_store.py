"""Tests for SQLite evaluation report persistence."""

from __future__ import annotations

import sqlite3

from mini_agent.evals import EvalCandidate, EvalResult, EvalRunReport, EvalScore, EvalSuite, EvalTask
from mini_agent.evals.sqlite_store import EvalSQLiteStore


def _sample_report(eval_run_id: str = "eval-1") -> EvalRunReport:
    suite = EvalSuite(
        suite_id="smoke",
        name="Smoke Suite",
        version="1",
        tasks=[
            EvalTask(
                task_id="direct",
                prompt="Answer directly",
                expected_output_contains=["done"],
            )
        ],
    )
    candidate = EvalCandidate(candidate_id="model-a", model="gpt-a", label="Model A")
    return EvalRunReport(
        eval_run_id=eval_run_id,
        suite=suite,
        candidates=[candidate],
        results=[
            EvalResult(
                eval_run_id=eval_run_id,
                suite_id=suite.suite_id,
                suite_version=suite.version,
                candidate_id=candidate.candidate_id,
                task_id="direct",
                agent_run_id="run-123",
                passed=True,
                score=EvalScore(
                    passed=True,
                    score=4,
                    max_score=4,
                    breakdown={"status": True, "output_contains": True},
                ),
                output="done",
                status="completed",
                duration_ms=42,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                total_cost=0.01,
                metadata={"source": "test"},
            )
        ],
    )


def test_eval_sqlite_store_round_trips_report_with_trace_link(tmp_path):
    store = EvalSQLiteStore(tmp_path / "evals.sqlite3")

    store.save_report(_sample_report())
    loaded = store.load_report("eval-1")

    assert loaded is not None
    assert loaded.eval_run_id == "eval-1"
    assert loaded.suite.suite_key == "smoke@1"
    assert loaded.candidates[0].label == "Model A"
    assert loaded.results[0].agent_run_id == "run-123"
    assert loaded.results[0].score.breakdown == {"status": True, "output_contains": True}
    assert loaded.results[0].metadata == {"source": "test"}


def test_eval_sqlite_store_loads_latest_report_by_created_order(tmp_path):
    store = EvalSQLiteStore(tmp_path / "evals.sqlite3")

    store.save_report(_sample_report("eval-old"))
    store.save_report(_sample_report("eval-new"))

    assert store.load_latest_report().eval_run_id == "eval-new"


def test_eval_sqlite_store_persists_score_breakdown_rows(tmp_path):
    db_path = tmp_path / "evals.sqlite3"
    store = EvalSQLiteStore(db_path)

    store.save_report(_sample_report())

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            select scorer, passed
            from eval_score_breakdowns
            where eval_run_id = ?
            order by scorer
            """,
            ("eval-1",),
        ).fetchall()

    assert rows == [("output_contains", 1), ("status", 1)]
