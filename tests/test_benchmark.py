import sqlite3

import pytest

from benchmarks.agent_benchmark import run_benchmark, run_eval_benchmark
from mini_agent.evals import EvalRunReport, EvalSQLiteStore


@pytest.mark.asyncio
async def test_deterministic_benchmark_passes():
    report = await run_benchmark()

    assert report["case_count"] == 10
    assert report["failed"] == 0
    assert report["pass_rate"] == 1.0
    assert report["total_tokens"] > 0


@pytest.mark.asyncio
async def test_deterministic_benchmark_produces_eval_report_with_trace_links(tmp_path):
    trace_db = tmp_path / "traces.sqlite3"

    report = await run_eval_benchmark(trace_db_path=trace_db)

    assert isinstance(report, EvalRunReport)
    assert report.suite.suite_key == "mini-agent-harness@deterministic"
    assert report.case_count == 10
    assert report.failed == 0
    assert report.pass_rate == 1.0
    assert all(result.agent_run_id for result in report.results)

    with sqlite3.connect(trace_db) as connection:
        persisted_run_ids = {
            row[0]
            for row in connection.execute("select run_id from agent_runs")
        }
    result_run_ids = {result.agent_run_id for result in report.results}

    assert result_run_ids <= persisted_run_ids


@pytest.mark.asyncio
async def test_deterministic_benchmark_persists_eval_and_trace_to_shared_db(tmp_path):
    db_path = tmp_path / "evals.sqlite3"

    report = await run_eval_benchmark(db_path=db_path)

    loaded = EvalSQLiteStore(db_path).load_report(report.eval_run_id)
    assert loaded is not None
    assert loaded.case_count == 10
    assert loaded.failed == 0

    with sqlite3.connect(db_path) as connection:
        eval_run_count = connection.execute("select count(*) from eval_runs").fetchone()[0]
        eval_result_count = connection.execute("select count(*) from eval_results").fetchone()[0]
        linked_count = connection.execute(
            """
            select count(*)
            from eval_results er
            join agent_runs ar on ar.run_id = er.agent_run_id
            """
        ).fetchone()[0]

    assert eval_run_count == 1
    assert eval_result_count == 10
    assert linked_count == 10
