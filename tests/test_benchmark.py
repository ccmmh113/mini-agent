import sqlite3

import pytest

from benchmarks.agent_benchmark import run_benchmark, run_eval_benchmark
from mini_agent.evals import EvalRunReport


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
