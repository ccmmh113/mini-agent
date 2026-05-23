import sqlite3

import pytest

from benchmarks.agent_benchmark import (
    RealBenchmarkCase,
    RealEvalCandidate,
    ScriptedLLM,
    ScriptedResponse,
    load_real_eval_candidates,
    run_benchmark,
    run_eval_benchmark,
    run_real_eval_benchmark,
)
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


def _write_candidate_config(path, *, provider: str, model: str, api_base: str = "https://example.test/v1"):
    path.write_text(
        f"""
api_key: test-key
provider: {provider}
api_base: {api_base}
model: {model}
tools:
  enable_mcp: false
""".strip(),
        encoding="utf-8",
    )


def test_load_real_eval_candidates_from_named_config_specs(tmp_path):
    gpt_config = tmp_path / "gpt.yaml"
    deepseek_config = tmp_path / "deepseek.yaml"
    claude_config = tmp_path / "claude.yaml"
    _write_candidate_config(gpt_config, provider="openai", model="gpt-4o")
    _write_candidate_config(deepseek_config, provider="openai", model="deepseek-chat")
    _write_candidate_config(claude_config, provider="anthropic", model="claude-3-5-sonnet")

    candidates = load_real_eval_candidates(
        [
            f"gpt={gpt_config}",
            f"deepseek={deepseek_config}",
            f"claude={claude_config}",
        ]
    )

    assert [candidate.candidate_id for candidate in candidates] == ["gpt", "deepseek", "claude"]
    assert [candidate.config.llm.model for candidate in candidates] == [
        "gpt-4o",
        "deepseek-chat",
        "claude-3-5-sonnet",
    ]
    assert candidates[2].config.llm.provider == "anthropic"


@pytest.mark.asyncio
async def test_real_eval_benchmark_runs_each_case_for_each_candidate_with_fake_runner(tmp_path):
    gpt_config = tmp_path / "gpt.yaml"
    deepseek_config = tmp_path / "deepseek.yaml"
    claude_config = tmp_path / "claude.yaml"
    _write_candidate_config(gpt_config, provider="openai", model="gpt-4o")
    _write_candidate_config(deepseek_config, provider="openai", model="deepseek-chat")
    _write_candidate_config(claude_config, provider="anthropic", model="claude-3-5-sonnet")
    candidates = load_real_eval_candidates(
        [f"gpt={gpt_config}", f"deepseek={deepseek_config}", f"claude={claude_config}"]
    )
    cases = [
        RealBenchmarkCase(
            name="real-direct",
            description="fake real model case",
            task="Answer",
            expect_output_contains=["ok"],
        )
    ]

    async def fake_runner(case: RealBenchmarkCase, candidate: RealEvalCandidate, output_root, trace_recorder):
        del output_root, trace_recorder
        return {
            "name": case.name,
            "description": case.description,
            "passed": True,
            "checks": {"output_contains": True, "completed": True},
            "status": "completed",
            "agent_run_id": f"run-{candidate.candidate_id}-{case.name}",
            "workspace_files": {},
            "tool_evidence": [],
            "elapsed_ms": 10,
            "llm_calls": 1,
            "tool_messages": 0,
            "message_count": 3,
            "tokens": {"prompt": 10, "completion": 2, "total": 12, "cached": 0, "cache_write": 0},
            "cost": {"total_cost": 0.01, "currency": "USD"},
            "output": f"{candidate.candidate_id} ok",
        }

    report = await run_real_eval_benchmark(
        candidates=candidates,
        output_root=tmp_path / "outputs",
        cases=cases,
        case_runner=fake_runner,
        db_path=tmp_path / "evals.sqlite3",
    )

    assert report.suite.suite_key == "mini-agent-real-model@real"
    assert report.case_count == 3
    assert report.failed == 0
    assert [candidate.candidate_id for candidate in report.candidates] == ["gpt", "deepseek", "claude"]
    assert {result.agent_run_id for result in report.results} == {
        "run-gpt-real-direct",
        "run-deepseek-real-direct",
        "run-claude-real-direct",
    }
    assert EvalSQLiteStore(tmp_path / "evals.sqlite3").load_report(report.eval_run_id).case_count == 3


@pytest.mark.asyncio
async def test_real_eval_benchmark_persists_trace_links_with_default_runner(tmp_path, monkeypatch):
    gpt_config = tmp_path / "gpt.yaml"
    _write_candidate_config(gpt_config, provider="openai", model="gpt-4o")
    candidates = load_real_eval_candidates([f"gpt={gpt_config}"])
    case = RealBenchmarkCase(
        name="real-direct",
        description="fake default runner case",
        task="Answer",
        expect_output_contains=["ok"],
        max_steps=2,
    )
    monkeypatch.setattr(
        "benchmarks.agent_benchmark._build_real_llm",
        lambda config: ScriptedLLM([ScriptedResponse(content="ok", prompt_tokens=7, completion_tokens=2)]),
    )
    db_path = tmp_path / "evals.sqlite3"

    report = await run_real_eval_benchmark(
        candidates=candidates,
        output_root=tmp_path / "outputs",
        cases=[case],
        db_path=db_path,
    )

    assert report.case_count == 1
    assert report.failed == 0
    assert report.results[0].agent_run_id
    with sqlite3.connect(db_path) as connection:
        linked_count = connection.execute(
            """
            select count(*)
            from eval_results er
            join agent_runs ar on ar.run_id = er.agent_run_id
            """
        ).fetchone()[0]
    assert linked_count == 1
