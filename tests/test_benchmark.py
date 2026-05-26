import sqlite3

import pytest

from benchmarks.agent_benchmark import (
    RealBenchmarkCase,
    RealEvalCandidate,
    ScriptedLLM,
    ScriptedResponse,
    _build_real_llm,
    load_real_eval_candidates,
    run_benchmark,
    run_eval_benchmark,
    run_real_eval_benchmark,
)
from mini_agent.evals import EvalRunReport, EvalSQLiteStore
from mini_agent.evals import EvalSuite, EvalTask


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


def test_build_real_llm_converts_loaded_retry_config_to_runtime_retry_config(tmp_path):
    config_path = tmp_path / "deepseek.yaml"
    _write_candidate_config(config_path, provider="openai", model="deepseek-chat")
    candidate = load_real_eval_candidates([f"deepseek={config_path}"])[0]

    llm = _build_real_llm(candidate.config)

    assert hasattr(llm._client.retry_config, "retryable_exceptions")
    assert llm._client.retry_config.retryable_exceptions == (Exception,)


def test_real_eval_tools_include_long_term_memory_tools(tmp_path):
    from benchmarks import agent_benchmark

    tool_names = [tool.name for tool in agent_benchmark._real_tools_for_workspace(tmp_path)]

    assert "record_note" in tool_names
    assert "recall_notes" in tool_names


def test_real_eval_tools_can_disable_memory_tools_for_baseline(tmp_path):
    from benchmarks import agent_benchmark

    tool_names = [tool.name for tool in agent_benchmark._real_tools_for_workspace(tmp_path, enable_memory=False)]

    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "record_note" not in tool_names
    assert "recall_notes" not in tool_names


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
async def test_real_eval_benchmark_can_add_memory_disabled_baseline_candidate(tmp_path):
    gpt_config = tmp_path / "gpt.yaml"
    _write_candidate_config(gpt_config, provider="openai", model="gpt-4o")
    candidates = load_real_eval_candidates([f"gpt={gpt_config}"])
    cases = [
        RealBenchmarkCase(
            name="reuse-large-source",
            description="fake memory case",
            task="Answer",
            expect_output_contains=["ok"],
        )
    ]

    async def fake_runner(case: RealBenchmarkCase, candidate: RealEvalCandidate, output_root, trace_recorder):
        del output_root, trace_recorder
        read_calls = 3 if candidate.memory_mode == "off" else 1
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
            "tool_messages": read_calls,
            "message_count": 3 + read_calls,
            "tokens": {"prompt": 100 * read_calls, "completion": 2, "total": 100 * read_calls + 2, "cached": 0, "cache_write": 0},
            "cost": {"total_cost": 0.01, "currency": "USD"},
            "metadata": {
                "memory_effectiveness": {
                    "read_file_calls": read_calls,
                    "recall_notes_calls": 0 if candidate.memory_mode == "off" else 1,
                    "record_note_calls": 0 if candidate.memory_mode == "off" else 1,
                }
            },
            "output": f"{candidate.candidate_id} ok",
        }

    report = await run_real_eval_benchmark(
        candidates=candidates,
        output_root=tmp_path / "outputs",
        cases=cases,
        case_runner=fake_runner,
        enable_memory_baseline=True,
    )

    assert [candidate.candidate_id for candidate in report.candidates] == ["gpt", "gpt-memory-off"]
    assert report.candidates[1].metadata["baseline_for"] == "gpt"
    assert report.candidates[1].metadata["memory_mode"] == "off"
    comparison = report.metadata["metrics"]["memory_effectiveness"]["baseline_comparison"]
    assert comparison["pair_count"] == 1
    assert comparison["read_file_call_delta"] == 2
    assert comparison["read_file_call_reduction_rate"] == 2 / 3


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


@pytest.mark.asyncio
async def test_real_eval_benchmark_uses_custom_suite_and_persists_metrics(tmp_path):
    gpt_config = tmp_path / "gpt.yaml"
    _write_candidate_config(gpt_config, provider="openai", model="gpt-4o")
    candidates = load_real_eval_candidates([f"gpt={gpt_config}"])
    suite = EvalSuite(
        suite_id="custom",
        name="Custom Suite",
        version="1",
        tasks=[
            EvalTask(
                task_id="custom-task",
                prompt="Answer custom task",
                expected_output_contains=["custom ok"],
            )
        ],
    )

    async def fake_runner(case: RealBenchmarkCase, candidate: RealEvalCandidate, output_root, trace_recorder):
        del candidate, output_root, trace_recorder
        return {
            "name": case.name,
            "description": case.description,
            "passed": True,
            "checks": {"output_contains": True, "completed": True},
            "status": "completed",
            "agent_run_id": "run-custom",
            "workspace_files": {},
            "tool_evidence": [],
            "elapsed_ms": 123,
            "llm_calls": 1,
            "tool_messages": 0,
            "message_count": 3,
            "tokens": {"prompt": 11, "completion": 4, "total": 15, "cached": 0, "cache_write": 0},
            "cost": {"total_cost": 0.02, "currency": "USD"},
            "output": "custom ok",
        }

    report = await run_real_eval_benchmark(
        candidates=candidates,
        output_root=tmp_path / "outputs",
        suite=suite,
        case_runner=fake_runner,
        db_path=tmp_path / "evals.sqlite3",
    )
    loaded = EvalSQLiteStore(tmp_path / "evals.sqlite3").load_report(report.eval_run_id)

    assert report.suite.suite_key == "custom@1"
    assert report.results[0].task_id == "custom-task"
    assert report.metadata["metrics"]["latency_ms"]["avg"] == 123
    assert loaded.metadata["metrics"]["tokens"]["total"] == 15


@pytest.mark.asyncio
async def test_real_eval_benchmark_converts_suite_fixtures_and_token_limit(tmp_path):
    gpt_config = tmp_path / "gpt.yaml"
    _write_candidate_config(gpt_config, provider="openai", model="gpt-4o")
    candidates = load_real_eval_candidates([f"gpt={gpt_config}"])
    suite = EvalSuite(
        suite_id="context",
        name="Context Governance",
        version="1",
        tasks=[
            EvalTask(
                task_id="context-task",
                prompt="Read fixture and report.",
                expected_output_contains=["ok"],
                scorers=["status", "output_contains", "metadata_contains"],
                metadata={
                    "fixtures": {"fixtures/source.md": "fixture sentinel"},
                    "agent_overrides": {"token_limit": 900},
                    "expected_metadata_contains": {
                        "context_governance.compression_triggered": True,
                        "context_governance.token_limit": 900,
                    },
                },
            )
        ],
    )

    async def fake_runner(case: RealBenchmarkCase, candidate: RealEvalCandidate, output_root, trace_recorder):
        del candidate, output_root, trace_recorder
        assert case.files == {"fixtures/source.md": "fixture sentinel"}
        assert case.token_limit == 900
        return {
            "name": case.name,
            "description": case.description,
            "passed": True,
            "checks": {"output_contains": True, "completed": True},
            "status": "completed",
            "agent_run_id": "run-context",
            "workspace_files": {},
            "tool_evidence": [],
            "elapsed_ms": 20,
            "llm_calls": 1,
            "tool_messages": 0,
            "message_count": 3,
            "tokens": {"prompt": 10, "completion": 2, "total": 12, "cached": 0, "cache_write": 0},
            "cost": {"total_cost": 0.01, "currency": "USD"},
            "metadata": {
                "context_governance": {
                    "compression_triggered": True,
                    "compression_markers": ["context_snip"],
                    "token_limit": 900,
                }
            },
            "output": "ok",
        }

    report = await run_real_eval_benchmark(
        candidates=candidates,
        output_root=tmp_path / "outputs",
        suite=suite,
        case_runner=fake_runner,
    )

    assert report.failed == 0
    assert report.results[0].score.breakdown["metadata_contains"] is True
    assert report.results[0].metadata["context_governance"]["token_limit"] == 900


@pytest.mark.asyncio
async def test_real_eval_benchmark_records_memory_effectiveness_metadata(tmp_path, monkeypatch):
    gpt_config = tmp_path / "gpt.yaml"
    _write_candidate_config(gpt_config, provider="openai", model="gpt-4o")
    candidates = load_real_eval_candidates([f"gpt={gpt_config}"])
    suite = EvalSuite(
        suite_id="memory",
        name="Memory Effectiveness",
        version="1",
        tasks=[
            EvalTask(
                task_id="memory-reuse",
                prompt="Read once, record memory, recall it, then write result.",
                expected_files={"memory-result.md": ["MEMORY_REUSE_OK"]},
                scorers=["status", "file_contains", "metadata_contains"],
                metadata={
                    "fixtures": {
                        "archive/huge_source.md": "MEMORY_KEY=zircon\n" + "large context\n" * 200,
                    },
                    "memory_effectiveness": {
                        "avoid_read_files": ["archive/huge_source.md"],
                        "allowed_read_calls_per_avoided_file": 1,
                    },
                    "expected_metadata_contains": {
                        "memory_effectiveness.recall_notes_calls": 1,
                        "memory_effectiveness.read_file_calls": 1,
                        "memory_effectiveness.record_note_calls": 1,
                        "memory_effectiveness.redundant_read_avoided": True,
                    },
                },
            )
        ],
    )
    monkeypatch.setattr(
        "benchmarks.agent_benchmark._build_real_llm",
        lambda config: ScriptedLLM(
            [
                ScriptedResponse(tool_name="read_file", arguments={"path": "archive/huge_source.md"}),
                ScriptedResponse(
                    tool_name="record_note",
                    arguments={
                        "content": "MEMORY_REUSE_OK MEMORY_KEY=zircon",
                        "type": "project",
                        "name": "memory-reuse",
                    },
                ),
                ScriptedResponse(tool_name="recall_notes", arguments={"query": "zircon", "limit": 1}),
                ScriptedResponse(
                    tool_name="write_file",
                    arguments={"path": "memory-result.md", "content": "MEMORY_REUSE_OK\n"},
                ),
                ScriptedResponse(content="MEMORY_REUSE_OK"),
            ]
        ),
    )

    report = await run_real_eval_benchmark(
        candidates=candidates,
        output_root=tmp_path / "outputs",
        suite=suite,
    )

    memory = report.results[0].metadata["memory_effectiveness"]
    assert memory["tool_calls_by_name"]["read_file"] == 1
    assert memory["tool_calls_by_name"]["record_note"] == 1
    assert memory["tool_calls_by_name"]["recall_notes"] == 1
    assert memory["redundant_read_avoided"] is True
    assert memory["avoided_read_token_estimate"] > 0
    assert report.metadata["metrics"]["memory_effectiveness"]["recall_notes_called"]["rate"] == 1.0


@pytest.mark.asyncio
async def test_real_eval_benchmark_enables_checkpoint_and_task_memory_from_suite(tmp_path, monkeypatch):
    gpt_config = tmp_path / "gpt.yaml"
    _write_candidate_config(gpt_config, provider="openai", model="gpt-4o")
    candidates = load_real_eval_candidates([f"gpt={gpt_config}"])
    suite = EvalSuite(
        suite_id="stateful",
        name="Stateful Runtime",
        version="1",
        tasks=[
            EvalTask(
                task_id="stateful-task",
                prompt="Create memory-output.md containing STATEFUL_OK.",
                expected_output_contains=["STATEFUL_OK"],
                expected_files={
                    "memory-output.md": ["STATEFUL_OK"],
                    ".mini_agent/checkpoints/latest.json": ['"reason": "completed"'],
                    ".mini_agent/task_memory.json": ['"status": "completed"', "memory-output.md"],
                    ".mini_agent/episodes.jsonl": ["STATEFUL_OK"],
                },
                metadata={
                    "agent_overrides": {
                        "enable_checkpoint": True,
                        "enable_task_memory": True,
                    },
                },
            )
        ],
    )
    monkeypatch.setattr(
        "benchmarks.agent_benchmark._build_real_llm",
        lambda config: ScriptedLLM(
            [
                ScriptedResponse(
                    tool_name="write_file",
                    arguments={"path": "memory-output.md", "content": "STATEFUL_OK\n"},
                    prompt_tokens=10,
                    completion_tokens=2,
                ),
                ScriptedResponse(content="STATEFUL_OK", prompt_tokens=8, completion_tokens=2),
            ]
        ),
    )

    report = await run_real_eval_benchmark(
        candidates=candidates,
        output_root=tmp_path / "outputs",
        suite=suite,
    )

    assert report.failed == 0
    assert report.results[0].score.breakdown["file_contains"] is True
