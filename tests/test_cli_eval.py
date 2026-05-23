"""Tests for eval CLI commands."""

from __future__ import annotations

import sqlite3

from mini_agent.cli import handle_eval_command, parse_args
from mini_agent.evals import EvalCandidate, EvalResult, EvalRunReport, EvalSQLiteStore, EvalScore, EvalSuite, EvalTask


def test_parse_eval_run_command():
    args = parse_args(["eval", "run", "--db", "out/evals.sqlite3"])

    assert args.command == "eval"
    assert args.eval_command == "run"
    assert args.db == "out/evals.sqlite3"


def test_parse_eval_report_command():
    args = parse_args(["eval", "report", "--db", "out/evals.sqlite3"])

    assert args.command == "eval"
    assert args.eval_command == "report"
    assert args.db == "out/evals.sqlite3"


def test_parse_eval_run_real_candidate_command():
    args = parse_args(
        [
            "eval",
            "run",
            "--real",
            "--db",
            "out/evals.sqlite3",
            "--candidate",
            "gpt=configs/gpt.yaml",
            "--candidate",
            "deepseek=configs/deepseek.yaml",
            "--candidate",
            "claude=configs/claude.yaml",
            "--output-root",
            "outputs/evals",
            "--suite",
            "eval_suites/smoke.yaml",
        ]
    )

    assert args.command == "eval"
    assert args.eval_command == "run"
    assert args.real is True
    assert args.candidate == [
        "gpt=configs/gpt.yaml",
        "deepseek=configs/deepseek.yaml",
        "claude=configs/claude.yaml",
    ]
    assert args.output_root == "outputs/evals"
    assert args.suite == "eval_suites/smoke.yaml"


def test_eval_run_handler_writes_eval_and_trace_tables(tmp_path, capsys):
    db_path = tmp_path / "evals.sqlite3"
    args = parse_args(["eval", "run", "--db", str(db_path)])

    exit_code = handle_eval_command(args)

    assert exit_code == 0
    assert "mini-agent-harness-deterministic" in capsys.readouterr().out
    assert EvalSQLiteStore(db_path).load_latest_report() is not None
    with sqlite3.connect(db_path) as connection:
        linked_count = connection.execute(
            """
            select count(*)
            from eval_results er
            join agent_runs ar on ar.run_id = er.agent_run_id
            """
        ).fetchone()[0]
    assert linked_count == 10


def test_eval_run_handler_routes_real_candidates(tmp_path, capsys, monkeypatch):
    import benchmarks.agent_benchmark as agent_benchmark

    captured = {}
    suite_yaml = tmp_path / "suite.yaml"
    suite_yaml.write_text(
        """
suite_id: custom
name: Custom Suite
version: v1
tasks:
  - task_id: custom-task
    prompt: Answer custom task
    expected_output_contains:
      - ok
""".strip(),
        encoding="utf-8",
    )
    candidate = EvalCandidate(candidate_id="gpt", model="gpt-4o", label="gpt")
    suite = EvalSuite(
        suite_id="mini-agent-real-model",
        name="Mini Agent Real Model",
        version="real",
        tasks=[EvalTask(task_id="real-direct", prompt="Answer")],
    )
    report = EvalRunReport(
        eval_run_id="mini-agent-real-model",
        suite=suite,
        candidates=[candidate],
        results=[
            EvalResult(
                eval_run_id="mini-agent-real-model",
                suite_id=suite.suite_id,
                suite_version=suite.version,
                candidate_id="gpt",
                task_id="real-direct",
                agent_run_id="run-gpt",
                passed=True,
                score=EvalScore(passed=True, score=4, max_score=4),
            )
        ],
    )

    def fake_load_real_eval_candidates(specs):
        captured["specs"] = specs
        return ["loaded-gpt"]

    async def fake_run_real_eval_benchmark(candidates, output_root, db_path, suite=None):
        captured["candidates"] = candidates
        captured["output_root"] = output_root
        captured["db_path"] = db_path
        captured["suite"] = suite
        return report

    monkeypatch.setattr(agent_benchmark, "load_real_eval_candidates", fake_load_real_eval_candidates)
    monkeypatch.setattr(agent_benchmark, "run_real_eval_benchmark", fake_run_real_eval_benchmark)
    db_path = tmp_path / "evals.sqlite3"
    args = parse_args(
        [
            "eval",
            "run",
            "--real",
            "--candidate",
            "gpt=configs/gpt.yaml",
            "--db",
            str(db_path),
            "--output-root",
            str(tmp_path / "outputs"),
            "--suite",
            str(suite_yaml),
        ]
    )

    assert handle_eval_command(args) == 0

    assert captured["specs"] == ["gpt=configs/gpt.yaml"]
    assert captured["candidates"] == ["loaded-gpt"]
    assert captured["db_path"] == db_path.absolute()
    assert captured["suite"].suite_key == "custom@v1"
    assert "mini-agent-real-model" in capsys.readouterr().out


def test_eval_run_handler_rejects_suite_without_real(tmp_path, capsys):
    suite_yaml = tmp_path / "suite.yaml"
    args = parse_args(["eval", "run", "--suite", str(suite_yaml)])

    assert handle_eval_command(args) == 2

    assert "--suite requires --real" in capsys.readouterr().out


def test_eval_report_handler_prints_latest_markdown(tmp_path, capsys):
    db_path = tmp_path / "evals.sqlite3"
    run_args = parse_args(["eval", "run", "--db", str(db_path)])
    report_args = parse_args(["eval", "report", "--db", str(db_path)])

    assert handle_eval_command(run_args) == 0
    capsys.readouterr()
    assert handle_eval_command(report_args) == 0

    output = capsys.readouterr().out
    assert "# Evaluation Report: Mini Agent Harness" in output
    assert "**Eval Run:** `mini-agent-harness-deterministic`" in output
