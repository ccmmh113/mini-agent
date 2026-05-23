"""Tests for eval CLI commands."""

from __future__ import annotations

import sqlite3

from mini_agent.cli import handle_eval_command, parse_args
from mini_agent.evals import EvalSQLiteStore


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
