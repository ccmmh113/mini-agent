"""SQLite persistence for evaluation reports."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .spec import EvalCandidate, EvalResult, EvalRunReport, EvalScore, EvalSuite, EvalTask

_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    eval_run_id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL,
    suite_name TEXT NOT NULL,
    suite_version TEXT NOT NULL,
    suite_description TEXT NOT NULL,
    suite_metadata_json TEXT NOT NULL,
    suite_tasks_json TEXT NOT NULL,
    candidates_json TEXT NOT NULL,
    case_count INTEGER NOT NULL,
    failed INTEGER NOT NULL,
    pass_rate REAL NOT NULL,
    total_duration_ms REAL NOT NULL,
    total_tokens INTEGER NOT NULL,
    total_cost REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
    eval_run_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    suite_id TEXT NOT NULL,
    suite_version TEXT NOT NULL,
    agent_run_id TEXT,
    passed INTEGER NOT NULL,
    score REAL NOT NULL,
    max_score REAL NOT NULL,
    score_passed INTEGER NOT NULL,
    score_failure_reasons_json TEXT NOT NULL,
    output TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    total_cost REAL NOT NULL,
    currency TEXT NOT NULL,
    failure_reason TEXT,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (eval_run_id, candidate_id, task_id)
);

CREATE TABLE IF NOT EXISTS eval_score_breakdowns (
    eval_run_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    scorer TEXT NOT NULL,
    passed INTEGER NOT NULL,
    PRIMARY KEY (eval_run_id, candidate_id, task_id, scorer)
);

CREATE INDEX IF NOT EXISTS idx_eval_results_agent_run
ON eval_results (agent_run_id);
"""


class EvalSQLiteStore:
    """Persist and reconstruct standard evaluation reports."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def save_report(self, report: EvalRunReport) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO eval_runs (
                    eval_run_id, suite_id, suite_name, suite_version,
                    suite_description, suite_metadata_json, suite_tasks_json,
                    candidates_json, case_count, failed, pass_rate,
                    total_duration_ms, total_tokens, total_cost, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(eval_run_id) DO UPDATE SET
                    suite_id = excluded.suite_id,
                    suite_name = excluded.suite_name,
                    suite_version = excluded.suite_version,
                    suite_description = excluded.suite_description,
                    suite_metadata_json = excluded.suite_metadata_json,
                    suite_tasks_json = excluded.suite_tasks_json,
                    candidates_json = excluded.candidates_json,
                    case_count = excluded.case_count,
                    failed = excluded.failed,
                    pass_rate = excluded.pass_rate,
                    total_duration_ms = excluded.total_duration_ms,
                    total_tokens = excluded.total_tokens,
                    total_cost = excluded.total_cost,
                    created_at = excluded.created_at
                """,
                (
                    report.eval_run_id,
                    report.suite.suite_id,
                    report.suite.name,
                    report.suite.version,
                    report.suite.description,
                    _json(report.suite.metadata),
                    _json([asdict(task) for task in report.suite.tasks]),
                    _json([asdict(candidate) for candidate in report.candidates]),
                    report.case_count,
                    report.failed,
                    report.pass_rate,
                    report.total_duration_ms,
                    report.total_tokens,
                    report.total_cost,
                    created_at,
                ),
            )
            connection.execute("DELETE FROM eval_results WHERE eval_run_id = ?", (report.eval_run_id,))
            connection.execute("DELETE FROM eval_score_breakdowns WHERE eval_run_id = ?", (report.eval_run_id,))
            for result in report.results:
                self._save_result(connection, result)

    def load_report(self, eval_run_id: str) -> EvalRunReport | None:
        with self._connect() as connection:
            run = connection.execute(
                "SELECT * FROM eval_runs WHERE eval_run_id = ?",
                (eval_run_id,),
            ).fetchone()
            if run is None:
                return None
            result_rows = connection.execute(
                """
                SELECT * FROM eval_results
                WHERE eval_run_id = ?
                ORDER BY rowid
                """,
                (eval_run_id,),
            ).fetchall()
            breakdown_rows = connection.execute(
                """
                SELECT candidate_id, task_id, scorer, passed
                FROM eval_score_breakdowns
                WHERE eval_run_id = ?
                """,
                (eval_run_id,),
            ).fetchall()
        return _report_from_rows(run, result_rows, breakdown_rows)

    def load_latest_report(self) -> EvalRunReport | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT eval_run_id
                FROM eval_runs
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return self.load_report(row["eval_run_id"])

    def _save_result(self, connection: sqlite3.Connection, result: EvalResult) -> None:
        connection.execute(
            """
            INSERT INTO eval_results (
                eval_run_id, candidate_id, task_id, suite_id, suite_version,
                agent_run_id, passed, score, max_score, score_passed,
                score_failure_reasons_json, output, status, duration_ms,
                prompt_tokens, completion_tokens, total_tokens, total_cost,
                currency, failure_reason, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.eval_run_id,
                result.candidate_id,
                result.task_id,
                result.suite_id,
                result.suite_version,
                result.agent_run_id,
                int(result.passed),
                result.score.score,
                result.score.max_score,
                int(result.score.passed),
                _json(result.score.failure_reasons),
                result.output,
                result.status,
                result.duration_ms,
                result.prompt_tokens,
                result.completion_tokens,
                result.total_tokens,
                result.total_cost,
                result.currency,
                result.failure_reason,
                _json(result.metadata),
            ),
        )
        for scorer, passed in result.score.breakdown.items():
            connection.execute(
                """
                INSERT INTO eval_score_breakdowns (
                    eval_run_id, candidate_id, task_id, scorer, passed
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (result.eval_run_id, result.candidate_id, result.task_id, scorer, int(passed)),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection


def _report_from_rows(
    run: sqlite3.Row,
    result_rows: list[sqlite3.Row],
    breakdown_rows: list[sqlite3.Row],
) -> EvalRunReport:
    tasks = [EvalTask(**task) for task in _loads(run["suite_tasks_json"])]
    candidates = [EvalCandidate(**candidate) for candidate in _loads(run["candidates_json"])]
    suite = EvalSuite(
        suite_id=run["suite_id"],
        name=run["suite_name"],
        version=run["suite_version"],
        description=run["suite_description"],
        tasks=tasks,
        metadata=_loads(run["suite_metadata_json"]),
    )
    breakdowns: dict[tuple[str, str], dict[str, bool]] = {}
    for row in breakdown_rows:
        breakdowns.setdefault((row["candidate_id"], row["task_id"]), {})[row["scorer"]] = bool(row["passed"])
    results = [
        EvalResult(
            eval_run_id=row["eval_run_id"],
            suite_id=row["suite_id"],
            suite_version=row["suite_version"],
            candidate_id=row["candidate_id"],
            task_id=row["task_id"],
            agent_run_id=row["agent_run_id"],
            passed=bool(row["passed"]),
            score=EvalScore(
                passed=bool(row["score_passed"]),
                score=row["score"],
                max_score=row["max_score"],
                breakdown=breakdowns.get((row["candidate_id"], row["task_id"]), {}),
                failure_reasons=_loads(row["score_failure_reasons_json"]),
            ),
            output=row["output"],
            status=row["status"],
            duration_ms=row["duration_ms"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
            total_cost=row["total_cost"],
            currency=row["currency"],
            failure_reason=row["failure_reason"],
            metadata=_loads(row["metadata_json"]),
        )
        for row in result_rows
    ]
    return EvalRunReport(eval_run_id=run["eval_run_id"], suite=suite, candidates=candidates, results=results)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _loads(value: str) -> Any:
    return json.loads(value)
