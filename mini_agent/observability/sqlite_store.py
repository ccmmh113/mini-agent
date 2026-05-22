import json
import sqlite3
from pathlib import Path
from typing import Any

from .events import LLMCallRecord, RunRecord, StepRecord, ToolCallRecord, TraceEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    workspace_dir TEXT NOT NULL,
    model TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_ms INTEGER,
    status TEXT NOT NULL,
    terminal_reason TEXT,
    total_steps INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    cache_write_tokens INTEGER NOT NULL,
    total_cost REAL NOT NULL,
    currency TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_steps (
    step_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_ms INTEGER,
    stop_reason TEXT
);

CREATE TABLE IF NOT EXISTS llm_calls (
    call_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_ms INTEGER,
    finish_reason TEXT,
    request_message_count INTEGER NOT NULL,
    request_tool_names_json TEXT NOT NULL,
    error TEXT,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    cache_write_tokens INTEGER NOT NULL,
    total_cost REAL NOT NULL,
    currency TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_calls (
    call_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step_index INTEGER,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_ms INTEGER,
    success INTEGER,
    policy_outcome TEXT,
    error TEXT,
    result_summary TEXT,
    affected_paths_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


class SQLiteTraceStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def save_run(self, run: RunRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (
                    run_id, workspace_dir, model, started_at, ended_at, duration_ms,
                    status, terminal_reason, total_steps, prompt_tokens,
                    completion_tokens, total_tokens, cached_tokens, cache_write_tokens,
                    total_cost, currency
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    workspace_dir = excluded.workspace_dir,
                    model = excluded.model,
                    started_at = excluded.started_at,
                    ended_at = excluded.ended_at,
                    duration_ms = excluded.duration_ms,
                    status = excluded.status,
                    terminal_reason = excluded.terminal_reason,
                    total_steps = excluded.total_steps,
                    prompt_tokens = excluded.prompt_tokens,
                    completion_tokens = excluded.completion_tokens,
                    total_tokens = excluded.total_tokens,
                    cached_tokens = excluded.cached_tokens,
                    cache_write_tokens = excluded.cache_write_tokens,
                    total_cost = excluded.total_cost,
                    currency = excluded.currency
                """,
                (
                    run.run_id,
                    run.workspace_dir,
                    run.model,
                    run.started_at,
                    run.ended_at,
                    run.duration_ms,
                    run.status.value,
                    run.terminal_reason,
                    run.total_steps,
                    run.prompt_tokens,
                    run.completion_tokens,
                    run.total_tokens,
                    run.cached_tokens,
                    run.cache_write_tokens,
                    run.total_cost,
                    run.currency,
                ),
            )

    def save_step(self, step: StepRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_steps (
                    step_id, run_id, step_index, started_at, ended_at, duration_ms,
                    stop_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(step_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    step_index = excluded.step_index,
                    started_at = excluded.started_at,
                    ended_at = excluded.ended_at,
                    duration_ms = excluded.duration_ms,
                    stop_reason = excluded.stop_reason
                """,
                (
                    step.step_id,
                    step.run_id,
                    step.step_index,
                    step.started_at,
                    step.ended_at,
                    step.duration_ms,
                    step.stop_reason,
                ),
            )

    def save_llm_call(self, call: LLMCallRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO llm_calls (
                    call_id, run_id, step_index, started_at, ended_at, duration_ms,
                    finish_reason, request_message_count, request_tool_names_json,
                    error, prompt_tokens, completion_tokens, total_tokens,
                    cached_tokens, cache_write_tokens, total_cost, currency
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(call_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    step_index = excluded.step_index,
                    started_at = excluded.started_at,
                    ended_at = excluded.ended_at,
                    duration_ms = excluded.duration_ms,
                    finish_reason = excluded.finish_reason,
                    request_message_count = excluded.request_message_count,
                    request_tool_names_json = excluded.request_tool_names_json,
                    error = excluded.error,
                    prompt_tokens = excluded.prompt_tokens,
                    completion_tokens = excluded.completion_tokens,
                    total_tokens = excluded.total_tokens,
                    cached_tokens = excluded.cached_tokens,
                    cache_write_tokens = excluded.cache_write_tokens,
                    total_cost = excluded.total_cost,
                    currency = excluded.currency
                """,
                (
                    call.call_id,
                    call.run_id,
                    call.step_index,
                    call.started_at,
                    call.ended_at,
                    call.duration_ms,
                    call.finish_reason,
                    call.request_message_count,
                    _json(call.request_tool_names),
                    call.error,
                    call.prompt_tokens,
                    call.completion_tokens,
                    call.total_tokens,
                    call.cached_tokens,
                    call.cache_write_tokens,
                    call.total_cost,
                    call.currency,
                ),
            )

    def save_tool_call(self, call: ToolCallRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tool_calls (
                    call_id, run_id, step_index, tool_name, arguments_json,
                    started_at, ended_at, duration_ms, success, policy_outcome,
                    error, result_summary, affected_paths_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(call_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    step_index = excluded.step_index,
                    tool_name = excluded.tool_name,
                    arguments_json = excluded.arguments_json,
                    started_at = excluded.started_at,
                    ended_at = excluded.ended_at,
                    duration_ms = excluded.duration_ms,
                    success = excluded.success,
                    policy_outcome = excluded.policy_outcome,
                    error = excluded.error,
                    result_summary = excluded.result_summary,
                    affected_paths_json = excluded.affected_paths_json
                """,
                (
                    call.call_id,
                    call.run_id,
                    call.step_index,
                    call.tool_name,
                    _json(call.arguments),
                    call.started_at,
                    call.ended_at,
                    call.duration_ms,
                    call.success,
                    call.policy_outcome,
                    call.error,
                    call.result_summary,
                    _json(call.affected_paths),
                ),
            )

    def save_event(self, event: TraceEvent) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_events (
                    event_id, run_id, kind, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.run_id,
                    event.kind.value,
                    event.created_at,
                    _json(event.payload),
                ),
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)
