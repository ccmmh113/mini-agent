import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from mini_agent.observability import (
    LLMCallRecord,
    RunRecord,
    RunStatus,
    SQLiteTraceStore,
    StepRecord,
    ToolCallRecord,
    TraceEvent,
    TraceEventKind,
)


def test_sqlite_trace_store_persists_trace_records(tmp_path):
    db_path = tmp_path / "traces" / "trace.sqlite3"
    store = SQLiteTraceStore(db_path)

    store.save_run(
        RunRecord(
            run_id="run-1",
            workspace_dir="workspace",
            model="gpt-5",
            started_at="2026-05-22T00:00:00Z",
        )
    )
    store.save_run(
        RunRecord(
            run_id="run-1",
            workspace_dir="workspace",
            model="gpt-5",
            started_at="2026-05-22T00:00:00Z",
            ended_at="2026-05-22T00:00:09Z",
            duration_ms=9000,
            status=RunStatus.COMPLETED,
            total_steps=1,
            total_tokens=25,
            total_cost=0.0025,
        )
    )
    store.save_step(
        StepRecord(
            step_id="step-1",
            run_id="run-1",
            step_index=0,
            started_at="2026-05-22T00:00:01Z",
            ended_at="2026-05-22T00:00:08Z",
            stop_reason="tool_calls",
        )
    )
    store.save_llm_call(
        LLMCallRecord(
            call_id="llm-1",
            run_id="run-1",
            step_index=0,
            started_at="2026-05-22T00:00:02Z",
            ended_at="2026-05-22T00:00:04Z",
            finish_reason="tool_calls",
            request_message_count=2,
            request_tool_names=["write_file", "read_file"],
            total_tokens=25,
        )
    )
    store.save_tool_call(
        ToolCallRecord(
            call_id="tool-1",
            run_id="run-1",
            step_index=0,
            tool_name="write_file",
            arguments={"path": "notes/trace.txt", "content": "done"},
            started_at="2026-05-22T00:00:05Z",
            ended_at="2026-05-22T00:00:07Z",
            success=True,
            affected_paths=["notes/trace.txt"],
        )
    )
    store.save_event(
        TraceEvent(
            event_id="event-1",
            run_id="run-1",
            kind=TraceEventKind.RUN_COMPLETED,
            created_at="2026-05-22T00:00:09Z",
            payload={"total_steps": 1},
        )
    )

    with sqlite3.connect(db_path) as connection:
        run = connection.execute(
            "SELECT status, total_steps FROM agent_runs WHERE run_id = ?",
            ("run-1",),
        ).fetchone()
        step = connection.execute(
            "SELECT step_index, stop_reason FROM agent_steps WHERE step_id = ?",
            ("step-1",),
        ).fetchone()
        llm_call = connection.execute(
            """
            SELECT finish_reason, request_tool_names_json
            FROM llm_calls
            WHERE call_id = ?
            """,
            ("llm-1",),
        ).fetchone()
        tool_call = connection.execute(
            "SELECT affected_paths_json FROM tool_calls WHERE call_id = ?",
            ("tool-1",),
        ).fetchone()
        event = connection.execute(
            "SELECT kind, payload_json FROM run_events WHERE event_id = ?",
            ("event-1",),
        ).fetchone()

    assert run == ("completed", 1)
    assert step == (0, "tool_calls")
    assert llm_call[0] == "tool_calls"
    assert json.loads(llm_call[1]) == ["write_file", "read_file"]
    assert json.loads(tool_call[0]) == ["notes/trace.txt"]
    assert event[0] == "run_completed"
    assert json.loads(event[1]) == {"total_steps": 1}


def test_sqlite_trace_store_persists_pydantic_serializable_structured_json(tmp_path):
    db_path = tmp_path / "trace.sqlite3"
    store = SQLiteTraceStore(db_path)

    store.save_tool_call(
        ToolCallRecord(
            call_id="tool-structured",
            run_id="run-1",
            step_index=0,
            tool_name="write_file",
            arguments={"path": Path("notes/trace.txt")},
            started_at="2026-05-22T00:00:05Z",
        )
    )
    store.save_event(
        TraceEvent(
            event_id="event-structured",
            run_id="run-1",
            kind=TraceEventKind.TOOL_COMPLETED,
            created_at="2026-05-22T00:00:06Z",
            payload={"completed_at": datetime(2026, 5, 22, tzinfo=timezone.utc)},
        )
    )

    with sqlite3.connect(db_path) as connection:
        arguments_json = connection.execute(
            "SELECT arguments_json FROM tool_calls WHERE call_id = ?",
            ("tool-structured",),
        ).fetchone()[0]
        payload_json = connection.execute(
            "SELECT payload_json FROM run_events WHERE event_id = ?",
            ("event-structured",),
        ).fetchone()[0]

    assert json.loads(arguments_json) == {"path": str(Path("notes/trace.txt"))}
    assert json.loads(payload_json) == {"completed_at": "2026-05-22T00:00:00Z"}


def test_sqlite_trace_store_rejects_duplicate_step_run_index(tmp_path):
    store = SQLiteTraceStore(tmp_path / "trace.sqlite3")

    store.save_step(
        StepRecord(
            step_id="step-1",
            run_id="run-1",
            step_index=0,
            started_at="2026-05-22T00:00:01Z",
        )
    )

    with pytest.raises(sqlite3.IntegrityError):
        store.save_step(
            StepRecord(
                step_id="step-replayed",
                run_id="run-1",
                step_index=0,
                started_at="2026-05-22T00:00:02Z",
            )
        )


def test_sqlite_trace_store_upserts_llm_call_updates(tmp_path):
    db_path = tmp_path / "trace.sqlite3"
    store = SQLiteTraceStore(db_path)

    store.save_llm_call(
        LLMCallRecord(
            call_id="llm-1",
            run_id="run-1",
            step_index=0,
            started_at="2026-05-22T00:00:02Z",
        )
    )
    store.save_llm_call(
        LLMCallRecord(
            call_id="llm-1",
            run_id="run-1",
            step_index=0,
            started_at="2026-05-22T00:00:02Z",
            ended_at="2026-05-22T00:00:04Z",
            finish_reason="stop",
            total_tokens=17,
        )
    )

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT finish_reason, total_tokens FROM llm_calls WHERE call_id = ?",
            ("llm-1",),
        ).fetchall()

    assert rows == [("stop", 17)]


def test_sqlite_trace_store_creates_replay_indexes(tmp_path):
    db_path = tmp_path / "trace.sqlite3"

    SQLiteTraceStore(db_path)

    with sqlite3.connect(db_path) as connection:
        indexes = {
            row[1]
            for table in ("agent_steps", "llm_calls", "tool_calls", "run_events")
            for row in connection.execute(f"PRAGMA index_list({table})")
        }

    assert {
        "idx_agent_steps_run_step",
        "idx_llm_calls_run_step",
        "idx_tool_calls_run_step",
        "idx_run_events_run_id",
    } <= indexes


def test_sqlite_trace_store_rejects_duplicate_events(tmp_path):
    store = SQLiteTraceStore(tmp_path / "trace.sqlite3")
    event = TraceEvent(
        event_id="event-1",
        run_id="run-1",
        kind=TraceEventKind.RUN_STARTED,
        created_at="2026-05-22T00:00:00Z",
    )

    store.save_event(event)

    with pytest.raises(sqlite3.IntegrityError):
        store.save_event(event)
