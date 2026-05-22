import json
import sqlite3

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
