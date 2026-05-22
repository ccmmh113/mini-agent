from mini_agent.observability import (
    LLMCallRecord,
    RunRecord,
    RunStatus,
    ToolCallRecord,
    TraceEvent,
    TraceEventKind,
)
from mini_agent.schema import TokenCost, TokenUsage


def test_run_record_defaults_to_running_with_zero_totals():
    run = RunRecord(
        run_id="run-1",
        workspace_dir="workspace",
        started_at="2026-05-22T00:00:00Z",
    )

    assert run.status is RunStatus.RUNNING
    assert run.total_steps == 0
    assert run.prompt_tokens == 0
    assert run.completion_tokens == 0
    assert run.total_tokens == 0
    assert run.cached_tokens == 0
    assert run.cache_write_tokens == 0
    assert run.total_cost == 0.0


def test_llm_call_record_copies_token_usage_and_cost_totals():
    usage = TokenUsage(
        prompt_tokens=120,
        completion_tokens=30,
        total_tokens=150,
        cached_tokens=20,
        cache_write_tokens=10,
    )
    cost = TokenCost(total_cost=0.0125, currency="EUR")

    call = LLMCallRecord(
        call_id="llm-1",
        run_id="run-1",
        step_index=1,
        started_at="2026-05-22T00:00:01Z",
        usage=usage,
        cost=cost,
    )

    assert call.prompt_tokens == 120
    assert call.completion_tokens == 30
    assert call.total_tokens == 150
    assert call.cached_tokens == 20
    assert call.cache_write_tokens == 10
    assert call.total_cost == 0.0125
    assert call.currency == "EUR"


def test_tool_call_record_keeps_affected_paths_without_result_summary():
    call = ToolCallRecord(
        call_id="tool-1",
        run_id="run-1",
        tool_name="write_file",
        started_at="2026-05-22T00:00:02Z",
        affected_paths=["notes/trace.jsonl"],
    )

    assert call.affected_paths == ["notes/trace.jsonl"]
    assert call.result_summary is None


def test_trace_event_keeps_kind_and_payload():
    event = TraceEvent(
        event_id="event-1",
        run_id="run-1",
        kind=TraceEventKind.RUN_STARTED,
        created_at="2026-05-22T00:00:00Z",
        payload={"workspace_dir": "workspace"},
    )

    assert event.kind is TraceEventKind.RUN_STARTED
    assert event.payload == {"workspace_dir": "workspace"}
