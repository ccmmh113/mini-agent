import pytest
from pydantic import ValidationError

from mini_agent.observability import (
    LLMCallRecord,
    RunRecord,
    RunStatus,
    StepRecord,
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


def test_llm_call_record_validates_mapping_token_usage_and_cost_inputs():
    call = LLMCallRecord.model_validate(
        {
            "call_id": "llm-1",
            "run_id": "run-1",
            "step_index": 1,
            "started_at": "2026-05-22T00:00:01Z",
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "cached_tokens": 20,
                "cache_write_tokens": 10,
            },
            "cost": {"total_cost": 0.0125, "currency": "EUR"},
        }
    )

    assert call.prompt_tokens == 120
    assert call.completion_tokens == 30
    assert call.total_tokens == 150
    assert call.cached_tokens == 20
    assert call.cache_write_tokens == 10
    assert call.total_cost == 0.0125
    assert call.currency == "EUR"


def test_step_record_keeps_step_identity():
    step = StepRecord(
        step_id="step-1",
        run_id="run-1",
        step_index=0,
        started_at="2026-05-22T00:00:00Z",
    )

    assert step.step_id == "step-1"
    assert step.step_index == 0


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


def test_trace_event_round_trips_through_serialized_payload():
    event = TraceEvent(
        event_id="event-1",
        run_id="run-1",
        kind=TraceEventKind.RUN_STARTED,
        created_at="2026-05-22T00:00:00Z",
        payload={"workspace_dir": "workspace"},
    )

    assert TraceEvent.model_validate_json(event.model_dump_json()) == event


@pytest.mark.parametrize(
    ("model_type", "payload", "field_name"),
    [
        (
            RunRecord,
            {
                "run_id": "run-1",
                "workspace_dir": "workspace",
                "started_at": "2026-05-22T00:00:00Z",
            },
            "duration_ms",
        ),
        (
            RunRecord,
            {
                "run_id": "run-1",
                "workspace_dir": "workspace",
                "started_at": "2026-05-22T00:00:00Z",
            },
            "total_steps",
        ),
        (
            StepRecord,
            {
                "step_id": "step-1",
                "run_id": "run-1",
                "step_index": 0,
                "started_at": "2026-05-22T00:00:00Z",
            },
            "step_index",
        ),
        (
            LLMCallRecord,
            {
                "call_id": "llm-1",
                "run_id": "run-1",
                "step_index": 0,
                "started_at": "2026-05-22T00:00:01Z",
            },
            "request_message_count",
        ),
        (
            LLMCallRecord,
            {
                "call_id": "llm-1",
                "run_id": "run-1",
                "step_index": 0,
                "started_at": "2026-05-22T00:00:01Z",
            },
            "prompt_tokens",
        ),
        (
            LLMCallRecord,
            {
                "call_id": "llm-1",
                "run_id": "run-1",
                "step_index": 0,
                "started_at": "2026-05-22T00:00:01Z",
            },
            "total_cost",
        ),
        (
            ToolCallRecord,
            {
                "call_id": "tool-1",
                "run_id": "run-1",
                "tool_name": "write_file",
                "started_at": "2026-05-22T00:00:02Z",
            },
            "duration_ms",
        ),
    ],
)
def test_trace_records_reject_negative_numeric_fields(model_type, payload, field_name):
    with pytest.raises(ValidationError):
        model_type.model_validate({**payload, field_name: -1})
