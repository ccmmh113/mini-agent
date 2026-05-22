from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ..schema import TokenCost, TokenUsage


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    MAX_STEPS = "max_steps"


class TraceEventKind(str, Enum):
    RUN_STARTED = "run_started"
    STEP_STARTED = "step_started"
    LLM_STARTED = "llm_started"
    LLM_COMPLETED = "llm_completed"
    LLM_FAILED = "llm_failed"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    TOOL_BLOCKED = "tool_blocked"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    RUN_MAX_STEPS = "run_max_steps"


class RunRecord(BaseModel):
    run_id: str
    workspace_dir: str
    model: str | None = None
    started_at: str
    ended_at: str | None = None
    duration_ms: int | None = None
    status: RunStatus = RunStatus.RUNNING
    terminal_reason: str | None = None
    total_steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    total_cost: float = 0.0
    currency: str = "USD"


class StepRecord(BaseModel):
    step_id: str
    run_id: str
    step_index: int
    started_at: str
    ended_at: str | None = None
    duration_ms: int | None = None
    stop_reason: str | None = None


class LLMCallRecord(BaseModel):
    call_id: str
    run_id: str
    step_index: int
    started_at: str
    ended_at: str | None = None
    duration_ms: int | None = None
    finish_reason: str | None = None
    request_message_count: int = 0
    request_tool_names: list[str] = Field(default_factory=list)
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    total_cost: float = 0.0
    currency: str = "USD"

    def __init__(
        self,
        *,
        usage: TokenUsage | None = None,
        cost: TokenCost | None = None,
        **data: Any,
    ) -> None:
        if usage is not None:
            data.update(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cached_tokens=usage.cached_tokens,
                cache_write_tokens=usage.cache_write_tokens,
            )
        if cost is not None:
            data.update(total_cost=cost.total_cost, currency=cost.currency)

        super().__init__(**data)


class ToolCallRecord(BaseModel):
    call_id: str
    run_id: str
    step_index: int | None = None
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    started_at: str
    ended_at: str | None = None
    duration_ms: int | None = None
    success: bool | None = None
    policy_outcome: str | None = None
    error: str | None = None
    result_summary: str | None = None
    affected_paths: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    event_id: str
    run_id: str
    kind: TraceEventKind
    created_at: str
    payload: dict[str, Any] = Field(default_factory=dict)
