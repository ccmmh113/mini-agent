from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    """LLM provider types."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class FunctionCall(BaseModel):
    """Function call details."""

    name: str
    arguments: dict[str, Any]  # Function arguments as dict


class ToolCall(BaseModel):
    """Tool call structure."""

    id: str
    type: str  # "function"
    function: FunctionCall


class Message(BaseModel):
    """Chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]]  # Can be string or list of content blocks
    thinking: str | None = None  # Extended thinking content for assistant messages
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # For tool role


class TokenUsage(BaseModel):
    """Token usage statistics from LLM API response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0


class TokenPricing(BaseModel):
    """Per-million-token pricing used for local cost estimates."""

    input_per_1m: float = 0.0
    output_per_1m: float = 0.0
    cache_read_per_1m: float = 0.0
    cache_write_per_1m: float = 0.0
    currency: str = "USD"

    @property
    def configured(self) -> bool:
        return any(
            rate > 0
            for rate in (
                self.input_per_1m,
                self.output_per_1m,
                self.cache_read_per_1m,
                self.cache_write_per_1m,
            )
        )


class TokenCost(BaseModel):
    """Estimated LLM cost for a token usage record."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"


class LLMResponse(BaseModel):
    """LLM response."""

    content: str
    thinking: str | None = None  # Extended thinking blocks
    tool_calls: list[ToolCall] | None = None
    finish_reason: str
    usage: TokenUsage | None = None  # Token usage from API response


class EpisodeRecord(BaseModel):
    """Task-level episodic memory captured when work completes."""

    episode_id: str
    kind: str = "episode_record"
    schema_version: str = "1.0"
    task_id: str
    goal: str
    task_type: str = "general"
    status: str = "completed"
    summary: str = ""
    completed_steps: list["TaskStepRecord"] = Field(default_factory=list)
    decisions: list["TaskDecisionRecord"] = Field(default_factory=list)
    artifacts: list["TaskArtifactRecord"] = Field(default_factory=list)
    archived_steps_summary: str = ""
    created_at: str
    updated_at: str
    completed_at: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeMemoryState(BaseModel):
    episodes: list[EpisodeRecord] = Field(default_factory=list)


class TaskStepRecord(BaseModel):
    description: str
    timestamp: str
    source: str | None = None


class TaskDecisionRecord(BaseModel):
    decision: str
    timestamp: str
    reason: str = ""


class TaskArtifactRecord(BaseModel):
    path: str
    timestamp: str
    artifact_type: str = "artifact"
    description: str = ""
    tool: str | None = None
    success: bool | None = None
    verification: dict[str, Any] | None = None
    source: str | None = None


class TaskQuestionRecord(BaseModel):
    question: str
    timestamp: str | None = None
    source: str | None = None


class TaskNextStepRecord(BaseModel):
    description: str
    timestamp: str | None = None
    source: str | None = None


class TaskResumeEvent(BaseModel):
    timestamp: str
    source: str
    goal: str


class TaskMemoryItem(BaseModel):
    task_id: str
    goal: str
    task_type: str = "general"
    status: str = "active"
    completed_steps: list[TaskStepRecord] = Field(default_factory=list)
    decisions: list[TaskDecisionRecord] = Field(default_factory=list)
    artifacts: list[TaskArtifactRecord] = Field(default_factory=list)
    open_questions: list[TaskQuestionRecord | dict[str, Any] | str] = Field(default_factory=list)
    next_steps: list[TaskNextStepRecord | dict[str, Any] | str] = Field(default_factory=list)
    archived_steps_summary: str = ""
    created_at: str
    updated_at: str
    summary: str = ""
    completed_by: str | None = None
    source: str | None = None
    resume_events: list[TaskResumeEvent] = Field(default_factory=list)


class TaskMemoryState(BaseModel):
    active_task_id: str | None = None
    tasks: list[TaskMemoryItem] = Field(default_factory=list)
