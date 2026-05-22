from typing import Protocol

from .events import LLMCallRecord, RunRecord, StepRecord, ToolCallRecord, TraceEvent


class TraceStore(Protocol):
    def save_run(self, run: RunRecord) -> None:
        """Persist one run record."""

    def save_step(self, step: StepRecord) -> None:
        """Persist one step record."""

    def save_llm_call(self, call: LLMCallRecord) -> None:
        """Persist one LLM call record."""

    def save_tool_call(self, call: ToolCallRecord) -> None:
        """Persist one tool call record."""

    def save_event(self, event: TraceEvent) -> None:
        """Persist one trace event."""
