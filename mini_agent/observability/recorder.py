import logging
from typing import Callable, Protocol, TypeVar

from .events import LLMCallRecord, RunRecord, StepRecord, ToolCallRecord, TraceEvent
from .store import TraceStore

RecordT = TypeVar("RecordT")
logger = logging.getLogger(__name__)


class TraceRecorder(Protocol):
    def record_run(self, run: RunRecord) -> None:
        """Record one run."""

    def record_step(self, step: StepRecord) -> None:
        """Record one step."""

    def record_llm_call(self, call: LLMCallRecord) -> None:
        """Record one LLM call."""

    def record_tool_call(self, call: ToolCallRecord) -> None:
        """Record one tool call."""

    def record_event(self, event: TraceEvent) -> None:
        """Record one trace event."""


class NullTraceRecorder:
    def record_run(self, run: RunRecord) -> None:
        pass

    def record_step(self, step: StepRecord) -> None:
        pass

    def record_llm_call(self, call: LLMCallRecord) -> None:
        pass

    def record_tool_call(self, call: ToolCallRecord) -> None:
        pass

    def record_event(self, event: TraceEvent) -> None:
        pass


class StoreTraceRecorder:
    def __init__(self, store: TraceStore) -> None:
        self._store = store

    def record_run(self, run: RunRecord) -> None:
        self._save("record_run", self._store.save_run, run)

    def record_step(self, step: StepRecord) -> None:
        self._save("record_step", self._store.save_step, step)

    def record_llm_call(self, call: LLMCallRecord) -> None:
        self._save("record_llm_call", self._store.save_llm_call, call)

    def record_tool_call(self, call: ToolCallRecord) -> None:
        self._save("record_tool_call", self._store.save_tool_call, call)

    def record_event(self, event: TraceEvent) -> None:
        self._save("record_event", self._store.save_event, event)

    @staticmethod
    def _save(method_name: str, save: Callable[[RecordT], None], record: RecordT) -> None:
        try:
            save(record)
        except Exception:
            logger.warning("Failed to save trace record in %s", method_name)
