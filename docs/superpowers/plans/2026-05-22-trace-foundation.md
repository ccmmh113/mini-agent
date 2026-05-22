# Trace Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a best-effort structured trace foundation for Mini Agent runs, LLM calls, tool calls, and run lifecycle events with local SQLite persistence.

**Architecture:** Introduce a small `mini_agent.observability` package with event models, a recorder interface, and a SQLite-backed store. Inject a recorder through `Agent` and `RunContext` so `Agent.run()` owns run, step, and LLM lifecycle events while `ToolRuntime.execute()` owns tool lifecycle events and policy outcomes. Keep the recorder best-effort so telemetry failures do not change Agent behavior.

**Tech Stack:** Python 3.12, Pydantic models already used by Mini Agent, built-in `sqlite3`, `pytest`, `pytest-asyncio`.

---

## Scope

This plan implements Phase 1 from the approved design spec:

- structured event contracts
- recorder/store boundary
- local SQLite persistence
- Agent run, step, and LLM-call instrumentation
- ToolRuntime tool-call and policy-result instrumentation
- regression coverage for best-effort telemetry behavior

This plan does not implement evaluation suites, score persistence, reports, CLI commands, or dashboard pages. Those are separate plans after this trace foundation is stable.

## File Structure

### New files

- `mini_agent/observability/__init__.py`
  - Export the public trace types and recorder constructors used by runtime code.
- `mini_agent/observability/events.py`
  - Define `RunStatus`, `TraceEventKind`, `RunRecord`, `StepRecord`, `LLMCallRecord`, `ToolCallRecord`, and `TraceEvent`.
- `mini_agent/observability/recorder.py`
  - Define the recorder protocol plus `NullTraceRecorder`, `StoreTraceRecorder`, and best-effort call protection.
- `mini_agent/observability/store.py`
  - Define the store protocol that accepts normalized trace records and events.
- `mini_agent/observability/sqlite_store.py`
  - Create SQLite schema and persist run, step, LLM call, tool call, and event records.
- `tests/test_observability_events.py`
  - Cover record defaults, redaction, and recorder best-effort behavior.
- `tests/test_observability_sqlite_store.py`
  - Cover SQLite schema writes and queryable persisted values.

### Modified files

- `mini_agent/agent.py`
  - Accept an optional recorder and emit run, step, LLM, terminal status, token, and cost trace data.
- `mini_agent/runtime.py`
  - Carry the recorder in `RunContext` and emit tool call lifecycle records from `ToolRuntime.execute()`.
- `tests/test_agent.py`
  - Verify completed, failed, cancelled, and max-step trace emission from Agent behavior.
- `tests/test_runtime.py`
  - Verify successful, blocked, failed, unknown, and workspace-diff tool trace data.

## Task 1: Define Trace Contracts

**Files:**
- Create: `mini_agent/observability/__init__.py`
- Create: `mini_agent/observability/events.py`
- Test: `tests/test_observability_events.py`

- [ ] **Step 1: Write the failing record-model tests**

Create `tests/test_observability_events.py` with these initial checks:

```python
from mini_agent.observability.events import (
    LLMCallRecord,
    RunRecord,
    RunStatus,
    StepRecord,
    ToolCallRecord,
    TraceEvent,
    TraceEventKind,
)
from mini_agent.schema import TokenCost, TokenUsage


def test_run_record_starts_running_with_workspace_and_model():
    run = RunRecord(
        run_id="run-1",
        workspace_dir="F:/workspace",
        model="gpt-test",
        started_at="2026-05-22T10:00:00+00:00",
    )

    assert run.status is RunStatus.RUNNING
    assert run.terminal_reason is None
    assert run.total_tokens == 0


def test_llm_call_record_copies_usage_and_cost_totals():
    call = LLMCallRecord(
        call_id="llm-1",
        run_id="run-1",
        step_index=1,
        started_at="2026-05-22T10:00:01+00:00",
        ended_at="2026-05-22T10:00:02+00:00",
        duration_ms=1000.0,
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=12, completion_tokens=4, total_tokens=16, cached_tokens=3),
        cost=TokenCost(total_cost=0.01, currency="USD"),
    )

    assert call.prompt_tokens == 12
    assert call.completion_tokens == 4
    assert call.total_tokens == 16
    assert call.cached_tokens == 3
    assert call.total_cost == 0.01
    assert call.currency == "USD"


def test_tool_call_record_keeps_affected_paths_without_raw_result():
    call = ToolCallRecord(
        call_id="tool-1",
        run_id="run-1",
        step_index=2,
        tool_name="write_file",
        arguments={"path": "result.md"},
        started_at="2026-05-22T10:00:03+00:00",
        ended_at="2026-05-22T10:00:04+00:00",
        duration_ms=1000.0,
        success=True,
        affected_paths=["result.md"],
    )

    assert call.affected_paths == ["result.md"]
    assert call.result_summary is None


def test_trace_event_has_kind_and_payload():
    event = TraceEvent(
        event_id="event-1",
        run_id="run-1",
        kind=TraceEventKind.RUN_STARTED,
        created_at="2026-05-22T10:00:00+00:00",
        payload={"workspace_dir": "F:/workspace"},
    )

    assert event.kind is TraceEventKind.RUN_STARTED
    assert event.payload["workspace_dir"] == "F:/workspace"
```

- [ ] **Step 2: Run the model tests and verify they fail**

Run:

```bash
uv run pytest tests/test_observability_events.py -q
```

Expected: FAIL because `mini_agent.observability` and its record classes do not exist yet.

- [ ] **Step 3: Implement the initial trace models**

Create `mini_agent/observability/events.py` with explicit status and event enums plus the record models:

```python
"""Structured trace records for Mini Agent runtime observability."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from mini_agent.schema import TokenCost, TokenUsage


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
    duration_ms: float | None = None
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
    duration_ms: float | None = None
    stop_reason: str | None = None


class LLMCallRecord(BaseModel):
    call_id: str
    run_id: str
    step_index: int
    started_at: str
    ended_at: str | None = None
    duration_ms: float | None = None
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
        usage: TokenUsage | None = None,
        cost: TokenCost | None = None,
        **data: Any,
    ):
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
    duration_ms: float | None = None
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
```

Create `mini_agent/observability/__init__.py` with stable exports:

```python
"""Mini Agent structured observability primitives."""

from .events import (
    LLMCallRecord,
    RunRecord,
    RunStatus,
    StepRecord,
    ToolCallRecord,
    TraceEvent,
    TraceEventKind,
)

__all__ = [
    "LLMCallRecord",
    "RunRecord",
    "RunStatus",
    "StepRecord",
    "ToolCallRecord",
    "TraceEvent",
    "TraceEventKind",
]
```

- [ ] **Step 4: Run the record-model tests and verify they pass**

Run:

```bash
uv run pytest tests/test_observability_events.py -q
```

Expected: PASS for the first four trace-model tests.

- [ ] **Step 5: Commit the trace contract**

```bash
git add mini_agent/observability/__init__.py mini_agent/observability/events.py tests/test_observability_events.py
git commit -m "feat: add trace record contracts"
```

## Task 2: Add Recorder And Best-Effort Store Boundary

**Files:**
- Create: `mini_agent/observability/recorder.py`
- Create: `mini_agent/observability/store.py`
- Modify: `mini_agent/observability/__init__.py`
- Modify: `tests/test_observability_events.py`

- [ ] **Step 1: Add failing recorder tests**

Append these tests to `tests/test_observability_events.py`:

```python
from mini_agent.observability.recorder import NullTraceRecorder, StoreTraceRecorder
from mini_agent.observability.store import TraceStore


class RecordingStore:
    def __init__(self):
        self.runs = []
        self.events = []

    def save_run(self, run):
        self.runs.append(run)

    def save_step(self, step):
        raise AssertionError("not used in this test")

    def save_llm_call(self, call):
        raise AssertionError("not used in this test")

    def save_tool_call(self, call):
        raise AssertionError("not used in this test")

    def save_event(self, event):
        self.events.append(event)


class FailingStore(RecordingStore):
    def save_run(self, run):
        raise RuntimeError("store unavailable")


def test_store_trace_recorder_writes_run_and_event():
    store = RecordingStore()
    recorder = StoreTraceRecorder(store)
    run = RunRecord(run_id="run-1", workspace_dir=".", started_at="2026-05-22T10:00:00+00:00")
    event = TraceEvent(
        event_id="event-1",
        run_id="run-1",
        kind=TraceEventKind.RUN_STARTED,
        created_at="2026-05-22T10:00:00+00:00",
    )

    recorder.record_run(run)
    recorder.record_event(event)

    assert store.runs == [run]
    assert store.events == [event]


def test_store_trace_recorder_swallow_store_failures():
    recorder = StoreTraceRecorder(FailingStore())
    run = RunRecord(run_id="run-1", workspace_dir=".", started_at="2026-05-22T10:00:00+00:00")

    recorder.record_run(run)


def test_null_trace_recorder_accepts_records_without_side_effects():
    recorder = NullTraceRecorder()
    run = RunRecord(run_id="run-1", workspace_dir=".", started_at="2026-05-22T10:00:00+00:00")

    recorder.record_run(run)
```

- [ ] **Step 2: Run the recorder tests and verify they fail**

Run:

```bash
uv run pytest tests/test_observability_events.py -q
```

Expected: FAIL because `recorder.py` and `store.py` do not exist.

- [ ] **Step 3: Add the store protocol**

Create `mini_agent/observability/store.py`:

```python
"""Storage boundary for structured trace records."""

from __future__ import annotations

from typing import Protocol

from .events import LLMCallRecord, RunRecord, StepRecord, ToolCallRecord, TraceEvent


class TraceStore(Protocol):
    def save_run(self, run: RunRecord) -> None: ...

    def save_step(self, step: StepRecord) -> None: ...

    def save_llm_call(self, call: LLMCallRecord) -> None: ...

    def save_tool_call(self, call: ToolCallRecord) -> None: ...

    def save_event(self, event: TraceEvent) -> None: ...
```

- [ ] **Step 4: Add a best-effort recorder**

Create `mini_agent/observability/recorder.py`:

```python
"""Trace recorders used by Agent and tool runtime instrumentation."""

from __future__ import annotations

from typing import Callable, Protocol, TypeVar

from .events import LLMCallRecord, RunRecord, StepRecord, ToolCallRecord, TraceEvent
from .store import TraceStore

T = TypeVar("T")


class TraceRecorder(Protocol):
    def record_run(self, run: RunRecord) -> None: ...

    def record_step(self, step: StepRecord) -> None: ...

    def record_llm_call(self, call: LLMCallRecord) -> None: ...

    def record_tool_call(self, call: ToolCallRecord) -> None: ...

    def record_event(self, event: TraceEvent) -> None: ...


class NullTraceRecorder:
    def record_run(self, run: RunRecord) -> None:
        del run

    def record_step(self, step: StepRecord) -> None:
        del step

    def record_llm_call(self, call: LLMCallRecord) -> None:
        del call

    def record_tool_call(self, call: ToolCallRecord) -> None:
        del call

    def record_event(self, event: TraceEvent) -> None:
        del event


class StoreTraceRecorder:
    def __init__(self, store: TraceStore):
        self.store = store

    def record_run(self, run: RunRecord) -> None:
        self._best_effort(self.store.save_run, run)

    def record_step(self, step: StepRecord) -> None:
        self._best_effort(self.store.save_step, step)

    def record_llm_call(self, call: LLMCallRecord) -> None:
        self._best_effort(self.store.save_llm_call, call)

    def record_tool_call(self, call: ToolCallRecord) -> None:
        self._best_effort(self.store.save_tool_call, call)

    def record_event(self, event: TraceEvent) -> None:
        self._best_effort(self.store.save_event, event)

    def _best_effort(self, write: Callable[[T], None], value: T) -> None:
        try:
            write(value)
        except Exception:
            return
```

Update `mini_agent/observability/__init__.py`:

```python
from .recorder import NullTraceRecorder, StoreTraceRecorder, TraceRecorder
```

Add these names to `__all__`:

```python
"NullTraceRecorder",
"StoreTraceRecorder",
"TraceRecorder",
```

- [ ] **Step 5: Run the recorder tests and verify they pass**

Run:

```bash
uv run pytest tests/test_observability_events.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the recorder boundary**

```bash
git add mini_agent/observability/__init__.py mini_agent/observability/recorder.py mini_agent/observability/store.py tests/test_observability_events.py
git commit -m "feat: add best effort trace recorder"
```

## Task 3: Persist Trace Records In SQLite

**Files:**
- Create: `mini_agent/observability/sqlite_store.py`
- Modify: `mini_agent/observability/__init__.py`
- Test: `tests/test_observability_sqlite_store.py`

- [ ] **Step 1: Write failing SQLite persistence tests**

Create `tests/test_observability_sqlite_store.py`:

```python
import sqlite3

from mini_agent.observability.events import (
    LLMCallRecord,
    RunRecord,
    StepRecord,
    ToolCallRecord,
    TraceEvent,
    TraceEventKind,
)
from mini_agent.observability.sqlite_store import SQLiteTraceStore


def test_sqlite_trace_store_persists_run_calls_steps_and_events(tmp_path):
    db_path = tmp_path / "traces.db"
    store = SQLiteTraceStore(db_path)

    store.save_run(RunRecord(run_id="run-1", workspace_dir=".", started_at="2026-05-22T10:00:00+00:00"))
    store.save_step(
        StepRecord(step_id="step-1", run_id="run-1", step_index=1, started_at="2026-05-22T10:00:01+00:00")
    )
    store.save_llm_call(
        LLMCallRecord(
            call_id="llm-1",
            run_id="run-1",
            step_index=1,
            started_at="2026-05-22T10:00:01+00:00",
            finish_reason="stop",
            request_tool_names=["read_file"],
        )
    )
    store.save_tool_call(
        ToolCallRecord(
            call_id="tool-1",
            run_id="run-1",
            step_index=1,
            tool_name="write_file",
            arguments={"path": "result.md"},
            started_at="2026-05-22T10:00:02+00:00",
            success=True,
            affected_paths=["result.md"],
        )
    )
    store.save_event(
        TraceEvent(
            event_id="event-1",
            run_id="run-1",
            kind=TraceEventKind.RUN_STARTED,
            created_at="2026-05-22T10:00:00+00:00",
            payload={"workspace_dir": "."},
        )
    )

    connection = sqlite3.connect(db_path)
    assert connection.execute("select status from agent_runs where run_id = 'run-1'").fetchone() == ("running",)
    assert connection.execute("select step_index from agent_steps where step_id = 'step-1'").fetchone() == (1,)
    assert connection.execute("select finish_reason from llm_calls where call_id = 'llm-1'").fetchone() == ("stop",)
    assert connection.execute("select affected_paths_json from tool_calls where call_id = 'tool-1'").fetchone()[0] == '["result.md"]'
    assert connection.execute("select kind from run_events where event_id = 'event-1'").fetchone() == ("run_started",)
```

- [ ] **Step 2: Run the SQLite tests and verify they fail**

Run:

```bash
uv run pytest tests/test_observability_sqlite_store.py -q
```

Expected: FAIL because `SQLiteTraceStore` does not exist.

- [ ] **Step 3: Add the SQLite schema and upsert writes**

Create `mini_agent/observability/sqlite_store.py`:

```python
"""SQLite persistence for local structured trace records."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .events import LLMCallRecord, RunRecord, StepRecord, ToolCallRecord, TraceEvent


class SQLiteTraceStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save_run(self, run: RunRecord) -> None:
        self._execute(
            """
            insert into agent_runs (
                run_id, workspace_dir, model, started_at, ended_at, duration_ms, status,
                terminal_reason, total_steps, prompt_tokens, completion_tokens, total_tokens,
                cached_tokens, cache_write_tokens, total_cost, currency
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(run_id) do update set
                ended_at=excluded.ended_at,
                duration_ms=excluded.duration_ms,
                status=excluded.status,
                terminal_reason=excluded.terminal_reason,
                total_steps=excluded.total_steps,
                prompt_tokens=excluded.prompt_tokens,
                completion_tokens=excluded.completion_tokens,
                total_tokens=excluded.total_tokens,
                cached_tokens=excluded.cached_tokens,
                cache_write_tokens=excluded.cache_write_tokens,
                total_cost=excluded.total_cost,
                currency=excluded.currency
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
        self._execute(
            """
            insert into agent_steps (step_id, run_id, step_index, started_at, ended_at, duration_ms, stop_reason)
            values (?, ?, ?, ?, ?, ?, ?)
            on conflict(step_id) do update set
                ended_at=excluded.ended_at,
                duration_ms=excluded.duration_ms,
                stop_reason=excluded.stop_reason
            """,
            (step.step_id, step.run_id, step.step_index, step.started_at, step.ended_at, step.duration_ms, step.stop_reason),
        )

    def save_llm_call(self, call: LLMCallRecord) -> None:
        self._execute(
            """
            insert into llm_calls (
                call_id, run_id, step_index, started_at, ended_at, duration_ms, finish_reason,
                request_message_count, request_tool_names_json, error, prompt_tokens,
                completion_tokens, total_tokens, cached_tokens, cache_write_tokens, total_cost, currency
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(call_id) do update set
                ended_at=excluded.ended_at,
                duration_ms=excluded.duration_ms,
                finish_reason=excluded.finish_reason,
                error=excluded.error,
                prompt_tokens=excluded.prompt_tokens,
                completion_tokens=excluded.completion_tokens,
                total_tokens=excluded.total_tokens,
                cached_tokens=excluded.cached_tokens,
                cache_write_tokens=excluded.cache_write_tokens,
                total_cost=excluded.total_cost,
                currency=excluded.currency
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
                json.dumps(call.request_tool_names, ensure_ascii=True),
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
        self._execute(
            """
            insert into tool_calls (
                call_id, run_id, step_index, tool_name, arguments_json, started_at, ended_at,
                duration_ms, success, policy_outcome, error, result_summary, affected_paths_json
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(call_id) do update set
                ended_at=excluded.ended_at,
                duration_ms=excluded.duration_ms,
                success=excluded.success,
                policy_outcome=excluded.policy_outcome,
                error=excluded.error,
                result_summary=excluded.result_summary,
                affected_paths_json=excluded.affected_paths_json
            """,
            (
                call.call_id,
                call.run_id,
                call.step_index,
                call.tool_name,
                json.dumps(call.arguments, ensure_ascii=True, sort_keys=True),
                call.started_at,
                call.ended_at,
                call.duration_ms,
                None if call.success is None else int(call.success),
                call.policy_outcome,
                call.error,
                call.result_summary,
                json.dumps(call.affected_paths, ensure_ascii=True),
            ),
        )

    def save_event(self, event: TraceEvent) -> None:
        self._execute(
            """
            insert into run_events (event_id, run_id, kind, created_at, payload_json)
            values (?, ?, ?, ?, ?)
            """,
            (event.event_id, event.run_id, event.kind.value, event.created_at, json.dumps(event.payload, ensure_ascii=True, sort_keys=True)),
        )

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.executescript(
                """
                create table if not exists agent_runs (
                    run_id text primary key,
                    workspace_dir text not null,
                    model text,
                    started_at text not null,
                    ended_at text,
                    duration_ms real,
                    status text not null,
                    terminal_reason text,
                    total_steps integer not null default 0,
                    prompt_tokens integer not null default 0,
                    completion_tokens integer not null default 0,
                    total_tokens integer not null default 0,
                    cached_tokens integer not null default 0,
                    cache_write_tokens integer not null default 0,
                    total_cost real not null default 0,
                    currency text not null default 'USD'
                );
                create table if not exists agent_steps (
                    step_id text primary key,
                    run_id text not null,
                    step_index integer not null,
                    started_at text not null,
                    ended_at text,
                    duration_ms real,
                    stop_reason text
                );
                create table if not exists llm_calls (
                    call_id text primary key,
                    run_id text not null,
                    step_index integer not null,
                    started_at text not null,
                    ended_at text,
                    duration_ms real,
                    finish_reason text,
                    request_message_count integer not null default 0,
                    request_tool_names_json text not null default '[]',
                    error text,
                    prompt_tokens integer not null default 0,
                    completion_tokens integer not null default 0,
                    total_tokens integer not null default 0,
                    cached_tokens integer not null default 0,
                    cache_write_tokens integer not null default 0,
                    total_cost real not null default 0,
                    currency text not null default 'USD'
                );
                create table if not exists tool_calls (
                    call_id text primary key,
                    run_id text not null,
                    step_index integer,
                    tool_name text not null,
                    arguments_json text not null default '{}',
                    started_at text not null,
                    ended_at text,
                    duration_ms real,
                    success integer,
                    policy_outcome text,
                    error text,
                    result_summary text,
                    affected_paths_json text not null default '[]'
                );
                create table if not exists run_events (
                    event_id text primary key,
                    run_id text not null,
                    kind text not null,
                    created_at text not null,
                    payload_json text not null default '{}'
                );
                """
            )

    def _execute(self, sql: str, params: tuple[object, ...]) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(sql, params)
```

Update `mini_agent/observability/__init__.py`:

```python
from .sqlite_store import SQLiteTraceStore
```

Add `"SQLiteTraceStore"` to `__all__`.

- [ ] **Step 4: Run the SQLite tests and verify they pass**

Run:

```bash
uv run pytest tests/test_observability_sqlite_store.py -q
```

Expected: PASS.

- [ ] **Step 5: Run observability tests together**

Run:

```bash
uv run pytest tests/test_observability_events.py tests/test_observability_sqlite_store.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit SQLite persistence**

```bash
git add mini_agent/observability/__init__.py mini_agent/observability/sqlite_store.py tests/test_observability_sqlite_store.py
git commit -m "feat: persist trace records in sqlite"
```

## Task 4: Instrument ToolRuntime

**Files:**
- Modify: `mini_agent/runtime.py`
- Modify: `tests/test_runtime.py`

- [ ] **Step 1: Add failing tool trace tests**

Extend `tests/test_runtime.py` with an in-memory recorder and these checks:

```python
from mini_agent.observability.events import TraceEventKind


class RuntimeTraceRecorder:
    def __init__(self):
        self.tool_calls = []
        self.events = []

    def record_run(self, run):
        raise AssertionError("not used in runtime tests")

    def record_step(self, step):
        raise AssertionError("not used in runtime tests")

    def record_llm_call(self, call):
        raise AssertionError("not used in runtime tests")

    def record_tool_call(self, call):
        self.tool_calls.append(call)

    def record_event(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_tool_runtime_records_successful_tool_trace(tmp_path):
    recorder = RuntimeTraceRecorder()
    context = RunContext(workspace_dir=tmp_path, run_id="run-1", step_index=3, trace_recorder=recorder)
    runtime = ToolRuntime({"write_file": WriteTool(workspace_dir=str(tmp_path))}, context)

    result = await runtime.execute("write_file", {"path": "result.txt", "content": "ok"})

    assert result.success
    call = recorder.tool_calls[-1]
    assert call.run_id == "run-1"
    assert call.step_index == 3
    assert call.tool_name == "write_file"
    assert call.success is True
    assert call.affected_paths == ["result.txt"]
    assert recorder.events[-1].kind is TraceEventKind.TOOL_COMPLETED


@pytest.mark.asyncio
async def test_tool_runtime_records_policy_block_as_tool_blocked(tmp_path):
    recorder = RuntimeTraceRecorder()
    context = RunContext(workspace_dir=tmp_path, run_id="run-1", step_index=1, trace_recorder=recorder)
    runtime = ToolRuntime({"bash": BashTool(workspace_dir=str(tmp_path))}, context)

    result = await runtime.execute("bash", {"command": "rm -rf ./important"})

    assert not result.success
    assert recorder.tool_calls[-1].policy_outcome == "blocked"
    assert recorder.events[-1].kind is TraceEventKind.TOOL_BLOCKED
```

- [ ] **Step 2: Run the new runtime trace tests and verify they fail**

Run:

```bash
uv run pytest tests/test_runtime.py::test_tool_runtime_records_successful_tool_trace tests/test_runtime.py::test_tool_runtime_records_policy_block_as_tool_blocked -q
```

Expected: FAIL because `RunContext` has no `run_id`, `step_index`, or `trace_recorder` fields.

- [ ] **Step 3: Carry recorder state through `RunContext`**

Modify the imports in `mini_agent/runtime.py`:

```python
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from .observability.events import ToolCallRecord, TraceEvent, TraceEventKind
from .observability.recorder import NullTraceRecorder, TraceRecorder
```

Extend `RunContext`:

```python
    run_id: str | None = None
    step_index: int | None = None
    trace_recorder: TraceRecorder = field(default_factory=NullTraceRecorder)
```

Add timestamp and event helpers near the runtime support functions:

```python
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tool_event_kind(result: ToolResult, policy_outcome: str | None) -> TraceEventKind:
    if policy_outcome == "blocked":
        return TraceEventKind.TOOL_BLOCKED
    return TraceEventKind.TOOL_COMPLETED if result.success else TraceEventKind.TOOL_FAILED
```

- [ ] **Step 4: Emit started and terminal tool traces**

Instrument `ToolRuntime.execute()` around its existing execution path:

```python
        call_id = f"tool-{uuid4().hex}"
        started_at = _now_iso()
        started = perf_counter()
        if self.context.run_id is not None:
            self.context.trace_recorder.record_event(
                TraceEvent(
                    event_id=f"event-{uuid4().hex}",
                    run_id=self.context.run_id,
                    kind=TraceEventKind.TOOL_STARTED,
                    created_at=started_at,
                    payload={"call_id": call_id, "tool_name": tool_name},
                )
            )
```

After `result = redact_tool_result(result)` and before `return result`, derive policy metadata and save the call:

```python
        metadata = result.metadata or {}
        policy_outcome = metadata.get("policy_outcome")
        if policy_outcome is None and result.error and "Command blocked by security policy" in result.error:
            policy_outcome = "blocked"
        if self.context.run_id is not None:
            ended_at = _now_iso()
            self.context.trace_recorder.record_tool_call(
                ToolCallRecord(
                    call_id=call_id,
                    run_id=self.context.run_id,
                    step_index=self.context.step_index,
                    tool_name=tool_name,
                    arguments=redact_data(arguments),
                    started_at=started_at,
                    ended_at=ended_at,
                    duration_ms=round((perf_counter() - started) * 1000, 3),
                    success=result.success,
                    policy_outcome=policy_outcome,
                    error=result.error if not result.success else None,
                    result_summary=(result.content[:240] if result.success and result.content else None),
                    affected_paths=list(metadata.get("affected_paths", [])),
                )
            )
            self.context.trace_recorder.record_event(
                TraceEvent(
                    event_id=f"event-{uuid4().hex}",
                    run_id=self.context.run_id,
                    kind=_tool_event_kind(result, policy_outcome),
                    created_at=ended_at,
                    payload={"call_id": call_id, "tool_name": tool_name, "success": result.success},
                )
            )
```

When `BashToolPolicy` blocks or denies a command, attach explicit policy metadata to the returned `ToolResult`:

```python
return ToolResult(
    success=False,
    content="",
    error=f"Command blocked by security policy: {decision.reason}",
    metadata={"policy_outcome": "blocked"},
)
```

```python
return ToolResult(
    success=False,
    content="",
    error="Command execution denied by user confirmation policy.",
    metadata={"policy_outcome": "confirmation_denied"},
)
```

- [ ] **Step 5: Run runtime trace tests and existing runtime tests**

Run:

```bash
uv run pytest tests/test_runtime.py -q
```

Expected: PASS. Existing Bash audit and workspace diff assertions still pass.

- [ ] **Step 6: Commit ToolRuntime instrumentation**

```bash
git add mini_agent/runtime.py tests/test_runtime.py
git commit -m "feat: trace tool runtime calls"
```

## Task 5: Instrument Agent Run, Step, And LLM Calls

**Files:**
- Modify: `mini_agent/agent.py`
- Modify: `tests/test_agent.py`

- [ ] **Step 1: Add failing Agent trace tests**

Extend `tests/test_agent.py`:

```python
from mini_agent.observability.events import RunStatus, TraceEventKind
from mini_agent.schema import TokenUsage


class AgentTraceRecorder:
    def __init__(self):
        self.runs = []
        self.steps = []
        self.llm_calls = []
        self.events = []

    def record_run(self, run):
        self.runs.append(run)

    def record_step(self, step):
        self.steps.append(step)

    def record_llm_call(self, call):
        self.llm_calls.append(call)

    def record_tool_call(self, call):
        raise AssertionError("tool calls are covered in runtime tests")

    def record_event(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_agent_records_completed_run_and_llm_usage(tmp_path):
    recorder = AgentTraceRecorder()
    llm_client = MagicMock(spec=LLMClient)
    llm_client.model = "gpt-test"
    llm_client.generate = AsyncMock(
        return_value=LLMResponse(
            content="done",
            tool_calls=None,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12, cached_tokens=3),
        )
    )
    agent = Agent(llm_client=llm_client, system_prompt="System", tools=[], workspace_dir=str(tmp_path), trace_recorder=recorder)
    agent.add_user_message("finish")

    assert await agent.run() == "done"
    assert recorder.runs[0].status is RunStatus.RUNNING
    assert recorder.runs[-1].status is RunStatus.COMPLETED
    assert recorder.runs[-1].total_tokens == 12
    assert recorder.llm_calls[-1].finish_reason == "stop"
    assert recorder.llm_calls[-1].cached_tokens == 3
    assert recorder.events[0].kind is TraceEventKind.RUN_STARTED
    assert recorder.events[-1].kind is TraceEventKind.RUN_COMPLETED


@pytest.mark.asyncio
async def test_agent_records_llm_failure_and_failed_run(tmp_path):
    recorder = AgentTraceRecorder()
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate = AsyncMock(side_effect=RuntimeError("provider down"))
    agent = Agent(llm_client=llm_client, system_prompt="System", tools=[], workspace_dir=str(tmp_path), trace_recorder=recorder)
    agent.add_user_message("fail")

    result = await agent.run()

    assert "LLM call failed" in result
    assert recorder.llm_calls[-1].error == "RuntimeError: provider down"
    assert recorder.runs[-1].status is RunStatus.FAILED
    assert recorder.events[-1].kind is TraceEventKind.RUN_FAILED
```

- [ ] **Step 2: Run the Agent trace tests and verify they fail**

Run:

```bash
uv run pytest tests/test_agent.py::test_agent_records_completed_run_and_llm_usage tests/test_agent.py::test_agent_records_llm_failure_and_failed_run -q
```

Expected: FAIL because `Agent.__init__()` has no `trace_recorder` argument and emits no trace records.

- [ ] **Step 3: Add Agent trace helpers and recorder injection**

Modify `mini_agent/agent.py` imports:

```python
from datetime import datetime, timezone
from uuid import uuid4

from .observability.events import (
    LLMCallRecord,
    RunRecord,
    RunStatus,
    StepRecord,
    TraceEvent,
    TraceEventKind,
)
from .observability.recorder import NullTraceRecorder, TraceRecorder
```

Add an optional constructor argument after `log_thinking`:

```python
        trace_recorder: TraceRecorder | None = None,
```

Wire it into the runtime context:

```python
        self.trace_recorder = trace_recorder or NullTraceRecorder()
        self.run_id: str | None = None
        self.runtime_context = RunContext(
            workspace_dir=Path(workspace_dir),
            checkpoint_store=checkpoint_store,
            task_memory_hook=task_memory_hook,
            tool_confirmation_callback=tool_confirmation_callback,
            trace_recorder=self.trace_recorder,
        )
```

Add helpers inside `Agent` before `run()`:

```python
    def _trace_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _trace_event(self, kind: TraceEventKind, payload: dict[str, Any] | None = None) -> None:
        if self.run_id is None:
            return
        self.trace_recorder.record_event(
            TraceEvent(
                event_id=f"event-{uuid4().hex}",
                run_id=self.run_id,
                kind=kind,
                created_at=self._trace_now(),
                payload=payload or {},
            )
        )

    def _run_record(
        self,
        *,
        started_at: str,
        status: RunStatus,
        total_steps: int,
        terminal_reason: str | None = None,
        ended_at: str | None = None,
        duration_ms: float | None = None,
    ) -> RunRecord:
        return RunRecord(
            run_id=self.run_id or "",
            workspace_dir=str(self.workspace_dir),
            model=getattr(self.llm, "model", None),
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            status=status,
            terminal_reason=terminal_reason,
            total_steps=total_steps,
            prompt_tokens=self.cumulative_prompt_tokens,
            completion_tokens=self.cumulative_completion_tokens,
            total_tokens=self.cumulative_total_tokens,
            cached_tokens=self.cumulative_cached_tokens,
            cache_write_tokens=self.cumulative_cache_write_tokens,
            total_cost=self.cumulative_token_cost.total_cost,
            currency=self.cumulative_token_cost.currency,
        )
```

- [ ] **Step 4: Emit run, step, and LLM trace records**

At the start of `run()` after cancellation state setup:

```python
        self.run_id = f"run-{uuid4().hex}"
        self.runtime_context.run_id = self.run_id
        run_started_at = self._trace_now()
        run_started_timer = perf_counter()
        self.trace_recorder.record_run(
            self._run_record(started_at=run_started_at, status=RunStatus.RUNNING, total_steps=0)
        )
        self._trace_event(TraceEventKind.RUN_STARTED, {"workspace_dir": str(self.workspace_dir)})
```

At the top of each step:

```python
            self.runtime_context.step_index = step + 1
            step_id = f"step-{uuid4().hex}"
            step_started_at = self._trace_now()
            self.trace_recorder.record_step(
                StepRecord(step_id=step_id, run_id=self.run_id, step_index=step + 1, started_at=step_started_at)
            )
            self._trace_event(TraceEventKind.STEP_STARTED, {"step_id": step_id, "step_index": step + 1})
```

Around the LLM call:

```python
            llm_call_id = f"llm-{uuid4().hex}"
            llm_started_at = self._trace_now()
            llm_started_timer = perf_counter()
            self._trace_event(
                TraceEventKind.LLM_STARTED,
                {"call_id": llm_call_id, "step_index": step + 1, "message_count": len(request_messages)},
            )
```

Record failures inside the current `except Exception as e` branch before returning:

```python
                self.trace_recorder.record_llm_call(
                    LLMCallRecord(
                        call_id=llm_call_id,
                        run_id=self.run_id,
                        step_index=step + 1,
                        started_at=llm_started_at,
                        ended_at=self._trace_now(),
                        duration_ms=round((perf_counter() - llm_started_timer) * 1000, 3),
                        request_message_count=len(request_messages),
                        request_tool_names=[tool.name for tool in tool_list],
                        error=f"{type(e).__name__}: {str(e)}",
                    )
                )
                self._trace_event(TraceEventKind.LLM_FAILED, {"call_id": llm_call_id, "step_index": step + 1})
                ended_at = self._trace_now()
                self.trace_recorder.record_run(
                    self._run_record(
                        started_at=run_started_at,
                        ended_at=ended_at,
                        duration_ms=round((perf_counter() - run_started_timer) * 1000, 3),
                        status=RunStatus.FAILED,
                        terminal_reason="llm_failed",
                        total_steps=step,
                    )
                )
                self._trace_event(TraceEventKind.RUN_FAILED, {"reason": "llm_failed"})
```

Record successful calls after token accounting:

```python
            self.trace_recorder.record_llm_call(
                LLMCallRecord(
                    call_id=llm_call_id,
                    run_id=self.run_id,
                    step_index=step + 1,
                    started_at=llm_started_at,
                    ended_at=self._trace_now(),
                    duration_ms=round((perf_counter() - llm_started_timer) * 1000, 3),
                    finish_reason=response.finish_reason,
                    request_message_count=len(request_messages),
                    request_tool_names=[tool.name for tool in tool_list],
                    usage=response.usage,
                    cost=self.last_token_cost,
                )
            )
            self._trace_event(TraceEventKind.LLM_COMPLETED, {"call_id": llm_call_id, "step_index": step + 1})
```

When a step completes, update its terminal record:

```python
                self.trace_recorder.record_step(
                    StepRecord(
                        step_id=step_id,
                        run_id=self.run_id,
                        step_index=step + 1,
                        started_at=step_started_at,
                        ended_at=self._trace_now(),
                        duration_ms=round(step_elapsed * 1000, 3),
                        stop_reason="completed",
                    )
                )
```

On the completed return path, persist the terminal run:

```python
                ended_at = self._trace_now()
                self.trace_recorder.record_run(
                    self._run_record(
                        started_at=run_started_at,
                        ended_at=ended_at,
                        duration_ms=round((perf_counter() - run_started_timer) * 1000, 3),
                        status=RunStatus.COMPLETED,
                        terminal_reason="completed",
                        total_steps=step + 1,
                    )
                )
                self._trace_event(TraceEventKind.RUN_COMPLETED, {"steps": step + 1})
```

- [ ] **Step 5: Run focused Agent trace tests**

Run:

```bash
uv run pytest tests/test_agent.py::test_agent_records_completed_run_and_llm_usage tests/test_agent.py::test_agent_records_llm_failure_and_failed_run -q
```

Expected: PASS.

- [ ] **Step 6: Run main Agent regression tests**

Run:

```bash
uv run pytest tests/test_agent.py::test_checkpoint_saved_for_tool_and_completion tests/test_agent.py::test_agent_uses_compression_pipeline_before_final_request -q
```

Expected: PASS. Checkpoint and compression-pipeline behavior remain unchanged.

- [ ] **Step 7: Commit Agent run and LLM instrumentation**

```bash
git add mini_agent/agent.py tests/test_agent.py
git commit -m "feat: trace agent run and llm lifecycle"
```

## Task 6: Cover Terminal Statuses And Best-Effort Behavior

**Files:**
- Modify: `mini_agent/agent.py`
- Modify: `tests/test_agent.py`
- Modify: `tests/test_observability_events.py`

- [ ] **Step 1: Add failing tests for terminal run states**

Append to `tests/test_agent.py`:

```python
@pytest.mark.asyncio
async def test_agent_records_cancelled_run(tmp_path):
    recorder = AgentTraceRecorder()
    llm_client = MagicMock(spec=LLMClient)
    agent = Agent(llm_client=llm_client, system_prompt="System", tools=[], workspace_dir=str(tmp_path), trace_recorder=recorder)
    agent.add_user_message("cancel")
    cancel_event = asyncio.Event()
    cancel_event.set()

    assert await agent.run(cancel_event=cancel_event) == "Task cancelled by user."
    assert recorder.runs[-1].status is RunStatus.CANCELLED
    assert recorder.events[-1].kind is TraceEventKind.RUN_CANCELLED


@pytest.mark.asyncio
async def test_agent_records_max_steps_run(tmp_path):
    recorder = AgentTraceRecorder()
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate = AsyncMock(
        return_value=LLMResponse(
            content="",
            tool_calls=[ToolCall(id="call-1", type="function", function=FunctionCall(name="missing_tool", arguments={}))],
            finish_reason="tool_calls",
        )
    )
    agent = Agent(
        llm_client=llm_client,
        system_prompt="System",
        tools=[],
        workspace_dir=str(tmp_path),
        max_steps=1,
        trace_recorder=recorder,
    )
    agent.add_user_message("loop")

    result = await agent.run()

    assert "couldn't be completed" in result
    assert recorder.runs[-1].status is RunStatus.MAX_STEPS
    assert recorder.events[-1].kind is TraceEventKind.RUN_MAX_STEPS
```

Append to `tests/test_observability_events.py`:

```python
def test_store_trace_recorder_keeps_agent_path_best_effort_for_all_record_types():
    class AlwaysFailingStore(FailingStore):
        def save_step(self, step):
            raise RuntimeError("step write failed")

        def save_llm_call(self, call):
            raise RuntimeError("llm write failed")

        def save_tool_call(self, call):
            raise RuntimeError("tool write failed")

        def save_event(self, event):
            raise RuntimeError("event write failed")

    recorder = StoreTraceRecorder(AlwaysFailingStore())
    recorder.record_step(StepRecord(step_id="step-1", run_id="run-1", step_index=1, started_at="2026-05-22T10:00:00+00:00"))
    recorder.record_llm_call(LLMCallRecord(call_id="llm-1", run_id="run-1", step_index=1, started_at="2026-05-22T10:00:00+00:00"))
    recorder.record_tool_call(ToolCallRecord(call_id="tool-1", run_id="run-1", tool_name="read_file", started_at="2026-05-22T10:00:00+00:00"))
    recorder.record_event(TraceEvent(event_id="event-1", run_id="run-1", kind=TraceEventKind.RUN_STARTED, created_at="2026-05-22T10:00:00+00:00"))
```

- [ ] **Step 2: Run the new status tests and verify they fail**

Run:

```bash
uv run pytest tests/test_agent.py::test_agent_records_cancelled_run tests/test_agent.py::test_agent_records_max_steps_run tests/test_observability_events.py::test_store_trace_recorder_keeps_agent_path_best_effort_for_all_record_types -q
```

Expected: cancellation and max-step assertions FAIL until terminal trace paths are implemented; best-effort recorder test PASS if Task 2 stayed generic.

- [ ] **Step 3: Add terminal-state run tracing**

In every cancellation return path in `Agent.run()`, record the terminal run and event before returning:

```python
                ended_at = self._trace_now()
                self.trace_recorder.record_run(
                    self._run_record(
                        started_at=run_started_at,
                        ended_at=ended_at,
                        duration_ms=round((perf_counter() - run_started_timer) * 1000, 3),
                        status=RunStatus.CANCELLED,
                        terminal_reason="cancelled",
                        total_steps=step,
                    )
                )
                self._trace_event(TraceEventKind.RUN_CANCELLED, {"step_index": step})
```

At the max-step path, record:

```python
        ended_at = self._trace_now()
        self.trace_recorder.record_run(
            self._run_record(
                started_at=run_started_at,
                ended_at=ended_at,
                duration_ms=round((perf_counter() - run_started_timer) * 1000, 3),
                status=RunStatus.MAX_STEPS,
                terminal_reason="max_steps",
                total_steps=step,
            )
        )
        self._trace_event(TraceEventKind.RUN_MAX_STEPS, {"steps": step})
```

Also complete non-terminal step records before incrementing `step`:

```python
            self.trace_recorder.record_step(
                StepRecord(
                    step_id=step_id,
                    run_id=self.run_id,
                    step_index=step + 1,
                    started_at=step_started_at,
                    ended_at=self._trace_now(),
                    duration_ms=round(step_elapsed * 1000, 3),
                    stop_reason="tool_calls",
                )
            )
```

- [ ] **Step 4: Run terminal-state tests**

Run:

```bash
uv run pytest tests/test_agent.py::test_agent_records_cancelled_run tests/test_agent.py::test_agent_records_max_steps_run tests/test_observability_events.py::test_store_trace_recorder_keeps_agent_path_best_effort_for_all_record_types -q
```

Expected: PASS.

- [ ] **Step 5: Run the focused trace suite**

Run:

```bash
uv run pytest tests/test_observability_events.py tests/test_observability_sqlite_store.py tests/test_runtime.py tests/test_agent.py -q
```

Expected: PASS for the trace additions and current runtime/Agent regression tests. If external LLM integration-style tests inside `tests/test_agent.py` are selected by default and require unavailable credentials, rerun the focused unit test node IDs from Tasks 4-6 and record the skipped external constraint in the final implementation note.

- [ ] **Step 6: Commit terminal-state coverage**

```bash
git add mini_agent/agent.py tests/test_agent.py tests/test_observability_events.py
git commit -m "test: cover trace terminal run states"
```

## Task 7: Verify SQLite Trace Linkage End To End

**Files:**
- Modify: `tests/test_agent.py`

- [ ] **Step 1: Add failing Agent-to-SQLite integration test**

Append to `tests/test_agent.py`:

```python
import sqlite3

from mini_agent.observability.recorder import StoreTraceRecorder
from mini_agent.observability.sqlite_store import SQLiteTraceStore


@pytest.mark.asyncio
async def test_agent_persists_completed_trace_to_sqlite(tmp_path):
    db_path = tmp_path / "traces.db"
    recorder = StoreTraceRecorder(SQLiteTraceStore(db_path))
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate = AsyncMock(return_value=LLMResponse(content="done", tool_calls=None, finish_reason="stop"))
    agent = Agent(llm_client=llm_client, system_prompt="System", tools=[], workspace_dir=str(tmp_path), trace_recorder=recorder)
    agent.add_user_message("persist trace")

    assert await agent.run() == "done"

    connection = sqlite3.connect(db_path)
    run = connection.execute("select status, terminal_reason from agent_runs").fetchone()
    llm_call_count = connection.execute("select count(*) from llm_calls").fetchone()[0]
    event_kinds = [row[0] for row in connection.execute("select kind from run_events order by created_at, event_id")]
    assert run == ("completed", "completed")
    assert llm_call_count == 1
    assert event_kinds[0] == "run_started"
    assert event_kinds[-1] == "run_completed"
```

- [ ] **Step 2: Run the integration test and verify it fails if wiring is incomplete**

Run:

```bash
uv run pytest tests/test_agent.py::test_agent_persists_completed_trace_to_sqlite -q
```

Expected: PASS if Tasks 1-6 wired the store and recorder correctly. If it FAILS, the failure should point to the missing store write or runtime wiring that must be fixed before this task continues.

- [ ] **Step 3: Fix only missing linkage revealed by the test**

If the store integration test fails because event ordering has identical timestamps, update the test query to order by SQLite insertion order:

```python
event_kinds = [row[0] for row in connection.execute("select kind from run_events order by rowid")]
```

If it fails because `Agent.run()` does not record a terminal run on the success path, apply the terminal `record_run(...)` block from Task 5 before `return response.content`.

Do not add CLI configuration, eval runtime, or dashboard code in this task.

- [ ] **Step 4: Run the integration test and the trace suite**

Run:

```bash
uv run pytest tests/test_agent.py::test_agent_persists_completed_trace_to_sqlite tests/test_observability_events.py tests/test_observability_sqlite_store.py tests/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 5: Run static syntax verification**

Run:

```bash
python -m py_compile mini_agent/agent.py mini_agent/runtime.py mini_agent/observability/events.py mini_agent/observability/recorder.py mini_agent/observability/store.py mini_agent/observability/sqlite_store.py
```

Expected: PASS with no output.

- [ ] **Step 6: Commit end-to-end trace verification**

```bash
git add tests/test_agent.py mini_agent/agent.py
git commit -m "test: verify sqlite trace persistence"
```

## Completion Criteria

Before claiming the Trace Foundation plan is complete:

1. `mini_agent.observability` exposes the trace contracts, recorder boundary, and SQLite store.
2. Agent completion, Agent LLM failure, cancellation, and max-step paths record terminal run statuses.
3. ToolRuntime records successful tool calls, blocked policy outcomes, failure status, affected paths, and replayable events.
4. Text logging and existing task behavior still work because observability is best-effort and additive.
5. These commands have been run and their results recorded:

```bash
uv run pytest tests/test_observability_events.py tests/test_observability_sqlite_store.py tests/test_runtime.py -q
uv run pytest tests/test_agent.py::test_agent_records_completed_run_and_llm_usage tests/test_agent.py::test_agent_records_llm_failure_and_failed_run tests/test_agent.py::test_agent_records_cancelled_run tests/test_agent.py::test_agent_records_max_steps_run tests/test_agent.py::test_agent_persists_completed_trace_to_sqlite -q
python -m py_compile mini_agent/agent.py mini_agent/runtime.py mini_agent/observability/events.py mini_agent/observability/recorder.py mini_agent/observability/store.py mini_agent/observability/sqlite_store.py
```
