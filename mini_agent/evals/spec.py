"""Evaluation runtime contracts for Mini Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_SCORERS = [
    "status",
    "output_contains",
    "file_contains",
    "tool_evidence_contains",
]


@dataclass(frozen=True)
class EvalTask:
    """One repeatable Agent task and its deterministic expectations."""

    task_id: str
    prompt: str
    description: str = ""
    expected_output_contains: list[str] = field(default_factory=list)
    expected_files: dict[str, str | list[str]] = field(default_factory=dict)
    expected_tool_evidence_contains: list[str] = field(default_factory=list)
    expected_status: str = "completed"
    scorers: list[str] = field(default_factory=lambda: list(DEFAULT_SCORERS))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalSuite:
    """A versioned collection of repeatable evaluation tasks."""

    suite_id: str
    name: str
    version: str
    tasks: list[EvalTask]
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def suite_key(self) -> str:
        return f"{self.suite_id}@{self.version}"


@dataclass(frozen=True)
class EvalCandidate:
    """One model or Agent configuration to evaluate."""

    candidate_id: str
    model: str
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalExecution:
    """Raw execution output produced by an evaluated candidate."""

    output: str
    status: str = "completed"
    agent_run_id: str | None = None
    workspace_files: dict[str, str] = field(default_factory=dict)
    tool_evidence: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    currency: str = "USD"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalScore:
    """Deterministic score summary for one task execution."""

    passed: bool
    score: float
    max_score: float
    breakdown: dict[str, bool] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalResult:
    """Scored result for one candidate on one task."""

    eval_run_id: str
    suite_id: str
    suite_version: str
    candidate_id: str
    task_id: str
    agent_run_id: str | None
    passed: bool
    score: EvalScore
    output: str = ""
    status: str = "completed"
    duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    currency: str = "USD"
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalRunReport:
    """Aggregate report for a concrete suite execution."""

    eval_run_id: str
    suite: EvalSuite
    candidates: list[EvalCandidate]
    results: list[EvalResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def case_count(self) -> int:
        return len(self.results)

    @property
    def failed(self) -> int:
        return sum(1 for result in self.results if not result.passed)

    @property
    def pass_rate(self) -> float:
        return 0.0 if not self.results else (len(self.results) - self.failed) / len(self.results)

    @property
    def total_duration_ms(self) -> float:
        return sum(result.duration_ms for result in self.results)

    @property
    def total_tokens(self) -> int:
        return sum(result.total_tokens for result in self.results)

    @property
    def total_cost(self) -> float:
        return sum(result.total_cost for result in self.results)
