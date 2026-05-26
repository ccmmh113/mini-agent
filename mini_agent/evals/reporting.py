"""Markdown reporting for evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass

from .spec import EvalCandidate, EvalResult, EvalRunReport


@dataclass(frozen=True)
class _CandidateSummary:
    cases: int = 0
    failed: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    duration_ms: float = 0.0

    @property
    def pass_rate(self) -> float:
        return 0.0 if self.cases == 0 else (self.cases - self.failed) / self.cases


def format_eval_report(report: EvalRunReport) -> str:
    """Render an evaluation report as deterministic Markdown."""

    lines = [
        f"# Evaluation Report: {report.suite.name}",
        "",
        f"**Suite:** `{report.suite.suite_key}`",
        f"**Eval Run:** `{report.eval_run_id}`",
        f"**Cases:** {report.case_count}",
        f"**Failed:** {report.failed}",
        f"**Pass Rate:** {_format_percent(report.pass_rate)}",
        f"**Tokens:** {report.total_tokens}",
        f"**Cost:** {report.total_cost:.4f}",
        f"**Duration:** {_format_duration(report.total_duration_ms)}",
        "",
    ]

    metrics = report.metadata.get("metrics") if isinstance(report.metadata, dict) else None
    if isinstance(metrics, dict):
        lines.extend(_format_metrics_section(metrics))

    lines.extend(
        [
            "## Candidate Comparison",
            "",
            "| Candidate | Model | Cases | Failed | Pass Rate | Tokens | Cost | Duration |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    summaries = _candidate_summaries(report.results)
    for candidate in report.candidates:
        summary = summaries.get(candidate.candidate_id, _CandidateSummary())
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate.label or candidate.candidate_id,
                    f"`{candidate.model}`",
                    str(summary.cases),
                    str(summary.failed),
                    _format_percent(summary.pass_rate),
                    str(summary.total_tokens),
                    f"{summary.total_cost:.4f}",
                    _format_duration(summary.duration_ms),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Task Results",
            "",
            "| Candidate | Task | Passed | Score | Status | Duration | Tokens | Cost | Trace Run | Failure |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for result in report.results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.candidate_id,
                    result.task_id,
                    "PASS" if result.passed else "FAIL",
                    _format_score(result),
                    result.status,
                    _format_duration(result.duration_ms),
                    str(result.total_tokens),
                    f"{result.total_cost:.4f}",
                    f"`{result.agent_run_id}`" if result.agent_run_id else "",
                    result.failure_reason or "",
                ]
            )
            + " |"
        )

    lines.append("")
    return "\n".join(lines)


def _format_metrics_section(metrics: dict) -> list[str]:
    latency = metrics.get("latency_ms", {})
    tokens = metrics.get("tokens", {})
    cost = metrics.get("cost", {})
    max_steps = metrics.get("max_steps", {})
    tool_failures = metrics.get("tool_evidence_failures", {})
    scorer_failures = metrics.get("scorer_failures", {})
    trace_linkage = metrics.get("trace_linkage", {})
    context = metrics.get("context_governance", {})
    compression_triggered = context.get("compression_triggered", {}) if isinstance(context, dict) else {}
    observability = metrics.get("observability", {})
    memory = metrics.get("memory_effectiveness", {})
    recall_called = memory.get("recall_notes_called", {}) if isinstance(memory, dict) else {}
    redundant_avoided = memory.get("redundant_read_avoided", {}) if isinstance(memory, dict) else {}
    scorer_text = ", ".join(f"{key}={value}" for key, value in scorer_failures.items()) or "none"
    return [
        "## Metrics",
        "",
        f"- Avg latency: {_format_duration(float(latency.get('avg', 0.0)))}",
        f"- P50 latency: {_format_duration(float(latency.get('p50', 0.0)))}",
        f"- P95 latency: {_format_duration(float(latency.get('p95', 0.0)))}",
        f"- Avg tokens: {float(tokens.get('avg', 0.0)):.2f}",
        f"- Cost per passed task: {float(cost.get('per_passed', 0.0)):.4f}",
        f"- Max-step rate: {_format_percent(float(max_steps.get('rate', 0.0)))} ({int(max_steps.get('count', 0))})",
        f"- Tool-evidence failure rate: {_format_percent(float(tool_failures.get('rate', 0.0)))} ({int(tool_failures.get('count', 0))})",
        f"- Scorer failures: {scorer_text}",
        f"- Trace linkage rate: {_format_percent(float(trace_linkage.get('rate', 0.0)))} ({int(trace_linkage.get('count', 0))})",
        f"- Compression trigger rate: {_format_percent(float(compression_triggered.get('rate', 0.0)))} ({int(compression_triggered.get('count', 0))})",
        f"- Avg compression ratio: {_format_percent(float(context.get('avg_compression_ratio', 0.0)))}",
        f"- Avg compression tokens: {float(context.get('avg_tokens_before_compression', 0.0)):.2f} -> {float(context.get('avg_tokens_after_compression', 0.0)):.2f}",
        f"- Avg LLM calls/task: {float(observability.get('avg_llm_calls', 0.0)):.2f}",
        f"- Avg tool calls/task: {float(observability.get('avg_tool_calls', 0.0)):.2f}",
        f"- Memory recall usage: {_format_percent(float(recall_called.get('rate', 0.0)))} ({int(recall_called.get('count', 0))})",
        f"- Redundant-read avoided rate: {_format_percent(float(redundant_avoided.get('rate', 0.0)))} ({int(redundant_avoided.get('count', 0))})",
        f"- Avg recall_notes calls/task: {float(memory.get('avg_recall_notes_calls', 0.0)):.2f}",
        f"- Avg read_file calls/task: {float(memory.get('avg_read_file_calls', 0.0)):.2f}",
        f"- Avg record_note calls/task: {float(memory.get('avg_record_note_calls', 0.0)):.2f}",
        f"- Estimated avoided read tokens: {int(memory.get('avoided_read_token_estimate', 0))}",
        "",
    ]


def _candidate_summaries(results: list[EvalResult]) -> dict[str, _CandidateSummary]:
    summaries: dict[str, _CandidateSummary] = {}
    for result in results:
        current = summaries.get(result.candidate_id, _CandidateSummary())
        summaries[result.candidate_id] = _CandidateSummary(
            cases=current.cases + 1,
            failed=current.failed + (0 if result.passed else 1),
            total_tokens=current.total_tokens + result.total_tokens,
            total_cost=current.total_cost + result.total_cost,
            duration_ms=current.duration_ms + result.duration_ms,
        )
    return summaries


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_duration(duration_ms: float) -> str:
    return f"{duration_ms:.0f}ms"


def _format_score(result: EvalResult) -> str:
    return f"{result.score.score:g}/{result.score.max_score:g}"
