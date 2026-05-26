"""Aggregate metrics for evaluation reports."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .spec import EvalResult, EvalRunReport


def compute_eval_metrics(report: EvalRunReport) -> dict[str, Any]:
    """Compute aggregate metrics from an evaluation report."""

    results = report.results
    case_count = len(results)
    failed = sum(1 for result in results if not result.passed)
    passed = case_count - failed
    durations = [result.duration_ms for result in results]
    total_tokens = sum(result.total_tokens for result in results)
    total_cost = sum(result.total_cost for result in results)
    status_failure_count = sum(1 for result in results if not result.score.breakdown.get("status", True))
    tool_failure_count = sum(
        1 for result in results if not result.score.breakdown.get("tool_evidence_contains", True)
    )
    max_steps_count = sum(1 for result in results if result.status == "max_steps")

    return {
        "case_count": case_count,
        "failed": failed,
        "pass_rate": _rate(passed, case_count),
        "latency_ms": {
            "total": sum(durations),
            "avg": _avg(durations),
            "p50": _percentile_nearest_rank(durations, 50),
            "p95": _percentile_nearest_rank(durations, 95),
        },
        "tokens": {
            "total": total_tokens,
            "avg": total_tokens / case_count if case_count else 0.0,
        },
        "cost": {
            "total": total_cost,
            "avg": total_cost / case_count if case_count else 0.0,
            "per_passed": total_cost / passed if passed else 0.0,
        },
        "max_steps": {"count": max_steps_count, "rate": _rate(max_steps_count, case_count)},
        "status_failures": {"count": status_failure_count, "rate": _rate(status_failure_count, case_count)},
        "tool_evidence_failures": {"count": tool_failure_count, "rate": _rate(tool_failure_count, case_count)},
        "trace_linkage": _trace_linkage_metrics(results),
        "context_governance": _context_governance_metrics(results),
        "observability": _observability_metrics(results),
        "memory_effectiveness": _memory_effectiveness_metrics(results, report.candidates),
        "scorer_failures": _scorer_failures(results),
        "candidates": _candidate_metrics(results),
    }


def with_eval_metrics(report: EvalRunReport) -> EvalRunReport:
    """Return a copy of the report with aggregate metrics in metadata."""

    metadata = dict(report.metadata)
    metadata["metrics"] = compute_eval_metrics(report)
    return replace(report, metadata=metadata)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _percentile_nearest_rank(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, round((percentile / 100) * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _scorer_failures(results: list[EvalResult]) -> dict[str, int]:
    failures: dict[str, int] = {}
    for result in results:
        for scorer, passed in result.score.breakdown.items():
            if not passed:
                failures[scorer] = failures.get(scorer, 0) + 1
    return dict(sorted(failures.items()))


def _trace_linkage_metrics(results: list[EvalResult]) -> dict[str, Any]:
    linked = sum(1 for result in results if result.agent_run_id)
    return {"count": linked, "rate": _rate(linked, len(results))}


def _context_governance_metrics(results: list[EvalResult]) -> dict[str, Any]:
    contexts = [
        result.metadata.get("context_governance")
        for result in results
        if isinstance(result.metadata.get("context_governance"), dict)
    ]
    triggered = sum(1 for context in contexts if context.get("compression_triggered") is True)
    ratios = [_number(context.get("compression_ratio")) for context in contexts if _number(context.get("compression_ratio")) is not None]
    before_tokens = [
        _number(context.get("before_tokens")) for context in contexts if _number(context.get("before_tokens")) is not None
    ]
    after_tokens = [
        _number(context.get("after_tokens")) for context in contexts if _number(context.get("after_tokens")) is not None
    ]
    return {
        "case_count": len(contexts),
        "compression_triggered": {"count": triggered, "rate": _rate(triggered, len(contexts))},
        "avg_compression_ratio": _avg(ratios),
        "avg_tokens_before_compression": _avg(before_tokens),
        "avg_tokens_after_compression": _avg(after_tokens),
    }


def _observability_metrics(results: list[EvalResult]) -> dict[str, Any]:
    observations = [
        result.metadata.get("observability")
        for result in results
        if isinstance(result.metadata.get("observability"), dict)
    ]
    llm_calls = [
        _number(observation.get("llm_call_count"))
        for observation in observations
        if _number(observation.get("llm_call_count")) is not None
    ]
    tool_calls = [
        _number(observation.get("tool_call_count"))
        for observation in observations
        if _number(observation.get("tool_call_count")) is not None
    ]
    return {
        "case_count": len(observations),
        "avg_llm_calls": _avg(llm_calls),
        "avg_tool_calls": _avg(tool_calls),
    }


def _memory_effectiveness_metrics(results: list[EvalResult], candidates: list[Any] | None = None) -> dict[str, Any]:
    memories = [
        result.metadata.get("memory_effectiveness")
        for result in results
        if isinstance(result.metadata.get("memory_effectiveness"), dict)
    ]
    recall_calls = [
        _number(memory.get("recall_notes_calls"))
        for memory in memories
        if _number(memory.get("recall_notes_calls")) is not None
    ]
    read_calls = [
        _number(memory.get("read_file_calls"))
        for memory in memories
        if _number(memory.get("read_file_calls")) is not None
    ]
    record_calls = [
        _number(memory.get("record_note_calls"))
        for memory in memories
        if _number(memory.get("record_note_calls")) is not None
    ]
    recall_used = sum(
        1
        for memory in memories
        if _number(memory.get("recall_notes_calls")) and memory.get("recall_notes_calls", 0) > 0
    )
    redundant_avoided = sum(1 for memory in memories if memory.get("redundant_read_avoided") is True)
    avoided_tokens = sum(
        _number(memory.get("avoided_read_token_estimate")) or 0.0
        for memory in memories
    )
    return {
        "case_count": len(memories),
        "recall_notes_called": {"count": recall_used, "rate": _rate(recall_used, len(memories))},
        "redundant_read_avoided": {"count": redundant_avoided, "rate": _rate(redundant_avoided, len(memories))},
        "avg_recall_notes_calls": _avg(recall_calls),
        "avg_read_file_calls": _avg(read_calls),
        "avg_record_note_calls": _avg(record_calls),
        "avoided_read_token_estimate": int(avoided_tokens),
        "baseline_comparison": _memory_baseline_comparison(results, candidates or []),
    }


def _number(value: Any) -> float | None:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


def _memory_baseline_comparison(results: list[EvalResult], candidates: list[Any]) -> dict[str, Any]:
    baseline_for: dict[str, str] = {}
    for candidate in candidates:
        metadata = getattr(candidate, "metadata", {})
        if isinstance(metadata, dict) and metadata.get("memory_mode") == "off":
            candidate_id = getattr(candidate, "candidate_id", "")
            baseline_for[candidate_id] = str(metadata.get("baseline_for") or _strip_memory_off_suffix(candidate_id))

    for result in results:
        if result.candidate_id.endswith("-memory-off"):
            baseline_for.setdefault(result.candidate_id, _strip_memory_off_suffix(result.candidate_id))

    keyed = {(result.candidate_id, result.task_id): result for result in results}
    pair_count = 0
    baseline_read_calls = 0
    memory_read_calls = 0
    baseline_tokens = 0
    memory_tokens = 0

    for baseline_id, memory_id in baseline_for.items():
        for result in results:
            if result.candidate_id != baseline_id:
                continue
            memory_result = keyed.get((memory_id, result.task_id))
            if memory_result is None:
                continue
            baseline_memory = result.metadata.get("memory_effectiveness")
            active_memory = memory_result.metadata.get("memory_effectiveness")
            if not isinstance(baseline_memory, dict) or not isinstance(active_memory, dict):
                continue
            baseline_read_calls += int(_number(baseline_memory.get("read_file_calls")) or 0)
            memory_read_calls += int(_number(active_memory.get("read_file_calls")) or 0)
            baseline_tokens += result.total_tokens
            memory_tokens += memory_result.total_tokens
            pair_count += 1

    read_delta = baseline_read_calls - memory_read_calls
    token_delta = baseline_tokens - memory_tokens
    return {
        "pair_count": pair_count,
        "baseline_read_file_calls": baseline_read_calls,
        "memory_read_file_calls": memory_read_calls,
        "read_file_call_delta": read_delta,
        "read_file_call_reduction_rate": read_delta / baseline_read_calls if baseline_read_calls else 0.0,
        "baseline_total_tokens": baseline_tokens,
        "memory_total_tokens": memory_tokens,
        "total_token_delta": token_delta,
        "total_token_reduction_rate": token_delta / baseline_tokens if baseline_tokens else 0.0,
    }


def _strip_memory_off_suffix(candidate_id: str) -> str:
    return candidate_id.removesuffix("-memory-off")


def _candidate_metrics(results: list[EvalResult]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[EvalResult]] = {}
    for result in results:
        grouped.setdefault(result.candidate_id, []).append(result)

    metrics: dict[str, dict[str, Any]] = {}
    for candidate_id, candidate_results in grouped.items():
        case_count = len(candidate_results)
        failed = sum(1 for result in candidate_results if not result.passed)
        durations = [result.duration_ms for result in candidate_results]
        metrics[candidate_id] = {
            "case_count": case_count,
            "failed": failed,
            "pass_rate": _rate(case_count - failed, case_count),
            "latency_ms": {
                "avg": _avg(durations),
                "p50": _percentile_nearest_rank(durations, 50),
                "p95": _percentile_nearest_rank(durations, 95),
            },
            "tokens": sum(result.total_tokens for result in candidate_results),
            "cost": sum(result.total_cost for result in candidate_results),
            "max_steps": sum(1 for result in candidate_results if result.status == "max_steps"),
        }
    return metrics
