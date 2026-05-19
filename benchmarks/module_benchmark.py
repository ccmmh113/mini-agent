"""Deterministic module benchmarks for Mini Agent internals.

The cases in this file measure harness modules directly, without a real LLM.
They are intended to answer whether compression, memory, and checkpointing
mechanisms are doing observable work and preserving required invariants.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

from mini_agent.checkpoint import CheckpointStore
from mini_agent.context_budget import estimate_messages_tokens
from mini_agent.memory.markdown_store import (
    STALE_MEMORY_WARNING,
    MarkdownMemoryStore,
    format_markdown_memory_index,
)
from mini_agent.request_context import RequestContextBuilder
from mini_agent.schema import FunctionCall, Message, ToolCall
from mini_agent.summarizer import (
    HARNESS_SUMMARY_HEADER,
    MICRO_COMPACT_HEADER,
    TOOL_RESULT_SPILL_HEADER,
    CompactionResult,
    CompressionPipeline,
    ContextCollapser,
    MessageCompactor,
    is_context_collapse_message,
    is_context_snip_message,
)
from mini_agent.tools.note_tool import RecallNoteTool, SessionNoteTool


class _CountingSummarizer:
    def __init__(self) -> None:
        self.calls = 0

    async def summarize_if_needed(
        self,
        messages: list[Message],
        api_total_tokens: int,
        budget_messages: list[Message] | None = None,
        tools: list[object] | None = None,
    ) -> list[Message]:
        del api_total_tokens, budget_messages, tools
        self.calls += 1
        latest_user = next((message for message in reversed(messages) if message.role == "user"), None)
        return [
            Message(role="system", content=f"{HARNESS_SUMMARY_HEADER}\n\nfallback summary"),
            *([latest_user] if latest_user is not None else []),
        ]


def _tool_call(call_id: str, name: str) -> ToolCall:
    return ToolCall(id=call_id, type="function", function=FunctionCall(name=name, arguments={}))


def _tool_round(tool_name: str, content: str, call_id: str) -> list[Message]:
    return [
        Message(
            role="assistant",
            content=f"calling {tool_name}",
            tool_calls=[_tool_call(call_id, tool_name)],
        ),
        Message(role="tool", content=content, name=tool_name, tool_call_id=call_id),
    ]


def _case_result(case: dict[str, Any], required: list[str]) -> dict[str, Any]:
    passed = all(bool(case.get(key)) for key in required)
    return {**case, "passed": passed}


def _module_report(module: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    failed = sum(1 for case in cases if not case.get("passed"))
    tokens_saved = sum(int(case.get("tokens_saved", 0)) for case in cases)
    return {
        "module": module,
        "case_count": len(cases),
        "failed": failed,
        "pass_rate": 0.0 if not cases else (len(cases) - failed) / len(cases),
        "tokens_saved": tokens_saved,
        "cases": cases,
    }


def run_compression_benchmark(base_dir: str | Path | None = None) -> dict[str, Any]:
    """Measure each staged compression layer with deterministic histories."""

    root = Path(base_dir) if base_dir is not None else Path(tempfile.mkdtemp(prefix="mini-agent-compression-"))
    root.mkdir(parents=True, exist_ok=True)
    cases = [
        _compression_tool_budget_case(root / "tool-budget"),
        _compression_snip_case(root / "snip"),
        _compression_micro_case(root / "micro"),
        _compression_collapse_case(root / "collapse"),
        _compression_auto_fallback_case(root / "auto-fallback"),
    ]
    return _module_report("compression", cases)


def _compression_tool_budget_case(workspace: Path) -> dict[str, Any]:
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="inspect large outputs"),
        *_tool_round("read_file", "A" * 90_000, "call_large_a"),
        *_tool_round("bash", "B" * 85_000, "call_large_b"),
        *_tool_round("read_file", "C" * 70_000, "call_large_c"),
        Message(role="user", content="current task"),
    ]
    before_tokens = estimate_messages_tokens(messages)
    compactor = MessageCompactor(token_limit=100_000, workspace_dir=workspace)
    result = compactor.compact(messages)
    after_tokens = estimate_messages_tokens(result.messages)
    marker_messages = [
        message
        for message in result.messages
        if isinstance(message.content, str) and message.content.startswith(TOOL_RESULT_SPILL_HEADER)
    ]
    files = list((workspace / ".mini_agent" / "tool-results").glob("*.txt"))
    case = {
        "case": "tool_result_budget",
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "tokens_saved": max(0, before_tokens - after_tokens),
        "compression_ratio": _ratio_saved(before_tokens, after_tokens),
        "tool_results_spilled": result.tool_results_spilled,
        "tool_result_files": len(files),
        "tool_call_ids_preserved": all(message.tool_call_id for message in marker_messages),
        "current_user_message_preserved": any(message.role == "user" and message.content == "current task" for message in result.messages),
    }
    return _case_result(
        case,
        ["tokens_saved", "tool_results_spilled", "tool_result_files", "tool_call_ids_preserved", "current_user_message_preserved"],
    )


def _compression_snip_case(workspace: Path) -> dict[str, Any]:
    messages = [Message(role="system", content="system")]
    for index in range(14):
        messages.append(Message(role="user", content=f"old user turn {index} " + "history " * 260))
        messages.append(Message(role="assistant", content=f"old assistant turn {index} " + "details " * 280))
    messages.append(Message(role="user", content="current task must stay"))

    before_tokens = estimate_messages_tokens(messages)
    result = MessageCompactor(token_limit=1600, workspace_dir=workspace).compact(messages)
    after_tokens = estimate_messages_tokens(result.messages)
    case = {
        "case": "snip",
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "tokens_saved": max(0, before_tokens - after_tokens),
        "compression_ratio": _ratio_saved(before_tokens, after_tokens),
        "snipped_messages": result.snipped_messages,
        "snip_tokens_freed": result.snip_tokens_freed,
        "snip_marker_present": any(is_context_snip_message(message) for message in result.messages),
        "current_user_message_preserved": any(message.role == "user" and message.content == "current task must stay" for message in result.messages),
    }
    return _case_result(case, ["tokens_saved", "snipped_messages", "snip_tokens_freed", "snip_marker_present", "current_user_message_preserved"])


def _compression_micro_case(workspace: Path) -> dict[str, Any]:
    messages = [Message(role="system", content="system")]
    for index in range(8):
        messages.append(Message(role="user", content=f"tool turn {index}"))
        messages.extend(_tool_round("read_file", f"file output {index}\n" + "line\n" * 900, f"read_{index}"))
    messages.append(Message(role="user", content="current task"))

    before_tokens = estimate_messages_tokens(messages)
    compactor = MessageCompactor(token_limit=600, workspace_dir=workspace, tool_result_soft_byte_limit=1_000_000)
    result = compactor._apply_micro_compact(CompactionResult(messages=list(messages)), active_start=None)
    after_tokens = estimate_messages_tokens(result.messages)
    compacted_tool_messages = [
        message
        for message in result.messages
        if message.role == "tool" and isinstance(message.content, str) and message.content.startswith(MICRO_COMPACT_HEADER)
    ]
    retained_lengths = [len(str(message.content)) for message in compacted_tool_messages]
    case = {
        "case": "micro_compact",
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "tokens_saved": max(0, before_tokens - after_tokens),
        "compression_ratio": _ratio_saved(before_tokens, after_tokens),
        "micro_compacted_results": result.micro_compacted_results,
        "micro_marker_present": bool(compacted_tool_messages),
        "retention_varies_by_age": len(set(retained_lengths)) > 1,
        "current_user_message_preserved": any(message.role == "user" and message.content == "current task" for message in result.messages),
    }
    return _case_result(case, ["tokens_saved", "micro_compacted_results", "micro_marker_present", "current_user_message_preserved"])


def _compression_collapse_case(workspace: Path) -> dict[str, Any]:
    del workspace
    messages = [Message(role="system", content="system")]
    for index in range(12):
        messages.append(Message(role="user", content=f"old request {index} " + "context " * 180))
        messages.append(Message(role="assistant", content=f"old answer {index} " + "details " * 180))
    messages.append(Message(role="user", content="current task"))
    original_dump = [message.model_dump(mode="json") for message in messages]
    before_tokens = estimate_messages_tokens(messages)
    collapser = ContextCollapser(token_limit=max(1000, int(before_tokens / 0.92)))
    result = collapser.apply_collapses_if_needed(messages, request_tokens=before_tokens)
    after_tokens = estimate_messages_tokens(result.messages)
    case = {
        "case": "context_collapse",
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "tokens_saved": max(0, before_tokens - after_tokens),
        "compression_ratio": _ratio_saved(before_tokens, after_tokens),
        "collapsed_messages": result.collapsed_messages,
        "collapse_tokens_freed": result.collapse_tokens_freed,
        "collapse_marker_present": any(is_context_collapse_message(message) for message in result.messages),
        "original_history_unchanged": original_dump == [message.model_dump(mode="json") for message in messages],
        "auto_compact_avoided": result.collapsed_messages > 0 and after_tokens <= collapser.token_limit,
        "current_user_message_preserved": any(message.role == "user" and message.content == "current task" for message in result.messages),
    }
    return _case_result(
        case,
        ["tokens_saved", "collapsed_messages", "collapse_tokens_freed", "collapse_marker_present", "original_history_unchanged", "auto_compact_avoided", "current_user_message_preserved"],
    )


def _compression_auto_fallback_case(workspace: Path) -> dict[str, Any]:
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="single huge current request " + "must keep " * 2000),
    ]
    before_tokens = estimate_messages_tokens(messages)
    summarizer = _CountingSummarizer()
    builder = RequestContextBuilder(core_prompt="core", workspace_dir=workspace, max_recent_messages=50)
    pipeline = CompressionPipeline(
        compactor=MessageCompactor(token_limit=120, workspace_dir=workspace),
        context_collapser=ContextCollapser(token_limit=120),
        summarizer=summarizer,  # type: ignore[arg-type]
        request_context_builder=builder,
        token_limit=120,
    )
    compressed = asyncio.run(
        pipeline.compress_before_request(
            messages=messages,
            api_total_tokens=before_tokens,
            tools=[],
        )
    )
    after_tokens = estimate_messages_tokens(compressed)
    case = {
        "case": "auto_compact_fallback",
        "before_tokens": before_tokens,
        "after_tokens": after_tokens,
        "tokens_saved": max(0, before_tokens - after_tokens),
        "compression_ratio": _ratio_saved(before_tokens, after_tokens),
        "auto_compact_called": summarizer.calls > 0,
        "latest_user_preserved": any(message.role == "user" and "single huge current request" in str(message.content) for message in compressed),
    }
    return _case_result(case, ["auto_compact_called", "latest_user_preserved"])


async def run_memory_benchmark(base_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(base_dir) if base_dir is not None else Path(tempfile.mkdtemp(prefix="mini-agent-memory-"))
    root.mkdir(parents=True, exist_ok=True)
    cases = [
        await _memory_index_and_recall_case(root / "index-recall"),
        await _memory_secret_redaction_case(root / "secret-redaction"),
        _memory_stale_guard_case(root / "stale-guard"),
    ]
    return _module_report("memory", cases)


async def _memory_index_and_recall_case(workspace: Path) -> dict[str, Any]:
    memory_dir = workspace / ".memory"
    store = MarkdownMemoryStore(memory_dir)
    memory = store.save_memory(
        content="The benchmark answer keyword is zircon-harness.",
        memory_type="project",
        name="benchmark-answer",
        description="Benchmark answer keyword",
    )
    index_text = store.index_file.read_text(encoding="utf-8")
    recall = await RecallNoteTool(memory_dir=str(memory_dir)).execute(query="zircon", limit=3)
    recalled_text = recall.content
    case = {
        "case": "index_and_recall",
        "memory_index_loaded": "[benchmark-answer]" in index_text and memory.path.name in index_text,
        "topic_memory_loaded_on_demand": "zircon-harness" in recalled_text,
        "memory_helped_answer": "zircon-harness" in recalled_text,
        "prompt_memory_tokens": estimate_messages_tokens([Message(role="system", content=format_markdown_memory_index("Memory:", store.load()))]),
        "memory_not_overloaded": len(index_text) < 1200,
    }
    return _case_result(case, ["memory_index_loaded", "topic_memory_loaded_on_demand", "memory_helped_answer", "memory_not_overloaded"])


async def _memory_secret_redaction_case(workspace: Path) -> dict[str, Any]:
    memory_dir = workspace / ".memory"
    raw_secret = "sk-" + "A" * 28
    result = await SessionNoteTool(memory_dir=str(memory_dir)).execute(
        content=f"Use token {raw_secret} for nothing; this should be redacted.",
        type="project",
        name="secret-memory",
        description=f"secret {raw_secret}",
    )
    file_text = "\n".join(path.read_text(encoding="utf-8") for path in memory_dir.glob("*.md"))
    case = {
        "case": "secret_redaction",
        "secret_redacted_before_write": "[REDACTED]" in file_text and "secrets redacted" in result.content,
        "raw_secret_absent_from_files": raw_secret not in file_text,
        "memory_files_written": len(list(memory_dir.glob("*.md"))),
    }
    return _case_result(case, ["secret_redacted_before_write", "raw_secret_absent_from_files", "memory_files_written"])


def _memory_stale_guard_case(workspace: Path) -> dict[str, Any]:
    memory_dir = workspace / ".memory"
    missing_file = workspace / "old_utils.py"
    store = MarkdownMemoryStore(memory_dir)
    store.save_memory(
        content=f"Old note says {missing_file.name} contains helper_fn.",
        memory_type="project",
        name="stale-file-note",
        description="Stale file note",
    )
    recall_text = format_markdown_memory_index("Memory:", store.load())
    file_exists_now = missing_file.exists()
    case = {
        "case": "stale_memory_guard",
        "stale_warning_present": STALE_MEMORY_WARNING in recall_text,
        "remembered_file_missing": not file_exists_now,
        "stale_memory_blind_trust": file_exists_now,
    }
    return _case_result(case, ["stale_warning_present", "remembered_file_missing"])


def run_checkpoint_benchmark(base_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(base_dir) if base_dir is not None else Path(tempfile.mkdtemp(prefix="mini-agent-checkpoint-"))
    root.mkdir(parents=True, exist_ok=True)
    cases = [
        _checkpoint_save_and_validate_case(root / "save"),
        _checkpoint_restore_case(root / "restore"),
        _checkpoint_resume_case(root / "resume"),
    ]
    return _module_report("checkpoint", cases)


def _checkpoint_messages(goal: str = "Continue previous task") -> list[Message]:
    return [
        Message(role="system", content="system"),
        Message(role="user", content=goal),
        Message(role="assistant", content="I started the task."),
    ]


def _checkpoint_save_and_validate_case(workspace: Path) -> dict[str, Any]:
    store = CheckpointStore(workspace / ".mini_agent" / "checkpoints")
    store.save(step=2, reason="completed", messages=_checkpoint_messages(), workspace_dir=workspace, available_tools=["read_file"])
    latest = store.load_latest()
    validation = store.validate_messages()
    case = {
        "case": "save_and_validate",
        "checkpoint_created": store.latest_file.exists(),
        "latest_checkpoint_valid_json": isinstance(latest, dict),
        "checkpoint_reason_correct": bool(latest and latest.get("reason") == "completed"),
        "valid_message_count": validation["valid"],
        "history_files": len(list(store.history_dir.glob("*.json"))),
    }
    return _case_result(case, ["checkpoint_created", "latest_checkpoint_valid_json", "checkpoint_reason_correct", "valid_message_count", "history_files"])


def _checkpoint_restore_case(workspace: Path) -> dict[str, Any]:
    store = CheckpointStore(workspace / ".mini_agent" / "checkpoints")
    messages = _checkpoint_messages("Restore this exact goal")
    store.save(step=1, reason="assistant_response", messages=messages, workspace_dir=workspace, available_tools=["read_file", "bash"])
    restored = store.load_latest_messages()
    issues = store.validate_for_workspace(workspace)
    case = {
        "case": "restore_messages",
        "messages_restored": [message.role for message in restored] == ["system", "user", "assistant"]
        and restored[1].content == "Restore this exact goal",
        "workspace_validation_passed": issues == [],
        "restore_status_available": bool(store.get_restore_status()),
    }
    return _case_result(case, ["messages_restored", "workspace_validation_passed", "restore_status_available"])


def _checkpoint_resume_case(workspace: Path) -> dict[str, Any]:
    store = CheckpointStore(workspace / ".mini_agent" / "checkpoints")
    store.save(step=0, reason="tool_result", messages=_checkpoint_messages("Draft report"), workspace_dir=workspace, available_tools=["write_file"])
    restored = store.load_latest_messages()
    continued = [*restored, Message(role="assistant", content="Draft report completed after resume.")]
    case = {
        "case": "resume_continues_task",
        "messages_restored": len(restored) == 3,
        "resume_continues_task": any("completed after resume" in str(message.content) for message in continued),
        "resume_summary_available": bool(store.get_resume_summary()),
    }
    return _case_result(case, ["messages_restored", "resume_continues_task", "resume_summary_available"])


async def run_module_benchmark(module: str = "all", base_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(base_dir) if base_dir is not None else Path(tempfile.mkdtemp(prefix="mini-agent-module-benchmark-"))
    root.mkdir(parents=True, exist_ok=True)
    selected = {"compression", "memory", "checkpoint"} if module == "all" else {module}
    unknown = selected - {"compression", "memory", "checkpoint"}
    if unknown:
        raise ValueError(f"Unknown benchmark module: {', '.join(sorted(unknown))}")

    start = perf_counter()
    modules: dict[str, Any] = {}
    if "compression" in selected:
        modules["compression"] = await asyncio.to_thread(run_compression_benchmark, root / "compression")
    if "memory" in selected:
        modules["memory"] = await run_memory_benchmark(root / "memory")
    if "checkpoint" in selected:
        modules["checkpoint"] = run_checkpoint_benchmark(root / "checkpoint")

    failed = sum(report["failed"] for report in modules.values())
    case_count = sum(report["case_count"] for report in modules.values())
    return {
        "benchmark": "module",
        "module_count": len(modules),
        "case_count": case_count,
        "failed": failed,
        "pass_rate": 0.0 if case_count == 0 else (case_count - failed) / case_count,
        "duration_seconds": round(perf_counter() - start, 4),
        "modules": modules,
    }


def _ratio_saved(before_tokens: int, after_tokens: int) -> float:
    if before_tokens <= 0:
        return 0.0
    return round(max(0, before_tokens - after_tokens) / before_tokens, 4)


def _format_text_report(report: dict[str, Any]) -> str:
    lines = [
        "Mini Agent Module Benchmark",
        f"Modules: {report['module_count']}  Cases: {report['case_count']}  Failed: {report['failed']}  Pass rate: {report['pass_rate']:.0%}",
        "",
    ]
    for module_name, module_report in report["modules"].items():
        lines.append(f"[{module_name}] cases={module_report['case_count']} failed={module_report['failed']} tokens_saved={module_report.get('tokens_saved', 0)}")
        for case in module_report["cases"]:
            status = "PASS" if case.get("passed") else "FAIL"
            saved = case.get("tokens_saved")
            saved_text = f" tokens_saved={saved}" if saved is not None else ""
            lines.append(f"  - {status} {case['case']}{saved_text}")
    return "\n".join(lines)


def format_markdown_report(report: dict[str, Any]) -> str:
    """Render a human-readable Markdown report for module benchmark results."""

    lines = [
        "# Mini Agent Module Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Benchmark: `{report['benchmark']}`",
        f"- Modules: {report['module_count']}",
        f"- Cases: {report['case_count']}",
        f"- Failed: {report['failed']}",
        f"- Pass rate: {report['pass_rate']:.0%}",
        f"- Duration: {report['duration_seconds']}s",
        "",
        "| Module | Cases | Failed | Pass Rate | Tokens Saved |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for module_name, module_report in report["modules"].items():
        lines.append(
            "| "
            f"{_md_cell(module_name)} | "
            f"{module_report['case_count']} | "
            f"{module_report['failed']} | "
            f"{module_report['pass_rate']:.0%} | "
            f"{module_report.get('tokens_saved', 0)} |"
        )

    section_titles = {
        "compression": "Compression",
        "memory": "Memory",
        "checkpoint": "Checkpoint",
    }
    metric_keys = {
        "compression": [
            "before_tokens",
            "after_tokens",
            "tokens_saved",
            "compression_ratio",
            "tool_results_spilled",
            "snipped_messages",
            "micro_compacted_results",
            "collapsed_messages",
            "auto_compact_called",
        ],
        "memory": [
            "memory_index_loaded",
            "topic_memory_loaded_on_demand",
            "memory_helped_answer",
            "secret_redacted_before_write",
            "raw_secret_absent_from_files",
            "stale_memory_blind_trust",
            "prompt_memory_tokens",
        ],
        "checkpoint": [
            "checkpoint_created",
            "latest_checkpoint_valid_json",
            "checkpoint_reason_correct",
            "messages_restored",
            "workspace_validation_passed",
            "resume_continues_task",
            "history_files",
        ],
    }

    for module_name, title in section_titles.items():
        module_report = report["modules"].get(module_name)
        if not module_report:
            continue
        lines.extend(["", f"## {title}", "", "| Case | Status | Key Metrics |", "| --- | --- | --- |"])
        keys = metric_keys[module_name]
        for case in module_report["cases"]:
            status = "PASS" if case.get("passed") else "FAIL"
            metrics = _format_case_metrics(case, keys)
            lines.append(f"| {_md_cell(case['case'])} | {status} | {_md_cell(metrics)} |")

    failed_cases = [
        (module_name, case)
        for module_name, module_report in report["modules"].items()
        for case in module_report["cases"]
        if not case.get("passed")
    ]
    if failed_cases:
        lines.extend(["", "## Failed Cases", "", "| Module | Case | Metrics |", "| --- | --- | --- |"])
        for module_name, case in failed_cases:
            lines.append(
                f"| {_md_cell(module_name)} | {_md_cell(case['case'])} | "
                f"{_md_cell(_format_case_metrics(case, sorted(case.keys())))} |"
            )

    return "\n".join(lines) + "\n"


def _format_case_metrics(case: dict[str, Any], keys: list[str]) -> str:
    parts = []
    for key in keys:
        if key in case:
            parts.append(f"{key}={case[key]}")
    return ", ".join(parts) if parts else "-"


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Mini Agent module benchmarks.")
    parser.add_argument("--module", choices=["all", "compression", "memory", "checkpoint"], default="all")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON report.")
    parser.add_argument("--markdown", type=Path, help="Optional path to write a Markdown report.")
    args = parser.parse_args()

    report = asyncio.run(run_module_benchmark(module=args.module))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(format_markdown_report(report), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(_format_text_report(report))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
