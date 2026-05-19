import pytest

from benchmarks.module_benchmark import (
    format_markdown_report,
    run_checkpoint_benchmark,
    run_compression_benchmark,
    run_memory_benchmark,
    run_module_benchmark,
)


@pytest.mark.asyncio
async def test_module_benchmark_reports_all_modules():
    report = await run_module_benchmark()

    assert report["module_count"] == 3
    assert report["failed"] == 0
    assert report["pass_rate"] == 1.0
    assert set(report["modules"]) == {"compression", "memory", "checkpoint"}


def test_compression_benchmark_measures_each_layer(tmp_path):
    report = run_compression_benchmark(tmp_path)
    by_case = {case["case"]: case for case in report["cases"]}

    assert report["failed"] == 0
    assert by_case["tool_result_budget"]["tool_results_spilled"] >= 2
    assert by_case["tool_result_budget"]["tool_result_files"] >= 2
    assert by_case["tool_result_budget"]["tokens_saved"] > 0
    assert by_case["snip"]["snipped_messages"] > 0
    assert by_case["snip"]["snip_tokens_freed"] > 0
    assert by_case["micro_compact"]["micro_compacted_results"] > 0
    assert by_case["micro_compact"]["micro_marker_present"] is True
    assert by_case["context_collapse"]["collapsed_messages"] > 0
    assert by_case["context_collapse"]["original_history_unchanged"] is True
    assert by_case["context_collapse"]["auto_compact_avoided"] is True
    assert by_case["auto_compact_fallback"]["auto_compact_called"] is True


@pytest.mark.asyncio
async def test_memory_benchmark_measures_index_recall_redaction_and_staleness(tmp_path):
    report = await run_memory_benchmark(tmp_path)
    by_case = {case["case"]: case for case in report["cases"]}

    assert report["failed"] == 0
    assert by_case["index_and_recall"]["memory_index_loaded"] is True
    assert by_case["index_and_recall"]["topic_memory_loaded_on_demand"] is True
    assert by_case["index_and_recall"]["memory_helped_answer"] is True
    assert by_case["secret_redaction"]["secret_redacted_before_write"] is True
    assert by_case["secret_redaction"]["raw_secret_absent_from_files"] is True
    assert by_case["stale_memory_guard"]["stale_warning_present"] is True
    assert by_case["stale_memory_guard"]["stale_memory_blind_trust"] is False


def test_checkpoint_benchmark_measures_save_validate_restore_and_resume(tmp_path):
    report = run_checkpoint_benchmark(tmp_path)
    by_case = {case["case"]: case for case in report["cases"]}

    assert report["failed"] == 0
    assert by_case["save_and_validate"]["checkpoint_created"] is True
    assert by_case["save_and_validate"]["latest_checkpoint_valid_json"] is True
    assert by_case["save_and_validate"]["checkpoint_reason_correct"] is True
    assert by_case["restore_messages"]["messages_restored"] is True
    assert by_case["restore_messages"]["workspace_validation_passed"] is True
    assert by_case["resume_continues_task"]["resume_continues_task"] is True


@pytest.mark.asyncio
async def test_module_benchmark_can_write_markdown_report(tmp_path):
    report = await run_module_benchmark()
    markdown = format_markdown_report(report)
    output = tmp_path / "module-benchmark.md"
    output.write_text(markdown, encoding="utf-8")

    saved = output.read_text(encoding="utf-8")
    assert saved.startswith("# Mini Agent Module Benchmark Report")
    assert "## Summary" in saved
    assert "| Module | Cases | Failed | Pass Rate | Tokens Saved |" in saved
    assert "## Compression" in saved
    assert "tool_result_budget" in saved
    assert "tokens_saved" in saved
    assert "## Memory" in saved
    assert "secret_redaction" in saved
    assert "secret_redacted_before_write" in saved
    assert "## Checkpoint" in saved
    assert "resume_continues_task" in saved
