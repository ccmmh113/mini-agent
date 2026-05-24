"""Tests for deterministic evaluation scorers."""

from __future__ import annotations

from mini_agent.evals import EvalExecution, EvalTask
from mini_agent.evals.scorers import score_task_result


def test_output_contains_scorer_records_missing_fragments():
    task = EvalTask(
        task_id="direct",
        prompt="Answer",
        expected_output_contains=["Mini Agent", "observable"],
        scorers=["output_contains"],
    )
    execution = EvalExecution(output="Mini Agent is local.")

    score = score_task_result(task, execution)

    assert score.passed is False
    assert score.score == 0
    assert score.max_score == 1
    assert score.breakdown["output_contains"] is False
    assert score.failure_reasons == ["output missing fragment: observable"]


def test_file_contains_scorer_checks_relative_artifacts():
    task = EvalTask(
        task_id="write",
        prompt="Write report",
        expected_files={"result.md": "passed"},
        scorers=["file_contains"],
    )
    execution = EvalExecution(output="done", workspace_files={"result.md": "# Result\n\npassed\n"})

    score = score_task_result(task, execution)

    assert score.passed is True
    assert score.score == 1
    assert score.breakdown["file_contains"] is True


def test_file_contains_scorer_reports_missing_file():
    task = EvalTask(
        task_id="write",
        prompt="Write report",
        expected_files={"result.md": "passed"},
        scorers=["file_contains"],
    )
    execution = EvalExecution(output="done", workspace_files={})

    score = score_task_result(task, execution)

    assert score.passed is False
    assert score.failure_reasons == ["expected file missing: result.md"]


def test_file_contains_scorer_accepts_multiple_expected_fragments_per_file():
    task = EvalTask(
        task_id="write",
        prompt="Write report",
        expected_files={"result.md": ["Mini Agent Benchmark", "passed", "real-model"]},
        scorers=["file_contains"],
    )
    execution = EvalExecution(
        output="done",
        workspace_files={"result.md": "# Mini Agent Benchmark\n\npassed\n\nreal-model\n"},
    )

    score = score_task_result(task, execution)

    assert score.passed is True
    assert score.breakdown["file_contains"] is True


def test_tool_evidence_scorer_matches_tool_message_fragments():
    task = EvalTask(
        task_id="tool",
        prompt="Use a tool",
        expected_tool_evidence_contains=["Command blocked by security policy"],
        scorers=["tool_evidence_contains"],
    )
    execution = EvalExecution(output="blocked", tool_evidence=["Command blocked by security policy: rm -rf"])

    score = score_task_result(task, execution)

    assert score.passed is True
    assert score.breakdown["tool_evidence_contains"] is True


def test_status_scorer_checks_terminal_state():
    task = EvalTask(task_id="status", prompt="Finish", expected_status="completed", scorers=["status"])
    execution = EvalExecution(output="stopped", status="max_steps")

    score = score_task_result(task, execution)

    assert score.passed is False
    assert score.failure_reasons == ["expected status completed, got max_steps"]


def test_combined_scoring_passes_only_when_all_requested_rules_pass():
    task = EvalTask(
        task_id="combined",
        prompt="Create result",
        expected_output_contains=["done"],
        expected_files={"result.md": "passed"},
        expected_tool_evidence_contains=["write_file"],
    )
    execution = EvalExecution(
        output="done",
        workspace_files={"result.md": "passed"},
        tool_evidence=["write_file wrote result.md"],
        status="completed",
    )

    score = score_task_result(task, execution)

    assert score.passed is True
    assert score.score == 4
    assert score.max_score == 4
    assert score.breakdown == {
        "status": True,
        "output_contains": True,
        "file_contains": True,
        "tool_evidence_contains": True,
    }


def test_metadata_contains_scorer_checks_nested_execution_metadata():
    task = EvalTask(
        task_id="context",
        prompt="Run context governance task",
        scorers=["metadata_contains"],
        metadata={
            "expected_metadata_contains": {
                "context_governance.compression_triggered": True,
                "context_governance.compression_markers": "context_snip",
                "context_governance.token_limit": 900,
            }
        },
    )
    execution = EvalExecution(
        output="done",
        metadata={
            "context_governance": {
                "compression_triggered": True,
                "compression_markers": ["context_snip", "micro_compact"],
                "token_limit": 900,
            }
        },
    )

    score = score_task_result(task, execution)

    assert score.passed is True
    assert score.breakdown["metadata_contains"] is True


def test_metadata_contains_scorer_reports_missing_nested_values():
    task = EvalTask(
        task_id="context",
        prompt="Run context governance task",
        scorers=["metadata_contains"],
        metadata={
            "expected_metadata_contains": {
                "context_governance.compression_triggered": True,
                "context_governance.compression_markers": "harness_summary",
            }
        },
    )
    execution = EvalExecution(
        output="done",
        metadata={
            "context_governance": {
                "compression_triggered": False,
                "compression_markers": ["context_snip"],
            }
        },
    )

    score = score_task_result(task, execution)

    assert score.passed is False
    assert score.breakdown["metadata_contains"] is False
    assert score.failure_reasons == [
        "metadata context_governance.compression_triggered expected True, got False",
        "metadata context_governance.compression_markers missing fragment: harness_summary",
    ]


def test_output_excludes_scorer_fails_on_stale_marker():
    task = EvalTask(
        task_id="stale-output",
        prompt="Prefer the latest instruction.",
        scorers=["output_excludes"],
        metadata={"expected_output_not_contains": ["STALE_OUTCOME"]},
    )
    execution = EvalExecution(output="The answer is STALE_OUTCOME.")

    score = score_task_result(task, execution)

    assert score.passed is False
    assert score.breakdown["output_excludes"] is False
    assert score.failure_reasons == ["output contains forbidden fragment: STALE_OUTCOME"]


def test_file_excludes_scorer_checks_artifacts_for_stale_marker():
    task = EvalTask(
        task_id="stale-file",
        prompt="Write latest decision.",
        scorers=["file_excludes"],
        metadata={"expected_files_not_contains": {"decision.md": ["STALE_OUTCOME"]}},
    )
    execution = EvalExecution(
        output="done",
        workspace_files={"decision.md": "LATEST_WINS\nSTALE_OUTCOME\n"},
    )

    score = score_task_result(task, execution)

    assert score.passed is False
    assert score.breakdown["file_excludes"] is False
    assert score.failure_reasons == ["file decision.md contains forbidden fragment: STALE_OUTCOME"]
