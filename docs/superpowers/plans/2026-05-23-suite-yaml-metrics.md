# Suite YAML And Eval Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users define evaluation suites in YAML and add richer aggregate metrics to evaluation reports.

**Architecture:** Add a focused `suite_loader.py` that maps YAML to existing eval contracts, and a `metrics.py` module that derives aggregate metrics from `EvalRunReport`. Integrate optional loaded suites into real-model benchmark execution and CLI `eval run --suite`, while keeping built-in deterministic benchmarks unchanged.

**Tech Stack:** Python 3.12, PyYAML already present, dataclasses, existing eval runtime contracts, pytest/pytest-asyncio.

---

## Task 1: Suite YAML Loader

**Files:**
- Create: `mini_agent/evals/suite_loader.py`
- Modify: `mini_agent/evals/__init__.py`
- Test: `tests/test_evals_suite_loader.py`

- [ ] Write failing tests for loading a suite YAML with output, file, tool-evidence, status, scorers, and metadata.
- [ ] Implement YAML validation and conversion to `EvalSuite`.
- [ ] Run suite loader tests and commit.

## Task 2: Eval Metrics Aggregation

**Files:**
- Create: `mini_agent/evals/metrics.py`
- Modify: `mini_agent/evals/__init__.py`
- Modify: `mini_agent/evals/reporting.py`
- Test: `tests/test_evals_metrics.py`
- Test: `tests/test_evals_reporting.py`

- [ ] Write failing tests for p50/p95 latency, cost per passed task, scorer failure distribution, status failures, max-step rate, and per-candidate summaries.
- [ ] Implement `compute_eval_metrics(report)`.
- [ ] Render a metrics section in Markdown reports when metadata contains metrics.
- [ ] Run metrics/reporting tests and commit.

## Task 3: Benchmark And CLI Suite Integration

**Files:**
- Modify: `benchmarks/agent_benchmark.py`
- Modify: `mini_agent/cli.py`
- Modify: `tests/test_benchmark.py`
- Modify: `tests/test_cli_eval.py`

- [ ] Write failing tests that real eval can use a loaded suite and CLI parses/routes `--suite`.
- [ ] Convert loaded `EvalSuite` tasks to real benchmark cases.
- [ ] Compute metrics before persisting eval reports.
- [ ] Reject `--suite` without `--real` with a clear message.
- [ ] Run benchmark/CLI tests and commit.

## Final Verification

Run:

```bash
uv run pytest -q
python -m py_compile mini_agent/evals/suite_loader.py mini_agent/evals/metrics.py benchmarks/agent_benchmark.py mini_agent/cli.py
git diff --check
git status --short
```

Expected: all tests pass, compilation passes, whitespace check passes, worktree clean after commits.
