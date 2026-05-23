# Evaluation Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable deterministic evaluation runtime that can run task suites across model/Agent candidates and produce comparable score, latency, token, cost, and trace-linked results.

**Architecture:** Add `mini_agent.evals` as a small domain layer independent of CLI and dashboard code. `spec.py` owns suite/candidate/result contracts, `scorers.py` owns deterministic scoring rules, `runner.py` orchestrates cases against injected candidate callables, and `reporting.py` renders aggregate Markdown summaries.

**Tech Stack:** Python 3.12, dataclasses, existing `mini_agent.schema.TokenUsage` and `TokenCost`, pytest/pytest-asyncio.

---

## Scope

This plan implements the first slice of Phase 2:

- evaluation suite and task-case contracts
- deterministic scoring rules for output, files, status, and tool evidence
- async runner over multiple candidates
- aggregate comparison metrics
- Markdown report rendering

This plan does not add CLI commands, real-model provider loading, dashboard pages, or SQLite persistence for evaluation tables. Those are follow-up plans after this runtime shape is stable.

## File Structure

### New files

- `mini_agent/evals/__init__.py`
  - Public exports for evaluation contracts, scorers, runner, and reporting.
- `mini_agent/evals/spec.py`
  - Dataclasses for `EvalTask`, `EvalSuite`, `EvalCandidate`, `EvalExecution`, `EvalScore`, `EvalResult`, and `EvalRunReport`.
- `mini_agent/evals/scorers.py`
  - Deterministic scoring based on expected output fragments, expected files, expected tool evidence, and expected terminal status.
- `mini_agent/evals/runner.py`
  - `run_eval_suite()` async orchestration over suite tasks and candidates.
- `mini_agent/evals/reporting.py`
  - Markdown rendering for suite-level and candidate-level comparison.
- `tests/test_evals_spec.py`
  - Contract and validation tests.
- `tests/test_evals_scorers.py`
  - Deterministic scorer tests.
- `tests/test_evals_runner.py`
  - Runner aggregation and trace-link tests.
- `tests/test_evals_reporting.py`
  - Markdown report tests.

## Task 1: Evaluation Contracts

**Files:**
- Create: `mini_agent/evals/__init__.py`
- Create: `mini_agent/evals/spec.py`
- Test: `tests/test_evals_spec.py`

- [ ] **Step 1: Write failing contract tests**

Create tests that build one suite with two tasks and one candidate, then assert stable IDs, default scorer list, and result pass-rate aggregation.

- [ ] **Step 2: Run contract tests and verify failure**

Run:

```bash
uv run pytest tests/test_evals_spec.py -q
```

Expected: FAIL because `mini_agent.evals` does not exist.

- [ ] **Step 3: Implement contracts**

Use frozen-lightweight dataclasses where practical. `EvalTask.expected_status` defaults to `"completed"`, `EvalTask.scorers` defaults to all deterministic scorer IDs, and `EvalRunReport.pass_rate` is derived from result count.

- [ ] **Step 4: Run contract tests**

Run:

```bash
uv run pytest tests/test_evals_spec.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mini_agent/evals/__init__.py mini_agent/evals/spec.py tests/test_evals_spec.py
git commit -m "feat: add evaluation contracts"
```

## Task 2: Deterministic Scorers

**Files:**
- Create: `mini_agent/evals/scorers.py`
- Test: `tests/test_evals_scorers.py`

- [ ] **Step 1: Write failing scorer tests**

Cover:
- output fragment matching contributes score and failure reason
- expected file fragment matching checks artifact text by relative path
- expected tool evidence checks tool message fragments
- expected terminal status checks `"completed"` versus runtime failures
- combined scoring returns pass only when every requested scorer passes

- [ ] **Step 2: Run scorer tests and verify failure**

Run:

```bash
uv run pytest tests/test_evals_scorers.py -q
```

Expected: FAIL because `mini_agent.evals.scorers` does not exist.

- [ ] **Step 3: Implement deterministic scoring**

Implement `score_task_result(task, execution)` returning an `EvalScore` with `passed`, `score`, `max_score`, `breakdown`, and `failure_reasons`.

- [ ] **Step 4: Run scorer tests**

Run:

```bash
uv run pytest tests/test_evals_scorers.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mini_agent/evals/scorers.py tests/test_evals_scorers.py
git commit -m "feat: add deterministic eval scorers"
```

## Task 3: Evaluation Runner

**Files:**
- Create: `mini_agent/evals/runner.py`
- Test: `tests/test_evals_runner.py`

- [ ] **Step 1: Write failing runner tests**

Create async candidate callables returning `EvalExecution` objects. Verify `run_eval_suite()` runs each candidate/task pair, links `agent_run_id`, aggregates tokens/cost/duration, and isolates candidate failures into failed `EvalResult` records.

- [ ] **Step 2: Run runner tests and verify failure**

Run:

```bash
uv run pytest tests/test_evals_runner.py -q
```

Expected: FAIL because runner does not exist.

- [ ] **Step 3: Implement runner**

Implement sequential async orchestration first. Keep callable signature explicit: `async def candidate(task: EvalTask) -> EvalExecution`.

- [ ] **Step 4: Run runner tests**

Run:

```bash
uv run pytest tests/test_evals_runner.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mini_agent/evals/runner.py tests/test_evals_runner.py
git commit -m "feat: add evaluation suite runner"
```

## Task 4: Markdown Reporting

**Files:**
- Create: `mini_agent/evals/reporting.py`
- Test: `tests/test_evals_reporting.py`

- [ ] **Step 1: Write failing reporting tests**

Build a report with two candidates and assert the Markdown includes suite name/version, aggregate pass rate, model comparison table, task rows, score values, cost, duration, and trace run IDs.

- [ ] **Step 2: Run reporting tests and verify failure**

Run:

```bash
uv run pytest tests/test_evals_reporting.py -q
```

Expected: FAIL because reporting does not exist.

- [ ] **Step 3: Implement Markdown renderer**

Implement `format_eval_report(report)` with deterministic ordering matching runner result order.

- [ ] **Step 4: Run reporting tests**

Run:

```bash
uv run pytest tests/test_evals_reporting.py -q
```

Expected: PASS.

- [ ] **Step 5: Run full eval suite**

Run:

```bash
uv run pytest tests/test_evals_spec.py tests/test_evals_scorers.py tests/test_evals_runner.py tests/test_evals_reporting.py -q
python -m py_compile mini_agent/evals/__init__.py mini_agent/evals/spec.py mini_agent/evals/scorers.py mini_agent/evals/runner.py mini_agent/evals/reporting.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mini_agent/evals tests/test_evals_spec.py tests/test_evals_scorers.py tests/test_evals_runner.py tests/test_evals_reporting.py
git commit -m "feat: render evaluation reports"
```

## Final Verification

Run:

```bash
uv run pytest -q
git diff --check
git status --short
```

Expected: all tests pass, no whitespace errors, clean worktree after commits.
