# Eval Store And CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist standard evaluation reports to SQLite and expose deterministic eval run/report commands through the CLI.

**Architecture:** Add `mini_agent.evals.sqlite_store` for eval tables and report reconstruction. Extend deterministic benchmark execution to optionally save eval reports alongside trace records in the same database. Add `mini-agent eval run/report` handlers that call the benchmark and store/report APIs.

**Tech Stack:** Python 3.12, built-in `sqlite3`, dataclasses, argparse, pytest/pytest-asyncio.

---

## Task 1: Eval SQLite Store

**Files:**
- Create: `mini_agent/evals/sqlite_store.py`
- Modify: `mini_agent/evals/__init__.py`
- Test: `tests/test_evals_sqlite_store.py`

- [ ] Write failing tests for saving and loading an `EvalRunReport`.
- [ ] Verify tests fail because `EvalSQLiteStore` does not exist.
- [ ] Implement schema, `save_report()`, `load_report()`, and `load_latest_report()`.
- [ ] Export `EvalSQLiteStore`.
- [ ] Run store tests and commit.

## Task 2: Benchmark Persistence

**Files:**
- Modify: `benchmarks/agent_benchmark.py`
- Modify: `tests/test_benchmark.py`

- [ ] Write failing test that `run_eval_benchmark(db_path=...)` writes eval tables and trace rows into the same database.
- [ ] Verify test fails because eval tables are not written.
- [ ] Add optional `db_path`/`eval_db_path` persistence to `run_eval_benchmark()`.
- [ ] Keep `run_benchmark()` dict compatibility.
- [ ] Run benchmark tests and commit.

## Task 3: CLI Eval Commands

**Files:**
- Modify: `mini_agent/cli.py`
- Create: `tests/test_cli_eval.py`

- [ ] Write failing tests for parsing `eval run/report` and for command handlers.
- [ ] Verify tests fail because the commands do not exist.
- [ ] Add `eval` subparsers with `run` and `report`.
- [ ] Implement handler functions that run deterministic benchmark and print Markdown report.
- [ ] Run CLI tests and commit.

## Final Verification

Run:

```bash
uv run pytest -q
python -m py_compile mini_agent/evals/sqlite_store.py benchmarks/agent_benchmark.py mini_agent/cli.py
git diff --check
git status --short
```

Expected: all tests pass, compilation passes, whitespace check passes, worktree clean after commits.
