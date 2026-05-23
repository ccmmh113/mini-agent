# Real Multi-Model Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow real GPT, DeepSeek, and Claude candidate configs to run the same benchmark suite and persist comparable `EvalRunReport` results linked to traces.

**Architecture:** Extend `benchmarks.agent_benchmark` with real eval candidate contracts and a `run_real_eval_benchmark()` orchestration path. Reuse existing real benchmark cases and `run_real_case()`, adding trace recorder and candidate workspace isolation. Extend CLI `eval run` with `--real` and repeated `--candidate name=path` arguments.

**Tech Stack:** Python 3.12, existing `Config`, `LLMClient`, SQLite eval/trace stores, argparse, pytest/pytest-asyncio.

---

## Task 1: Real Eval Candidate API

**Files:**
- Modify: `benchmarks/agent_benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] Write failing tests for `load_real_eval_candidates()` and `run_real_eval_benchmark()` with fake runner.
- [ ] Implement `RealEvalCandidate`, candidate spec parsing, real suite conversion, and fake-runner-friendly orchestration.
- [ ] Run benchmark tests and commit.

## Task 2: Real Case Trace Integration

**Files:**
- Modify: `benchmarks/agent_benchmark.py`
- Test: `tests/test_benchmark.py`

- [ ] Write failing test that real eval persistence stores eval rows for multiple candidates.
- [ ] Add trace recorder and candidate workspace isolation to `run_real_case()`.
- [ ] Persist real eval reports through `EvalSQLiteStore`.
- [ ] Run benchmark tests and commit.

## Task 3: CLI Real Candidate Flags

**Files:**
- Modify: `mini_agent/cli.py`
- Modify: `tests/test_cli_eval.py`

- [ ] Write failing parse tests for `mini-agent eval run --real --candidate gpt=...`.
- [ ] Add `--real`, repeated `--candidate`, and `--output-root` arguments.
- [ ] Route real eval runs to `run_real_eval_benchmark()`.
- [ ] Run CLI tests and commit.

## Final Verification

Run:

```bash
uv run pytest -q
python -m py_compile benchmarks/agent_benchmark.py mini_agent/cli.py
git diff --check
git status --short
```

Expected: all tests pass, compilation passes, whitespace check passes, worktree clean after commits.
