# Context Governance Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real context-governance evaluation suite that can verify compression evidence, current-instruction retention, and stale-context rejection.

**Architecture:** Keep YAML loading generic by storing fixtures and overrides in `EvalTask.metadata`. Convert those metadata fields inside the real benchmark runner, record context evidence in `EvalExecution.metadata`, and add opt-in scorer rules for metadata and forbidden fragments.

**Tech Stack:** Python dataclasses, YAML eval suites, pytest, Mini Agent real benchmark runtime.

---

### Task 1: Metadata Scoring Rules

**Files:**
- Modify: `mini_agent/evals/scorers.py`
- Test: `tests/test_evals_scorers.py`

- [x] **Step 1: Write failing tests for `metadata_contains`**

Add tests that expect nested metadata paths such as `context_governance.compression_triggered` and list membership checks such as `context_governance.compression_markers`.

- [x] **Step 2: Implement `metadata_contains`**

Add a scorer branch that reads `task.metadata["expected_metadata_contains"]`, resolves dot paths in `execution.metadata`, and compares booleans/numbers exactly while matching strings as fragments.

- [x] **Step 3: Add stale-context exclusion scorers**

Add `output_excludes` for `metadata.expected_output_not_contains` and `file_excludes` for `metadata.expected_files_not_contains`.

- [x] **Step 4: Verify scorer tests**

Run: `uv run pytest tests\test_evals_scorers.py -q`

Expected: all scorer tests pass.

### Task 2: Real Benchmark Suite Metadata

**Files:**
- Modify: `benchmarks/agent_benchmark.py`
- Test: `tests/test_benchmark.py`

- [x] **Step 1: Write failing test for fixtures and token override**

Add a custom `EvalSuite` task with `metadata.fixtures`, `metadata.agent_overrides.token_limit`, and `metadata.expected_metadata_contains`.

- [x] **Step 2: Convert suite metadata into runnable cases**

Map `metadata.fixtures` into `RealBenchmarkCase.files` and `metadata.agent_overrides.token_limit` into `RealBenchmarkCase.token_limit`.

- [x] **Step 3: Record context governance metadata**

After `run_real_case`, inspect Agent messages for context snip, context collapse, harness summary, tool result spill, and micro compact markers.

- [x] **Step 4: Preserve runner-provided metadata**

Merge `result["metadata"]` into `EvalExecution.metadata` before adding legacy fields.

### Task 3: Context Governance Suite

**Files:**
- Create: `eval_suites/context_governance_suite.yaml`
- Create: `eval_suites/comprehensive_agent_suite.yaml`

- [x] **Step 1: Add broad comprehensive suite**

Create a general suite for tool, artifact, shell, security, missing-file, and structure smoke coverage.

- [x] **Step 2: Add dedicated context governance suite**

Create five pressure cases covering stale instruction override, needle retention, evidence re-grounding, multi-step state integrity, and compression boundary awareness.

- [x] **Step 3: Validate YAML loading**

Run: `uv run python -c "from mini_agent.evals import load_eval_suite_yaml; s=load_eval_suite_yaml('eval_suites/context_governance_suite.yaml'); print(s.suite_key); print(len(s.tasks))"`

Expected: `mini-agent-context-governance@v1` and `5`.

### Task 4: Final Verification

**Files:**
- All modified files

- [x] **Step 1: Run focused tests**

Run: `uv run pytest tests\test_evals_scorers.py tests\test_benchmark.py tests\test_cli_eval.py tests\test_evals_suite_loader.py -q`

Expected: all focused tests pass.

- [x] **Step 2: Run full test suite**

Run: `uv run pytest -q`

Expected: all tests pass except expected skips.

- [x] **Step 3: Commit**

Commit the eval suite and runtime/scorer changes.
