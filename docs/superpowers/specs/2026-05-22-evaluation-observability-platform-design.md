# Evaluation And Observability Platform Design

## Summary

Mini Agent will evolve from a CLI Agent harness with benchmark scripts and text logs into a local evaluation and observability platform that can later be serviceized. The first release keeps the existing CLI and benchmark workflows, standardizes trace data emitted by Agent execution, persists evaluation and run records locally, and adds a local dashboard for model comparison and single-run replay.

The platform closes two loops:

- Evaluation: define task suites, run the same Agent tasks against multiple model configurations, score the results, and compare quality, cost, latency, and runtime behavior.
- Observability: capture structured traces for each Agent run, aggregate metrics, and replay the LLM/tool execution chain behind a benchmark result or failure.

## Goals

- Provide an automated Agent evaluation framework for task suites and multi-model comparison.
- Make Agent runtime behavior observable through structured events, metrics, and run-level trace replay.
- Reuse the current `Agent`, `ToolRuntime`, benchmark runners, logger, token accounting, and workspace artifact behavior.
- Deliver a local demonstrable version first while keeping storage and event boundaries suitable for a later backend service.
- Keep benchmark results and runtime traces connected through stable run identifiers.

## Non-Goals

- Do not build distributed scheduling in the first release.
- Do not add multi-user permissions, remote tenancy, or online alerting.
- Do not require Prometheus, Grafana, or external telemetry infrastructure.
- Do not make LLM-as-judge the primary first-release scorer.
- Do not replace the current CLI Agent workflow with a service-only workflow.

## Current State

Mini Agent already has useful foundations for both evaluation and observability:

- `benchmarks/agent_benchmark.py` has deterministic harness benchmarks and a real-model benchmark mode.
- `benchmarks/module_benchmark.py` measures compression, memory, and checkpoint modules.
- Benchmark reports already expose pass rates, elapsed time, token usage, and cost in some real-model reports.
- `Agent` accumulates API-reported token usage and estimated cost.
- `AgentLogger` records LLM requests, LLM responses, and tool results as text log sections.
- `ToolRuntime`, policies, and observers already provide a boundary for tool execution, safety decisions, side effects, and workspace diffs.

These foundations are fragmented:

- Benchmark output is report-oriented rather than a reusable evaluation domain model.
- Runtime logs are useful for debugging but are not query-friendly trace data.
- CLI stats, text logs, benchmark reports, tool audit events, and workspace artifacts are not unified by a platform data contract.
- A failed benchmark result cannot yet link directly to a structured timeline of the underlying Agent run.

## Recommended Approach

The first implementation should use a trace-centered platform shape:

```text
Task Suite + Model Configs
          |
          v
   Evaluation Runtime
          |
          v
     Agent Runtime ---> Trace Events ---> Trace Store
          |                                |
          +-------- Scores/Metrics --------+
                                           |
                                           v
                                   Local Dashboard
```

This approach keeps evaluation and observability mutually reinforcing:

- Evaluation creates repeatable workloads and model comparisons.
- Structured traces explain the behavior behind each metric and score.
- The dashboard uses the same persisted run model for overview charts, evaluation drill-down, and execution replay.

## Architecture

### Agent Runtime

The existing runtime remains the execution engine. It gains structured observability hooks at the main boundaries:

- Run lifecycle in `mini_agent/agent.py`.
- Step and LLM-call lifecycle in `mini_agent/agent.py`.
- Tool-call lifecycle, policy outcomes, and workspace effects in `mini_agent/runtime.py`.
- Token and cost accounting reuse from `mini_agent/token_accounting.py`.

Observability must be additive. Trace recording failures should not fail the Agent task unless the caller explicitly configures strict telemetry behavior later.

### Evaluation Runtime

Evaluation code should move from one-off benchmark scripts toward reusable modules under `mini_agent/evals/`:

- `spec.py`: task suite, task case, model selection, fixture, and scorer definitions.
- `runner.py`: batch orchestration over tasks and model configurations.
- `scorers.py`: deterministic rule scorers for first-release checks.
- `reporting.py`: aggregate metrics and report rendering.

The existing `benchmarks/` directory remains useful as:

- runnable demo suites
- regression fixtures
- examples of how to invoke the reusable evaluation runtime

Migration should be incremental. Existing benchmark scripts do not need to be deleted in the first implementation.

### Trace Store

The local release should use a replaceable storage boundary:

- SQLite for queryable evaluation, run, step, LLM call, tool call, event, score, and metric records.
- JSON or Markdown artifacts for exported reports, raw event payload snapshots when needed, and workspace artifact references.

The storage interface should keep dashboard and evaluation code independent from the concrete SQLite implementation. A later service can replace SQLite with Postgres plus object storage without forcing a redesign of the runtime event contract.

### Local Dashboard

The first dashboard should read the local store and expose three primary pages:

- Overview: recent evaluations, recent Agent runs, quality/cost/latency summaries, and trend entry points.
- Eval Detail: suite metadata, model comparison, task-level results, scores, and failure reasons.
- Run Trace: run metadata, timeline replay, LLM calls, tool calls, policy blocks, errors, metrics, and related artifacts.

The dashboard should be useful for inspection rather than become a benchmark configuration admin in the first release.

## Data Model

### Evaluation Records

The evaluation layer should distinguish a reusable suite definition from a concrete batch execution:

- `eval_suite`: suite name, version, task-set metadata, and source reference.
- `eval_task`: task prompt, fixture declaration, expected output or artifact conditions, and scorer declaration.
- `eval_run`: one batch execution over a suite and one or more model configurations.
- `eval_result`: one model/task result linked to the underlying `agent_run_id`, scores, pass state, and failure reason.

### Runtime Records

The execution layer should retain both normalized call records and a replayable event stream:

- `agent_run`: run identifier, model, workspace, status, start/end timestamps, duration, total tokens, total cost, and terminal reason.
- `agent_step`: step index, request sizing metadata, state transition, and stop reason.
- `llm_call`: call timing, request summary, finish reason, usage, cost, error status, and model metadata.
- `tool_call`: tool name, argument summary, success state, duration, error summary, affected files, and policy outcome where applicable.
- `run_event`: ordered event stream for trace replay and future streaming use cases.

The event stream should include at least:

- `run_started`
- `step_started`
- `llm_started`
- `llm_completed`
- `llm_failed`
- `tool_started`
- `tool_completed`
- `tool_failed`
- `tool_blocked`
- `run_completed`
- `run_failed`
- `run_cancelled`
- `run_max_steps`

### Metrics

The first release should persist and aggregate:

- task pass rate
- model success rate
- score distribution
- run duration and percentile-ready latency samples
- LLM call count
- tool call count
- tool failure rate
- safety policy block count
- max-step termination count
- prompt, completion, cached, cache-write, and total tokens
- estimated cost

Dashboard pages should query structured records rather than parse text logs. Existing text logs remain valuable for audits and low-level debugging.

## Evaluation Scope

The first release should support task specs that describe:

- task prompt
- workspace fixtures
- expected output fragments
- expected files or expected file fragments
- expected tool-result fragments when needed
- expected completion state
- selected scoring rules

Initial scorers should be deterministic:

- output contains
- file artifact exists or contains expected fragments
- tool-chain observation contains expected evidence
- completion-state check
- aggregation into pass/fail and score breakdown

Real-model suites can reuse the same result shape as deterministic suites. Deterministic suites remain the regression baseline for harness behavior.

## Observability Scope

Each observable Agent run should provide:

- stable run identifier
- step timeline
- LLM latency and usage
- estimated cost when pricing is configured
- tool latency and result state
- policy block and confirmation-denial visibility
- unknown-tool, max-step, cancellation, and runtime failure visibility
- workspace artifact or affected-file references when available

Sensitive data handling must preserve current redaction expectations. Structured event payloads should store summaries or redacted payloads by default, with raw request/response handling explicitly designed before any future expansion.

## Module Placement

### Observability

```text
mini_agent/observability/
  events.py
  recorder.py
  store.py
  sqlite_store.py
```

### Evaluation

```text
mini_agent/evals/
  spec.py
  runner.py
  scorers.py
  reporting.py
```

### Dashboard

```text
mini_agent/dashboard/
  app.py
  queries.py
  templates/ or static assets
```

### CLI Surface

The CLI should grow a small platform-oriented surface:

- `mini-agent eval run ...`
- `mini-agent eval report ...`
- `mini-agent dashboard`

The first release should not promise a separate `trace show` CLI because the Run Trace dashboard page is the intended replay surface. Implementation planning can decide how these commands fit the current parser while preserving the three user-visible capabilities above.

## Failure Handling

- Agent execution should continue when telemetry persistence fails in best-effort mode.
- A run must end in an explicit terminal status such as `completed`, `failed`, `max_steps`, or `cancelled`.
- Evaluation results must preserve scoring failures separately from runtime failures.
- Dashboard queries should tolerate partial traces from interrupted processes.
- Store writes should be batched or scoped so a single malformed event does not corrupt the entire local database.

## Testing Strategy

- Unit-test event creation, status transitions, redaction boundaries, and SQLite persistence.
- Add runtime tests for LLM-call and tool-call trace emission around existing Agent and ToolRuntime behavior.
- Add evaluation integration tests using deterministic scripted LLM cases.
- Keep benchmark regression tests for the current harness behavior.
- Add dashboard query tests against a temporary SQLite store seeded with representative eval and run records.
- Add smoke coverage for report generation and evaluation-to-trace linkage.

## Delivery Phases

### Phase 1: Trace Foundation

- Define event and run data contracts.
- Implement local recorder/store boundary.
- Instrument Agent and tool runtime lifecycle.
- Persist run-level, LLM-call, tool-call, and event data.

### Phase 2: Reusable Evaluation Runtime

- Define task suite and scorer specs.
- Migrate enough benchmark behavior into reusable eval modules.
- Support multi-model batch execution and persisted eval results.
- Produce structured summaries and report exports.

### Phase 3: Local Dashboard

- Add store-backed overview, evaluation detail, and run trace pages.
- Link evaluation failures to single-run trace replay.
- Validate the dashboard against deterministic and real-model report data.

## Project Narrative

After this work, the project can credibly demonstrate:

1. Mini Agent provides an automated evaluation framework that compares Agent capability across model configurations with repeatable task suites, scoring rules, reports, and historical results.
2. Mini Agent provides performance monitoring and end-to-end observability through structured runtime traces, metrics, local persistence, and run replay from evaluation outcome down to LLM and tool execution.
