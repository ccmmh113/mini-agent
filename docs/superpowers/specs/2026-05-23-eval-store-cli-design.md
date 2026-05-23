# Eval Store And CLI Design

## Summary

Mini Agent now has structured trace persistence and a reusable evaluation runtime. This change adds the missing local product loop: persist evaluation reports to SQLite and expose deterministic benchmark evaluation through CLI commands.

## Scope

- Add SQLite persistence for `EvalRunReport`.
- Store enough suite, candidate, result, score, and score-breakdown data to reconstruct the latest report.
- Allow deterministic benchmark runs to write trace records and eval records into the same SQLite database.
- Add CLI commands:
  - `mini-agent eval run --db path/to/evals.sqlite3`
  - `mini-agent eval report --db path/to/evals.sqlite3`

The first implementation only runs the deterministic harness benchmark. Real-model multi-candidate evaluation and dashboard pages remain follow-up work.

## Architecture

`mini_agent.evals.sqlite_store` will own eval-specific SQLite tables. It will not modify the existing trace schema; instead, `eval_results.agent_run_id` references the trace layer's `agent_runs.run_id` by value so callers can join them in one database file.

`benchmarks.agent_benchmark.run_eval_benchmark()` will gain optional eval persistence. When passed one database path for both trace and eval storage, the same file will contain trace tables and eval tables.

`mini_agent.cli` will add a small `eval` subcommand group. The CLI handler will call benchmark APIs and store/report helpers rather than duplicating evaluation logic.

## Data Model

- `eval_runs`
  - `eval_run_id`
  - suite id/name/version
  - serialized suite tasks
  - serialized candidates
  - aggregate counts, tokens, cost, duration
  - created timestamp
- `eval_results`
  - one row per candidate/task result
  - includes `agent_run_id`, pass state, score, status, tokens, cost, duration, failure reason, output, metadata, and score failure reasons
- `eval_score_breakdowns`
  - one row per scorer result for a candidate/task

## CLI Behavior

`mini-agent eval run --db evals.sqlite3` runs the deterministic benchmark, writes trace and eval data into that database, and prints a compact summary including eval run id, pass rate, and database path.

`mini-agent eval report --db evals.sqlite3` loads the latest eval run from the database and prints the existing Markdown report format.

If no eval run exists, `eval report` exits normally with a clear message.

## Testing

- Store tests verify save/load round-trip, latest-run selection, score breakdown rows, and `agent_run_id` preservation.
- Benchmark tests verify running with a shared DB creates both eval rows and trace rows linked by `agent_run_id`.
- CLI tests verify parser support and command handlers for `eval run` / `eval report`.
