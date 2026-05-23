# Real Multi-Model Eval Design

## Summary

Add real-model evaluation across multiple named model candidates. Users can compare GPT, DeepSeek, and Claude style configurations by passing existing Mini Agent `config.yaml` files to the eval CLI.

## User Interface

```bash
mini-agent eval run \
  --real \
  --db evals.sqlite3 \
  --candidate gpt=./configs/gpt.yaml \
  --candidate deepseek=./configs/deepseek.yaml \
  --candidate claude=./configs/claude.yaml
```

Each config file uses the existing configuration format. GPT and DeepSeek should use `provider: openai` with their own `api_base`; Claude should use `provider: anthropic`.

## Architecture

`benchmarks.agent_benchmark` will expose a real eval API:

- `RealEvalCandidate`: named candidate with a loaded `Config`.
- `load_real_eval_candidates()`: parse `name=path` specs into candidates.
- `run_real_eval_benchmark()`: run every real benchmark case for every candidate and return `EvalRunReport`.

The function will accept an injectable case runner for tests. The default case runner will call the current `run_real_case()` and pass a trace recorder so real eval results link to SQLite trace rows.

## Persistence

When `db_path` is provided, `run_real_eval_benchmark()` writes both trace records and eval records into the same SQLite file. `eval_results.agent_run_id` remains the link from model/task result to the underlying trace.

## Scope

This change does not add dashboard views or parallel scheduling. Execution remains sequential and local.

## Testing

Tests will use fake real-case runners instead of calling external LLM APIs. They will verify:

- candidate specs parse `gpt=path`, `deepseek=path`, and `claude=path`
- multi-candidate real eval creates one result per candidate/task
- saved eval reports preserve candidate IDs, model names, and agent run IDs
- CLI parses `--real --candidate name=path`
