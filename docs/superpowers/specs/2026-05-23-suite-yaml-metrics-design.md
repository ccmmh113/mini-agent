# Suite YAML And Eval Metrics Design

## Summary

Add user-configurable evaluation suites and richer aggregate metrics. This turns the current built-in benchmark set into an extensible eval workflow where users can define task prompts and deterministic expectations in YAML, then compare GPT, DeepSeek, Claude, or future candidates with the same suite.

## Suite YAML

The suite file maps directly to `EvalSuite` and `EvalTask`:

```yaml
suite_id: agent-core
name: Agent Core Capability
version: 2026-05-23
description: Core Agent task evaluation
tasks:
  - task_id: write-report
    prompt: "创建 report.md，总结项目能力"
    expected_output_contains:
      - "完成"
    expected_files:
      report.md:
        - "Agent"
        - "评测"
    expected_tool_evidence_contains:
      - "write_file"
    expected_status: completed
```

The first implementation uses YAML suites for real-model eval runs. Deterministic built-in harness tests remain available when `--suite` is omitted.

## Metrics

Add `mini_agent.evals.metrics.compute_eval_metrics(report)` and store the result in `report.metadata["metrics"]`. Metrics include:

- case count, failed count, pass rate
- total, average, p50, and p95 latency
- total and average tokens
- total and average cost
- cost per passed task
- max-step count and rate
- status failure count and rate
- tool evidence failure count and rate
- scorer failure distribution
- per-candidate summaries with pass rate, latency, token, and cost totals

These metrics are saved through the existing metadata persistence in `EvalSQLiteStore`.

## CLI

Add `--suite` to `mini-agent eval run`:

```bash
mini-agent eval run --real \
  --suite evals/agent-core.yaml \
  --db evals.sqlite3 \
  --candidate gpt=configs/gpt.yaml \
  --candidate deepseek=configs/deepseek.yaml \
  --candidate claude=configs/claude.yaml
```

If `--suite` is provided without `--real`, the command returns a clear error because arbitrary user tasks require a real candidate runner.

## Reporting

`format_eval_report()` will include a metrics section before candidate comparison. Existing candidate and task tables remain unchanged.
