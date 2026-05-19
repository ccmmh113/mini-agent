# Mini Agent Module Benchmark Report

## Summary

- Benchmark: `module`
- Modules: 3
- Cases: 11
- Failed: 0
- Pass rate: 100%
- Duration: 13.4101s

| Module | Cases | Failed | Pass Rate | Tokens Saved |
| --- | ---: | ---: | ---: | ---: |
| compression | 5 | 0 | 100% | 65158 |
| memory | 3 | 0 | 100% | 0 |
| checkpoint | 3 | 0 | 100% | 0 |

## Compression

| Case | Status | Key Metrics |
| --- | --- | --- |
| tool_result_budget | PASS | before_tokens=50141, after_tokens=779, tokens_saved=49362, compression_ratio=0.9845, tool_results_spilled=3 |
| snip | PASS | before_tokens=7853, after_tokens=1164, tokens_saved=6689, compression_ratio=0.8518, snipped_messages=24 |
| micro_compact | PASS | before_tokens=14851, after_tokens=6019, tokens_saved=8832, compression_ratio=0.5947, micro_compacted_results=8 |
| context_collapse | PASS | before_tokens=4547, after_tokens=4272, tokens_saved=275, compression_ratio=0.0605, collapsed_messages=2 |
| auto_compact_fallback | PASS | before_tokens=4014, after_tokens=4020, tokens_saved=0, compression_ratio=0.0, auto_compact_called=True |

## Memory

| Case | Status | Key Metrics |
| --- | --- | --- |
| index_and_recall | PASS | memory_index_loaded=True, topic_memory_loaded_on_demand=True, memory_helped_answer=True, prompt_memory_tokens=79 |
| secret_redaction | PASS | secret_redacted_before_write=True, raw_secret_absent_from_files=True |
| stale_memory_guard | PASS | stale_memory_blind_trust=False |

## Checkpoint

| Case | Status | Key Metrics |
| --- | --- | --- |
| save_and_validate | PASS | checkpoint_created=True, latest_checkpoint_valid_json=True, checkpoint_reason_correct=True, history_files=1 |
| restore_messages | PASS | messages_restored=True, workspace_validation_passed=True |
| resume_continues_task | PASS | messages_restored=True, resume_continues_task=True |
