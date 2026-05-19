# Mini Agent Benchmark

这个目录用于评估 Mini Agent 的 harness 能力，而不是评估某个真实模型的智力。

当前 benchmark 使用确定性的 `ScriptedLLM`，所以每次结果应该稳定一致，适合面试展示、回归测试和架构改动后的验证。

## 运行

```bash
uv run python benchmarks/agent_benchmark.py
```

输出 JSON 报告：

```bash
uv run python benchmarks/agent_benchmark.py --json
```

写入文件：

```bash
uv run python benchmarks/agent_benchmark.py --output outputs/benchmark-report.json
```

模块有效性 benchmark：

```bash
uv run python benchmarks/module_benchmark.py
```

输出 JSON：

```bash
uv run python benchmarks/module_benchmark.py --json
```

只跑某个模块：

```bash
uv run python benchmarks/module_benchmark.py --module compression
uv run python benchmarks/module_benchmark.py --module memory
uv run python benchmarks/module_benchmark.py --module checkpoint
```

写入文件：

```bash
uv run python benchmarks/module_benchmark.py --output outputs/module-benchmark.json
```

写入 Markdown 报告：

```bash
uv run python benchmarks/module_benchmark.py --markdown outputs/module-benchmark.md
```

同时写入 JSON 和 Markdown：

```bash
uv run python benchmarks/module_benchmark.py \
  --output outputs/module-benchmark.json \
  --markdown outputs/module-benchmark.md
```

## 接入真实大模型

真实模型 benchmark 会读取项目的 `config.yaml`，使用里面的 `provider`、`api_base`、`model`、`api_key`、`token_pricing`。

```bash
uv run python -m benchmarks.agent_benchmark --real
```

默认会输出：

```text
outputs/benchmarks/<timestamp>/report.json
outputs/benchmarks/<timestamp>/report.md
outputs/benchmarks/<timestamp>/workspaces/
```

也可以指定报告路径：

```bash
uv run python -m benchmarks.agent_benchmark --real \
  --output outputs/benchmarks/real-report.json \
  --markdown outputs/benchmarks/real-report.md
```

真实模型模式当前覆盖：

- 直接架构回答
- 读取 fixture 文件并总结
- 写入 Markdown 报告文件
- 修改已有文件
- 多文件读取后综合写入
- 缺失文件恢复
- 低风险 Bash 命令
- 危险 Bash 命令安全拦截

真实模型结果会受模型版本、温度、网络、API 返回 token usage 影响，所以适合看趋势和成本，不适合作为完全确定的单元测试。

## 当前覆盖点

`agent_benchmark.py` 覆盖完整 Agent harness 流程：

- `direct_answer`：不调用工具直接完成。
- `read_file`：验证工具调用和工具结果链路。
- `write_file`：验证 workspace 内文件写入。
- `bash_policy_blocks_dangerous_command`：验证 Bash 安全策略会拦截危险命令。
- `max_steps_guard`：验证模型一直请求工具时，Agent 能通过 `max_steps` 停止。
- `prompt_compression_keeps_current_task`：验证长历史会压缩，当前任务仍被保留。
- `unknown_tool_is_reported_cleanly`：验证未知工具不会导致 Agent 崩溃。
- `checkpoint_saved_during_run`：验证完成运行后会保存 checkpoint。
- `subagent_disabled_by_default`：验证 subagent 默认关闭时 `task` 工具不可用。
- `task_memory_records_completion`：验证任务完成后会记录 task memory 和 episode memory。

`module_benchmark.py` 覆盖内部模块有效性：

- `compression/tool_result_budget`：验证大工具结果会落盘、保留 `tool_call_id` 和可恢复路径，并统计节省 token。
- `compression/snip`：验证旧消息会被直接移除、插入 snip 边界、保留当前用户问题，并统计释放 token。
- `compression/micro_compact`：验证可恢复工具结果会按时间衰减裁剪，插入 `Old tool result content shortened` 标记。
- `compression/context_collapse`：验证读时投影不修改原始历史，插入 collapse 标记，并在阈值内避免 Auto-Compact。
- `compression/auto_compact_fallback`：验证前几层仍不足时会调用 Auto-Compact 兜底。
- `memory/index_and_recall`：验证热索引存在、冷记忆可按需召回，并能用记忆回答构造问题。
- `memory/secret_redaction`：验证写入记忆前会脱敏，原始 secret 不落盘。
- `memory/stale_memory_guard`：验证记忆提示包含过期警告，避免把旧文件事实当成当前事实。
- `checkpoint/save_and_validate`：验证 `latest.json` 和历史 checkpoint 可写、可解析、reason 正确。
- `checkpoint/restore_messages`：验证消息可恢复，workspace 校验通过。
- `checkpoint/resume_continues_task`：验证恢复后的消息能继续追加后续任务结果。

## 指标

`agent_benchmark.py` 报告会包含：

- 通过率
- 每个 case 的耗时
- LLM 调用次数
- tool message 数量
- message count
- prompt / completion / total / cached token

`module_benchmark.py` 报告会包含：

- 每个模块的 `case_count`、`failed`、`pass_rate`
- 压缩模块的 `before_tokens`、`after_tokens`、`tokens_saved`、`compression_ratio`
- 每层专属指标，例如 `tool_results_spilled`、`snipped_messages`、`micro_compacted_results`、`collapsed_messages`、`auto_compact_called`
- 记忆模块的 `memory_index_loaded`、`topic_memory_loaded_on_demand`、`secret_redacted_before_write`、`stale_memory_blind_trust`
- checkpoint 模块的 `checkpoint_created`、`latest_checkpoint_valid_json`、`messages_restored`、`resume_continues_task`

Markdown 报告包含同样的模块和 case 指标，但用表格展示，适合直接放进评审记录或面试材料。

> 我把 benchmark 分成两层。第一层是确定性 harness benchmark，用 Fake LLM 验证工具执行、安全策略、状态流转和 token 统计是否稳定。第二层以后可以接真实模型，评估任务完成率和成本。这样可以把模型能力和 Agent 工程能力分开看。
