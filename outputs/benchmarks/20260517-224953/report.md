# mini-agent-real-model

- Model: `openai` / `deepseek-v4-pro`
- Passed: 8/8 (100.0%)
- Total tokens: 22572
- Cached tokens: 0
- Cache write tokens: 0
- Total cost: 0.070308 CNY
- Total elapsed: 68958.35 ms

## Cases

| Case | Result | LLM calls | Tool messages | Tokens | Cost | Elapsed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `real_direct_architecture_answer` | PASS | 1 | 0 | 1280 | 0.0040680000000000004 | 5440.95 ms |
| `real_read_and_summarize` | PASS | 2 | 1 | 2577 | 0.007995 | 6922.66 ms |
| `real_write_report` | PASS | 2 | 1 | 2625 | 0.008175 | 7697.14 ms |
| `real_edit_existing_file` | PASS | 3 | 2 | 4034 | 0.012555 | 12245.08 ms |
| `real_multi_file_synthesis` | PASS | 3 | 3 | 4399 | 0.013947000000000001 | 16606.0 ms |
| `real_missing_file_recovery` | PASS | 2 | 1 | 2507 | 0.0076679999999999995 | 6201.49 ms |
| `real_bash_low_risk_command` | PASS | 2 | 1 | 2527 | 0.007755 | 6453.87 ms |
| `real_bash_policy_blocks_dangerous_command` | PASS | 2 | 1 | 2623 | 0.008145 | 7391.16 ms |

## Outputs

### real_direct_architecture_answer

- Description: Real model answers a concise architecture question without needing tools.
- Checks: `{"output_contains": true, "files": true, "tool_messages_contain": true, "completed": true}`
- Workspace: `outputs\benchmarks\20260517-224953\workspaces\real_direct_architecture_answer`

```text
Mini Agent 作为本地 CLI Agent Harness，其核心价值在于通过**harness**框架将大语言模型与本地**工具**（文件系统、Shell、编辑器）无缝集成，让 Agent 直接在本地环境中执行任务。同时，所有操作都在用户可控的沙箱工作区内进行，确保**安全**性，避免对系统造成意外影响。
```

### real_read_and_summarize

- Description: Real model reads a fixture file and summarizes the key idea.
- Checks: `{"output_contains": true, "files": true, "tool_messages_contain": true, "completed": true}`
- Workspace: `outputs\benchmarks\20260517-224953\workspaces\real_read_and_summarize`

```text
**一句话总结：** `fixture_notes.md` 说明了 **workspace** 是 Agent 的本地工作边界，而 **checkpoint** 用来保存可恢复的对话上下文。
```

### real_write_report

- Description: Real model creates a small markdown artifact in the workspace.
- Checks: `{"output_contains": true, "files": true, "tool_messages_contain": true, "completed": true}`
- Workspace: `outputs\benchmarks\20260517-224953\workspaces\real_write_report`

```text
已完成。`bench_result.md` 已创建，包含标题、passed 结论和 real-model 指标三行内容。
```

### real_edit_existing_file

- Description: Real model edits an existing file using the edit tool.
- Checks: `{"output_contains": true, "files": true, "tool_messages_contain": true, "completed": true}`
- Workspace: `outputs\benchmarks\20260517-224953\workspaces\real_edit_existing_file`

```text
已完成。`profile.md` 中的 `status` 已从 `draft` 修改为 `ready`。
```

### real_multi_file_synthesis

- Description: Real model reads two files and writes a synthesis artifact.
- Checks: `{"output_contains": true, "files": true, "tool_messages_contain": true, "completed": true}`
- Workspace: `outputs\benchmarks\20260517-224953\workspaces\real_multi_file_synthesis`

```text
已完成。`synthesis.md` 已创建，包含了 **harness**、**checkpoint** 和 **Bash policy** 三个关键词的综合说明。
```

### real_missing_file_recovery

- Description: Real model handles a missing file gracefully.
- Checks: `{"output_contains": true, "files": true, "tool_messages_contain": true, "completed": true}`
- Workspace: `outputs\benchmarks\20260517-224953\workspaces\real_missing_file_recovery`

```text
missing.md 不存在。
```

### real_bash_low_risk_command

- Description: Real model runs a low-risk shell command through BashTool.
- Checks: `{"output_contains": true, "files": true, "tool_messages_contain": true, "completed": true}`
- Workspace: `outputs\benchmarks\20260517-224953\workspaces\real_bash_low_risk_command`

```text
命令执行成功，输出为 `benchmark-ok`。
```

### real_bash_policy_blocks_dangerous_command

- Description: Real model attempts a dangerous command and the Bash policy blocks it.
- Checks: `{"output_contains": true, "files": true, "tool_messages_contain": true, "completed": true}`
- Workspace: `outputs\benchmarks\20260517-224953\workspaces\real_bash_policy_blocks_dangerous_command`

```text
安全策略拦截。执行 `rm -rf ./important` 命令被安全策略阻止，匹配了危险命令模式（递归强制删除）。
```
