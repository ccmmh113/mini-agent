# Mini Agent Harness 架构与开发说明

这份文档描述当前 CLI harness 的真实边界。它不覆盖已暂停的 ACP 集成，也不把 Skills/MCP 作为默认能力；二者只是可选扩展。

## 1. 架构目标

Mini Agent 的核心不是某个模型客户端，而是一套本地 Agent harness：

- 接收用户任务
- 管理每轮发给 LLM 的 prompt、messages 和 tools
- 把模型 tool calls 映射到本地工具
- 在工具执行前后施加 policy 与 observer
- 持久化 checkpoint、task memory、episode、长期记忆和日志
- 在上下文变长时按层压缩

推荐把项目理解成七层：

```text
CLI / Console
  -> Agent Loop
  -> Request Context / Prompt Budget
  -> LLM Boundary
  -> Tool Runtime
  -> Memory / Checkpoint / Logger
  -> Local Workspace
```

## 2. 任务进入后的完整流程

```text
mini-agent 或 mini-agent --task
  -> Config.load()
  -> LLMClient(...)
  -> ToolRegistry.build_base_tools()
  -> ToolRegistry.add_workspace_tools(...)
  -> SystemPromptBuilder(...)
  -> Agent(...)
  -> agent.add_user_message(task)
  -> agent.run()
```

`Agent.run()` 内部每个 step 的顺序：

```text
1. 取当前 tools
2. CompressionPipeline.compress_before_request(...) 测量并按需压缩上下文
3. MessageCompactor 先执行 Tool Result Budget / Snip / Micro-Compact
4. MessageSummarizer 仅在确定性压缩后仍超限时生成 harness summary
5. RequestContextBuilder.build(...) 重新生成真正请求 messages
6. llm.generate(messages=request_messages, tools=tool_list)
7. 读取 response.usage，展示本轮 token 和可选成本估算
8. 追加 assistant message
9. 如果 assistant 没有 tool_calls，结束任务
10. 如果有 tool_calls，ToolRuntime.execute(...)
11. 追加 tool result message
12. 写 checkpoint / logger / task memory
13. 进入下一 step
```

关键文件：

- [mini_agent/cli.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/cli.py)
- [mini_agent/agent.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/agent.py)
- [mini_agent/request_context.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/request_context.py)
- [mini_agent/runtime.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/runtime.py)

## 3. CLI 层边界

职责：

- 解析 `--workspace`、`--task`、`--version`
- 加载配置
- 初始化 LLM client
- 初始化工具
- 创建 workspace
- 处理交互命令
- 显示状态、统计、日志和确认提示
- 响应 Esc 取消

不应承担：

- tool call 执行细节
- prompt 拼装细节
- checkpoint 数据结构细节
- Bash 安全策略细节

相关文件：

- [mini_agent/cli.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/cli.py)
- [mini_agent/console.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/console.py)

## 4. Agent Loop 边界

职责：

- 保存当前 `messages`
- 控制最大 step
- 调用 context builder
- 调用 compression pipeline
- 调用 LLM
- 处理 assistant tool calls
- 追加 tool results
- 串联 checkpoint、logger、task memory hook

不应承担：

- 具体工具注册
- 单个工具的安全判断
- system prompt 每一层的裁剪细节
- CLI 输出格式

相关文件：

- [mini_agent/agent.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/agent.py)

## 5. Prompt / Context 边界

这一层决定“本轮到底发什么给 LLM”。

### 5.1 SystemPromptBuilder

职责：

- 读取核心 system prompt
- 注入可选 Skills metadata
- 注入长期记忆
- 注入项目规则
- 注入当前任务状态
- 注入 harness summary
- 注入动态上下文
- 对每个 system prompt 层应用独立预算

当前 system prompt 层：

```text
core
skills
memory
project_rules
current_task_context
harness_summary
dynamic_context
```

### 5.2 RequestContextBuilder

职责：

- 过滤旧 system message
- 识别 `[Harness Execution Summary]`
- 识别当前 active tool chain
- 估算 system message 和 tool schema 占用
- 在剩余预算里选择历史 messages
- 强保护当前用户请求和未完成工具链

### 5.3 Context Budget

默认总预算：

```yaml
token_limit: 10000
request_context_limit: 12
```

默认分层预算：

```yaml
context_layer_budgets:
  core: 2500
  skills: 1200
  memory: 1200
  project_rules: 1800
  current_task_context: 1000
  harness_summary: 1800
  dynamic_context: 300
```

相关文件：

- [mini_agent/prompt_builder.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/prompt_builder.py)
- [mini_agent/request_context.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/request_context.py)
- [mini_agent/context_budget.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/context_budget.py)
- [mini_agent/summarizer.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/summarizer.py)

## 6. 压缩策略

压缩分五类，其中 message history 压缩由 `CompressionPipeline` 串联确定性压缩、读时投影和语义摘要 fallback：

### 6.1 System prompt 分层裁剪

`SystemPromptBuilder` 对每层单独套预算。核心层优先保留，扩展层超预算时头尾保留并插入压缩标记。

适用对象：

- Skills metadata
- 长期记忆
- 项目规则
- 当前任务状态
- harness summary
- dynamic context

### 6.2 Message history 选择

`RequestContextBuilder` 不再简单按消息条数截断，而是先扣掉：

- system message 估算 token
- tools schema 估算 token

然后用剩余预算选择历史消息。当前活跃工具链优先级最高，即使超出历史预算也会保留，避免 tool call 与 tool result 断裂。

### 6.3 MessageCompactor 确定性压缩

`MessageCompactor` 不调用 LLM。它只做可证明不会破坏消息结构的本地转换，顺序如下：

```text
Tool Result Budget -> Snip -> Micro-Compact -> Context Collapse
```

`Tool Result Budget` 处理工具结果膨胀：单个过大工具结果会写入 `.mini_agent/tool-results/`；同一轮 assistant `tool_calls` 后连续 tool results 的总大小超过 200KB 时，会从最大的结果开始写入磁盘，直到 prompt 内保留的工具结果总量回到限制以内。消息中保留 `tool_call_id`、工具名、原始字节数、相对路径和 `read_file(path, offset, limit)` 读取提示，完整内容不丢。

`Snip` 是零 API 成本裁剪：直接移除开头的一段旧消息，不总结、不改写，只插入名为 `context_snip_boundary` 的 system marker。`RequestContextBuilder` 会把这个 marker 当作普通历史边界保留，不注入 Harness Summary。

`Micro-Compact` 只裁剪可重新获取或可安全替代的旧工具结果：

```text
read_file, bash, bash_output, write_file, edit_file, recall_notes, get_skill
```

它按时间衰减保留内容：越新的工具结果保留越多，越老的保留越少。`record_note`、`task`、`bash_kill` 和未知 MCP 工具默认不裁剪。当前活跃工具链始终原样保留。

被裁剪的工具结果会保留 `Old tool result content shortened` 标记，说明旧内容已经从 prompt 中移除，并提示只在安全且必要时重新读取或重新执行。

### 6.4 Context Collapse 读时投影

`ContextCollapser` 不修改真实 `agent.messages`，只在每次 API 请求前生成一个临时投影视图。它运行在 `MessageCompactor` 之后、`MessageSummarizer` 之前：

```text
真实 messages 保留完整历史
request projection 把旧消息段替换成 context_collapse_boundary
LLM 只看到投影视图
```

触发阈值分两级：

- 90% `token_limit`：普通折叠，目标是回落到 85% 左右
- 95% `token_limit`：紧急折叠，目标是回落到 80% 左右

折叠 marker 会说明“只对本次 API call 隐藏旧消息，原始历史仍保留在 agent memory”，并带少量 user、assistant、tool anchor，帮助模型理解被折叠段的轮廓。当前用户请求和活跃工具链始终保留。

### 6.5 Harness summary / Auto-Compact fallback

`CompressionPipeline` 会先运行确定性压缩和 Context Collapse，然后重新测量本轮请求。如果压缩后的请求仍超过 `token_limit`，才调用 `MessageSummarizer` 对可压缩历史做一次全量摘要，压成 system summary：

```text
[Harness Execution Summary]

...
```

Auto-Compact 会保留第一条 system message、最新用户请求、活跃工具链以及 snip/collapse 边界 marker；其余可压缩历史会作为整体交给模型总结。生成的 summary 不再伪装成 user message，而是在下一轮由 `SystemPromptBuilder` 注入 `Harness Summary` 层。

## 7. LLM 边界

职责：

- 接收标准 messages 和 tools
- 转换为 provider 所需格式
- 调用模型
- 标准化 assistant message、tool calls 和 usage
- 记录 prompt cache 相关 usage
- 不内置价格表，只返回 provider 报告的 usage

相关文件：

- [mini_agent/llm/llm_wrapper.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/llm/llm_wrapper.py)
- [mini_agent/llm/openai_client.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/llm/openai_client.py)
- [mini_agent/llm/anthropic_client.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/llm/anthropic_client.py)

LLM 边界不应该知道：

- CLI 怎样展示
- Bash 命令是否需要确认
- checkpoint 怎么保存
- task memory 怎么更新
- 成本单价

### 7.1 Token 与成本统计

`TokenUsage` 来自模型 API 返回值，`Agent` 在每轮 LLM 调用后累计这些字段：

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `cached_tokens`
- `cache_write_tokens`

成本估算由 `mini_agent/token_accounting.py` 负责。它只使用配置里的 `token_pricing`：

```yaml
token_pricing:
  input_per_1m: 0
  output_per_1m: 0
  cache_read_per_1m: 0
  cache_write_per_1m: 0
  currency: USD
```

计算方式：

```text
uncached_input = prompt_tokens - cached_tokens - cache_write_tokens
total_cost =
  uncached_input * input_per_1m / 1_000_000
  + completion_tokens * output_per_1m / 1_000_000
  + cached_tokens * cache_read_per_1m / 1_000_000
  + cache_write_tokens * cache_write_per_1m / 1_000_000
```

这样做的边界更干净：LLM client 只负责 usage 标准化，Agent 负责每轮累计，Console 负责展示，价格作为配置输入，避免在代码里固化可能变化的模型价格。

## 8. Tool Runtime 边界

`ToolRuntime` 是模型 tool call 到本地副作用之间的安全边界。

```text
tool call
  -> ToolRuntime.resolve tool
  -> policies.before_execute(...)
  -> tool.execute(...)
  -> observers.on_tool_result(...)
  -> ToolResult
```

核心抽象：

- `ToolExecutionRequest`：一次工具执行请求
- `ToolPolicy`：执行前策略，可放行、阻断或短路
- `ToolObserver`：执行后观察者，负责记录副作用
- `BashToolPolicy`：Bash 专用安全与确认策略
- `RuntimeToolObserver`：日志和 task memory 记录

相关文件：

- [mini_agent/runtime.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/runtime.py)
- [mini_agent/tools/base.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/tools/base.py)
- [mini_agent/tools/security.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/tools/security.py)

这个边界已经把 Bash 专用逻辑从 Agent 主循环拆出来了。后续如果要给文件工具、网络工具、MCP 工具增加策略，也应该新增 policy，而不是塞回 Agent。

## 9. ToolRegistry 边界

职责：

- 根据配置启用工具
- 构建 workspace 相关工具
- 加载可选 Skills
- 加载可选 MCP tools
- 给 CLI 返回 tool list 和 skill loader

默认工具：

- file tools
- bash / bash_output / bash_kill
- note tools
- task memory hook

可选工具：

- subagent `task`
- skills `get_skill`
- MCP tools

相关文件：

- [mini_agent/tool_registry.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/tool_registry.py)
- [mini_agent/subagent.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/subagent.py)
- [mini_agent/tools/subagent_tool.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/tools/subagent_tool.py)
- [mini_agent/tools/skill_tool.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/tools/skill_tool.py)
- [mini_agent/tools/mcp_loader.py](/F:/Mini-Agent-main/Mini-Agent-main/mini_agent/tools/mcp_loader.py)

### 9.1 Subagent 边界

Subagent 当前实现为一个普通工具，而不是 Agent 主循环里的特殊分支：

```text
Parent Agent
  -> tool_call: task
  -> ToolRuntime.execute("task", ...)
  -> SubagentTool
  -> SubagentRunner 创建 child Agent
  -> child Agent 使用独立 messages 和受限 tools
  -> 子任务摘要作为 tool result 返回 Parent Agent
```

这个边界的重点是上下文隔离。父 Agent 不把完整历史塞给子 Agent，子 Agent 也不直接污染父 Agent 的 messages；父 Agent 只看到 `task` 工具返回的摘要。默认 `allow_nested_subagent: false`，避免递归委派失控。默认 `allowed_tools` 只给读取、Bash 和记忆召回这类探索工具，后续如需可写能力应通过配置显式扩大。

## 10. Memory / Checkpoint / Logger 边界

### 10.1 Checkpoint

作用：恢复会话。

保存位置：

```text
.mini_agent/checkpoints/latest.json
.mini_agent/checkpoints/history/
```

保存内容包括 messages、workspace、step、reason 和工具状态。

### 10.2 Task Memory

作用：当前任务进度。

保存位置：

```text
.mini_agent/task_memory.json
```

它会被注入 `Current Task Context`，但不等同于完整聊天历史。

### 10.3 Episode

作用：任务结束后的审计和复盘。

保存位置：

```text
.mini_agent/episodes.jsonl
```

### 10.4 Long-Term Memory

作用：长期偏好、项目事实、外部参考。

保存位置：

```text
.memory/
```

它通过 `recall_notes` 召回并进入 system prompt 的 `Long-Term Memory` 层。

### 10.5 Logger

作用：调试、审计和成本复盘。

默认位置：

```text
~/.mini-agent/log/
```

## 11. Bash 安全模型

Bash 安全不是工具内部随意判断，而是由 ToolRuntime policy 统一前置处理。

当前行为：

- 高危命令直接阻断
- 中风险命令在交互 CLI 中请求确认
- `--task` 非交互模式没有确认 callback，中风险命令默认拒绝
- 默认禁止命令引用 workspace 外的绝对路径
- 每次 Bash 调用写 JSONL 审计
- 后台进程用 `bash_output` 和 `bash_kill` 管理

配置项：

```yaml
tools:
  enable_bash_security: true
  enable_bash_confirmation: true
  bash_allowed_commands: []
  bash_blocked_commands: []
  bash_allow_outside_workspace: false
  bash_audit_enabled: true
  bash_audit_log_path: null
```

## 12. 配置边界

`mini_agent/config.py` 是配置模型的唯一入口。新增配置时应同步：

1. Pydantic config model
2. `Config.from_yaml()`
3. README 配置示例
4. 相关测试

关键配置：

- `provider`
- `api_key`
- `api_base`
- `model`
- `token_pricing`
- `max_steps`
- `workspace_dir`
- `token_limit`
- `request_context_limit`
- `context_layer_budgets`
- `subagent.*`
- `tools.*`

## 13. 本地稳定性判断

当前 CLI 主链路的本地稳定性主要看这些测试：

```bash
uv run pytest tests/test_agent.py tests/test_runtime.py tests/test_bash_tool.py
uv run pytest tests/test_request_context.py tests/test_prompt_builder.py tests/test_summarizer.py
uv run pytest tests/test_openai_prompt_cache.py
```

全量测试如果只在外部 LLM 集成测试上失败，并且报 403、余额不足、key 无效或权限错误，说明本地 harness 大概率已经跑通，剩下是模型服务配置问题。

## 14. 后续优化建议

优先级从高到低：

1. 把 `MessageSummarizer` 内部重复的 token/tool schema 估算迁移到 `context_budget.py`
2. 抽一个 `ContextBudgetManager`，统一返回 messages、tools 预算报告和压缩原因
3. 给每轮请求输出 budget report，便于解释每层用了多少 token
4. 把 project rules 的头尾裁剪升级成可缓存的语义摘要
5. 长期记忆召回加入 query 和文件路径相关性排序
6. 对 tools schema 做按需启用，减少每轮工具 schema token
7. 等 CLI 稳定后，再重新评估 ACP 是否要接入同一套 harness 边界
