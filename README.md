# Mini Agent

Mini Agent 是一个面向本地项目工作的 CLI Agent harness。它负责把用户任务、系统提示词、项目上下文、记忆、历史摘要和工具 schema 组装成每一轮 LLM 请求，再把模型返回的 tool calls 安全地落到本地工具执行。

当前主线聚焦 CLI。ACP 入口仍保留在代码中，但不作为推荐使用路径；Skills 和 MCP 也默认关闭，只有明确启用时才加载。

## 当前能力

- 交互式 CLI 与 `--task` 非交互任务模式
- OpenAI / Anthropic 风格 LLM 调用封装
- 文件读写编辑工具
- Bash 工具、安全策略、确认机制和审计日志
- 轻量长期记忆：`.memory/*.md`
- 当前任务状态：`.mini_agent/task_memory.json`
- 会话 checkpoint 恢复：`.mini_agent/checkpoints/`
- 任务完成 episode 记录：`.mini_agent/episodes.jsonl`
- 分层 prompt 拼装与 10000 token 默认预算
- 历史 assistant/tool 执行轨迹摘要压缩
- 可选 Subagent：通过 `task` 工具把局部任务委派给隔离子 Agent
- 可选 Skills 与 MCP 工具加载

## 快速开始

### 1. 安装依赖

开发模式推荐使用 uv：

```bash
uv sync
```

如需启用 MCP 相关依赖：

```bash
uv sync --extra mcp
```

`--extra mcp` 安装的是 `pyproject.toml` 里 `[project.optional-dependencies].mcp` 声明的 `mcp>=1.0.0`，用于连接和加载 MCP server 暴露的工具。

### 2. 配置模型

创建或编辑配置文件：

```bash
mini_agent/config/config.yaml
```

最小配置示例：

```yaml
api_key: "YOUR_API_KEY_HERE"
api_base: "https://api.openai.com/v1"
model: "gpt-4o-mini"
provider: "openai"

token_pricing:
  input_per_1m: 0
  output_per_1m: 0
  cache_read_per_1m: 0
  cache_write_per_1m: 0
  currency: "USD"

max_steps: 50
workspace_dir: "."
token_limit: 10000
request_context_limit: 12

context_layer_budgets:
  core: 2500
  skills: 1200
  memory: 1200
  project_rules: 1800
  current_task_context: 1000
  harness_summary: 1800
  dynamic_context: 300

tools:
  enable_file_tools: true
  enable_bash: true
  enable_note: true
  enable_task_memory: true
  enable_bash_security: true
  enable_bash_confirmation: true
  bash_allow_outside_workspace: false
  bash_audit_enabled: true
  enable_subagent: false
  enable_skills: false
  enable_mcp: false

subagent:
  max_steps: 12
  token_limit: 6000
  request_context_limit: 8
  allowed_tools:
    - read_file
    - bash
    - recall_notes
  allow_nested_subagent: false
```

配置加载优先级见 `Config.find_config_file()`：当前开发目录、用户目录 `~/.mini-agent/config/`、包内默认目录。

### 3. 启动 CLI

```bash
uv run mini-agent
uv run mini-agent --workspace F:\path\to\project
uv run mini-agent --task "分析这个项目还有哪些优化点"
```

安装为命令后也可以直接运行：

```bash
mini-agent
mini-agent --workspace /path/to/project
mini-agent --task "修复测试失败"
```

## CLI 命令

交互模式下支持：

- `/help`：查看帮助
- `/clear`：清空当前会话历史
- `/history`：查看当前消息数量
- `/resume`：从最新 checkpoint 恢复
- `/task`：查看当前任务状态
- `/memory`：查看长期记忆概况
- `/memory review`：列出长期记忆文件
- `/memory delete <name>`：删除一条长期记忆
- `/stats`：查看 token、cache 与成本统计
- `/log`：查看日志目录
- `/log <file>`：查看指定日志
- `/exit`：退出

## 任务执行流程

```text
用户输入任务
  -> CLI 加载配置、LLM、工具、记忆和 checkpoint
  -> Agent.add_user_message()
  -> Agent.run()
  -> RequestContextBuilder 预构建预算上下文
  -> MessageSummarizer 必要时压缩旧 assistant/tool 轨迹
  -> RequestContextBuilder 再构建真正发送给 LLM 的 messages
  -> LLMClient.generate(messages, tools)
  -> Agent 追加 assistant 消息
  -> 如有 tool_calls，ToolRuntime 执行工具
  -> ToolPolicy 做执行前安全/确认判断
  -> ToolObserver 记录日志、任务状态和审计
  -> tool result 写回 messages，进入下一轮
  -> 无 tool_calls 时结束任务并写 episode
```

LLM 实际收到的不是一个拼好的字符串，而是：

- `messages`：第一条是 system message，后面是用户、assistant、tool 历史
- `tools`：当前启用工具的 JSON schema

## Prompt 与压缩策略

默认总预算是 `token_limit: 10000`。预算不是平均切分，而是按层和优先级处理：

1. `core`：核心 system prompt，最高优先级，尽量保留
2. `skills`：可选 skills metadata，默认关闭
3. `memory`：长期记忆召回内容
4. `project_rules`：项目规则文件内容
5. `current_task_context`：当前任务状态
6. `harness_summary`：旧 assistant/tool 执行轨迹摘要
7. `dynamic_context`：运行时动态上下文
8. recent messages：最近用户消息、assistant 回复和 tool 结果
9. active tool chain：当前未完成的工具链强保护

## Token 与成本显示

每次 LLM 调用返回后，终端会显示本轮 API reported token：

- `prompt`：本轮输入 token
- `completion`：本轮输出 token
- `cached`：命中 prompt cache 的输入 token
- `cache_write`：写入 cache 的输入 token

如果配置了 `token_pricing`，还会按每百万 token 单价估算本轮成本，并在 `/stats` 中展示累计成本。`prompt_tokens` 通常包含 cache read/write token，因此成本估算会先计算：

```text
uncached_input = prompt_tokens - cached_tokens - cache_write_tokens
```

然后分别估算 `input`、`output`、`cache_read`、`cache_write` 成本。价格由模型服务商决定，建议面试或演示前按实际模型价格填写，不要把它写死在代码里。

每轮请求前会先估算 system prompt、tool schema 和历史消息占用。旧执行过程过长时，会把 assistant/tool 轨迹压成带有 `[Harness Execution Summary]` 标记的 system message，再由 `SystemPromptBuilder` 注入 `Harness Summary` 层。

## 工具边界

工具由 `ToolRegistry` 根据配置加载。

默认工具：

- `read_file` / `write_file` / `edit_file`
- `bash`
- `bash_output`
- `bash_kill`
- `record_note`
- `recall_notes`
- task memory hook

可选工具：

- `task`：`tools.enable_subagent: true` 时加载，用隔离子 Agent 完成局部任务
- `get_skill`：`tools.enable_skills: true` 且存在可用 `SKILL.md` 时加载
- MCP tools：`tools.enable_mcp: true` 且 `mcp_config_path` 可解析时加载

Bash 执行会经过 `ToolRuntime` 的 policy/observer 机制。当前内置 `BashToolPolicy` 负责命令安全检查、用户确认和拒绝审计；`RuntimeToolObserver` 负责记录工具结果和任务状态。

## Subagent

Subagent 默认关闭。启用后会多一个 `task` 工具，父 Agent 可以把独立的分析、搜索或局部调查任务交给子 Agent 执行：

```yaml
tools:
  enable_subagent: true

subagent:
  max_steps: 12
  token_limit: 6000
  request_context_limit: 8
  allowed_tools:
    - read_file
    - bash
    - recall_notes
  allow_nested_subagent: false
```

第一版 Subagent 的边界是“上下文隔离”：子 Agent 使用独立 `messages`，不继承父 Agent 完整历史；结束后只把摘要作为 `task` 的 tool result 返回父 Agent。默认不允许递归创建 Subagent，默认工具也限制在读取、Bash 和记忆召回这类探索能力上。

## 记忆与状态文件

工作区下会产生这些运行态文件：

```text
.mini_agent/
  checkpoints/
    latest.json
    history/
  task_memory.json
  episodes.jsonl
  bash_audit.jsonl

.memory/
  MEMORY.md
  *.md
```

这些文件职责不同：

- `messages`：当前会话上下文，存在内存中，可被 checkpoint 保存
- `checkpoint`：会话恢复用，保存消息、step、workspace 和工具状态
- `task_memory`：当前任务进度、决策、artifact 与 next steps
- `episode`：任务完成后的复盘记录
- `.memory`：用户偏好、项目事实、外部参考等长期记忆
- `bash_audit`：Bash 安全审计

## Skills 与 MCP

默认不启用：

```yaml
tools:
  enable_skills: false
  enable_mcp: false
```

启用 Skills：

```yaml
tools:
  enable_skills: true
  skills_dir: "./skills"
```

`skills_dir` 下每个技能目录需要包含 `SKILL.md`。启用后，system prompt 只注入技能名称和描述，完整内容通过 `get_skill(skill_name)` 按需获取。

启用 MCP：

```yaml
tools:
  enable_mcp: true
  mcp_config_path: "mcp.json"
  mcp:
    connect_timeout: 10
    execute_timeout: 60
    sse_read_timeout: 120
```

MCP 需要先安装额外依赖：

```bash
uv sync --extra mcp
```

当前建议先把 CLI 主链路跑稳，再按任务需要逐个打开 Skills 或 MCP。

## 开发与测试

常用命令：

```bash
uv run pytest
uv run pytest tests/test_request_context.py tests/test_prompt_builder.py tests/test_summarizer.py
uv run python -m py_compile mini_agent/agent.py mini_agent/cli.py mini_agent/config.py
```

如果本地全量测试中只有外部 LLM 集成测试失败，并返回余额、权限或 API 403 一类错误，通常说明代码链路已跑到外部服务调用阶段，需要检查模型服务配置或账户状态。

## 主要源码入口

- `mini_agent/cli.py`：CLI 入口、命令处理、工具构建、交互循环
- `mini_agent/agent.py`：Agent 主循环
- `mini_agent/request_context.py`：每轮请求上下文选择
- `mini_agent/prompt_builder.py`：system prompt 分层拼装
- `mini_agent/summarizer.py`：历史执行轨迹压缩
- `mini_agent/context_budget.py`：token 估算和分层裁剪
- `mini_agent/runtime.py`：ToolRuntime、policy、observer
- `mini_agent/tool_registry.py`：工具注册与可选扩展加载
- `mini_agent/config.py`：配置模型和 YAML 解析
