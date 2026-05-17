"""Tool registry and factories for the Mini-Agent harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import Config
from .llm import LLMClient
from .subagent import SubagentRunner
from .tools.base import Tool
from .tools.bash_tool import BashKillTool, BashOutputTool, BashTool
from .tools.file_tools import EditTool, ReadTool, WriteTool
from .tools.note_tool import RecallNoteTool, SessionNoteTool
from .tools.security import BashSecurityPolicy
from .tools.skill_tool import create_skill_tools
from .tools.subagent_tool import SubagentTool

StatusCallback = Callable[[str, str], None]


@dataclass
class ToolBuildResult:
    """Tools and loader state produced by the registry."""

    tools: list[Tool] = field(default_factory=list)
    skill_loader: object | None = None


class ToolRegistry:
    """Build enabled tools from configuration and runtime workspace."""

    def __init__(self, config: Config, notify: StatusCallback | None = None):
        self.config = config
        self.notify = notify or (lambda _level, _message: None)

    async def build_base_tools(self) -> ToolBuildResult:
        """Build tools that do not need a workspace path."""

        result = ToolBuildResult()

        if self.config.tools.enable_bash:
            result.tools.extend([BashOutputTool(), BashKillTool()])
            self.notify("success", "Loaded Bash Output tool")
            self.notify("success", "Loaded Bash Kill tool")

        if self.config.tools.enable_skills:
            self.notify("info", "Loading Skills...")
            try:
                skill_tools, skill_loader = create_skill_tools(self._resolve_skills_dir())
                result.skill_loader = skill_loader
                if skill_tools:
                    result.tools.extend(skill_tools)
                    self.notify("success", "Loaded Skill tool (get_skill)")
                else:
                    self.notify("warning", "No available Skills found")
            except Exception as exc:
                self.notify("warning", f"Failed to load Skills: {exc}")

        if self.config.tools.enable_mcp:
            self.notify("info", "Loading MCP tools...")
            try:
                from mini_agent.tools.mcp_loader import load_mcp_tools_async, set_mcp_timeout_config

                mcp_config = self.config.tools.mcp
                set_mcp_timeout_config(
                    connect_timeout=mcp_config.connect_timeout,
                    execute_timeout=mcp_config.execute_timeout,
                    sse_read_timeout=mcp_config.sse_read_timeout,
                )
                self.notify(
                    "detail",
                    "MCP timeouts: "
                    f"connect={mcp_config.connect_timeout}s, "
                    f"execute={mcp_config.execute_timeout}s, "
                    f"sse_read={mcp_config.sse_read_timeout}s",
                )

                mcp_config_path = Config.find_config_file(self.config.tools.mcp_config_path)
                if mcp_config_path:
                    mcp_tools = await load_mcp_tools_async(str(mcp_config_path))
                    if mcp_tools:
                        result.tools.extend(mcp_tools)
                        self.notify("success", f"Loaded {len(mcp_tools)} MCP tools (from: {mcp_config_path})")
                    else:
                        self.notify("warning", "No available MCP tools found")
                else:
                    self.notify("warning", f"MCP config file not found: {self.config.tools.mcp_config_path}")
            except Exception as exc:
                self.notify("warning", f"Failed to load MCP tools: {exc}")

        return result

    def add_workspace_tools(
        self,
        tools: list[Tool],
        workspace_dir: Path,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Append tools that depend on the active workspace."""

        workspace_dir.mkdir(parents=True, exist_ok=True)

        if self.config.tools.enable_bash:
            bash_policy = self._build_bash_security_policy()
            tools.append(BashTool(workspace_dir=str(workspace_dir), security_policy=bash_policy))
            self.notify("success", f"Loaded Bash tool (cwd: {workspace_dir})")
            if bash_policy.enabled:
                self.notify("success", "Loaded Bash security policy")

        if self.config.tools.enable_file_tools:
            tools.extend(
                [
                    ReadTool(workspace_dir=str(workspace_dir)),
                    WriteTool(workspace_dir=str(workspace_dir)),
                    EditTool(workspace_dir=str(workspace_dir)),
                ]
            )
            self.notify("success", f"Loaded file operation tools (workspace: {workspace_dir})")

        if self.config.tools.enable_note:
            memory_dir = workspace_dir / ".memory"
            tools.append(SessionNoteTool(memory_dir=str(memory_dir)))
            tools.append(RecallNoteTool(memory_dir=str(memory_dir), workspace_dir=str(workspace_dir)))
            self.notify("success", "Loaded lightweight memory tools")

        if self.config.tools.enable_subagent:
            if llm_client is None:
                self.notify("warning", "Subagent tool skipped because no LLM client was provided")
                return
            runner = SubagentRunner(
                llm_client=llm_client,
                workspace_dir=workspace_dir,
                tools_provider=lambda: tools,
                config=self.config.subagent,
                context_layer_budgets=self.config.agent.context_layer_budgets.to_prompt_layer_budgets(),
                token_pricing=self.config.llm.token_pricing,
            )
            tools.append(SubagentTool(runner))
            self.notify("success", "Loaded Subagent task tool")

    def _resolve_skills_dir(self) -> str:
        skills_path = Path(self.config.tools.skills_dir).expanduser()
        if skills_path.is_absolute():
            return str(skills_path)

        search_paths = [
            skills_path,
            Path("mini_agent") / skills_path,
            Config.get_package_dir() / skills_path,
        ]

        for path in search_paths:
            if path.exists():
                return str(path.resolve())
        return str(skills_path)

    def _build_bash_security_policy(self) -> BashSecurityPolicy:
        blocked_commands = self.config.tools.bash_blocked_commands or BashSecurityPolicy().blocked_commands
        return BashSecurityPolicy(
            enabled=self.config.tools.enable_bash_security,
            allowed_commands=self.config.tools.bash_allowed_commands,
            blocked_commands=blocked_commands,
            allow_outside_workspace=self.config.tools.bash_allow_outside_workspace,
            audit_enabled=self.config.tools.bash_audit_enabled,
            audit_log_path=self.config.tools.bash_audit_log_path,
        )
