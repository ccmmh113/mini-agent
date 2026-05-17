"""Runtime context and tool execution harness."""

from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from .checkpoint import CheckpointStore
from .logger import AgentLogger
from .tools.base import Tool, ToolResult
from .tools.security import CommandSecurityDecision, check_command_security, requires_bash_confirmation, write_bash_audit_event
from .tools.task_memory_tool import TaskMemoryHook

ToolConfirmationCallback = Callable[[str, dict[str, Any], CommandSecurityDecision], Awaitable[bool]]


@dataclass
class RunContext:
    """Shared harness state for one agent runtime."""

    workspace_dir: Path
    logger: AgentLogger = field(default_factory=AgentLogger)
    checkpoint_store: CheckpointStore | None = None
    task_memory_hook: TaskMemoryHook | None = None
    tool_confirmation_callback: ToolConfirmationCallback | None = None
    cancel_event: asyncio.Event | None = None

    def __post_init__(self) -> None:
        self.workspace_dir = Path(self.workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ToolExecutionRequest:
    """A model-requested tool call after runtime resolution."""

    tool_name: str
    arguments: dict[str, Any]
    tool: Tool | None
    context: RunContext


class ToolPolicy(Protocol):
    """Pre-execution hook that may allow, block, or short-circuit a tool call."""

    async def before_execute(self, request: ToolExecutionRequest) -> ToolResult | None:
        """Return a ToolResult to stop execution, or None to continue."""


class ToolObserver(Protocol):
    """Observer for tool execution results and side effects."""

    def on_tool_result(self, request: ToolExecutionRequest, result: ToolResult) -> None:
        """Record side effects after a tool result is available."""


class BashToolPolicy:
    """Bash-specific safety and confirmation policy for the generic tool runtime."""

    async def before_execute(self, request: ToolExecutionRequest) -> ToolResult | None:
        tool = request.tool
        if tool is None or tool.name != "bash":
            return None

        decision = self._preflight(request)
        if decision and not decision.allowed:
            self._record_blocked(request, decision)
            return ToolResult(success=False, content="", error=f"Command blocked by security policy: {decision.reason}")

        if decision and decision.requires_confirmation:
            approved = False
            if request.context.tool_confirmation_callback is not None:
                approved = await request.context.tool_confirmation_callback(
                    request.tool_name,
                    request.arguments,
                    decision,
                )

            if not approved:
                self._record_confirmation_denied(request, decision)
                return ToolResult(
                    success=False,
                    content="",
                    error="Command execution denied by user confirmation policy.",
                )

        return None

    def _preflight(self, request: ToolExecutionRequest) -> CommandSecurityDecision | None:
        tool = request.tool
        if tool is None:
            return None

        policy = getattr(tool, "security_policy", None)
        if policy is not None and not getattr(policy, "enabled", True):
            return None

        command = str(request.arguments.get("command", ""))
        run_in_background = bool(request.arguments.get("run_in_background", False))
        workspace_dir = getattr(tool, "workspace_dir", str(request.context.workspace_dir))

        security_decision = check_command_security(command, workspace_dir, policy)
        if not security_decision.allowed:
            return security_decision

        confirmation_decision = requires_bash_confirmation(command, run_in_background)
        if confirmation_decision.requires_confirmation:
            return confirmation_decision

        return confirmation_decision

    def _record_confirmation_denied(
        self,
        request: ToolExecutionRequest,
        decision: CommandSecurityDecision,
    ) -> None:
        self._write_audit_event(request, decision, "confirmation_denied")

    def _record_blocked(
        self,
        request: ToolExecutionRequest,
        decision: CommandSecurityDecision,
    ) -> None:
        self._write_audit_event(request, decision, "blocked")

    def _write_audit_event(
        self,
        request: ToolExecutionRequest,
        decision: CommandSecurityDecision,
        outcome: str,
    ) -> None:
        tool = request.tool
        if tool is None:
            return

        policy = getattr(tool, "security_policy", None)
        workspace_dir = getattr(tool, "workspace_dir", str(request.context.workspace_dir))
        write_bash_audit_event(
            {
                "tool": tool.name,
                "command": str(request.arguments.get("command", "")),
                "run_in_background": bool(request.arguments.get("run_in_background", False)),
                "decision": outcome,
                "risk_level": decision.risk_level,
                "matched_rules": decision.matched_rules,
                "reason": decision.reason,
            },
            workspace_dir,
            policy,
        )


class RuntimeToolObserver:
    """Persist standard runtime side effects after each tool call."""

    def on_tool_result(self, request: ToolExecutionRequest, result: ToolResult) -> None:
        request.context.logger.log_tool_result(
            tool_name=request.tool_name,
            arguments=request.arguments,
            result_success=result.success,
            result_content=result.content if result.success else None,
            result_error=result.error if not result.success else None,
        )

        if request.context.task_memory_hook is not None:
            request.context.task_memory_hook.record_tool_result(request.tool_name, request.arguments, result)


class ToolRuntime:
    """Resolve, preflight, execute, and observe model-requested tools."""

    def __init__(
        self,
        tools: dict[str, Tool],
        context: RunContext,
        policies: list[ToolPolicy] | None = None,
        observers: list[ToolObserver] | None = None,
    ):
        self.tools = tools
        self.context = context
        self.policies = policies if policies is not None else [BashToolPolicy()]
        self.observers = observers if observers is not None else [RuntimeToolObserver()]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool call through the harness boundary."""

        tool = self.tools.get(tool_name)
        request = ToolExecutionRequest(tool_name=tool_name, arguments=arguments, tool=tool, context=self.context)
        if tool is None:
            result = ToolResult(success=False, content="", error=f"Unknown tool: {tool_name}")
            self._notify_observers(request, result)
            return result

        try:
            result = await self._run_policies(request)
            if result is None:
                result = await tool.execute(**arguments)
        except Exception as exc:
            error_detail = f"{type(exc).__name__}: {str(exc)}"
            error_trace = traceback.format_exc()
            result = ToolResult(
                success=False,
                content="",
                error=f"Tool execution failed: {error_detail}\n\nTraceback:\n{error_trace}",
            )

        self._notify_observers(request, result)
        return result

    async def _run_policies(self, request: ToolExecutionRequest) -> ToolResult | None:
        for policy in self.policies:
            result = await policy.before_execute(request)
            if result is not None:
                return result
        return None

    def _notify_observers(self, request: ToolExecutionRequest, result: ToolResult) -> None:
        for observer in self.observers:
            observer.on_tool_result(request, result)
