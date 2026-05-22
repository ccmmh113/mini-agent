"""Runtime context and tool execution harness."""

from __future__ import annotations

import asyncio
import hashlib
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable, Protocol
from uuid import uuid4

from .checkpoint import CheckpointStore
from .logger import AgentLogger
from .observability import NullTraceRecorder, ToolCallRecord, TraceEvent, TraceEventKind, TraceRecorder
from .redaction import redact_data, redact_tool_result
from .tools.base import Tool, ToolResult
from .tools.file_tools import resolve_workspace_path
from .tools.security import CommandSecurityDecision, check_command_security, requires_bash_confirmation, write_bash_audit_event
from .tools.task_memory_tool import TaskMemoryHook

ToolConfirmationCallback = Callable[[str, dict[str, Any], CommandSecurityDecision], Awaitable[bool]]


@dataclass(frozen=True)
class FileFingerprint:
    """Stable evidence that a file was read at a specific content version."""

    path: str
    size_bytes: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class WorkspaceFileState:
    """Lightweight file state used for before/after workspace diffs."""

    size_bytes: int
    mtime_ns: int


@dataclass
class RunContext:
    """Shared harness state for one agent runtime."""

    workspace_dir: Path
    logger: AgentLogger = field(default_factory=AgentLogger)
    checkpoint_store: CheckpointStore | None = None
    task_memory_hook: TaskMemoryHook | None = None
    tool_confirmation_callback: ToolConfirmationCallback | None = None
    cancel_event: asyncio.Event | None = None
    read_snapshots: dict[str, FileFingerprint] = field(default_factory=dict)
    run_id: str | None = None
    step_index: int | None = None
    trace_recorder: TraceRecorder = field(default_factory=NullTraceRecorder)

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


def _fingerprint_file(path: Path) -> FileFingerprint:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    stat = path.stat()
    return FileFingerprint(
        path=str(path),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=digest.hexdigest(),
    )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tool_workspace_dir(request: ToolExecutionRequest) -> Path:
    tool_workspace = getattr(request.tool, "workspace_dir", None) if request.tool is not None else None
    return Path(tool_workspace) if tool_workspace else request.context.workspace_dir


def _resolve_tool_path(request: ToolExecutionRequest) -> Path:
    path = request.arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Missing required file path")
    return resolve_workspace_path(path, _tool_workspace_dir(request))


def _workspace_snapshot(workspace_dir: Path) -> dict[str, WorkspaceFileState]:
    workspace = workspace_dir.resolve(strict=False)
    ignored_dirs = {".git", ".mini_agent", "__pycache__", ".pytest_cache"}
    snapshot: dict[str, WorkspaceFileState] = {}

    if not workspace.exists():
        return snapshot

    for path in workspace.rglob("*"):
        try:
            relative = path.relative_to(workspace)
        except ValueError:
            continue
        if any(part in ignored_dirs for part in relative.parts):
            continue
        if not path.is_file():
            continue

        try:
            stat = path.stat()
        except OSError:
            continue
        snapshot[relative.as_posix()] = WorkspaceFileState(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)

    return snapshot


def _workspace_diff(
    before: dict[str, WorkspaceFileState],
    after: dict[str, WorkspaceFileState],
) -> dict[str, list[str]]:
    before_paths = set(before)
    after_paths = set(after)
    shared_paths = before_paths & after_paths
    return {
        "created": sorted(after_paths - before_paths),
        "modified": sorted(path for path in shared_paths if before[path] != after[path]),
        "deleted": sorted(before_paths - after_paths),
    }


def _affected_paths(diff: dict[str, list[str]]) -> list[str]:
    return sorted({path for paths in diff.values() for path in paths})


def _format_workspace_diff(diff: dict[str, list[str]]) -> str:
    lines = ["[workspace_diff]"]
    for key in ("created", "modified", "deleted"):
        values = diff.get(key, [])
        if values:
            lines.append(f"{key}: {', '.join(values)}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _attach_workspace_diff(result: ToolResult, diff: dict[str, list[str]]) -> ToolResult:
    affected = _affected_paths(diff)
    if not affected:
        return result

    metadata = {
        **result.metadata,
        "workspace_diff": diff,
        "affected_paths": affected,
    }
    diff_summary = _format_workspace_diff(diff)
    updates: dict[str, Any] = {"metadata": metadata}

    if result.success:
        content = result.content or ""
        updates["content"] = f"{content}\n\n{diff_summary}" if content else diff_summary
    else:
        error = result.error or "Tool failed"
        updates["error"] = f"{error}\n\n{diff_summary}"

    return result.model_copy(update=updates)


class FreshReadPolicy:
    """Require a current read_file snapshot before modifying existing files."""

    MUTATING_FILE_TOOLS = {"write_file", "edit_file"}

    async def before_execute(self, request: ToolExecutionRequest) -> ToolResult | None:
        if request.tool_name not in self.MUTATING_FILE_TOOLS:
            return None

        try:
            file_path = _resolve_tool_path(request)
        except ValueError as exc:
            return ToolResult(success=False, content="", error=str(exc))

        if not file_path.exists():
            return None
        if not file_path.is_file():
            return ToolResult(success=False, content="", error=f"Path is not a file: {file_path}")

        current = _fingerprint_file(file_path)
        previous = request.context.read_snapshots.get(str(file_path))
        if previous is None:
            return ToolResult(
                success=False,
                content="",
                error=f"Fresh read required before modifying existing file: {file_path}",
            )

        if previous.sha256 != current.sha256:
            return ToolResult(
                success=False,
                content="",
                error=f"File changed since last read; read_file must be called again before modifying: {file_path}",
            )

        return None


class BashToolPolicy:
    """Bash-specific safety and confirmation policy for the generic tool runtime."""

    async def before_execute(self, request: ToolExecutionRequest) -> ToolResult | None:
        tool = request.tool
        if tool is None or tool.name != "bash":
            return None

        decision = self._preflight(request)
        if decision and not decision.allowed:
            self._record_blocked(request, decision)
            return ToolResult(
                success=False,
                content="",
                error=f"Command blocked by security policy: {decision.reason}",
                metadata={"policy_outcome": "blocked"},
            )

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
                    metadata={"policy_outcome": "confirmation_denied"},
                )

        if decision:
            self._write_audit_event(request, decision, "allowed")

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
        if request.tool_name == "read_file" and result.success:
            self._record_read_snapshot(request)

        request.context.logger.log_tool_result(
            tool_name=request.tool_name,
            arguments=redact_data(request.arguments),
            result_success=result.success,
            result_content=result.content if result.success else None,
            result_error=result.error if not result.success else None,
            metadata=result.metadata,
        )

        if request.context.task_memory_hook is not None:
            request.context.task_memory_hook.record_tool_result(
                request.tool_name,
                redact_data(request.arguments),
                result,
            )

    def _record_read_snapshot(self, request: ToolExecutionRequest) -> None:
        try:
            file_path = _resolve_tool_path(request)
        except ValueError:
            return
        if file_path.exists() and file_path.is_file():
            request.context.read_snapshots[str(file_path)] = _fingerprint_file(file_path)


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
        self.policies = policies if policies is not None else [FreshReadPolicy(), BashToolPolicy()]
        self.observers = observers if observers is not None else [RuntimeToolObserver()]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool call through the harness boundary."""

        tool = self.tools.get(tool_name)
        request = ToolExecutionRequest(tool_name=tool_name, arguments=arguments, tool=tool, context=self.context)
        trace_call_id = self._start_trace(request)
        trace_started_at = _utc_timestamp()
        trace_start_time = perf_counter()
        if tool is None:
            result = ToolResult(success=False, content="", error=f"Unknown tool: {tool_name}")
            self._notify_observers(request, result)
            self._finish_trace(request, result, trace_call_id, trace_started_at, trace_start_time)
            return result

        before_snapshot: dict[str, WorkspaceFileState] | None = None
        try:
            result = await self._run_policies(request)
            if result is None:
                before_snapshot = _workspace_snapshot(self.context.workspace_dir)
                result = await tool.execute(**arguments)
                after_snapshot = _workspace_snapshot(self.context.workspace_dir)
                result = _attach_workspace_diff(result, _workspace_diff(before_snapshot, after_snapshot))
        except Exception as exc:
            error_detail = f"{type(exc).__name__}: {str(exc)}"
            error_trace = traceback.format_exc()
            result = ToolResult(
                success=False,
                content="",
                error=f"Tool execution failed: {error_detail}\n\nTraceback:\n{error_trace}",
            )
            if before_snapshot is not None:
                after_snapshot = _workspace_snapshot(self.context.workspace_dir)
                result = _attach_workspace_diff(result, _workspace_diff(before_snapshot, after_snapshot))

        result = redact_tool_result(result)
        self._notify_observers(request, result)
        self._finish_trace(request, result, trace_call_id, trace_started_at, trace_start_time)
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

    def _start_trace(self, request: ToolExecutionRequest) -> str | None:
        run_id = request.context.run_id
        if run_id is None:
            return None

        call_id = f"tool-{uuid4().hex}"
        self._record_trace_event(
            request,
            TraceEventKind.TOOL_STARTED,
            {"call_id": call_id, "tool_name": request.tool_name, "step_index": request.context.step_index},
        )
        return call_id

    def _finish_trace(
        self,
        request: ToolExecutionRequest,
        result: ToolResult,
        call_id: str | None,
        started_at: str,
        start_time: float,
    ) -> None:
        run_id = request.context.run_id
        if run_id is None or call_id is None:
            return

        policy_outcome = result.metadata.get("policy_outcome")
        affected_paths = result.metadata.get("affected_paths", [])
        if not isinstance(affected_paths, list):
            affected_paths = []

        duration_ms = int((perf_counter() - start_time) * 1000)
        call = ToolCallRecord(
            call_id=call_id,
            run_id=run_id,
            step_index=request.context.step_index,
            tool_name=request.tool_name,
            arguments=redact_data(request.arguments),
            started_at=started_at,
            ended_at=_utc_timestamp(),
            duration_ms=duration_ms,
            success=result.success,
            policy_outcome=policy_outcome if isinstance(policy_outcome, str) else None,
            error=result.error if not result.success else None,
            result_summary=result.content if result.success else None,
            affected_paths=affected_paths,
        )
        try:
            request.context.trace_recorder.record_tool_call(call)
        except Exception:
            pass

        kind = TraceEventKind.TOOL_COMPLETED if result.success else TraceEventKind.TOOL_FAILED
        if policy_outcome == "blocked":
            kind = TraceEventKind.TOOL_BLOCKED
        self._record_trace_event(request, kind, {"call_id": call_id, "tool_name": request.tool_name})

    def _record_trace_event(
        self,
        request: ToolExecutionRequest,
        kind: TraceEventKind,
        payload: dict[str, Any],
    ) -> None:
        run_id = request.context.run_id
        if run_id is None:
            return

        event = TraceEvent(
            event_id=f"event-{uuid4().hex}",
            run_id=run_id,
            kind=kind,
            created_at=_utc_timestamp(),
            payload=payload,
        )
        try:
            request.context.trace_recorder.record_event(event)
        except Exception:
            pass
