"""Tests for harness runtime boundaries."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from mini_agent.config import AgentConfig, Config, LLMConfig, ToolsConfig
from mini_agent.observability import TraceEventKind
from mini_agent.runtime import RunContext, ToolExecutionRequest, ToolRuntime
from mini_agent.tool_registry import ToolRegistry
from mini_agent.tools import EditTool, ReadTool, WriteTool
from mini_agent.tools.base import Tool, ToolResult
from mini_agent.tools.bash_tool import BashTool


class RuntimeDummyTool(Tool):
    @property
    def name(self) -> str:
        return "runtime_dummy"

    @property
    def description(self) -> str:
        return "Runtime dummy tool."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {"value": {"type": "string"}}}

    async def execute(self, value: str) -> ToolResult:
        return ToolResult(success=True, content=f"ok {value}")


class BlockingPolicy:
    async def before_execute(self, request: ToolExecutionRequest) -> ToolResult | None:
        if request.tool_name == "runtime_dummy" and request.arguments.get("value") == "blocked":
            return ToolResult(success=False, error="blocked by test policy")
        return None


class RecordingObserver:
    def __init__(self):
        self.events: list[tuple[str, bool]] = []

    def on_tool_result(self, request: ToolExecutionRequest, result: ToolResult) -> None:
        self.events.append((request.tool_name, result.success))


class RuntimeTraceRecorder:
    def __init__(self):
        self.tool_calls = []
        self.events = []

    def record_run(self, run):
        pass

    def record_step(self, step):
        pass

    def record_llm_call(self, call):
        pass

    def record_tool_call(self, call):
        self.tool_calls.append(call)

    def record_event(self, event):
        self.events.append(event)


def _config(**tool_overrides) -> Config:
    tool_values = {
        "enable_bash": False,
        "enable_file_tools": False,
        "enable_note": False,
        "enable_task_memory": False,
        "enable_skills": False,
        "enable_mcp": False,
        **tool_overrides,
    }
    tools = ToolsConfig(**tool_values)
    return Config(
        llm=LLMConfig(api_key="test-key"),
        agent=AgentConfig(),
        tools=tools,
    )


@pytest.mark.asyncio
async def test_tool_runtime_executes_and_logs_result(tmp_path):
    context = RunContext(workspace_dir=tmp_path)
    runtime = ToolRuntime({"runtime_dummy": RuntimeDummyTool()}, context)

    result = await runtime.execute("runtime_dummy", {"value": "sample"})

    assert result.success
    assert result.content == "ok sample"


@pytest.mark.asyncio
async def test_tool_runtime_blocks_dangerous_bash_before_execution(tmp_path):
    context = RunContext(workspace_dir=tmp_path)
    runtime = ToolRuntime({"bash": BashTool(workspace_dir=str(tmp_path))}, context)

    result = await runtime.execute("bash", {"command": "rm -rf ./important"})

    assert not result.success
    assert "Command blocked by security policy" in result.error
    audit_log = tmp_path / ".mini_agent" / "bash_audit.jsonl"
    assert "blocked" in audit_log.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_tool_runtime_uses_confirmation_callback(tmp_path):
    callback = AsyncMock(return_value=False)
    context = RunContext(workspace_dir=tmp_path, tool_confirmation_callback=callback)
    runtime = ToolRuntime({"bash": BashTool(workspace_dir=str(tmp_path))}, context)

    result = await runtime.execute("bash", {"command": "echo wait", "run_in_background": True})

    assert not result.success
    assert "denied by user confirmation policy" in result.error
    callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_runtime_policy_can_short_circuit_execution(tmp_path):
    context = RunContext(workspace_dir=tmp_path)
    observer = RecordingObserver()
    runtime = ToolRuntime(
        {"runtime_dummy": RuntimeDummyTool()},
        context,
        policies=[BlockingPolicy()],
        observers=[observer],
    )

    result = await runtime.execute("runtime_dummy", {"value": "blocked"})

    assert not result.success
    assert result.error == "blocked by test policy"
    assert observer.events == [("runtime_dummy", False)]


@pytest.mark.asyncio
async def test_tool_runtime_observer_records_successful_result(tmp_path):
    context = RunContext(workspace_dir=tmp_path)
    observer = RecordingObserver()
    runtime = ToolRuntime(
        {"runtime_dummy": RuntimeDummyTool()},
        context,
        policies=[BlockingPolicy()],
        observers=[observer],
    )

    result = await runtime.execute("runtime_dummy", {"value": "sample"})

    assert result.success
    assert result.content == "ok sample"
    assert observer.events == [("runtime_dummy", True)]


@pytest.mark.asyncio
async def test_tool_runtime_requires_fresh_read_before_editing_existing_file(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("status: draft\n", encoding="utf-8")
    context = RunContext(workspace_dir=tmp_path)
    runtime = ToolRuntime(
        {
            "read_file": ReadTool(workspace_dir=str(tmp_path)),
            "edit_file": EditTool(workspace_dir=str(tmp_path)),
        },
        context,
    )

    denied = await runtime.execute("edit_file", {"path": "sample.txt", "old_str": "draft", "new_str": "ready"})
    assert not denied.success
    assert "Fresh read required" in (denied.error or "")
    assert test_file.read_text(encoding="utf-8") == "status: draft\n"

    read_result = await runtime.execute("read_file", {"path": "sample.txt"})
    assert read_result.success

    edit_result = await runtime.execute("edit_file", {"path": "sample.txt", "old_str": "draft", "new_str": "ready"})
    assert edit_result.success
    assert test_file.read_text(encoding="utf-8") == "status: ready\n"
    assert edit_result.metadata["affected_paths"] == ["sample.txt"]
    assert edit_result.metadata["workspace_diff"]["modified"] == ["sample.txt"]


@pytest.mark.asyncio
async def test_tool_runtime_rejects_stale_read_before_editing(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("status: draft\n", encoding="utf-8")
    context = RunContext(workspace_dir=tmp_path)
    runtime = ToolRuntime(
        {
            "read_file": ReadTool(workspace_dir=str(tmp_path)),
            "edit_file": EditTool(workspace_dir=str(tmp_path)),
        },
        context,
    )

    read_result = await runtime.execute("read_file", {"path": "sample.txt"})
    assert read_result.success

    test_file.write_text("status: changed\n", encoding="utf-8")
    edit_result = await runtime.execute("edit_file", {"path": "sample.txt", "old_str": "changed", "new_str": "ready"})

    assert not edit_result.success
    assert "File changed since last read" in (edit_result.error or "")
    assert test_file.read_text(encoding="utf-8") == "status: changed\n"


@pytest.mark.asyncio
async def test_tool_runtime_records_workspace_diff_for_created_files(tmp_path):
    context = RunContext(workspace_dir=tmp_path)
    runtime = ToolRuntime({"write_file": WriteTool(workspace_dir=str(tmp_path))}, context)

    result = await runtime.execute("write_file", {"path": "created.txt", "content": "hello"})

    assert result.success
    assert result.metadata["affected_paths"] == ["created.txt"]
    assert result.metadata["workspace_diff"]["created"] == ["created.txt"]
    assert "[workspace_diff]" in result.content


@pytest.mark.asyncio
async def test_tool_runtime_traces_successful_write_file_calls(tmp_path):
    recorder = RuntimeTraceRecorder()
    context = RunContext(workspace_dir=tmp_path, run_id="run-1", step_index=2, trace_recorder=recorder)
    runtime = ToolRuntime({"write_file": WriteTool(workspace_dir=str(tmp_path))}, context)

    result = await runtime.execute(
        "write_file",
        {"path": "created.txt", "content": "sk-testsecret12345678901234567890"},
    )

    assert result.success
    assert [event.kind for event in recorder.events] == [
        TraceEventKind.TOOL_STARTED,
        TraceEventKind.TOOL_COMPLETED,
    ]
    call = recorder.tool_calls[0]
    assert call.run_id == "run-1"
    assert call.step_index == 2
    assert call.tool_name == "write_file"
    assert call.arguments == {"path": "created.txt", "content": "[REDACTED]"}
    assert call.started_at
    assert call.ended_at
    assert call.duration_ms >= 0
    assert call.success is True
    assert call.error is None
    assert call.result_summary
    assert call.affected_paths == ["created.txt"]


@pytest.mark.asyncio
async def test_tool_runtime_traces_blocked_bash_policy_outcomes(tmp_path):
    recorder = RuntimeTraceRecorder()
    context = RunContext(workspace_dir=tmp_path, run_id="run-1", trace_recorder=recorder)
    runtime = ToolRuntime({"bash": BashTool(workspace_dir=str(tmp_path))}, context)

    result = await runtime.execute("bash", {"command": "rm -rf ./important"})

    assert not result.success
    assert [event.kind for event in recorder.events] == [
        TraceEventKind.TOOL_STARTED,
        TraceEventKind.TOOL_BLOCKED,
    ]
    call = recorder.tool_calls[0]
    assert call.tool_name == "bash"
    assert call.success is False
    assert call.policy_outcome == "blocked"
    assert call.error == result.error


@pytest.mark.asyncio
async def test_tool_runtime_redacts_secret_tool_results(tmp_path):
    context = RunContext(workspace_dir=tmp_path)
    runtime = ToolRuntime({"runtime_dummy": RuntimeDummyTool()}, context)

    result = await runtime.execute("runtime_dummy", {"value": "sk-testsecret12345678901234567890"})

    assert result.success
    assert "sk-testsecret" not in result.content
    assert "[REDACTED]" in result.content


@pytest.mark.asyncio
async def test_tool_registry_separates_base_and_workspace_tools(tmp_path):
    config = _config(enable_bash=True, enable_file_tools=True, enable_note=True)
    messages: list[tuple[str, str]] = []
    registry = ToolRegistry(config, notify=lambda level, message: messages.append((level, message)))

    base = await registry.build_base_tools()
    tool_names = [tool.name for tool in base.tools]
    assert tool_names == ["bash_output", "bash_kill"]

    registry.add_workspace_tools(base.tools, Path(tmp_path))
    tool_names = [tool.name for tool in base.tools]

    assert "bash" in tool_names
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "edit_file" in tool_names
    assert "record_note" in tool_names
    assert "recall_notes" in tool_names
    assert any(message == "Loaded Bash security policy" for _level, message in messages)
