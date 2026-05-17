"""Tests for harness runtime boundaries."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from mini_agent.config import AgentConfig, Config, LLMConfig, ToolsConfig
from mini_agent.runtime import RunContext, ToolExecutionRequest, ToolRuntime
from mini_agent.tool_registry import ToolRegistry
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
