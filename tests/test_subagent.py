"""Tests for delegated subagent harness behavior."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_agent.config import AgentConfig, Config, LLMConfig, SubagentConfig, ToolsConfig
from mini_agent.llm import LLMClient
from mini_agent.schema import LLMResponse
from mini_agent.subagent import SubagentResult, SubagentRunner
from mini_agent.tool_registry import ToolRegistry
from mini_agent.tools.base import Tool, ToolResult
from mini_agent.tools.subagent_tool import SubagentTool


class NamedDummyTool(Tool):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} test tool"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self) -> ToolResult:
        return ToolResult(success=True, content="ok")


class FakeRunner:
    async def run(self, description: str, prompt: str) -> SubagentResult:
        return SubagentResult(content=f"{description}: {prompt}", completed=True, total_tokens=10)


def _mock_llm(response: str = "child summary") -> LLMClient:
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate = AsyncMock(return_value=LLMResponse(content=response, finish_reason="stop"))
    return llm_client


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
    return Config(
        llm=LLMConfig(api_key="test-key"),
        agent=AgentConfig(),
        tools=ToolsConfig(**tool_values),
    )


@pytest.mark.asyncio
async def test_subagent_runner_uses_isolated_messages_and_filtered_tools(tmp_path):
    llm_client = _mock_llm()
    config = SubagentConfig(
        allowed_tools=["read_file", "task", "dummy"],
        allow_nested_subagent=False,
    )
    runner = SubagentRunner(
        llm_client=llm_client,
        workspace_dir=tmp_path,
        tools_provider=lambda: [
            NamedDummyTool("dummy"),
            NamedDummyTool("task"),
            NamedDummyTool("write_file"),
        ],
        config=config,
    )

    result = await runner.run(description="inspect area", prompt="find the useful files")

    assert result.completed
    assert result.content == "child summary"
    call_kwargs = llm_client.generate.await_args.kwargs
    sent_messages = call_kwargs["messages"]
    sent_tools = call_kwargs["tools"]
    assert [tool.name for tool in sent_tools] == ["dummy"]
    assert [message.role for message in sent_messages] == ["system", "user"]
    assert "inspect area" in sent_messages[-1].content
    assert "find the useful files" in sent_messages[-1].content


@pytest.mark.asyncio
async def test_subagent_tool_returns_summary_as_tool_result():
    tool = SubagentTool(FakeRunner())

    result = await tool.execute(description="analyze", prompt="look around")

    assert result.success
    assert "Subagent status: completed" in result.content
    assert "analyze: look around" in result.content


@pytest.mark.asyncio
async def test_tool_registry_registers_subagent_tool_when_enabled(tmp_path):
    config = _config(enable_subagent=True)
    tools: list[Tool] = []

    ToolRegistry(config).add_workspace_tools(tools, tmp_path, llm_client=_mock_llm())

    assert [tool.name for tool in tools] == ["task"]
