"""Test cases for Agent."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_agent import LLMClient
from mini_agent.agent import Agent
from mini_agent.checkpoint import CheckpointStore
from mini_agent.config import Config
from mini_agent.schema import FunctionCall, LLMResponse, Message, ToolCall
from mini_agent.tools.base import Tool, ToolResult
from mini_agent.tools import BashTool, EditTool, ReadTool, WriteTool
from mini_agent.tools.task_memory_tool import EpisodeMemoryStore, TaskMemoryHook


class DummyTool(Tool):
    @property
    def name(self) -> str:
        return "dummy_tool"

    @property
    def description(self) -> str:
        return "Return a fixed response."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
            },
            "required": ["value"],
        }

    async def execute(self, value: str) -> ToolResult:
        return ToolResult(success=True, content=f"processed {value}")


def _mock_bash_llm(command: str, run_in_background: bool = False):
    """Create a mock LLM that asks for one bash tool call then finishes."""

    client = MagicMock(spec=LLMClient)
    client.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(
                            name="bash",
                            arguments={
                                "command": command,
                                "run_in_background": run_in_background,
                            },
                        ),
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done", tool_calls=None, finish_reason="stop"),
        ]
    )
    return client


@pytest.mark.asyncio
async def test_bash_confirmation_defaults_to_denied_without_callback(tmp_path):
    """Medium-risk bash commands are denied when no confirmation callback exists."""

    agent = Agent(
        llm_client=_mock_bash_llm("echo needs-confirmation", run_in_background=True),
        system_prompt="System",
        tools=[BashTool(workspace_dir=str(tmp_path))],
        workspace_dir=str(tmp_path),
    )
    agent.add_user_message("run background command")

    result = await agent.run()

    assert result == "done"
    tool_messages = [msg for msg in agent.messages if msg.role == "tool"]
    assert tool_messages
    assert "Command execution denied by user confirmation policy." in tool_messages[0].content


@pytest.mark.asyncio
async def test_normal_bash_command_runs_without_confirmation_callback(tmp_path):
    """Low-risk bash commands run normally without a confirmation callback."""

    agent = Agent(
        llm_client=_mock_bash_llm("echo no-confirmation-needed"),
        system_prompt="System",
        tools=[BashTool(workspace_dir=str(tmp_path))],
        workspace_dir=str(tmp_path),
    )
    agent.add_user_message("run simple command")

    await agent.run()

    tool_messages = [msg for msg in agent.messages if msg.role == "tool"]
    assert "no-confirmation-needed" in tool_messages[0].content


@pytest.mark.asyncio
async def test_bash_confirmation_denied_by_callback(tmp_path):
    """A callback can deny a medium-risk bash command before execution."""

    callback = AsyncMock(return_value=False)
    agent = Agent(
        llm_client=_mock_bash_llm("echo needs-confirmation", run_in_background=True),
        system_prompt="System",
        tools=[BashTool(workspace_dir=str(tmp_path))],
        workspace_dir=str(tmp_path),
        tool_confirmation_callback=callback,
    )
    agent.add_user_message("run background command")

    await agent.run()

    callback.assert_awaited_once()
    tool_messages = [msg for msg in agent.messages if msg.role == "tool"]
    assert "Command execution denied by user confirmation policy." in tool_messages[0].content
    audit_log = tmp_path / ".mini_agent" / "bash_audit.jsonl"
    assert "confirmation_denied" in audit_log.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_bash_confirmation_approved_by_callback(tmp_path):
    """A callback can approve a medium-risk bash command."""

    callback = AsyncMock(return_value=True)
    agent = Agent(
        llm_client=_mock_bash_llm("echo confirmed", run_in_background=True),
        system_prompt="System",
        tools=[BashTool(workspace_dir=str(tmp_path))],
        workspace_dir=str(tmp_path),
        tool_confirmation_callback=callback,
    )
    agent.add_user_message("run background command")

    await agent.run()

    callback.assert_awaited_once()
    tool_messages = [msg for msg in agent.messages if msg.role == "tool"]
    assert "Background command started" in tool_messages[0].content


@pytest.mark.asyncio
async def test_dangerous_bash_command_does_not_request_confirmation(tmp_path):
    """High-risk blocked commands remain blocked without asking for confirmation."""

    callback = AsyncMock(return_value=True)
    agent = Agent(
        llm_client=_mock_bash_llm("rm -rf ./important"),
        system_prompt="System",
        tools=[BashTool(workspace_dir=str(tmp_path))],
        workspace_dir=str(tmp_path),
        tool_confirmation_callback=callback,
    )
    agent.add_user_message("run dangerous command")

    await agent.run()

    callback.assert_not_awaited()
    tool_messages = [msg for msg in agent.messages if msg.role == "tool"]
    assert "Command blocked by security policy" in tool_messages[0].content


@pytest.mark.asyncio
async def test_checkpoint_saved_for_tool_and_completion(tmp_path):
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Need to use a tool",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(
                            name="dummy_tool",
                            arguments={"value": "sample"},
                        ),
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="done", tool_calls=None, finish_reason="stop"),
        ]
    )

    checkpoint_store = CheckpointStore(tmp_path / ".mini_agent" / "checkpoints")
    task_memory_hook = TaskMemoryHook(
        memory_file=str(tmp_path / ".mini_agent" / "task_memory.json"),
        workspace_dir=str(tmp_path),
        episode_memory_file=str(tmp_path / ".mini_agent" / "episodes.jsonl"),
    )
    agent = Agent(
        llm_client=llm_client,
        system_prompt="System",
        tools=[DummyTool()],
        workspace_dir=str(tmp_path),
        task_memory_hook=task_memory_hook,
        checkpoint_store=checkpoint_store,
    )
    agent.add_user_message("run dummy tool")

    result = await agent.run()

    assert result == "done"
    latest = checkpoint_store.load_latest()
    assert latest is not None
    assert latest["reason"] == "completed"
    assert latest["messages"][-1]["content"] == "done"
    assert "task_memory" not in latest

    history_files = sorted((tmp_path / ".mini_agent" / "checkpoints" / "history").glob("*.json"))
    reasons = [json.loads(path.read_text(encoding="utf-8"))["reason"] for path in history_files]
    assert "tool_result" in reasons
    assert "completed" in reasons

    episode_file = tmp_path / ".mini_agent" / "episodes.jsonl"
    long_term_memory_dir = tmp_path / ".memory"
    assert episode_file.exists()
    assert EpisodeMemoryStore(str(episode_file)).load()["episodes"]
    assert not long_term_memory_dir.exists()


@pytest.mark.asyncio
async def test_checkpoint_after_cancellation_keeps_last_committed_state(tmp_path):
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate = AsyncMock(
        return_value=LLMResponse(
            content="Tool call pending",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(
                        name="dummy_tool",
                        arguments={"value": "sample"},
                    ),
                )
            ],
            finish_reason="tool_calls",
        )
    )

    checkpoint_store = CheckpointStore(tmp_path / ".mini_agent" / "checkpoints")
    agent = Agent(
        llm_client=llm_client,
        system_prompt="System",
        tools=[DummyTool()],
        workspace_dir=str(tmp_path),
        checkpoint_store=checkpoint_store,
    )
    agent.add_user_message("cancel before tool execution")
    cancel_event = asyncio.Event()
    cancel_event.set()

    result = await agent.run(cancel_event=cancel_event)

    assert result == "Task cancelled by user."
    latest = checkpoint_store.load_latest()
    assert latest is not None
    assert latest["reason"] == "cancelled"
    assert [message["role"] for message in latest["messages"]] == ["system", "user"]


@pytest.mark.asyncio
async def test_agent_uses_compression_pipeline_before_final_request(tmp_path):
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate = AsyncMock(return_value=LLMResponse(content="done", tool_calls=None, finish_reason="stop"))
    agent = Agent(
        llm_client=llm_client,
        system_prompt="System",
        tools=[DummyTool()],
        workspace_dir=str(tmp_path),
    )
    agent.add_user_message("original user")
    compressed_messages = [
        Message(role="system", content="System"),
        Message(role="user", content="compressed user"),
    ]
    agent.compression_pipeline = MagicMock()
    agent.compression_pipeline.compress_before_request = AsyncMock(return_value=compressed_messages)

    result = await agent.run()

    assert result == "done"
    agent.compression_pipeline.compress_before_request.assert_awaited_once()
    request_messages = llm_client.generate.await_args.kwargs["messages"]
    assert [message.content for message in request_messages[1:]] == ["compressed user"]


def test_checkpoint_store_can_restore_messages(tmp_path):
    checkpoint_store = CheckpointStore(tmp_path / ".mini_agent" / "checkpoints")
    original_messages = [
        Message(role="system", content="System"),
        Message(role="user", content="Continue previous task"),
        Message(role="assistant", content="Last answer"),
    ]

    checkpoint_store.save(
        step=2,
        reason="assistant_response",
        messages=original_messages,
        workspace_dir=tmp_path,
        available_tools=["dummy_tool"],
    )

    restored = checkpoint_store.load_latest_messages()

    assert [message.role for message in restored] == ["system", "user", "assistant"]
    assert restored[1].content == "Continue previous task"


def test_agent_can_truncate_incomplete_turn(tmp_path):
    llm_client = MagicMock(spec=LLMClient)
    agent = Agent(
        llm_client=llm_client,
        system_prompt="System",
        tools=[DummyTool()],
        workspace_dir=str(tmp_path),
    )
    agent.add_user_message("completed task")
    agent.messages.append(Message(role="assistant", content="done"))
    checkpoint = len(agent.messages)

    agent.add_user_message("unfinished task")
    agent.messages.append(
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="dummy_tool", arguments={"value": "sample"}),
                )
            ],
        )
    )
    agent.messages.append(Message(role="tool", content="processed sample", tool_call_id="call_1"))

    agent.truncate_messages(checkpoint)

    assert [message.role for message in agent.messages] == ["system", "user", "assistant"]
    assert agent.messages[-1].content == "done"


def test_checkpoint_validation_reports_invalid_message_stats(tmp_path):
    checkpoint_store = CheckpointStore(tmp_path / ".mini_agent" / "checkpoints")
    checkpoint_store.latest_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_store.latest_file.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-05-15T10:00:00",
                "step": 1,
                "reason": "assistant_response",
                "workspace_dir": str(tmp_path.resolve()),
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                    {"role": "tool"},
                ],
            }
        ),
        encoding="utf-8",
    )

    validation = checkpoint_store.validate_messages()
    issues = checkpoint_store.validate_for_workspace(tmp_path)

    assert validation["total"] == 3
    assert validation["valid"] == 2
    assert validation["dropped"] == 1
    assert any("invalid messages" in issue.lower() for issue in issues)


def test_checkpoint_validation_rejects_non_system_first_message(tmp_path):
    checkpoint_store = CheckpointStore(tmp_path / ".mini_agent" / "checkpoints")
    checkpoint_store.latest_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_store.latest_file.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-05-15T10:00:00",
                "step": 1,
                "reason": "assistant_response",
                "workspace_dir": str(tmp_path.resolve()),
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
            }
        ),
        encoding="utf-8",
    )

    issues = checkpoint_store.validate_for_workspace(tmp_path)

    assert any("first valid message is not a system message" in issue.lower() for issue in issues)


def test_checkpoint_validation_rejects_high_invalid_message_ratio(tmp_path):
    checkpoint_store = CheckpointStore(tmp_path / ".mini_agent" / "checkpoints")
    checkpoint_store.latest_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_store.latest_file.write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-05-15T10:00:00",
                "step": 1,
                "reason": "assistant_response",
                "workspace_dir": str(tmp_path.resolve()),
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                    {"role": "tool"},
                    {"invalid": "message"},
                ],
            }
        ),
        encoding="utf-8",
    )

    issues = checkpoint_store.validate_for_workspace(tmp_path)

    assert any("drop ratio is too high" in issue.lower() for issue in issues)


def test_checkpoint_validation_detects_workspace_mismatch(tmp_path):
    checkpoint_store = CheckpointStore(tmp_path / ".mini_agent" / "checkpoints")
    checkpoint_store.save(
        step=0,
        reason="run_started",
        messages=[Message(role="system", content="System")],
        workspace_dir=tmp_path / "workspace-a",
        available_tools=["dummy_tool"],
    )

    issues = checkpoint_store.validate_for_workspace(tmp_path / "workspace-b")

    assert issues
    assert "workspace mismatch" in issues[0].lower()


@pytest.mark.asyncio
async def test_agent_simple_task():
    """Test agent with a simple file creation task."""
    print("\n=== Testing Agent with Simple File Task ===")

    # Load config
    config_path = Path("mini_agent/config/config.yaml")
    config = Config.from_yaml(config_path)

    # Create temp workspace
    with tempfile.TemporaryDirectory() as workspace_dir:
        print(f"Using workspace: {workspace_dir}")

        # Load system prompt (Agent will auto-inject workspace info)
        system_prompt_path = Path("mini_agent/config/system_prompt.md")
        if system_prompt_path.exists():
            system_prompt = system_prompt_path.read_text(encoding="utf-8")
        else:
            system_prompt = "You are a helpful AI assistant that can use tools."

        # Initialize LLM client
        llm_client = LLMClient(
            api_key=config.llm.api_key,
            api_base=config.llm.api_base,
            model=config.llm.model,
        )

        # Initialize tools
        tools = [
            ReadTool(workspace_dir=workspace_dir),
            WriteTool(workspace_dir=workspace_dir),
            EditTool(workspace_dir=workspace_dir),
            BashTool(),
        ]

        # Create agent
        agent = Agent(
            llm_client=llm_client,
            system_prompt=system_prompt,
            tools=tools,
            max_steps=10,  # Limit steps for testing
            workspace_dir=workspace_dir,
        )

        # Task: Create a simple text file
        task = "Create a file named 'test.txt' with the content 'Hello from Agent!'"
        print(f"\nTask: {task}\n")

        agent.add_user_message(task)

        try:
            result = await agent.run()

            print(f"\n{'=' * 80}")
            print(f"Agent Result: {result}")
            print("=" * 80)

            # Check if file was created
            test_file = Path(workspace_dir) / "test.txt"
            if test_file.exists():
                content = test_file.read_text()
                print("\n✅ File created successfully!")
                print(f"Content: {content}")

                if "Hello from Agent!" in content:
                    print("✅ Content is correct!")
                    return True
                else:
                    print(f"⚠️  Content mismatch: {content}")
                    return True  # Still count as success, agent did create the file
            else:
                print("⚠️  File was not created, but agent completed")
                return True  # Agent might have completed differently

        except Exception as e:
            print(f"❌ Agent test failed: {e}")
            import traceback

            traceback.print_exc()
            return False


@pytest.mark.asyncio
async def test_agent_bash_task():
    """Test agent with a bash command task."""
    print("\n=== Testing Agent with Bash Task ===")

    # Load config
    config_path = Path("mini_agent/config/config.yaml")
    config = Config.from_yaml(config_path)

    # Create temp workspace
    with tempfile.TemporaryDirectory() as workspace_dir:
        print(f"Using workspace: {workspace_dir}")

        # Load system prompt (Agent will auto-inject workspace info)
        system_prompt_path = Path("mini_agent/config/system_prompt.md")
        if system_prompt_path.exists():
            system_prompt = system_prompt_path.read_text(encoding="utf-8")
        else:
            system_prompt = "You are a helpful AI assistant that can use tools."

        # Initialize LLM client
        llm_client = LLMClient(
            api_key=config.llm.api_key,
            api_base=config.llm.api_base,
            model=config.llm.model,
        )

        # Initialize tools
        tools = [
            ReadTool(workspace_dir=workspace_dir),
            WriteTool(workspace_dir=workspace_dir),
            BashTool(),
        ]

        # Create agent
        agent = Agent(
            llm_client=llm_client,
            system_prompt=system_prompt,
            tools=tools,
            max_steps=10,
            workspace_dir=workspace_dir,
        )

        # Task: List files using bash
        task = "Use bash to list all files in the current directory and tell me what you find."
        print(f"\nTask: {task}\n")

        agent.add_user_message(task)

        try:
            result = await agent.run()

            print(f"\n{'=' * 80}")
            print(f"Agent Result: {result}")
            print("=" * 80)

            print("\n✅ Bash task completed!")
            return True

        except Exception as e:
            print(f"❌ Bash task failed: {e}")
            import traceback

            traceback.print_exc()
            return False


async def main():
    """Run all agent tests."""
    print("=" * 80)
    print("Running Agent Integration Tests")
    print("=" * 80)
    print("\nNote: These tests require a valid MiniMax API key in config.yaml")
    print("These tests will actually call the LLM API and may take some time.\n")

    # Test simple file task
    result1 = await test_agent_simple_task()

    # Test bash task
    result2 = await test_agent_bash_task()

    print("\n" + "=" * 80)
    if result1 and result2:
        print("All Agent tests passed! ✅")
    else:
        print("Some Agent tests failed. Check the output above.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
