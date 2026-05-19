from mini_agent.request_context import RequestContextBuilder
from mini_agent.schema import FunctionCall, Message, ToolCall
from mini_agent.summarizer import (
    CONTEXT_COLLAPSE_MESSAGE_NAME,
    CONTEXT_SNIP_MESSAGE_NAME,
    HARNESS_SUMMARY_HEADER,
    HARNESS_SUMMARY_MESSAGE_NAME,
)


def test_request_context_rebuilds_system_prompt_from_memory(tmp_path):
    memory_dir = tmp_path / ".memory"
    memory_dir.mkdir()
    (memory_dir / "project-note.md").write_text(
        "---\ntype: project\nupdated_at: 2026-05-17T00:00:00\n---\nUse pytest for verification.",
        encoding="utf-8",
    )
    builder = RequestContextBuilder(core_prompt="System", workspace_dir=tmp_path)

    request = builder.build([Message(role="system", content="old"), Message(role="user", content="hi")])

    assert request[0].role == "system"
    assert "Use pytest for verification." in request[0].content
    assert request[1:] == [Message(role="user", content="hi")]


def test_request_context_sanitizes_thinking_and_trims_history(tmp_path):
    builder = RequestContextBuilder(core_prompt="System", workspace_dir=tmp_path, max_recent_messages=3)
    messages = [
        Message(role="system", content="old"),
        Message(role="user", content="old question"),
        Message(role="assistant", content="old answer", thinking="hidden"),
        Message(role="user", content="new question"),
        Message(role="assistant", content="new answer", thinking="hidden"),
    ]

    request = builder.build(messages)

    assert [message.content for message in request[1:]] == ["old answer", "new question", "new answer"]
    assert all(message.thinking is None for message in request)


def test_request_context_preserves_active_tool_chain(tmp_path):
    builder = RequestContextBuilder(core_prompt="System", workspace_dir=tmp_path, max_recent_messages=2)
    tool_call = ToolCall(
        id="call-1",
        type="function",
        function=FunctionCall(name="write_file", arguments={"path": "x", "content": "y"}),
    )
    messages = [
        Message(role="system", content="old"),
        Message(role="user", content="old question"),
        Message(role="assistant", content="old answer"),
        Message(role="user", content="current question"),
        Message(role="assistant", content="", tool_calls=[tool_call], thinking="hidden"),
        Message(role="tool", content="wrote file", tool_call_id="call-1"),
    ]

    request = builder.build(messages)

    assert [message.role for message in request] == ["system", "user", "assistant", "tool"]
    assert request[1].content == "current question"
    assert request[2].tool_calls == [tool_call]
    assert request[3].tool_call_id == "call-1"


def test_request_context_injects_harness_summary_into_system_prompt(tmp_path):
    builder = RequestContextBuilder(core_prompt="System", workspace_dir=tmp_path)
    messages = [
        Message(role="system", content="old"),
        Message(role="user", content="original question"),
        Message(
            role="system",
            content=f"{HARNESS_SUMMARY_HEADER}\n\nRead files and found the failing test.",
            name=HARNESS_SUMMARY_MESSAGE_NAME,
        ),
        Message(role="user", content="follow up"),
    ]

    request = builder.build(messages)

    assert "## Harness Summary" in request[0].content
    assert "Read files and found the failing test." in request[0].content
    assert [message.content for message in request[1:]] == ["original question", "follow up"]


def test_request_context_preserves_snip_boundary_as_history_message(tmp_path):
    builder = RequestContextBuilder(core_prompt="System", workspace_dir=tmp_path, max_recent_messages=1)
    messages = [
        Message(role="system", content="old"),
        Message(
            role="system",
            content="[Context Snipped: 4 older messages removed, approximately 100 tokens freed. Earlier context is unavailable.]",
            name=CONTEXT_SNIP_MESSAGE_NAME,
        ),
        Message(role="user", content="follow up"),
    ]

    request = builder.build(messages)

    assert "## Harness Summary" not in request[0].content
    assert [message.name for message in request[1:]] == [CONTEXT_SNIP_MESSAGE_NAME, None]
    assert request[1].role == "system"
    assert request[2].content == "follow up"


def test_request_context_preserves_context_collapse_boundary_as_history_message(tmp_path):
    builder = RequestContextBuilder(core_prompt="System", workspace_dir=tmp_path, max_recent_messages=1)
    messages = [
        Message(role="system", content="old"),
        Message(
            role="system",
            content="[Context Collapsed: 4 older messages hidden for this API call only.]",
            name=CONTEXT_COLLAPSE_MESSAGE_NAME,
        ),
        Message(role="user", content="follow up"),
    ]

    request = builder.build(messages)

    assert "## Harness Summary" not in request[0].content
    assert [message.name for message in request[1:]] == [CONTEXT_COLLAPSE_MESSAGE_NAME, None]
    assert request[1].role == "system"
    assert request[2].content == "follow up"


def test_request_context_uses_token_budget_for_recent_messages(tmp_path):
    builder = RequestContextBuilder(core_prompt="System", workspace_dir=tmp_path, max_recent_messages=10, token_budget=45)
    messages = [
        Message(role="system", content="old"),
        Message(role="user", content="old question " + "alpha " * 80),
        Message(role="assistant", content="old answer " + "beta " * 80),
        Message(role="user", content="latest question"),
    ]

    request = builder.build(messages, token_budget=45)

    assert [message.content for message in request[1:]] == ["latest question"]


def test_request_context_preserves_active_tool_chain_over_token_budget(tmp_path):
    builder = RequestContextBuilder(core_prompt="System", workspace_dir=tmp_path, max_recent_messages=10, token_budget=20)
    tool_call = ToolCall(
        id="call-1",
        type="function",
        function=FunctionCall(name="read_file", arguments={"path": "large.txt"}),
    )
    messages = [
        Message(role="system", content="old"),
        Message(role="user", content="old question " + "alpha " * 80),
        Message(role="assistant", content="old answer " + "beta " * 80),
        Message(role="user", content="current question"),
        Message(role="assistant", content="", tool_calls=[tool_call]),
        Message(role="tool", content="large tool output " + "gamma " * 80, tool_call_id="call-1"),
    ]

    request = builder.build(messages, token_budget=20)

    assert [message.role for message in request] == ["system", "user", "assistant", "tool"]
    assert request[1].content == "current question"
    assert request[2].tool_calls == [tool_call]
    assert request[3].tool_call_id == "call-1"
