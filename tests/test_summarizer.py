from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_agent.llm import LLMClient
from mini_agent.schema import FunctionCall, LLMResponse, Message, ToolCall
from mini_agent.summarizer import CONTEXT_SNIP_MESSAGE_NAME, HARNESS_SUMMARY_MESSAGE_NAME, MessageSummarizer


@pytest.mark.asyncio
async def test_summarizer_writes_harness_summary_as_system_message():
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate = AsyncMock(return_value=LLMResponse(content="Compressed execution.", finish_reason="stop"))
    summarizer = MessageSummarizer(llm_client=llm_client, token_limit=1)
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="Do work"),
        Message(role="assistant", content="I read the file."),
    ]

    compacted = await summarizer.summarize_if_needed(messages, api_total_tokens=0)

    assert compacted[0] == messages[0]
    assert compacted[1] == messages[1]
    assert compacted[2].role == "system"
    assert compacted[2].name == HARNESS_SUMMARY_MESSAGE_NAME
    assert "Compressed execution." in compacted[2].content


@pytest.mark.asyncio
async def test_summarizer_preserves_active_tool_round_raw():
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate = AsyncMock(return_value=LLMResponse(content="Old round summary.", finish_reason="stop"))
    summarizer = MessageSummarizer(llm_client=llm_client, token_limit=1)
    tool_call = ToolCall(
        id="call-1",
        type="function",
        function=FunctionCall(name="read_file", arguments={"path": "a.py"}),
    )
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="Old request"),
        Message(role="assistant", content="Old work done."),
        Message(role="user", content="Current request"),
        Message(role="assistant", content="", tool_calls=[tool_call]),
        Message(role="tool", content="file content", tool_call_id="call-1", name="read_file"),
    ]

    compacted = await summarizer.summarize_if_needed(messages, api_total_tokens=0)

    assert [message.content for message in compacted[-3:]] == ["Current request", "", "file content"]
    assert compacted[-2].tool_calls == [tool_call]
    assert compacted[-1].tool_call_id == "call-1"


@pytest.mark.asyncio
async def test_summarizer_preserves_snip_boundary_when_nothing_can_be_summarized():
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate = AsyncMock()
    summarizer = MessageSummarizer(llm_client=llm_client, token_limit=1)
    messages = [
        Message(role="system", content="System"),
        Message(
            role="system",
            content="[Context Snipped: 2 older messages removed, approximately 100 tokens freed. Earlier context is unavailable.]",
            name=CONTEXT_SNIP_MESSAGE_NAME,
        ),
        Message(role="user", content="Current request"),
    ]

    compacted = await summarizer.summarize_if_needed(messages, api_total_tokens=0)

    assert compacted == messages
    llm_client.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_summarizer_full_history_fallback_handles_prefix_without_old_user_turns():
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate = AsyncMock(return_value=LLMResponse(content="Recovered checkpoint summary.", finish_reason="stop"))
    summarizer = MessageSummarizer(llm_client=llm_client, token_limit=1)
    tool_call = ToolCall(
        id="call-1",
        type="function",
        function=FunctionCall(name="read_file", arguments={"path": "a.py"}),
    )
    messages = [
        Message(role="system", content="System"),
        Message(role="assistant", content="Restored checkpoint context without a user turn."),
        Message(role="user", content="Current request"),
        Message(role="assistant", content="", tool_calls=[tool_call]),
        Message(role="tool", content="file content", tool_call_id="call-1", name="read_file"),
    ]

    compacted = await summarizer.summarize_if_needed(messages, api_total_tokens=0)

    assert compacted[0] == messages[0]
    assert compacted[1].role == "system"
    assert compacted[1].name == HARNESS_SUMMARY_MESSAGE_NAME
    assert "Recovered checkpoint summary." in compacted[1].content
    assert [message.content for message in compacted[-3:]] == ["Current request", "", "file content"]
    assert compacted[-2].tool_calls == [tool_call]
    llm_client.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_summarizer_full_history_fallback_uses_one_global_summary():
    llm_client = MagicMock(spec=LLMClient)
    llm_client.generate = AsyncMock(return_value=LLMResponse(content="Global history summary.", finish_reason="stop"))
    summarizer = MessageSummarizer(llm_client=llm_client, token_limit=1)
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="Old request 1"),
        Message(role="assistant", content="Old work 1"),
        Message(role="user", content="Old request 2"),
        Message(role="assistant", content="Old work 2"),
        Message(role="user", content="Current request"),
    ]

    compacted = await summarizer.summarize_if_needed(messages, api_total_tokens=0)

    assert llm_client.generate.await_count == 1
    assert [message.role for message in compacted] == ["system", "system", "user"]
    assert compacted[1].name == HARNESS_SUMMARY_MESSAGE_NAME
    assert "Global history summary." in compacted[1].content
    assert compacted[-1].content == "Current request"
