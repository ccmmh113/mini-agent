from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_agent.llm import LLMClient
from mini_agent.schema import FunctionCall, LLMResponse, Message, ToolCall
from mini_agent.summarizer import HARNESS_SUMMARY_MESSAGE_NAME, MessageSummarizer


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
