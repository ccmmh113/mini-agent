from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_agent.schema import FunctionCall, Message, ToolCall
from mini_agent.summarizer import (
    CONTEXT_COLLAPSE_MESSAGE_NAME,
    CONTEXT_SNIP_MESSAGE_NAME,
    HARNESS_SUMMARY_HEADER,
    CompactionResult,
    CompressionPipeline,
    ContextCollapser,
    MessageCompactor,
)


def _tool_call(call_id: str, name: str = "read_file", arguments: dict | None = None) -> ToolCall:
    return ToolCall(
        id=call_id,
        type="function",
        function=FunctionCall(name=name, arguments=arguments or {"path": "large.txt"}),
    )


def _tool_round(
    call_id: str,
    content: str,
    *,
    name: str = "read_file",
    final_assistant: bool = True,
) -> list[Message]:
    messages = [
        Message(role="assistant", content="", tool_calls=[_tool_call(call_id, name=name)]),
        Message(role="tool", content=content, tool_call_id=call_id, name=name),
    ]
    if final_assistant:
        messages.append(Message(role="assistant", content=f"Observed {call_id}."))
    return messages


def _tool_contents(messages: list[Message]) -> list[str]:
    return [message.content for message in messages if message.role == "tool" and isinstance(message.content, str)]


def test_tool_result_budget_spills_large_result_and_preserves_tool_identity(tmp_path: Path):
    large_content = "\n".join(f"line {index}: {'x' * 80}" for index in range(80))
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="Read a large file."),
        *_tool_round("call-1", large_content),
    ]
    compactor = MessageCompactor(
        token_limit=10_000,
        workspace_dir=tmp_path,
        tool_result_soft_byte_limit=1_000,
        tool_round_total_byte_limit=200_000,
    )

    result = compactor.compact(messages)

    tool_messages = [message for message in result.messages if message.role == "tool"]
    assert len(tool_messages) == 1
    compacted_tool = tool_messages[0]
    assert compacted_tool.tool_call_id == "call-1"
    assert compacted_tool.name == "read_file"
    assert "[Tool result stored on disk:" in compacted_tool.content
    assert "read_file" in compacted_tool.content
    assert "offset" in compacted_tool.content
    assert "limit" in compacted_tool.content

    spill_files = list((tmp_path / ".mini_agent" / "tool-results").glob("*.txt"))
    assert len(spill_files) == 1
    assert spill_files[0].read_text(encoding="utf-8") == large_content
    assert result.tool_results_spilled == 1


def test_tool_round_total_budget_spills_largest_results_until_under_limit(tmp_path: Path):
    outputs = {
        "small": "s" * 80,
        "medium": "m" * 100,
        "large": "l" * 150,
    }
    tool_calls = [_tool_call("small"), _tool_call("medium"), _tool_call("large")]
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="Run several tools."),
        Message(role="assistant", content="", tool_calls=tool_calls),
        Message(role="tool", content=outputs["small"], tool_call_id="small", name="read_file"),
        Message(role="tool", content=outputs["medium"], tool_call_id="medium", name="read_file"),
        Message(role="tool", content=outputs["large"], tool_call_id="large", name="read_file"),
        Message(role="assistant", content="Done."),
    ]
    compactor = MessageCompactor(
        token_limit=10_000,
        workspace_dir=tmp_path,
        tool_result_soft_byte_limit=10_000,
        tool_round_total_byte_limit=200,
    )

    result = compactor.compact(messages)

    tool_messages = [message for message in result.messages if message.role == "tool"]
    assert "[Tool result stored on disk:" in tool_messages[2].content
    assert tool_messages[0].content == outputs["small"]
    assert tool_messages[1].content == outputs["medium"]
    remaining_bytes = sum(
        len(message.content.encode("utf-8"))
        for message in tool_messages
        if isinstance(message.content, str) and not message.content.startswith("[Tool result stored on disk:")
    )
    assert remaining_bytes <= 200
    assert result.tool_results_spilled == 1


def test_compactor_preserves_active_tool_chain_raw(tmp_path: Path):
    large_content = "active-result " * 1_000
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="Current request"),
        *_tool_round("call-active", large_content, final_assistant=False),
    ]
    compactor = MessageCompactor(
        token_limit=20,
        workspace_dir=tmp_path,
        tool_result_soft_byte_limit=100,
        tool_round_total_byte_limit=200,
    )

    result = compactor.compact(messages)

    assert result.messages == messages
    assert not (tmp_path / ".mini_agent" / "tool-results").exists()


def test_snip_removes_old_block_and_inserts_boundary_without_summary(tmp_path: Path):
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="old question 1 " + "alpha " * 80),
        Message(role="assistant", content="old answer 1 " + "beta " * 80),
        Message(role="user", content="old question 2 " + "gamma " * 80),
        Message(role="assistant", content="old answer 2 " + "delta " * 80),
        Message(role="user", content="latest question"),
    ]
    compactor = MessageCompactor(
        token_limit=80,
        workspace_dir=tmp_path,
        tool_result_soft_byte_limit=100_000,
        tool_round_total_byte_limit=200_000,
    )

    result = compactor.compact(messages)

    assert result.snipped_messages > 0
    assert result.snip_tokens_freed > 0
    marker_messages = [message for message in result.messages if message.name == CONTEXT_SNIP_MESSAGE_NAME]
    assert len(marker_messages) == 1
    assert "Context Snipped" in marker_messages[0].content
    assert HARNESS_SUMMARY_HEADER not in marker_messages[0].content
    assert result.messages[-1].content == "latest question"
    assert all("old answer 1" not in str(message.content) for message in result.messages)


def test_micro_compact_clips_only_compactable_tools_with_time_decay(tmp_path: Path):
    old = "old-token " * 400
    middle = "middle-token " * 400
    newest = "newest-token " * 400
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="Current request"),
        *_tool_round("old", old),
        *_tool_round("middle", middle),
        *_tool_round("newest", newest),
    ]
    compactor = MessageCompactor(
        token_limit=60,
        workspace_dir=tmp_path,
        tool_result_soft_byte_limit=100_000,
        tool_round_total_byte_limit=200_000,
    )
    compactor.newest_tool_result_count = 1
    compactor.middle_tool_result_count = 1
    compactor.newest_tool_result_token_limit = 40
    compactor.middle_tool_result_token_limit = 20
    compactor.old_tool_result_token_limit = 8

    result = compactor.compact(messages)

    contents = _tool_contents(result.messages)
    assert all("[Old tool result content shortened:" in content for content in contents)
    assert all("Earlier content was removed from this prompt" in content for content in contents)
    assert len(contents[0]) < len(contents[1]) < len(contents[2])
    assert "tool=read_file" in contents[0]
    assert result.micro_compacted_results == 3


def test_micro_compact_leaves_non_compactable_tools_unchanged(tmp_path: Path):
    note_result = "note persisted " * 400
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="Current request"),
        *_tool_round("note", note_result, name="record_note"),
    ]
    compactor = MessageCompactor(
        token_limit=30,
        workspace_dir=tmp_path,
        tool_result_soft_byte_limit=100_000,
        tool_round_total_byte_limit=200_000,
    )

    result = compactor.compact(messages)

    assert _tool_contents(result.messages) == [note_result]
    assert result.micro_compacted_results == 0


class EchoRequestContextBuilder:
    def build(self, messages: list[Message], *, tools=None, token_budget=None) -> list[Message]:
        return messages


def test_context_collapse_projects_old_history_without_mutating_original():
    old_user = Message(role="user", content="old request " + "alpha " * 100)
    old_assistant = Message(role="assistant", content="old answer " + "beta " * 100)
    messages = [
        Message(role="system", content="System"),
        old_user,
        old_assistant,
        Message(role="user", content="latest request"),
    ]
    original_contents = [message.content for message in messages]
    collapser = ContextCollapser(token_limit=120, collapse_ratio=0.1, emergency_ratio=0.2)

    result = collapser.apply_collapses_if_needed(messages, request_tokens=50)

    assert [message.content for message in messages] == original_contents
    assert result.collapsed_messages == 2
    assert any(message.name == CONTEXT_COLLAPSE_MESSAGE_NAME for message in result.messages)
    assert result.messages[-1].content == "latest request"
    assert not any(message.role == "assistant" and str(message.content).startswith("old answer") for message in result.messages)


def test_context_collapse_preserves_active_tool_chain():
    tool_call = _tool_call("active")
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="old request " + "alpha " * 100),
        Message(role="assistant", content="old answer " + "beta " * 100),
        Message(role="user", content="latest request"),
        Message(role="assistant", content="", tool_calls=[tool_call]),
        Message(role="tool", content="active result " * 100, tool_call_id="active", name="read_file"),
    ]
    collapser = ContextCollapser(token_limit=120, collapse_ratio=0.1, emergency_ratio=0.2)

    result = collapser.apply_collapses_if_needed(messages, request_tokens=50)

    assert result.messages[-3].content == "latest request"
    assert result.messages[-2].tool_calls == [tool_call]
    assert result.messages[-1].tool_call_id == "active"


@pytest.mark.asyncio
async def test_pipeline_skips_compaction_below_near_limit():
    compactor = MagicMock()
    summarizer = MagicMock()
    summarizer.summarize_if_needed = AsyncMock()
    pipeline = CompressionPipeline(
        compactor=compactor,
        context_collapser=MagicMock(),
        summarizer=summarizer,
        request_context_builder=EchoRequestContextBuilder(),
        token_limit=1_000,
    )
    messages = [Message(role="system", content="System"), Message(role="user", content="small")]

    compacted = await pipeline.compress_before_request(messages=messages, api_total_tokens=0, tools=[])

    assert compacted == messages
    compactor.compact.assert_not_called()
    summarizer.summarize_if_needed.assert_not_awaited()


@pytest.mark.asyncio
async def test_pipeline_uses_context_collapse_before_summarizing():
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="old " + "alpha " * 300),
        Message(role="assistant", content="old answer " + "beta " * 300),
        Message(role="user", content="latest"),
    ]
    summarizer = MagicMock()
    summarizer.summarize_if_needed = AsyncMock(return_value=[Message(role="system", content="summary")])
    compactor = MagicMock()
    compactor.compact.return_value = CompactionResult(messages=list(messages))
    pipeline = CompressionPipeline(
        compactor=compactor,
        context_collapser=ContextCollapser(token_limit=140, collapse_ratio=0.1, emergency_ratio=0.2),
        summarizer=summarizer,
        request_context_builder=EchoRequestContextBuilder(),
        token_limit=140,
        near_limit_ratio=0.1,
    )

    projected = await pipeline.compress_before_request(messages=messages, api_total_tokens=0, tools=[])

    assert projected is not messages
    assert messages[1].content.startswith("old alpha")
    assert any(message.name == CONTEXT_COLLAPSE_MESSAGE_NAME for message in projected)
    summarizer.summarize_if_needed.assert_not_awaited()


@pytest.mark.asyncio
async def test_pipeline_records_compression_stats(tmp_path: Path):
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="old " + "alpha " * 150),
        Message(role="assistant", content="old answer " + "beta " * 150),
        Message(role="user", content="latest"),
    ]
    summarizer = MagicMock()
    summarizer.summarize_if_needed = AsyncMock(return_value=[Message(role="system", content="summary")])
    pipeline = CompressionPipeline(
        compactor=MessageCompactor(token_limit=80, workspace_dir=tmp_path),
        context_collapser=ContextCollapser(token_limit=80),
        summarizer=summarizer,
        request_context_builder=EchoRequestContextBuilder(),
        token_limit=80,
    )

    compacted = await pipeline.compress_before_request(messages=messages, api_total_tokens=10_000, tools=[])

    assert compacted != messages
    assert pipeline.stats
    latest = pipeline.stats[-1]
    assert latest["compression_triggered"] is True
    assert latest["before_tokens"] > latest["after_tokens"]
    assert 0 < latest["compression_ratio"] < 1
    assert latest["before_message_count"] == len(messages)
    assert latest["after_message_count"] == len(compacted)


@pytest.mark.asyncio
async def test_pipeline_uses_post_compaction_estimate_before_summarizing(tmp_path: Path):
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="old " + "alpha " * 150),
        Message(role="assistant", content="old answer " + "beta " * 150),
        Message(role="user", content="latest"),
    ]
    summarizer = MagicMock()
    summarizer.summarize_if_needed = AsyncMock(return_value=[Message(role="system", content="summary")])
    pipeline = CompressionPipeline(
        compactor=MessageCompactor(token_limit=80, workspace_dir=tmp_path),
        context_collapser=ContextCollapser(token_limit=80),
        summarizer=summarizer,
        request_context_builder=EchoRequestContextBuilder(),
        token_limit=80,
    )

    compacted = await pipeline.compress_before_request(messages=messages, api_total_tokens=10_000, tools=[])

    assert compacted != messages
    assert any(message.name == CONTEXT_SNIP_MESSAGE_NAME for message in compacted)
    summarizer.summarize_if_needed.assert_not_awaited()


@pytest.mark.asyncio
async def test_pipeline_calls_summarizer_when_deterministic_compaction_is_insufficient(tmp_path: Path):
    messages = [
        Message(role="system", content="System"),
        Message(role="user", content="latest"),
        Message(role="assistant", content="still too large " + "zeta " * 400),
    ]
    summarized = [Message(role="system", content="System"), Message(role="user", content="latest")]
    summarizer = MagicMock()
    summarizer.summarize_if_needed = AsyncMock(return_value=summarized)
    pipeline = CompressionPipeline(
        compactor=MessageCompactor(token_limit=40, workspace_dir=tmp_path),
        context_collapser=ContextCollapser(token_limit=40),
        summarizer=summarizer,
        request_context_builder=EchoRequestContextBuilder(),
        token_limit=40,
    )

    compacted = await pipeline.compress_before_request(messages=messages, api_total_tokens=0, tools=[])

    assert compacted == summarized
    summarizer.summarize_if_needed.assert_awaited_once()
