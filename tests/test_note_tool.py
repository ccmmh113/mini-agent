"""Test cases for lightweight Markdown memory recording and recall."""

from pathlib import Path

import pytest

from mini_agent.cli import build_memory_review_rows
from mini_agent.memory.markdown_store import MarkdownMemoryStore
from mini_agent.tools.note_tool import RecallNoteTool, SessionNoteTool


@pytest.mark.asyncio
async def test_record_and_recall_markdown_memory(tmp_path: Path):
    memory_dir = tmp_path / ".memory"
    record_tool = SessionNoteTool(memory_dir=str(memory_dir))
    recall_tool = RecallNoteTool(memory_dir=str(memory_dir))

    result = await record_tool.execute(
        content="The user prefers concise responses.",
        type="user",
        name="concise-responses",
        description="User prefers concise responses",
    )

    assert result.success
    assert (memory_dir / "concise-responses.md").exists()
    assert (memory_dir / "MEMORY.md").exists()

    result = await recall_tool.execute(query="concise", type="user")

    assert result.success
    assert "concise-responses" in result.content
    assert "The user prefers concise responses." in result.content


@pytest.mark.asyncio
async def test_empty_memory(tmp_path: Path):
    recall_tool = RecallNoteTool(memory_dir=str(tmp_path / ".memory"))

    result = await recall_tool.execute()

    assert result.success
    assert "No memory recorded yet" in result.content


@pytest.mark.asyncio
async def test_memory_persistence_across_tool_instances(tmp_path: Path):
    memory_dir = tmp_path / ".memory"

    first = SessionNoteTool(memory_dir=str(memory_dir))
    result = await first.execute(
        content="Project deploys through the internal release script.",
        type="project",
        name="release-script",
    )
    assert result.success

    second = RecallNoteTool(memory_dir=str(memory_dir))
    result = await second.execute(query="release")

    assert result.success
    assert "internal release script" in result.content


@pytest.mark.asyncio
async def test_category_backcompat_maps_to_lightweight_type(tmp_path: Path):
    memory_dir = tmp_path / ".memory"
    record_tool = SessionNoteTool(memory_dir=str(memory_dir))
    recall_tool = RecallNoteTool(memory_dir=str(memory_dir))

    result = await record_tool.execute(
        content="User wants explicit warnings before destructive actions.",
        category="user_preference",
        name="destructive-warning",
    )

    assert result.success
    memories = MarkdownMemoryStore(memory_dir).load()
    assert memories[0].type == "user"

    result = await recall_tool.execute(category="user_preference")
    assert result.success
    assert "destructive actions" in result.content


def test_memory_review_rows_show_markdown_memory(tmp_path: Path):
    store = MarkdownMemoryStore(tmp_path / ".memory")
    store.save_memory(
        content="Use pytest for focused verification before broad test runs.",
        memory_type="feedback",
        name="focused-pytest",
        description="Prefer focused pytest verification",
    )

    rows = build_memory_review_rows(tmp_path)

    assert rows[0]["name"] == "focused-pytest"
    assert rows[0]["type"] == "feedback"
    assert "focused pytest" in rows[0]["description"].lower()
    assert rows[0]["path"].endswith("focused-pytest.md")


def test_markdown_memory_delete_updates_index(tmp_path: Path):
    store = MarkdownMemoryStore(tmp_path / ".memory")
    store.save_memory(
        content="External incident board: https://example.invalid/incidents",
        memory_type="reference",
        name="incident-board",
    )

    assert store.delete("incident-board")
    assert not (tmp_path / ".memory" / "incident-board.md").exists()
    assert "incident-board" not in (tmp_path / ".memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert not store.delete("missing")
