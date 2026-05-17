"""Explicit long-term memory recording and recall tools."""

from pathlib import Path
from typing import Any

from ..memory.markdown_store import MEMORY_TYPES, MarkdownMemoryStore, format_markdown_memories
from .base import Tool, ToolResult


class SessionNoteTool(Tool):
    """Tool for explicitly recording durable long-term memory."""

    def __init__(self, memory_dir: str = "./workspace/.memory", memory_file: str | None = None):
        # memory_file is accepted for older call sites/tests, but the new store is directory-backed.
        base = Path(memory_file).parent if memory_file else Path(memory_dir)
        self.store = MarkdownMemoryStore(base)

    @property
    def name(self) -> str:
        return "record_note"

    @property
    def description(self) -> str:
        return (
            "Explicitly record durable cross-session memory. "
            "Use only for stable user preferences, explicit feedback, durable project facts, or reference links. "
            "Do not record temporary task progress, file structure, function signatures, branch names, or secrets."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The durable memory content. Keep it concise and reusable across future sessions.",
                },
                "type": {
                    "type": "string",
                    "description": "Memory type: user, feedback, project, or reference.",
                    "enum": sorted(MEMORY_TYPES),
                    "default": "project",
                },
                "name": {
                    "type": "string",
                    "description": "Optional stable file-friendly memory name.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional short index description.",
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        content: str,
        type: str | None = None,
        name: str | None = None,
        description: str | None = None,
        category: str | None = None,
    ) -> ToolResult:
        try:
            memory_type = type if type in MEMORY_TYPES else (_category_to_type(category) or "project")
            memory = self.store.save_memory(
                content=content,
                memory_type=memory_type,
                name=name,
                description=description,
            )
            return ToolResult(
                success=True,
                content=f"Recorded memory: {memory.name} ({memory.type})",
            )
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to record memory: {str(e)}")


class RecallNoteTool(Tool):
    """Tool for recalling lightweight Markdown memories."""

    def __init__(
        self,
        memory_dir: str = "./workspace/.memory",
        memory_file: str | None = None,
        workspace_dir: str | None = None,
    ):
        base = Path(memory_file).parent if memory_file else Path(memory_dir)
        self.store = MarkdownMemoryStore(base)
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir is not None else None

    @property
    def name(self) -> str:
        return "recall_notes"

    @property
    def description(self) -> str:
        return (
            "Recall durable Markdown memories from the local workspace. "
            "Memory is intentionally limited to user preferences, feedback, project facts, and references."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional keyword query over durable memory files.",
                },
                "type": {
                    "type": "string",
                    "description": "Optional memory type filter: user, feedback, project, or reference.",
                    "enum": sorted(MEMORY_TYPES),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memory files to return.",
                    "default": 10,
                },
            },
        }

    async def execute(
        self,
        query: str | None = None,
        type: str | None = None,
        category: str | None = None,
        source: str | None = None,
        limit: int = 10,
    ) -> ToolResult:
        try:
            memory_type = type if type in MEMORY_TYPES else _category_to_type(category)
            if source and source not in {"explicit_memory", "markdown_memory"}:
                return ToolResult(success=True, content=f"No lightweight memory source matches: {source}")

            memories = self.store.search(query=query, memory_type=memory_type, limit=limit)
            if not memories:
                return ToolResult(success=True, content="No memory recorded yet.")
            title = f"Relevant Memory for: {query}" if query else "Recorded Memory:"
            return ToolResult(success=True, content=format_markdown_memories(title, memories))
        except Exception as e:
            return ToolResult(success=False, content="", error=f"Failed to recall memory: {str(e)}")


def _category_to_type(category: str | None) -> str | None:
    if not category:
        return None
    mapping = {
        "user_preference": "user",
        "preference": "user",
        "feedback": "feedback",
        "project_info": "project",
        "project": "project",
        "reference": "reference",
        "link": "reference",
    }
    return mapping.get(category, "project")
