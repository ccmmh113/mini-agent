"""Tool wrapper that delegates a bounded task to an isolated subagent."""

from __future__ import annotations

from .base import Tool, ToolResult
from ..subagent import SubagentRunner


class SubagentTool(Tool):
    """Delegate a focused task to a child Agent with separate message history."""

    def __init__(self, runner: SubagentRunner):
        self.runner = runner

    @property
    def name(self) -> str:
        return "task"

    @property
    def description(self) -> str:
        return (
            "Delegate a focused, self-contained investigation or analysis task to an isolated subagent. "
            "Use this when a subtask can be solved independently and summarized back to the main agent."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A short name for the delegated subtask.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Detailed instructions for the subagent.",
                },
            },
            "required": ["description", "prompt"],
        }

    async def execute(self, description: str, prompt: str) -> ToolResult:
        result = await self.runner.run(description=description, prompt=prompt)
        content = result.to_tool_content()
        if not result.completed:
            return ToolResult(success=False, content=content, error=content)
        return ToolResult(success=True, content=content)
