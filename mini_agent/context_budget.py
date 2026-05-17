"""Token budgeting helpers for prompt and request assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tiktoken

from .schema import Message


@dataclass(frozen=True)
class PromptLayerBudgets:
    """Per-layer token budgets for the system prompt."""

    core: int = 2500
    skills: int = 1200
    memory: int = 1200
    project_rules: int = 1800
    current_task_context: int = 1000
    harness_summary: int = 1800
    dynamic_context: int = 300


def count_text_tokens(text: str) -> int:
    """Estimate token count for text with the repository's default tokenizer."""

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        return int(len(text) / 2.5)
    return len(encoding.encode(text))


def estimate_message_tokens(message: Message) -> int:
    """Estimate the request token footprint of one chat message."""

    total = 4
    if isinstance(message.content, str):
        total += count_text_tokens(message.content)
    elif isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, dict):
                total += count_text_tokens(str(block))

    if message.thinking:
        total += count_text_tokens(message.thinking)
    if message.tool_calls:
        total += count_text_tokens(str(message.tool_calls))
    if message.tool_call_id:
        total += count_text_tokens(message.tool_call_id)
    if message.name:
        total += count_text_tokens(message.name)
    return total


def estimate_messages_tokens(messages: list[Message]) -> int:
    """Estimate token usage for a message list."""

    return sum(estimate_message_tokens(message) for message in messages)


def estimate_tool_tokens(tools: list[object] | None) -> int:
    """Estimate token usage for tool schemas sent with a request."""

    if not tools:
        return 0
    return sum(count_text_tokens(str(_tool_schema_for_estimate(tool))) for tool in tools)


def clip_text_to_token_budget(text: str, max_tokens: int, *, label: str = "content") -> str:
    """Clip text to a token budget using head/tail retention."""

    if max_tokens <= 0:
        return ""
    if count_text_tokens(text) <= max_tokens:
        return text

    note = f"\n\n... [{label} compressed to fit {max_tokens} token budget] ...\n\n"
    note_tokens = count_text_tokens(note)
    remaining = max(1, max_tokens - note_tokens)
    head_budget = max(1, int(remaining * 0.65))
    tail_budget = max(1, remaining - head_budget)

    head = _clip_from_start(text, head_budget).rstrip()
    tail = _clip_from_end(text, tail_budget).lstrip()
    return head + note + tail


def _clip_from_start(text: str, max_tokens: int) -> str:
    if count_text_tokens(text) <= max_tokens:
        return text
    low = 0
    high = len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid]
        if count_text_tokens(candidate) <= max_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    newline = best.rfind("\n")
    return best[:newline] if newline > 0 else best


def _clip_from_end(text: str, max_tokens: int) -> str:
    if count_text_tokens(text) <= max_tokens:
        return text
    low = 0
    high = len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[len(text) - mid :]
        if count_text_tokens(candidate) <= max_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    newline = best.find("\n")
    return best[newline + 1 :] if newline > 0 else best


def _tool_schema_for_estimate(tool: object) -> Any:
    if hasattr(tool, "to_openai_schema"):
        return tool.to_openai_schema()
    if hasattr(tool, "to_schema"):
        return tool.to_schema()
    return tool
