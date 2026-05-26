"""Message history summarization for the agent harness."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken

from .console import AgentConsoleRenderer
from .context_budget import (
    clip_text_to_token_budget,
    count_text_tokens,
    estimate_messages_tokens,
    estimate_tool_tokens as estimate_request_tool_tokens,
)
from .llm import LLMClient
from .schema import Message

HARNESS_SUMMARY_MESSAGE_NAME = "harness_execution_summary"
HARNESS_SUMMARY_HEADER = "[Harness Execution Summary]"
CONTEXT_SNIP_MESSAGE_NAME = "context_snip_boundary"
CONTEXT_SNIP_HEADER = "[Context Snipped:"
CONTEXT_COLLAPSE_MESSAGE_NAME = "context_collapse_boundary"
CONTEXT_COLLAPSE_HEADER = "[Context Collapsed:"
TOOL_RESULT_SPILL_HEADER = "[Tool result stored on disk:"
MICRO_COMPACT_HEADER = "[Old tool result content shortened:"

COMPACTABLE_TOOL_NAMES = {
    "read_file",
    "bash",
    "bash_output",
    "write_file",
    "edit_file",
    "recall_notes",
    "get_skill",
}


def is_harness_summary_message(message: Message) -> bool:
    """Return true for internal compressed execution summaries."""

    if message.name == HARNESS_SUMMARY_MESSAGE_NAME:
        return True
    return isinstance(message.content, str) and message.content.startswith(HARNESS_SUMMARY_HEADER)


def strip_harness_summary_header(content: str) -> str:
    """Remove the internal summary marker before injecting it into system context."""

    text = content.strip()
    if text.startswith(HARNESS_SUMMARY_HEADER):
        return text[len(HARNESS_SUMMARY_HEADER) :].strip()
    return text


def is_context_snip_message(message: Message) -> bool:
    """Return true for deterministic context boundary markers."""

    if message.name == CONTEXT_SNIP_MESSAGE_NAME:
        return True
    return isinstance(message.content, str) and message.content.startswith(CONTEXT_SNIP_HEADER)


def is_context_collapse_message(message: Message) -> bool:
    """Return true for read-time context collapse markers."""

    if message.name == CONTEXT_COLLAPSE_MESSAGE_NAME:
        return True
    return isinstance(message.content, str) and message.content.startswith(CONTEXT_COLLAPSE_HEADER)


@dataclass
class CompactionResult:
    """Result metadata from deterministic message compaction."""

    messages: list[Message]
    tool_results_spilled: int = 0
    snipped_messages: int = 0
    snip_tokens_freed: int = 0
    micro_compacted_results: int = 0


class MessageCompactor:
    """Apply deterministic context compression without calling an LLM."""

    default_tool_result_soft_byte_limit = 64 * 1024
    default_tool_round_total_byte_limit = 200 * 1024

    newest_tool_result_count = 2
    middle_tool_result_count = 4
    newest_tool_result_token_limit = 1200
    middle_tool_result_token_limit = 600
    old_tool_result_token_limit = 250

    def __init__(
        self,
        *,
        token_limit: int,
        workspace_dir: str | Path = ".",
        near_limit_ratio: float = 0.85,
        tool_result_soft_byte_limit: int | None = None,
        tool_round_total_byte_limit: int | None = None,
    ):
        self.token_limit = token_limit
        self.workspace_dir = Path(workspace_dir).resolve()
        self.near_limit_ratio = near_limit_ratio
        self.tool_result_soft_byte_limit = (
            tool_result_soft_byte_limit
            if tool_result_soft_byte_limit is not None
            else self.default_tool_result_soft_byte_limit
        )
        self.tool_round_total_byte_limit = (
            tool_round_total_byte_limit
            if tool_round_total_byte_limit is not None
            else self.default_tool_round_total_byte_limit
        )

    def compact(self, messages: list[Message]) -> CompactionResult:
        """Run deterministic compaction layers in order."""

        result = CompactionResult(messages=list(messages))
        active_start = self._find_active_tool_chain_start(result.messages)
        result = self._apply_tool_result_budget(result, active_start)

        target_tokens = max(0, int(self.token_limit * self.near_limit_ratio))
        if estimate_messages_tokens(result.messages) > target_tokens:
            result = self._apply_snip(result, target_tokens)

        if estimate_messages_tokens(result.messages) > target_tokens:
            active_start = self._find_active_tool_chain_start(result.messages)
            result = self._apply_micro_compact(result, active_start)

        return result

    def _apply_tool_result_budget(self, result: CompactionResult, active_start: int | None) -> CompactionResult:
        messages = list(result.messages)
        active_indices = self._active_indices(messages, active_start)

        for assistant_index, tool_indices in self._tool_rounds(messages):
            if assistant_index in active_indices or any(index in active_indices for index in tool_indices):
                continue

            sizes: dict[int, int] = {}
            spill_indices: set[int] = set()
            for index in tool_indices:
                message = messages[index]
                if not isinstance(message.content, str) or self._is_spill_marker(message.content):
                    continue
                size = len(message.content.encode("utf-8"))
                sizes[index] = size
                if size > self.tool_result_soft_byte_limit:
                    spill_indices.add(index)

            remaining_total = sum(size for index, size in sizes.items() if index not in spill_indices)
            if remaining_total > self.tool_round_total_byte_limit:
                for index, size in sorted(sizes.items(), key=lambda item: item[1], reverse=True):
                    if remaining_total <= self.tool_round_total_byte_limit:
                        break
                    if index in spill_indices:
                        continue
                    spill_indices.add(index)
                    remaining_total -= size

            for index in sorted(spill_indices):
                message = messages[index]
                if not isinstance(message.content, str) or self._is_spill_marker(message.content):
                    continue
                tool_name = self._tool_name_for_result(messages, index)
                relative_path = self._spill_tool_result(
                    content=message.content,
                    assistant_index=assistant_index,
                    tool_call_id=message.tool_call_id,
                    tool_name=tool_name,
                )
                marker = self._format_spill_marker(
                    content=message.content,
                    path=relative_path,
                    tool_name=tool_name,
                    tool_call_id=message.tool_call_id,
                )
                messages[index] = message.model_copy(update={"content": marker})
                result.tool_results_spilled += 1

        result.messages = messages
        return result

    def _apply_snip(self, result: CompactionResult, target_tokens: int) -> CompactionResult:
        messages = list(result.messages)
        current_tokens = estimate_messages_tokens(messages)
        if current_tokens <= target_tokens:
            return result

        active_start = self._find_active_tool_chain_start(messages)
        latest_user_start = self._find_latest_user_start(messages)
        protected_starts = [index for index in (active_start, latest_user_start) if index is not None]
        protected_start = min(protected_starts) if protected_starts else len(messages)

        groups = self._snip_candidate_groups(messages, protected_start)
        if not groups:
            return result

        remove_indices: set[int] = set()
        removed_tokens = 0
        for group in groups:
            group_messages = [messages[index] for index in group]
            group_tokens = estimate_messages_tokens(group_messages)
            remove_indices.update(group)
            removed_tokens += group_tokens
            current_tokens -= group_tokens
            if current_tokens <= target_tokens:
                break

        if not remove_indices:
            return result

        marker = Message(
            role="system",
            content=(
                f"[Context Snipped: {len(remove_indices)} older messages removed, "
                f"approximately {max(0, removed_tokens)} tokens freed. Earlier context is unavailable.]"
            ),
            name=CONTEXT_SNIP_MESSAGE_NAME,
        )
        marker_tokens = estimate_messages_tokens([marker])
        new_messages: list[Message] = []
        inserted_marker = False
        for index, message in enumerate(messages):
            if index in remove_indices:
                if not inserted_marker:
                    new_messages.append(marker)
                    inserted_marker = True
                continue
            new_messages.append(message)

        result.messages = new_messages
        result.snipped_messages += len(remove_indices)
        result.snip_tokens_freed += max(0, removed_tokens - marker_tokens)
        return result

    def _apply_micro_compact(self, result: CompactionResult, active_start: int | None) -> CompactionResult:
        messages = list(result.messages)
        active_indices = self._active_indices(messages, active_start)
        candidate_indices: list[int] = []

        for index, message in enumerate(messages):
            if index in active_indices or message.role != "tool" or not isinstance(message.content, str):
                continue
            if self._is_spill_marker(message.content) or self._is_micro_compact_marker(message.content):
                continue
            tool_name = self._tool_name_for_result(messages, index)
            if tool_name in COMPACTABLE_TOOL_NAMES:
                candidate_indices.append(index)

        for rank, index in enumerate(reversed(candidate_indices)):
            message = messages[index]
            if not isinstance(message.content, str):
                continue
            original_tokens = count_text_tokens(message.content)
            retention = self._micro_retention_for_rank(rank)
            if original_tokens <= retention:
                continue

            tool_name = self._tool_name_for_result(messages, index)
            clipped = clip_text_to_token_budget(message.content, retention, label=f"{tool_name} result")
            retained_tokens = count_text_tokens(clipped)
            marker = (
                f"[Old tool result content shortened: tool={tool_name}, original_tokens={original_tokens}, "
                f"retained_tokens={retained_tokens}. Earlier content was removed from this prompt. "
                "Re-run or re-read only if safe and needed.]\n\n"
                f"{clipped}"
            )
            messages[index] = message.model_copy(update={"content": marker})
            result.micro_compacted_results += 1

        result.messages = messages
        return result

    def _tool_rounds(self, messages: list[Message]) -> list[tuple[int, list[int]]]:
        rounds: list[tuple[int, list[int]]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.role != "assistant" or not message.tool_calls:
                index += 1
                continue
            tool_indices: list[int] = []
            cursor = index + 1
            while cursor < len(messages) and messages[cursor].role == "tool":
                tool_indices.append(cursor)
                cursor += 1
            if tool_indices:
                rounds.append((index, tool_indices))
            index = cursor
        return rounds

    def _snip_candidate_groups(self, messages: list[Message], protected_start: int) -> list[list[int]]:
        groups: list[list[int]] = []
        index = 1 if messages and messages[0].role == "system" else 0
        protected_start = min(protected_start, len(messages))

        while index < protected_start:
            message = messages[index]
            if self._is_protected_system_message(message):
                index += 1
                continue

            start = index
            if message.role == "user":
                index += 1
                while index < protected_start and messages[index].role != "user":
                    index += 1
            elif message.role == "assistant" and message.tool_calls:
                index += 1
                while index < protected_start and messages[index].role == "tool":
                    index += 1
            else:
                index += 1

            group = list(range(start, index))
            if not any(self._is_protected_system_message(messages[group_index]) for group_index in group):
                groups.append(group)

        return groups

    def _is_protected_system_message(self, message: Message) -> bool:
        return message.role == "system" or is_harness_summary_message(message) or is_context_snip_message(message)

    def _spill_tool_result(
        self,
        *,
        content: str,
        assistant_index: int,
        tool_call_id: str | None,
        tool_name: str,
    ) -> str:
        spill_dir = self.workspace_dir / ".mini_agent" / "tool-results"
        spill_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"step-{assistant_index}-"
            f"{self._safe_filename_part(tool_call_id or 'unknown-call')}-"
            f"{self._safe_filename_part(tool_name)}.txt"
        )
        path = (spill_dir / filename).resolve()
        try:
            path.relative_to(self.workspace_dir)
        except ValueError as exc:
            raise ValueError("Tool result spill path escaped workspace") from exc

        if not path.exists() or path.read_text(encoding="utf-8", errors="replace") != content:
            path.write_text(content, encoding="utf-8")
        return path.relative_to(self.workspace_dir).as_posix()

    def _format_spill_marker(
        self,
        *,
        content: str,
        path: str,
        tool_name: str,
        tool_call_id: str | None,
    ) -> str:
        byte_size = len(content.encode("utf-8"))
        preview = self._preview(content)
        return (
            f"[Tool result stored on disk: tool={tool_name}, tool_call_id={tool_call_id or 'unknown'}, "
            f"original_bytes={byte_size}, path={path}. "
            "Use read_file(path, offset, limit) to inspect specific ranges.]\n"
            f"{preview}"
        )

    def _preview(self, content: str) -> str:
        if len(content) <= 800:
            return content
        head = content[:350].rstrip()
        tail = content[-350:].lstrip()
        return f"[Preview head]\n{head}\n\n[Preview tail]\n{tail}"

    def _tool_name_for_result(self, messages: list[Message], index: int) -> str:
        message = messages[index]
        if message.name:
            return message.name
        if message.tool_call_id:
            for cursor in range(index - 1, -1, -1):
                previous = messages[cursor]
                if previous.role == "assistant" and previous.tool_calls:
                    for tool_call in previous.tool_calls:
                        if tool_call.id == message.tool_call_id:
                            return tool_call.function.name
        return "unknown"

    def _micro_retention_for_rank(self, rank: int) -> int:
        if rank < self.newest_tool_result_count:
            return self.newest_tool_result_token_limit
        if rank < self.newest_tool_result_count + self.middle_tool_result_count:
            return self.middle_tool_result_token_limit
        return self.old_tool_result_token_limit

    def _find_active_tool_chain_start(self, messages: list[Message]) -> int | None:
        tool_call_index: int | None = None
        saw_tool_result = False

        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if is_harness_summary_message(message) or is_context_snip_message(message) or is_context_collapse_message(message):
                continue
            if message.role == "tool":
                saw_tool_result = True
                continue
            if message.role == "assistant":
                if message.tool_calls:
                    tool_call_index = index
                    continue
                return None
            if message.role == "user":
                if tool_call_index is not None or saw_tool_result:
                    return index
                return None
        return None

    def _find_latest_user_start(self, messages: list[Message]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user":
                return index
        return None

    def _active_indices(self, messages: list[Message], active_start: int | None) -> set[int]:
        if active_start is None:
            return set()
        return set(range(active_start, len(messages)))

    def _is_spill_marker(self, content: str) -> bool:
        return content.startswith(TOOL_RESULT_SPILL_HEADER)

    def _is_micro_compact_marker(self, content: str) -> bool:
        return content.startswith(MICRO_COMPACT_HEADER)

    def _safe_filename_part(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
        return cleaned.strip(".-") or "unknown"


@dataclass
class CollapseResult:
    """Read-time context projection result."""

    messages: list[Message]
    collapsed_messages: int = 0
    collapse_tokens_freed: int = 0
    emergency: bool = False


class ContextCollapser:
    """Create a compressed request view without mutating durable history."""

    collapse_ratio = 0.90
    emergency_ratio = 0.95
    normal_target_ratio = 0.85
    emergency_target_ratio = 0.80
    max_marker_tokens = 220

    def __init__(
        self,
        *,
        token_limit: int,
        collapse_ratio: float | None = None,
        emergency_ratio: float | None = None,
    ):
        self.token_limit = token_limit
        if collapse_ratio is not None:
            self.collapse_ratio = collapse_ratio
        if emergency_ratio is not None:
            self.emergency_ratio = emergency_ratio

    def apply_collapses_if_needed(
        self,
        messages: list[Message],
        *,
        request_tokens: int,
    ) -> CollapseResult:
        """Return a collapsed request projection when the request is near the limit."""

        if request_tokens < int(self.token_limit * self.collapse_ratio):
            return CollapseResult(messages=messages)

        emergency = request_tokens >= int(self.token_limit * self.emergency_ratio)
        target_ratio = self.emergency_target_ratio if emergency else self.normal_target_ratio
        target_tokens = max(0, int(self.token_limit * target_ratio))
        active_start = self._find_active_tool_chain_start(messages)
        latest_user_start = self._find_latest_user_start(messages)
        protected_starts = [index for index in (active_start, latest_user_start) if index is not None]
        protected_start = min(protected_starts) if protected_starts else len(messages)
        groups = self._collapse_candidate_groups(messages, protected_start)
        if not groups:
            return CollapseResult(messages=messages, emergency=emergency)

        projected = list(messages)
        remove_indices: set[int] = set()
        removed_tokens = 0
        current_tokens = estimate_messages_tokens(projected)
        for group in groups:
            group_messages = [messages[index] for index in group]
            group_tokens = estimate_messages_tokens(group_messages)
            remove_indices.update(group)
            removed_tokens += group_tokens
            current_tokens -= group_tokens
            if current_tokens <= target_tokens:
                break

        if not remove_indices:
            return CollapseResult(messages=messages, emergency=emergency)

        marker = self._collapse_marker(
            [messages[index] for index in sorted(remove_indices)],
            removed_tokens=removed_tokens,
            emergency=emergency,
        )
        marker_tokens = estimate_messages_tokens([marker])
        new_messages: list[Message] = []
        inserted_marker = False
        for index, message in enumerate(projected):
            if index in remove_indices:
                if not inserted_marker:
                    new_messages.append(marker)
                    inserted_marker = True
                continue
            new_messages.append(message)

        return CollapseResult(
            messages=new_messages,
            collapsed_messages=len(remove_indices),
            collapse_tokens_freed=max(0, removed_tokens - marker_tokens),
            emergency=emergency,
        )

    def _collapse_marker(self, messages: list[Message], *, removed_tokens: int, emergency: bool) -> Message:
        anchors = self._anchors(messages)
        mode = "emergency" if emergency else "normal"
        lines = [
            (
                f"[Context Collapsed: {len(messages)} older messages hidden for this API call only, "
                f"approximately {removed_tokens} tokens removed, mode={mode}. "
                "Original history is preserved in agent memory.]"
            )
        ]
        if anchors:
            lines.extend(["Visible anchors from collapsed segment:", *anchors])
        content = "\n".join(lines)
        content = clip_text_to_token_budget(content, self.max_marker_tokens, label="context collapse marker")
        return Message(role="system", content=content, name=CONTEXT_COLLAPSE_MESSAGE_NAME)

    def _anchors(self, messages: list[Message]) -> list[str]:
        anchors: list[str] = []
        for message in messages:
            if message.role == "user":
                anchors.append(f"- user: {self._short_text(message.content)}")
            elif message.role == "assistant" and message.tool_calls:
                tool_names = ", ".join(tool_call.function.name for tool_call in message.tool_calls)
                anchors.append(f"- assistant tool calls: {tool_names}")
            elif message.role == "assistant":
                anchors.append(f"- assistant: {self._short_text(message.content)}")
            elif message.role == "tool":
                anchors.append(f"- tool result: {message.name or 'unknown'} ({message.tool_call_id or 'no id'})")
            if len(anchors) >= 8:
                break
        return anchors

    def _short_text(self, content: str | list[dict[str, Any]]) -> str:
        text = content if isinstance(content, str) else str(content)
        text = " ".join(text.strip().split())
        return text[:160] + ("..." if len(text) > 160 else "")

    def _collapse_candidate_groups(self, messages: list[Message], protected_start: int) -> list[list[int]]:
        groups: list[list[int]] = []
        index = 1 if messages and messages[0].role == "system" else 0
        protected_start = min(protected_start, len(messages))

        while index < protected_start:
            message = messages[index]
            if self._is_protected_system_message(message):
                index += 1
                continue

            start = index
            if message.role == "user":
                index += 1
                while index < protected_start and messages[index].role != "user":
                    index += 1
            elif message.role == "assistant" and message.tool_calls:
                index += 1
                while index < protected_start and messages[index].role == "tool":
                    index += 1
            else:
                index += 1

            group = list(range(start, index))
            if not any(self._is_protected_system_message(messages[group_index]) for group_index in group):
                groups.append(group)

        return groups

    def _is_protected_system_message(self, message: Message) -> bool:
        return (
            message.role == "system"
            or is_harness_summary_message(message)
            or is_context_snip_message(message)
            or is_context_collapse_message(message)
        )

    def _find_active_tool_chain_start(self, messages: list[Message]) -> int | None:
        tool_call_index: int | None = None
        saw_tool_result = False

        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if self._is_protected_system_message(message):
                continue
            if message.role == "tool":
                saw_tool_result = True
                continue
            if message.role == "assistant":
                if message.tool_calls:
                    tool_call_index = index
                    continue
                return None
            if message.role == "user":
                if tool_call_index is not None or saw_tool_result:
                    return index
                return None
        return None

    def _find_latest_user_start(self, messages: list[Message]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user":
                return index
        return None


class CompressionPipeline:
    """Run deterministic compaction before falling back to semantic summarization."""

    def __init__(
        self,
        *,
        compactor: MessageCompactor,
        context_collapser: ContextCollapser | None = None,
        summarizer: "MessageSummarizer",
        request_context_builder: Any,
        token_limit: int,
        renderer: AgentConsoleRenderer | None = None,
        near_limit_ratio: float = 0.85,
        hard_limit_ratio: float = 1.0,
    ):
        self.compactor = compactor
        self.context_collapser = context_collapser or ContextCollapser(token_limit=token_limit)
        self.summarizer = summarizer
        self.request_context_builder = request_context_builder
        self.token_limit = token_limit
        self.renderer = renderer or AgentConsoleRenderer()
        self.near_limit_ratio = near_limit_ratio
        self.hard_limit_ratio = hard_limit_ratio
        self.stats: list[dict[str, Any]] = []

    async def compress_before_request(
        self,
        *,
        messages: list[Message],
        api_total_tokens: int,
        tools: list[object] | None = None,
    ) -> list[Message]:
        """Return request-ready history after staged compression."""

        estimated_tokens = self._estimate_request_tokens(messages, tools)
        if estimated_tokens <= int(self.token_limit * self.near_limit_ratio):
            self._record_stats(
                compression_triggered=False,
                stage="none",
                before_messages=messages,
                after_messages=messages,
                before_tokens=estimated_tokens,
                after_tokens=estimated_tokens,
            )
            return messages

        compaction = self.compactor.compact(messages)
        compacted_messages = compaction.messages
        compacted_estimate = self._estimate_request_tokens(compacted_messages, tools)
        if compacted_estimate <= int(self.token_limit * self.hard_limit_ratio):
            self._record_stats(
                compression_triggered=True,
                stage="compaction",
                before_messages=messages,
                after_messages=compacted_messages,
                before_tokens=estimated_tokens,
                after_tokens=compacted_estimate,
            )
            return compacted_messages

        collapse = self.context_collapser.apply_collapses_if_needed(
            compacted_messages,
            request_tokens=compacted_estimate,
        )
        collapsed_messages = collapse.messages
        collapsed_estimate = self._estimate_request_tokens(collapsed_messages, tools)
        if collapsed_estimate <= int(self.token_limit * self.hard_limit_ratio):
            self._record_stats(
                compression_triggered=True,
                stage="collapse",
                before_messages=messages,
                after_messages=collapsed_messages,
                before_tokens=estimated_tokens,
                after_tokens=collapsed_estimate,
            )
            return collapsed_messages

        budget_messages = self.request_context_builder.build(
            collapsed_messages,
            tools=tools,
            token_budget=self.token_limit,
        )
        summarized_messages = await self.summarizer.summarize_if_needed(
            collapsed_messages,
            api_total_tokens,
            budget_messages=budget_messages,
            tools=tools,
        )
        summarized_estimate = self._estimate_request_tokens(summarized_messages, tools)
        self._record_stats(
            compression_triggered=True,
            stage="summarization",
            before_messages=messages,
            after_messages=summarized_messages,
            before_tokens=estimated_tokens,
            after_tokens=summarized_estimate,
        )
        return summarized_messages

    def _estimate_request_tokens(self, messages: list[Message], tools: list[object] | None) -> int:
        request_messages = self.request_context_builder.build(
            messages,
            tools=tools,
            token_budget=self.token_limit,
        )
        return estimate_messages_tokens(request_messages) + estimate_request_tool_tokens(tools)

    def _record_stats(
        self,
        *,
        compression_triggered: bool,
        stage: str,
        before_messages: list[Message],
        after_messages: list[Message],
        before_tokens: int,
        after_tokens: int,
    ) -> None:
        ratio = 0.0 if before_tokens <= 0 else max(0.0, 1.0 - (after_tokens / before_tokens))
        self.stats.append(
            {
                "compression_triggered": compression_triggered,
                "stage": stage,
                "before_tokens": before_tokens,
                "after_tokens": after_tokens,
                "compression_ratio": ratio,
                "before_message_count": len(before_messages),
                "after_message_count": len(after_messages),
            }
        )


class MessageSummarizer:
    """Compact agent message history when it approaches the context limit."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        token_limit: int,
        renderer: AgentConsoleRenderer | None = None,
    ):
        self.llm = llm_client
        self.token_limit = token_limit
        self.renderer = renderer or AgentConsoleRenderer()
        self._skip_next_token_check = False

    async def summarize_if_needed(
        self,
        messages: list[Message],
        api_total_tokens: int,
        budget_messages: list[Message] | None = None,
        tools: list[object] | None = None,
    ) -> list[Message]:
        """Return a compacted message history when local/API tokens exceed the limit."""

        if self._skip_next_token_check:
            self._skip_next_token_check = False
            return messages

        estimated_tokens = self.estimate_tokens(messages)
        if budget_messages is not None:
            estimated_tokens = max(estimated_tokens, self.estimate_tokens(budget_messages))
        if tools:
            estimated_tokens += self.estimate_tool_tokens(tools)
        should_summarize = estimated_tokens > self.token_limit or api_total_tokens > self.token_limit
        if not should_summarize:
            return messages

        self.renderer.summary_triggered(estimated_tokens, api_total_tokens, self.token_limit)

        active_start = self._find_active_tool_round_start(messages)
        latest_user_start = None if active_start is not None else self._find_latest_user_message_start(messages)

        if active_start is not None:
            protected_tail = messages[active_start:]
            preserved_user: Message | None = None
            append_summary_after_user = False
            historical_source = messages[1:active_start]
        elif latest_user_start is not None:
            protected_tail = []
            preserved_user = messages[latest_user_start]
            suffix_after_user = messages[latest_user_start + 1 :]
            append_summary_after_user = bool(suffix_after_user)
            historical_source = [*messages[1:latest_user_start], *suffix_after_user]
        else:
            protected_tail = []
            preserved_user = None
            append_summary_after_user = False
            historical_source = messages[1:]

        boundary_messages = [
            message
            for message in historical_source
            if is_context_snip_message(message) or is_context_collapse_message(message)
        ]
        summarizable_messages = [
            message
            for message in historical_source
            if not self._is_preserved_projection_marker(message) and message.role != "system"
        ]
        if not summarizable_messages:
            self.renderer.summary_insufficient_messages()
            return messages

        new_messages = [messages[0], *boundary_messages]
        if preserved_user is not None and append_summary_after_user:
            new_messages.append(preserved_user)
        summary_text = await self._create_full_history_summary(summarizable_messages)
        summary_count = 0
        if summary_text:
            new_messages.append(
                Message(
                    role="system",
                    content=f"{HARNESS_SUMMARY_HEADER}\n\n{summary_text}",
                    name=HARNESS_SUMMARY_MESSAGE_NAME,
                )
            )
            summary_count = 1

        if preserved_user is not None and not append_summary_after_user:
            new_messages.append(preserved_user)
        new_messages.extend(protected_tail)

        self._skip_next_token_check = True
        new_tokens = self.estimate_tokens(new_messages)
        self.renderer.summary_completed(estimated_tokens, new_tokens, len(summarizable_messages), summary_count)
        return new_messages

    def estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate token count for message history using tiktoken when available."""

        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return self._estimate_tokens_fallback(messages)

        total_tokens = 0
        for msg in messages:
            if isinstance(msg.content, str):
                total_tokens += len(encoding.encode(msg.content))
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        total_tokens += len(encoding.encode(str(block)))

            if msg.thinking:
                total_tokens += len(encoding.encode(msg.thinking))

            if msg.tool_calls:
                total_tokens += len(encoding.encode(str(msg.tool_calls)))

            total_tokens += 4

        return total_tokens

    def estimate_tool_tokens(self, tools: list[object]) -> int:
        """Estimate token usage for tool schemas sent alongside the request."""

        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            total_chars = 0
            for tool in tools:
                total_chars += len(str(self._tool_schema_for_estimate(tool)))
            return int(total_chars / 2.5)

        total_tokens = 0
        for tool in tools:
            total_tokens += len(encoding.encode(str(self._tool_schema_for_estimate(tool))))
        return total_tokens

    def _tool_schema_for_estimate(self, tool: object) -> object:
        if hasattr(tool, "to_openai_schema"):
            return tool.to_openai_schema()
        if hasattr(tool, "to_schema"):
            return tool.to_schema()
        return tool

    def _estimate_tokens_fallback(self, messages: list[Message]) -> int:
        total_chars = 0
        for msg in messages:
            if isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        total_chars += len(str(block))

            if msg.thinking:
                total_chars += len(msg.thinking)

            if msg.tool_calls:
                total_chars += len(str(msg.tool_calls))

        return int(total_chars / 2.5)

    async def _create_summary(self, messages: list[Message], round_num: int) -> str:
        if not messages:
            return ""

        summary_content = f"第 {round_num} 轮执行过程：\n\n"
        for msg in messages:
            if msg.role == "assistant":
                content_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                summary_content += f"Assistant: {content_text}\n"
                if msg.tool_calls:
                    tool_names = [tc.function.name for tc in msg.tool_calls]
                    summary_content += f"  -> 调用工具: {', '.join(tool_names)}\n"
            elif msg.role == "tool":
                result_preview = msg.content if isinstance(msg.content, str) else str(msg.content)
                summary_content += f"  <- 工具返回: {result_preview}...\n"

        try:
            summary_prompt = f"""请简洁总结下面这段 Agent 执行过程：

{summary_content}

要求：
1. 聚焦已完成的任务、调用过的工具和关键结果
2. 保留重要发现、文件路径、错误信息和决策
3. 控制在 1000 字以内
4. 使用中文
5. 不要添加新的用户指令，只总结 Agent 的执行过程"""

            response = await self.llm.generate(
                messages=[
                    Message(
                        role="system",
                        content="你擅长把 Agent 的工具调用和执行过程压缩成可靠的上下文摘要。",
                    ),
                    Message(role="user", content=summary_prompt),
                ]
            )

            self.renderer.summary_round_success(round_num)
            return response.content

        except Exception as exc:
            self.renderer.summary_round_failed(round_num, exc)
            return summary_content

    async def _create_full_history_summary(self, messages: list[Message]) -> str:
        if not messages:
            return ""

        summary_content = "需要压缩的历史上下文：\n\n"
        for index, msg in enumerate(messages, 1):
            content_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            summary_content += f"{index}. {msg.role}: {content_text}\n"
            if msg.thinking:
                summary_content += f"   thinking: {msg.thinking}\n"
            if msg.tool_calls:
                tool_names = [tc.function.name for tc in msg.tool_calls]
                summary_content += f"   tool calls: {', '.join(tool_names)}\n"
            if msg.tool_call_id:
                summary_content += f"   tool_call_id: {msg.tool_call_id}\n"
            if msg.name:
                summary_content += f"   name: {msg.name}\n"

        try:
            summary_prompt = f"""请将下面这段 Agent 历史上下文压缩成一个可靠的全量摘要：

{summary_content}

要求：
1. 只总结已经发生的历史事实，不添加新的用户指令
2. 保留任务目标、关键文件路径、工具调用、错误、决策和已完成结果
3. 如果历史里有工具结果被裁剪、落盘或上下文边界标记，保留这些恢复线索
4. 控制在 1200 字以内
5. 使用中文"""

            response = await self.llm.generate(
                messages=[
                    Message(
                        role="system",
                        content="你擅长把 Agent 的完整历史上下文压缩成可恢复、可继续执行的可靠摘要。",
                    ),
                    Message(role="user", content=summary_prompt),
                ]
            )

            self.renderer.summary_round_success(1)
            return response.content

        except Exception as exc:
            self.renderer.summary_round_failed(1, exc)
            return summary_content

    def _find_active_tool_round_start(self, messages: list[Message]) -> int | None:
        """Find the user message that started an unfinished tool round, if any."""

        saw_tool_result = False
        saw_tool_call = False

        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if is_harness_summary_message(message):
                continue
            if message.role == "tool":
                saw_tool_result = True
                continue
            if message.role == "assistant":
                if message.tool_calls:
                    saw_tool_call = True
                    continue
                return None
            if message.role == "user":
                if saw_tool_call or saw_tool_result:
                    return index
                return None
        return None

    def _find_latest_user_message_start(self, messages: list[Message]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user":
                return index
        return None

    def _is_preserved_projection_marker(self, message: Message) -> bool:
        return is_context_snip_message(message) or is_context_collapse_message(message)
