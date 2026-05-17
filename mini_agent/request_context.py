from __future__ import annotations

from pathlib import Path

from .context_budget import PromptLayerBudgets, estimate_message_tokens, estimate_messages_tokens, estimate_tool_tokens
from .prompt_builder import SystemPromptBuilder
from .schema import Message
from .summarizer import is_harness_summary_message, strip_harness_summary_header


class RequestContextBuilder:
    """Build a fresh LLM request context from durable state and session history."""

    def __init__(
        self,
        *,
        core_prompt: str,
        workspace_dir: str | Path,
        skill_loader=None,
        max_recent_messages: int = 12,
        token_budget: int | None = None,
        layer_budgets: PromptLayerBudgets | None = None,
    ):
        self.core_prompt = core_prompt
        self.workspace_dir = Path(workspace_dir).resolve()
        self.skill_loader = skill_loader
        self.max_recent_messages = max_recent_messages
        self.token_budget = token_budget
        self.layer_budgets = layer_budgets or PromptLayerBudgets()

    def build(
        self,
        messages: list[Message],
        *,
        tools: list[object] | None = None,
        token_budget: int | None = None,
    ) -> list[Message]:
        harness_summaries = self._extract_harness_summaries(messages)
        system_prompt = SystemPromptBuilder(
            core_prompt=self.core_prompt,
            workspace_dir=self.workspace_dir,
            skill_loader=self.skill_loader,
            harness_summaries=harness_summaries,
            layer_budgets=self.layer_budgets,
        ).build()

        system_message = Message(role="system", content=system_prompt)
        effective_budget = token_budget if token_budget is not None else self.token_budget
        history_budget = self._history_budget(effective_budget, system_message, tools)
        selected = self._select_messages(messages, history_budget=history_budget)
        return [system_message, *selected]

    def _select_messages(self, messages: list[Message], history_budget: int | None = None) -> list[Message]:
        non_system = [
            self._sanitize_message(message)
            for message in messages
            if message.role != "system" and not is_harness_summary_message(message)
        ]
        if len(non_system) <= self.max_recent_messages:
            if history_budget is None or estimate_messages_tokens(non_system) <= history_budget:
                return non_system

        tool_chain_start = self._find_active_tool_chain_start(non_system)
        protected_start = tool_chain_start if tool_chain_start is not None else self._find_latest_user_start(non_system)
        protected = non_system[protected_start:] if protected_start is not None else []

        if history_budget is not None:
            return self._select_by_token_budget(non_system, protected_start, protected, history_budget)

        if tool_chain_start is not None:
            recent_start = max(0, len(non_system) - self.max_recent_messages)
            start = min(tool_chain_start, recent_start)
            return non_system[start:]

        return non_system[-self.max_recent_messages :]

    def _find_active_tool_chain_start(self, messages: list[Message]) -> int | None:
        tool_call_index: int | None = None
        saw_tool_result = False

        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
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

    def _extract_harness_summaries(self, messages: list[Message]) -> list[str]:
        summaries: list[str] = []
        for message in messages:
            if not is_harness_summary_message(message) or not isinstance(message.content, str):
                continue
            summaries.append(strip_harness_summary_header(message.content))
        return summaries

    def _history_budget(self, token_budget: int | None, system_message: Message, tools: list[object] | None) -> int | None:
        if token_budget is None:
            return None
        reserved = estimate_message_tokens(system_message) + estimate_tool_tokens(tools)
        return max(0, token_budget - reserved)

    def _find_latest_user_start(self, messages: list[Message]) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user":
                return index
        return None

    def _select_by_token_budget(
        self,
        messages: list[Message],
        protected_start: int | None,
        protected: list[Message],
        history_budget: int,
    ) -> list[Message]:
        if protected_start is None:
            protected_start = len(messages)
            protected = []

        selected = list(protected)
        used = estimate_messages_tokens(selected)
        available_prefix = messages[:protected_start]

        for message in reversed(available_prefix):
            if len(selected) >= self.max_recent_messages:
                break
            message_tokens = estimate_message_tokens(message)
            if used + message_tokens > history_budget and selected:
                break
            if used + message_tokens > history_budget and not selected:
                selected.insert(0, message)
                break
            selected.insert(0, message)
            used += message_tokens

        if not selected and messages:
            return [messages[-1]]
        return selected

    def _sanitize_message(self, message: Message) -> Message:
        return Message(
            role=message.role,
            content=message.content,
            thinking=None,
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id,
            name=message.name,
        )
