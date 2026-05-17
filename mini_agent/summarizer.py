"""Message history summarization for the agent harness."""

from __future__ import annotations

import tiktoken

from .console import AgentConsoleRenderer
from .llm import LLMClient
from .schema import Message

HARNESS_SUMMARY_MESSAGE_NAME = "harness_execution_summary"
HARNESS_SUMMARY_HEADER = "[Harness Execution Summary]"


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

        preserve_from = self._find_active_tool_round_start(messages)
        summarizable_end = preserve_from if preserve_from is not None else len(messages)
        user_indices = [i for i, msg in enumerate(messages[:summarizable_end]) if msg.role == "user" and i > 0]
        if len(user_indices) < 1:
            self.renderer.summary_insufficient_messages()
            return messages

        new_messages = [messages[0]]
        summary_count = 0

        for i, user_idx in enumerate(user_indices):
            new_messages.append(messages[user_idx])
            next_user_idx = user_indices[i + 1] if i < len(user_indices) - 1 else summarizable_end
            execution_messages = messages[user_idx + 1 : next_user_idx]

            if execution_messages:
                summary_text = await self._create_summary(execution_messages, i + 1)
                if summary_text:
                    new_messages.append(
                        Message(
                            role="system",
                            content=f"{HARNESS_SUMMARY_HEADER}\n\n{summary_text}",
                            name=HARNESS_SUMMARY_MESSAGE_NAME,
                        )
                    )
                    summary_count += 1

        if preserve_from is not None:
            new_messages.extend(messages[preserve_from:])

        self._skip_next_token_check = True
        new_tokens = self.estimate_tokens(new_messages)
        self.renderer.summary_completed(estimated_tokens, new_tokens, len(user_indices), summary_count)
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
