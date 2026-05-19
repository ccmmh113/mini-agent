"""Core Agent implementation."""

import asyncio
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable, Optional

from .checkpoint import CheckpointStore
from .checkpointing import CheckpointCoordinator
from .console import AgentConsoleRenderer, Colors
from .context_budget import PromptLayerBudgets
from .llm import LLMClient
from .request_context import RequestContextBuilder
from .runtime import RunContext, ToolRuntime
from .schema import Message, TokenCost, TokenPricing
from .summarizer import CompressionPipeline, ContextCollapser, MessageCompactor, MessageSummarizer
from .token_accounting import estimate_token_cost
from .tools.base import Tool
from .tools.security import CommandSecurityDecision
from .tools.task_memory_tool import TaskMemoryHook


class Agent:
    """Single agent with basic tools and MCP support."""

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        tools: list[Tool],
        max_steps: int = 50,
        workspace_dir: str = "./workspace",
        token_limit: int = 10000,  # Summary triggered when tokens exceed this value
        core_system_prompt: str | None = None,
        skill_loader: Any | None = None,
        request_context_limit: int = 12,
        context_layer_budgets: PromptLayerBudgets | None = None,
        tool_confirmation_callback: Callable[[str, dict[str, Any], CommandSecurityDecision], Awaitable[bool]]
        | None = None,
        task_memory_hook: TaskMemoryHook | None = None,
        checkpoint_store: CheckpointStore | None = None,
        token_pricing: TokenPricing | None = None,
        preserve_thinking: bool = False,
        show_thinking: bool = False,
        log_thinking: bool = False,
    ):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.token_limit = token_limit
        self.runtime_context = RunContext(
            workspace_dir=Path(workspace_dir),
            checkpoint_store=checkpoint_store,
            task_memory_hook=task_memory_hook,
            tool_confirmation_callback=tool_confirmation_callback,
        )
        self.tool_runtime = ToolRuntime(self.tools, self.runtime_context)
        self.workspace_dir = self.runtime_context.workspace_dir
        self.tool_confirmation_callback = self.runtime_context.tool_confirmation_callback
        self.task_memory_hook = self.runtime_context.task_memory_hook
        self.checkpoint_store = self.runtime_context.checkpoint_store
        self.checkpoint_coordinator = CheckpointCoordinator(self.checkpoint_store)
        self.renderer = AgentConsoleRenderer()
        self.message_summarizer = MessageSummarizer(
            llm_client=self.llm,
            token_limit=self.token_limit,
            renderer=self.renderer,
        )
        self.preserve_thinking = preserve_thinking
        self.show_thinking = show_thinking
        self.log_thinking = log_thinking
        # Cancellation event for interrupting agent execution (set externally, e.g., by Esc key)
        self.cancel_event: Optional[asyncio.Event] = None

        # Compatibility fallback for direct Agent construction without SystemPromptBuilder.
        if "Current Workspace" not in system_prompt and "Current workspace" not in system_prompt:
            workspace_info = f"\n\n## Current Workspace\nYou are currently working in: `{self.workspace_dir.absolute()}`\nAll relative paths will be resolved relative to this directory."
            system_prompt = system_prompt + workspace_info

        self.system_prompt = system_prompt
        self.core_system_prompt = core_system_prompt or system_prompt
        self.request_context_builder = RequestContextBuilder(
            core_prompt=self.core_system_prompt,
            workspace_dir=self.workspace_dir,
            skill_loader=skill_loader,
            max_recent_messages=request_context_limit,
            token_budget=self.token_limit,
            layer_budgets=context_layer_budgets,
        )
        self.compression_pipeline = CompressionPipeline(
            compactor=MessageCompactor(token_limit=self.token_limit, workspace_dir=self.workspace_dir),
            context_collapser=ContextCollapser(token_limit=self.token_limit),
            summarizer=self.message_summarizer,
            request_context_builder=self.request_context_builder,
            token_limit=self.token_limit,
            renderer=self.renderer,
        )

        # Initialize message history
        self.messages: list[Message] = [Message(role="system", content=system_prompt)]

        # Initialize logger
        self.logger = self.runtime_context.logger

        # Token usage from last API response (updated after each LLM call)
        self.api_total_tokens: int = 0
        self.api_prompt_tokens: int = 0
        self.api_completion_tokens: int = 0
        self.api_cached_tokens: int = 0
        self.api_cache_write_tokens: int = 0
        self.cumulative_prompt_tokens: int = 0
        self.cumulative_completion_tokens: int = 0
        self.cumulative_total_tokens: int = 0
        self.cumulative_cached_tokens: int = 0
        self.cumulative_cache_write_tokens: int = 0
        self.token_pricing = token_pricing
        self.last_token_cost: TokenCost | None = None
        self.cumulative_token_cost = TokenCost(
            currency=(token_pricing.currency if token_pricing else "USD"),
        )
        self.last_run_completed: bool = False

    def add_user_message(self, content: str):
        """Add a user message to history."""
        self.messages.append(Message(role="user", content=content))

    def restore_messages(self, messages: list[Message]) -> None:
        """Restore message history from a checkpoint snapshot."""
        if not messages:
            return
        self.messages = messages.copy()

    def truncate_messages(self, length: int) -> None:
        """Truncate message history to a previous length."""
        if length < 1:
            length = 1
        self.messages = self.messages[:length]

    def _extract_last_user_goal(self) -> str:
        """Extract the last user message content as the task goal.

        Used by TaskMemoryHook to name the task on auto-start/resume.
        """
        for msg in reversed(self.messages):
            if msg.role == "user":
                content = msg.content
                if isinstance(content, str):
                    # Use first line or first 120 chars as goal summary
                    first_line = content.strip().split("\n")[0].strip()
                    return first_line[:120] if first_line else "general task"
                return "general task"
        return "general task"

    def _check_cancelled(self) -> bool:
        """Check if agent execution has been cancelled.

        Returns:
            True if cancelled, False otherwise.
        """
        active_cancel_event = self.runtime_context.cancel_event or self.cancel_event
        if active_cancel_event is not None and active_cancel_event.is_set():
            return True
        return False

    def _save_checkpoint(self, step: int, reason: str) -> None:
        """Persist a lightweight recovery snapshot when checkpointing is enabled."""
        self.checkpoint_coordinator.save(
            step=step,
            reason=reason,
            messages=self.messages,
            workspace_dir=self.workspace_dir,
            available_tools=list(self.tools.keys()),
        )

    def _cleanup_incomplete_messages(self):
        """Remove the incomplete assistant message and its partial tool results.

        This ensures message consistency after cancellation by removing
        only the current step's incomplete messages, preserving completed steps.
        """
        # Find the index of the last assistant message
        last_assistant_idx = -1
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].role == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx == -1:
            # No assistant message found, nothing to clean
            return

        # Remove the last assistant message and all tool results after it
        removed_count = len(self.messages) - last_assistant_idx
        if removed_count > 0:
            self.messages = self.messages[:last_assistant_idx]
            self.renderer.incomplete_messages_cleaned(removed_count)

    async def run(self, cancel_event: Optional[asyncio.Event] = None) -> str:
        """Execute agent loop until task is complete or max steps reached.

        Args:
            cancel_event: Optional asyncio.Event that can be set to cancel execution.
                          When set, the agent will stop at the next safe checkpoint
                          (after completing the current step to keep messages consistent).

        Returns:
            The final response content, or error message (including cancellation message).
        """
        # Set cancellation event (can also be set via self.cancel_event before calling run())
        self.last_run_completed = False
        if cancel_event is not None:
            self.cancel_event = cancel_event
        self.runtime_context.cancel_event = self.cancel_event

        # Auto-start/resume task memory if hook is configured
        if self.task_memory_hook is not None:
            goal = self._extract_last_user_goal()
            self.task_memory_hook.start_or_resume_task(goal=goal)
        self._save_checkpoint(step=0, reason="run_started")

        # Start new run, initialize log file
        self.logger.start_new_run()
        self.renderer.log_file(self.logger.get_log_file_path())

        step = 0
        run_start_time = perf_counter()

        while step < self.max_steps:
            # Check for cancellation at start of each step
            if self._check_cancelled():
                self._cleanup_incomplete_messages()
                self._save_checkpoint(step, "cancelled")
                cancel_msg = "Task cancelled by user."
                self.renderer.cancellation(cancel_msg)
                return cancel_msg

            step_start_time = perf_counter()
            # Get tool list for LLM call
            tool_list = list(self.tools.values())

            self.messages = await self.compression_pipeline.compress_before_request(
                messages=self.messages,
                api_total_tokens=self.api_total_tokens,
                tools=tool_list,
            )

            self.renderer.step_header(step + 1, self.max_steps)

            request_messages = self.request_context_builder.build(
                self.messages,
                tools=tool_list,
                token_budget=self.token_limit,
            )

            # Log LLM request and call LLM with Tool objects directly
            self.logger.log_request(messages=request_messages, tools=tool_list)

            try:
                response = await self.llm.generate(messages=request_messages, tools=tool_list)
            except Exception as e:
                # Check if it's a retry exhausted error
                from .retry import RetryExhaustedError

                if isinstance(e, RetryExhaustedError):
                    error_msg = f"LLM call failed after {e.attempts} retries\nLast error: {str(e.last_exception)}"
                    self.renderer.llm_error(error_msg, retry_exhausted=True)
                else:
                    error_msg = f"LLM call failed: {str(e)}"
                    self.renderer.llm_error(error_msg)
                self._save_checkpoint(step, "failed")
                return error_msg

            # Accumulate API reported token usage
            if response.usage:
                token_cost = estimate_token_cost(response.usage, self.token_pricing)
                self.api_prompt_tokens = response.usage.prompt_tokens
                self.api_completion_tokens = response.usage.completion_tokens
                self.api_total_tokens = response.usage.total_tokens
                self.api_cached_tokens = response.usage.cached_tokens
                self.api_cache_write_tokens = response.usage.cache_write_tokens
                self.last_token_cost = token_cost
                self.cumulative_prompt_tokens += response.usage.prompt_tokens
                self.cumulative_completion_tokens += response.usage.completion_tokens
                self.cumulative_total_tokens += response.usage.total_tokens
                self.cumulative_cached_tokens += response.usage.cached_tokens
                self.cumulative_cache_write_tokens += response.usage.cache_write_tokens
                if token_cost is not None:
                    self.cumulative_token_cost.input_cost += token_cost.input_cost
                    self.cumulative_token_cost.output_cost += token_cost.output_cost
                    self.cumulative_token_cost.cache_read_cost += token_cost.cache_read_cost
                    self.cumulative_token_cost.cache_write_cost += token_cost.cache_write_cost
                    self.cumulative_token_cost.total_cost += token_cost.total_cost
                self.renderer.token_usage(
                    step=step + 1,
                    prompt_tokens=self.api_prompt_tokens,
                    completion_tokens=self.api_completion_tokens,
                    total_tokens=self.api_total_tokens,
                    cached_tokens=self.api_cached_tokens,
                    cache_write_tokens=self.api_cache_write_tokens,
                    cost=token_cost,
                )

            # Log LLM response
            self.logger.log_response(
                content=response.content,
                thinking=response.thinking if self.log_thinking else None,
                tool_calls=response.tool_calls,
                finish_reason=response.finish_reason,
                usage=response.usage,
            )

            # Add assistant message
            thinking_for_history = response.thinking if self.preserve_thinking else None
            assistant_msg = Message(
                role="assistant",
                content=response.content,
                thinking=thinking_for_history,
                tool_calls=response.tool_calls,
            )
            self.messages.append(assistant_msg)
            self._save_checkpoint(step, "assistant_response")

            # Print thinking if present
            if self.show_thinking and response.thinking:
                self.renderer.thinking(response.thinking)

            # Print assistant response
            if response.content:
                self.renderer.assistant_response(response.content)

            # Check if task is complete (no tool calls)
            if not response.tool_calls:
                step_elapsed = perf_counter() - step_start_time
                total_elapsed = perf_counter() - run_start_time
                self.renderer.step_completed(step + 1, step_elapsed, total_elapsed)
                if self.task_memory_hook is not None:
                    self.task_memory_hook.finish_task(summary=response.content)
                self._save_checkpoint(step, "completed")
                self.last_run_completed = True
                return response.content

            # Check for cancellation before executing tools
            if self._check_cancelled():
                self._cleanup_incomplete_messages()
                self._save_checkpoint(step, "cancelled")
                cancel_msg = "Task cancelled by user."
                self.renderer.cancellation(cancel_msg)
                return cancel_msg

            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_call_id = tool_call.id
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments

                self.renderer.tool_call(function_name, arguments)

                result = await self.tool_runtime.execute(function_name, arguments)

                self.renderer.tool_result(result)

                # Add tool result message
                tool_msg = Message(
                    role="tool",
                    content=result.content if result.success else f"Error: {result.error}",
                    tool_call_id=tool_call_id,
                    name=function_name,
                )
                self.messages.append(tool_msg)
                self._save_checkpoint(step, "tool_result")

                # Check for cancellation after each tool execution
                if self._check_cancelled():
                    self._cleanup_incomplete_messages()
                    self._save_checkpoint(step, "cancelled")
                    cancel_msg = "Task cancelled by user."
                    self.renderer.cancellation(cancel_msg)
                    return cancel_msg

            step_elapsed = perf_counter() - step_start_time
            total_elapsed = perf_counter() - run_start_time
            self.renderer.step_completed(step + 1, step_elapsed, total_elapsed)

            step += 1

        # Max steps reached
        error_msg = f"Task couldn't be completed after {self.max_steps} steps."
        self.renderer.max_steps_reached(error_msg)
        self._save_checkpoint(step, "max_steps")
        return error_msg

    def get_history(self) -> list[Message]:
        """Get message history."""
        return self.messages.copy()
