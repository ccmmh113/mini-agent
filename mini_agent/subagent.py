"""Subagent runner for delegated, isolated tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import SubagentConfig
from .context_budget import PromptLayerBudgets
from .llm import LLMClient
from .schema import TokenCost, TokenPricing
from .tools.base import Tool


DEFAULT_SUBAGENT_SYSTEM_PROMPT = """你是 Mini-Agent 的子代理，负责完成父 Agent 委派的局部任务。

工作边界：
- 你只处理当前子任务，不继承父 Agent 的完整对话历史。
- 优先收集必要上下文，避免扩大任务范围。
- 不要调用未提供给你的工具。
- 除非任务明确要求，否则不要修改文件。
- 最终返回简洁、可交给父 Agent 继续推理的中文总结。
"""


@dataclass
class SubagentResult:
    """Result returned from an isolated subagent run."""

    content: str
    completed: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    cost: TokenCost | None = None

    def to_tool_content(self) -> str:
        status = "completed" if self.completed else "stopped"
        lines = [
            f"Subagent status: {status}",
            "",
            "Subagent result:",
            self.content.strip() or "(empty result)",
        ]
        if self.total_tokens:
            lines.extend(
                [
                    "",
                    "Subagent token usage:",
                    (
                        f"prompt={self.prompt_tokens}, completion={self.completion_tokens}, "
                        f"total={self.total_tokens}, cached={self.cached_tokens}, "
                        f"cache_write={self.cache_write_tokens}"
                    ),
                ]
            )
        if self.cost is not None:
            lines.append(f"Subagent estimated cost: {self.cost.total_cost:.6f} {self.cost.currency}")
        return "\n".join(lines)


class SubagentRunner:
    """Run a child Agent with isolated messages and a constrained tool set."""

    def __init__(
        self,
        llm_client: LLMClient,
        workspace_dir: str | Path,
        tools_provider: Callable[[], list[Tool]],
        config: SubagentConfig,
        system_prompt: str = DEFAULT_SUBAGENT_SYSTEM_PROMPT,
        core_system_prompt: str | None = None,
        context_layer_budgets: PromptLayerBudgets | None = None,
        tool_confirmation_callback=None,
        token_pricing: TokenPricing | None = None,
        cancel_event=None,
    ):
        self.llm_client = llm_client
        self.workspace_dir = Path(workspace_dir)
        self.tools_provider = tools_provider
        self.config = config
        self.system_prompt = system_prompt
        self.core_system_prompt = core_system_prompt or system_prompt
        self.context_layer_budgets = context_layer_budgets
        self.tool_confirmation_callback = tool_confirmation_callback
        self.token_pricing = token_pricing
        self.cancel_event = cancel_event

    async def run(self, description: str, prompt: str) -> SubagentResult:
        """Execute a delegated task in an isolated child Agent."""

        from .agent import Agent

        child_tools = self._select_tools()
        child_agent = Agent(
            llm_client=self.llm_client,
            system_prompt=self.system_prompt,
            tools=child_tools,
            max_steps=self.config.max_steps,
            workspace_dir=str(self.workspace_dir),
            token_limit=self.config.token_limit,
            core_system_prompt=self.core_system_prompt,
            request_context_limit=self.config.request_context_limit,
            context_layer_budgets=self.context_layer_budgets,
            tool_confirmation_callback=self.tool_confirmation_callback,
            token_pricing=self.token_pricing,
        )
        child_agent.add_user_message(self._build_child_task(description, prompt))
        content = await child_agent.run(cancel_event=self.cancel_event)
        return SubagentResult(
            content=content,
            completed=child_agent.last_run_completed,
            prompt_tokens=child_agent.cumulative_prompt_tokens,
            completion_tokens=child_agent.cumulative_completion_tokens,
            total_tokens=child_agent.cumulative_total_tokens,
            cached_tokens=child_agent.cumulative_cached_tokens,
            cache_write_tokens=child_agent.cumulative_cache_write_tokens,
            cost=child_agent.cumulative_token_cost if child_agent.cumulative_token_cost.total_cost else None,
        )

    def _select_tools(self) -> list[Tool]:
        allowed = set(self.config.allowed_tools)
        selected: list[Tool] = []
        for tool in self.tools_provider():
            if tool.name not in allowed:
                continue
            if tool.name == "task" and not self.config.allow_nested_subagent:
                continue
            selected.append(tool)
        return selected

    def _build_child_task(self, description: str, prompt: str) -> str:
        return (
            f"子任务说明：{description.strip()}\n\n"
            f"具体要求：\n{prompt.strip()}\n\n"
            "请只围绕这个子任务工作，最后返回可以给父 Agent 使用的结论、依据和限制。"
        )
