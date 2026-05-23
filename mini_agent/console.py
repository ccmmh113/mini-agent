"""Console rendering helpers for the agent harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .schema import TokenCost
from .tools.base import ToolResult
from .utils import calculate_display_width


def _safe_print(text: str = "") -> None:
    """Print text while degrading characters unsupported by the terminal."""

    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe_text)


class Colors:
    """Terminal color definitions."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class AgentConsoleRenderer:
    """Render agent runtime events to a terminal."""

    def log_file(self, path: Path) -> None:
        _safe_print(f"{Colors.DIM}📝 Log file: {path}{Colors.RESET}")

    def incomplete_messages_cleaned(self, count: int) -> None:
        _safe_print(f"{Colors.DIM}   Cleaned up {count} incomplete message(s){Colors.RESET}")

    def summary_triggered(self, estimated_tokens: int, api_total_tokens: int, token_limit: int) -> None:
        _safe_print(
            f"\n{Colors.BRIGHT_YELLOW}📊 Token usage - Local estimate: {estimated_tokens}, "
            f"API reported: {api_total_tokens}, Limit: {token_limit}{Colors.RESET}"
        )
        _safe_print(f"{Colors.BRIGHT_YELLOW}🔄 L5 Auto-Compact: summarizing full historical context...{Colors.RESET}")

    def summary_insufficient_messages(self) -> None:
        _safe_print(f"{Colors.BRIGHT_YELLOW}⚠️  L5 Auto-Compact skipped: no compressible historical context{Colors.RESET}")

    def summary_round_success(self, round_num: int) -> None:
        _safe_print(f"{Colors.BRIGHT_GREEN}✓ Summary for round {round_num} generated successfully{Colors.RESET}")

    def summary_round_failed(self, round_num: int, error: Exception) -> None:
        _safe_print(f"{Colors.BRIGHT_RED}✗ Summary generation failed for round {round_num}: {error}{Colors.RESET}")

    def summary_completed(
        self,
        estimated_tokens: int,
        new_tokens: int,
        user_count: int,
        summary_count: int,
    ) -> None:
        _safe_print(f"{Colors.BRIGHT_GREEN}✓ Summary completed, local tokens: {estimated_tokens} → {new_tokens}{Colors.RESET}")
        _safe_print(f"{Colors.DIM}  Structure: system + {user_count} user messages + {summary_count} summaries{Colors.RESET}")
        _safe_print(f"{Colors.DIM}  Note: API token count will update on next LLM call{Colors.RESET}")

    def step_header(self, step: int, max_steps: int) -> None:
        box_width = 58
        step_text = f"{Colors.BOLD}{Colors.BRIGHT_CYAN}💭 Step {step}/{max_steps}{Colors.RESET}"
        step_display_width = calculate_display_width(step_text)
        padding = max(0, box_width - 1 - step_display_width)

        _safe_print(f"\n{Colors.DIM}╭{'─' * box_width}╮{Colors.RESET}")
        _safe_print(f"{Colors.DIM}│{Colors.RESET} {step_text}{' ' * padding}{Colors.DIM}│{Colors.RESET}")
        _safe_print(f"{Colors.DIM}╰{'─' * box_width}╯{Colors.RESET}")

    def llm_error(self, message: str, retry_exhausted: bool = False) -> None:
        label = "❌ Retry failed:" if retry_exhausted else "❌ Error:"
        _safe_print(f"\n{Colors.BRIGHT_RED}{label}{Colors.RESET} {message}")

    def token_usage(
        self,
        *,
        step: int | None = None,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
        cost: TokenCost | None = None,
    ) -> None:
        label = f"Step {step} tokens" if step is not None else "Tokens"
        _safe_print(
            f"{Colors.DIM}   {label}: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
            + (f", cached={cached_tokens}" if cached_tokens or cache_write_tokens else "")
            + (f", cache_write={cache_write_tokens}" if cache_write_tokens else "")
            + f"{Colors.RESET}"
        )
        if cost is not None:
            _safe_print(
                f"{Colors.DIM}   Cost estimate: input={cost.input_cost:.6f}, "
                f"output={cost.output_cost:.6f}, cache_read={cost.cache_read_cost:.6f}, "
                f"cache_write={cost.cache_write_cost:.6f}, total={cost.total_cost:.6f} "
                f"{cost.currency}{Colors.RESET}"
            )

    def thinking(self, text: str) -> None:
        _safe_print(f"\n{Colors.BOLD}{Colors.MAGENTA}🧠 Thinking:{Colors.RESET}")
        _safe_print(f"{Colors.DIM}{text}{Colors.RESET}")

    def assistant_response(self, text: str) -> None:
        _safe_print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}🤖 Assistant:{Colors.RESET}")
        _safe_print(text)

    def step_completed(self, step: int, step_elapsed: float, total_elapsed: float) -> None:
        _safe_print(f"\n{Colors.DIM}⏱️  Step {step} completed in {step_elapsed:.2f}s (total: {total_elapsed:.2f}s){Colors.RESET}")

    def cancellation(self, message: str) -> None:
        _safe_print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {message}{Colors.RESET}")

    def tool_call(self, function_name: str, arguments: dict[str, Any]) -> None:
        _safe_print(f"\n{Colors.BRIGHT_YELLOW}🔧 Tool Call:{Colors.RESET} {Colors.BOLD}{Colors.CYAN}{function_name}{Colors.RESET}")
        _safe_print(f"{Colors.DIM}   Arguments:{Colors.RESET}")

        truncated_args: dict[str, Any] = {}
        for key, value in arguments.items():
            value_str = str(value)
            truncated_args[key] = value_str[:200] + "..." if len(value_str) > 200 else value

        args_json = json.dumps(truncated_args, indent=2, ensure_ascii=False)
        for line in args_json.split("\n"):
            _safe_print(f"   {Colors.DIM}{line}{Colors.RESET}")

    def tool_result(self, result: ToolResult) -> None:
        if result.success:
            result_text = result.content
            if len(result_text) > 300:
                result_text = result_text[:300] + f"{Colors.DIM}...{Colors.RESET}"
            _safe_print(f"{Colors.BRIGHT_GREEN}✓ Result:{Colors.RESET} {result_text}")
        else:
            _safe_print(f"{Colors.BRIGHT_RED}✗ Error:{Colors.RESET} {Colors.RED}{result.error}{Colors.RESET}")

    def max_steps_reached(self, message: str) -> None:
        _safe_print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {message}{Colors.RESET}")
