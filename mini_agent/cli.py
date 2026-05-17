"""
Mini Agent - Interactive Runtime Example

Usage:
    mini-agent [--workspace DIR] [--task TASK]

Examples:
    mini-agent                              # Use current directory as workspace (interactive mode)
    mini-agent --workspace /path/to/dir     # Use specific workspace directory (interactive mode)
    mini-agent --task "create a file"       # Execute a task non-interactively
"""

import argparse
import asyncio
import hashlib
import platform
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from mini_agent import LLMClient
from mini_agent.agent import Agent
from mini_agent.checkpoint import CheckpointStore
from mini_agent.config import Config
from mini_agent.memory.markdown_store import MarkdownMemoryStore
from mini_agent.prompt_builder import SystemPromptBuilder
from mini_agent.schema import LLMProvider
from mini_agent.tool_registry import ToolRegistry
from mini_agent.tools.base import Tool
from mini_agent.tools.security import CommandSecurityDecision
from mini_agent.tools.task_memory_tool import (
    EpisodeMemoryStore,
    TaskMemoryHook,
    TaskMemoryStore,
    format_task_memory,
    verify_task_artifacts,
)
from mini_agent.utils import calculate_display_width, pad_to_width, truncate_with_ellipsis


# ANSI color codes
class Colors:
    """Terminal color definitions"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


DEFAULT_PANEL_WIDTH = 76
MAX_PANEL_WIDTH = 96


def terminal_panel_width() -> int:
    """Return a comfortable panel width for the current terminal."""

    columns = shutil.get_terminal_size((DEFAULT_PANEL_WIDTH, 20)).columns
    return max(64, min(MAX_PANEL_WIDTH, columns - 4))


def color(text: object, value: str) -> str:
    """Wrap text in a terminal color."""

    return f"{value}{text}{Colors.RESET}"


def dim(text: object) -> str:
    return color(text, Colors.DIM)


def label(text: str, width: int = 16) -> str:
    return f"{Colors.DIM}{pad_to_width(text, width)}{Colors.RESET}"


def status_badge(text: str, color_value: str = Colors.BRIGHT_GREEN) -> str:
    return f"{color_value}{text}{Colors.RESET}"


def print_panel(
    title: str,
    lines: Sequence[str],
    *,
    accent: str = Colors.BRIGHT_CYAN,
    width: int | None = None,
) -> None:
    """Print a compact terminal panel with CJK/emoji-aware padding."""

    panel_width = width or terminal_panel_width()
    inner_width = panel_width
    title_text = f" {title} "
    title_width = calculate_display_width(title_text)
    right_rule = max(1, inner_width - title_width)

    print(f"\n{accent}╭{title_text}{'─' * right_rule}╮{Colors.RESET}")
    for raw_line in lines:
        line = truncate_with_ellipsis(raw_line, inner_width - 2)
        padding = max(0, inner_width - 2 - calculate_display_width(line))
        print(f"{accent}│{Colors.RESET} {line}{' ' * padding} {accent}│{Colors.RESET}")
    print(f"{accent}╰{'─' * inner_width}╯{Colors.RESET}\n")


def print_rule(width: int | None = None) -> None:
    print(f"{Colors.DIM}{'─' * (width or terminal_panel_width())}{Colors.RESET}")


def format_duration(started_at: datetime) -> str:
    duration = datetime.now() - started_at
    hours, remainder = divmod(int(duration.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_size(size: int) -> str:
    if size < 1024:
        return f"{size:,} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.1f} MB"


def configure_output_encoding() -> None:
    """Keep Windows terminals from crashing on Unicode status output."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            encoding = (getattr(stream, "encoding", None) or "").lower()
            if "utf" in encoding:
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def get_log_directory() -> Path:
    """Get the log directory path."""
    return Path.home() / ".mini-agent" / "log"


def show_log_directory(open_file_manager: bool = True) -> None:
    """Show log directory contents and optionally open file manager.

    Args:
        open_file_manager: Whether to open the system file manager
    """
    log_dir = get_log_directory()

    if not log_dir.exists() or not log_dir.is_dir():
        print_panel(
            "日志目录",
            [
                f"{label('路径')} {log_dir}",
                f"{label('状态')} {status_badge('不存在', Colors.BRIGHT_RED)}",
            ],
            accent=Colors.BRIGHT_RED,
        )
        return

    log_files = list(log_dir.glob("*.log"))

    if not log_files:
        print_panel(
            "日志目录",
            [
                f"{label('路径')} {log_dir}",
                f"{label('状态')} {status_badge('暂无日志', Colors.BRIGHT_YELLOW)}",
            ],
            accent=Colors.BRIGHT_YELLOW,
        )
        return

    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    lines = [
        f"{label('路径')} {log_dir}",
        f"{label('最近文件')} {len(log_files)} 个日志文件，按更新时间排序",
        "",
    ]
    for i, log_file in enumerate(log_files[:10], 1):
        mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
        size = log_file.stat().st_size
        lines.append(
            f"{Colors.BRIGHT_GREEN}{i:2d}.{Colors.RESET} "
            f"{Colors.BRIGHT_WHITE}{log_file.name}{Colors.RESET} "
            f"{dim(mtime.strftime('%Y-%m-%d %H:%M:%S'))} {dim(format_size(size))}"
        )

    if len(log_files) > 10:
        lines.append(dim(f"... 还有 {len(log_files) - 10} 个文件"))
    print_panel("日志目录", lines, accent=Colors.BRIGHT_CYAN)

    # Open file manager
    if open_file_manager:
        _open_directory_in_file_manager(log_dir)

    print()


def _open_directory_in_file_manager(directory: Path) -> None:
    """Open directory in system file manager (cross-platform)."""
    system = platform.system()

    try:
        if system == "Darwin":
            subprocess.run(["open", str(directory)], check=False)
        elif system == "Windows":
            subprocess.run(["explorer", str(directory)], check=False)
        elif system == "Linux":
            subprocess.run(["xdg-open", str(directory)], check=False)
    except FileNotFoundError:
        print(f"{Colors.YELLOW}Could not open file manager. Please navigate manually.{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.YELLOW}Error opening file manager: {e}{Colors.RESET}")


def read_log_file(filename: str) -> None:
    """Read and display a specific log file.

    Args:
        filename: The log filename to read
    """
    log_dir = get_log_directory()
    log_file = log_dir / filename

    if not log_file.exists() or not log_file.is_file():
        print_panel("读取日志", [f"{label('文件')} {log_file}", f"{label('状态')} 未找到"], accent=Colors.BRIGHT_RED)
        return

    print_panel("读取日志", [f"{label('文件')} {log_file}", f"{label('状态')} 开始输出"], accent=Colors.BRIGHT_CYAN)

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        print(content)
        print_panel("读取日志", [f"{label('状态')} 已到达文件末尾"], accent=Colors.BRIGHT_GREEN)
    except Exception as e:
        print_panel("读取日志", [f"{label('错误')} {e}"], accent=Colors.BRIGHT_RED)


def print_banner():
    """Print welcome banner with proper alignment"""
    print_panel(
        "Mini Agent",
        [
            f"{Colors.BOLD}{Colors.BRIGHT_CYAN}本地 Agent Harness · CLI 工作台{Colors.RESET}",
            f"{dim('输入任务直接执行；输入 /help 查看命令；Esc 可中断正在运行的 Agent。')}",
        ],
        accent=Colors.BRIGHT_CYAN,
    )


def print_help():
    """Print help information"""
    print_panel(
        "命令",
        [
            f"{status_badge('/help', Colors.BRIGHT_GREEN)}              显示帮助",
            f"{status_badge('/clear', Colors.BRIGHT_GREEN)}             清空当前会话历史，保留 system prompt",
            f"{status_badge('/history', Colors.BRIGHT_GREEN)}           查看当前会话消息数量",
            f"{status_badge('/resume', Colors.BRIGHT_GREEN)}            从最新 checkpoint 恢复上下文",
            f"{status_badge('/task', Colors.BRIGHT_GREEN)}              查看当前任务记忆与产物校验",
            f"{status_badge('/memory', Colors.BRIGHT_GREEN)}            查看长期记忆状态",
            f"{status_badge('/memory review', Colors.BRIGHT_GREEN)}     浏览长期记忆文件",
            f"{status_badge('/memory delete <name>', Colors.BRIGHT_GREEN)} 删除长期记忆文件",
            f"{status_badge('/stats', Colors.BRIGHT_GREEN)}             查看消息、token、cache、成本统计",
            f"{status_badge('/log', Colors.BRIGHT_GREEN)}               查看日志目录",
            f"{status_badge('/log <file>', Colors.BRIGHT_GREEN)}        读取指定日志",
            f"{status_badge('/exit', Colors.BRIGHT_GREEN)}              退出程序，也可用 exit/quit/q",
        ],
        accent=Colors.BRIGHT_YELLOW,
    )
    print_panel(
        "快捷键",
        [
            f"{status_badge('Esc', Colors.BRIGHT_CYAN)}       中断当前 Agent 执行",
            f"{status_badge('Ctrl+C', Colors.BRIGHT_CYAN)}    退出程序",
            f"{status_badge('Ctrl+U', Colors.BRIGHT_CYAN)}    清空当前输入行",
            f"{status_badge('Ctrl+L', Colors.BRIGHT_CYAN)}    清屏",
            f"{status_badge('Ctrl+J', Colors.BRIGHT_CYAN)}    输入换行",
            f"{status_badge('Tab', Colors.BRIGHT_CYAN)}       命令补全",
            f"{status_badge('↑/↓', Colors.BRIGHT_CYAN)}       浏览历史输入",
            f"{status_badge('→', Colors.BRIGHT_CYAN)}         接受自动建议",
        ],
        accent=Colors.BRIGHT_CYAN,
    )


def print_session_info(agent: Agent, workspace_dir: Path, model: str):
    """Print session information with proper alignment"""
    tool_name_list = list(agent.tools.keys()) if isinstance(agent.tools, dict) else [tool.name for tool in agent.tools]
    tool_names = ", ".join(tool_name_list[:8])
    if len(tool_name_list) > 8:
        tool_names += f", +{len(tool_name_list) - 8}"
    print_panel(
        "当前会话",
        [
            f"{label('模型')} {Colors.BRIGHT_WHITE}{model}{Colors.RESET}",
            f"{label('工作区')} {workspace_dir}",
            f"{label('消息')} {len(agent.messages)} 条",
            f"{label('工具')} {len(tool_name_list)} 个 {dim(tool_names)}",
            "",
            f"{dim('提示：输入 /help 查看命令，输入 /exit 退出。')}",
        ],
        accent=Colors.BRIGHT_BLUE,
    )


def print_stats(agent: Agent, session_start: datetime):
    """Print session statistics"""
    # Count different types of messages
    user_msgs = sum(1 for m in agent.messages if m.role == "user")
    assistant_msgs = sum(1 for m in agent.messages if m.role == "assistant")
    tool_msgs = sum(1 for m in agent.messages if m.role == "tool")

    lines = [
        f"{label('运行时长')} {format_duration(session_start)}",
        f"{label('消息')} 总计 {len(agent.messages)} 条 "
        f"{dim(f'user={user_msgs}, assistant={assistant_msgs}, tool={tool_msgs}')}",
        f"{label('可用工具')} {len(agent.tools)} 个",
    ]
    if agent.cumulative_total_tokens > 0:
        lines.append(
            f"{label('Token')} {Colors.BRIGHT_MAGENTA}{agent.cumulative_total_tokens:,}{Colors.RESET} "
            f"{dim(f'prompt={agent.cumulative_prompt_tokens:,}, completion={agent.cumulative_completion_tokens:,}')}"
        )
        if agent.cumulative_cached_tokens or agent.cumulative_cache_write_tokens:
            lines.append(
                f"{label('Cache')} read={agent.cumulative_cached_tokens:,}, "
                f"write={agent.cumulative_cache_write_tokens:,}"
            )
        if agent.cumulative_token_cost.total_cost > 0:
            cost = agent.cumulative_token_cost
            lines.append(
                f"{label('成本估算')} {Colors.BRIGHT_MAGENTA}{cost.total_cost:.6f} {cost.currency}{Colors.RESET}"
            )
            lines.append(
                f"{label('成本明细')} "
                f"{dim(f'input={cost.input_cost:.6f}, output={cost.output_cost:.6f}, cache_read={cost.cache_read_cost:.6f}, cache_write={cost.cache_write_cost:.6f}')}"
            )
    else:
        lines.append(f"{label('Token')} 尚未产生 API token 统计")
    print_panel("会话统计", lines, accent=Colors.BRIGHT_MAGENTA)


def print_current_task(workspace_dir: Path) -> None:
    """Print the active structured task memory."""

    memory_file = workspace_dir / ".mini_agent" / "task_memory.json"
    store = TaskMemoryStore(str(memory_file))
    data = store.load()
    artifact_verification = verify_task_artifacts(data, workspace_dir)
    print_panel("任务进度", [f"{label('工作记忆')} {memory_file}"], accent=Colors.BRIGHT_CYAN)
    print(format_task_memory(data, artifact_verification=artifact_verification))
    if artifact_verification["total"]:
        print_panel(
            "产物校验",
            [
                f"{label('matched')} {artifact_verification['matched']}",
                f"{label('changed')} {artifact_verification['changed']}",
                f"{label('missing')} {artifact_verification['missing']}",
                f"{label('outside')} {artifact_verification['outside_workspace']}",
                f"{label('not verified')} {artifact_verification['not_verifiable']}",
            ],
            accent=Colors.BRIGHT_YELLOW,
        )


def print_memory_status(workspace_dir: Path) -> None:
    """Print lightweight long-term memory status."""

    memory_dir = workspace_dir / ".memory"
    episode_file = workspace_dir / ".mini_agent" / "episodes.jsonl"
    memories = MarkdownMemoryStore(memory_dir).load()
    episodes = EpisodeMemoryStore(str(episode_file)).load().get("episodes", [])

    by_type: dict[str, int] = {}
    for memory in memories:
        by_type[memory.type] = by_type.get(memory.type, 0) + 1

    lines = [
        f"{label('长期记忆')} {len(memories)} 条",
        f"{label('记忆目录')} {memory_dir}",
        f"{label('Episodes')} {len(episodes)} 条可复盘记录",
    ]
    if by_type:
        lines.append(f"{label('类型')} {', '.join(f'{key}={value}' for key, value in sorted(by_type.items()))}")
    print_panel("长期记忆", lines, accent=Colors.BRIGHT_CYAN)


def build_memory_review_rows(workspace_dir: Path, limit: int = 20) -> list[dict[str, object]]:
    """Build read-only lightweight memory review rows for CLI display."""

    rows: list[dict[str, object]] = []
    for memory in MarkdownMemoryStore(workspace_dir / ".memory").search(limit=limit):
        rows.append(
            {
                "name": memory.name,
                "type": memory.type,
                "description": memory.description,
                "created_at": memory.created_at,
                "updated_at": memory.updated_at,
                "path": str(memory.path),
                "content": memory.content,
            }
        )
    return rows


def print_memory_review(workspace_dir: Path, limit: int = 20) -> None:
    """Print lightweight long-term memory documents."""

    rows = build_memory_review_rows(workspace_dir, limit=limit)
    if not rows:
        print_panel("记忆浏览", ["暂无长期记忆文件。"], accent=Colors.BRIGHT_YELLOW)
        return

    lines = []
    for index, row in enumerate(rows, 1):
        updated = row["updated_at"] or "unknown"
        memory_type = row["type"]
        lines.append(
            f"{Colors.BRIGHT_GREEN}{index:2d}.{Colors.RESET} "
            f"{Colors.BRIGHT_WHITE}{row['name']}{Colors.RESET} "
            f"{dim(f'[{memory_type}] updated={updated}')}"
        )
        if row["description"]:
            lines.append(f"    {row['description']}")
        lines.append(f"    {dim(row['path'])}")
    print_panel("记忆浏览", lines, accent=Colors.BRIGHT_CYAN)


def delete_memory(workspace_dir: Path, memory_id: str) -> None:
    """Delete a lightweight long-term memory document by file stem."""

    deleted = MarkdownMemoryStore(workspace_dir / ".memory").delete(memory_id)
    if deleted:
        print_panel("长期记忆", [f"{label('已删除')} {memory_id}"], accent=Colors.BRIGHT_GREEN)
    else:
        print_panel("长期记忆", [f"{label('未找到')} {memory_id}"], accent=Colors.BRIGHT_YELLOW)


def print_resume_checkpoint_summary(summary: str) -> None:
    """Print the latest checkpoint summary."""
    print_panel("Checkpoint", [dim(line) for line in summary.splitlines()], accent=Colors.BRIGHT_YELLOW)


def print_checkpoint_validation(issues: list[str]) -> None:
    """Print checkpoint validation warnings."""
    if not issues:
        print_panel("Checkpoint 校验", [f"{label('状态')} {status_badge('通过', Colors.BRIGHT_GREEN)}"], accent=Colors.BRIGHT_GREEN)
        return
    print_panel("Checkpoint 校验", [dim(f"- {issue}") for issue in issues], accent=Colors.BRIGHT_YELLOW)


def print_checkpoint_message_stats(total: int, valid: int, dropped: int) -> None:
    """Print checkpoint message validation statistics."""
    print_panel(
        "Checkpoint 消息",
        [f"{label('total')} {total}", f"{label('valid')} {valid}", f"{label('dropped')} {dropped}"],
        accent=Colors.BRIGHT_CYAN,
    )
    if dropped > 0:
        print(f"{Colors.YELLOW}⚠️  部分非法消息会在恢复时丢弃{Colors.RESET}")


def try_restore_checkpoint(agent: Agent, checkpoint_store: CheckpointStore, workspace_dir: Path) -> bool:
    """Restore latest checkpoint into the current agent if validation passes."""
    validation = checkpoint_store.validate_messages()
    print_checkpoint_message_stats(
        validation["total"],
        validation["valid"],
        validation["dropped"],
    )
    issues = checkpoint_store.validate_for_workspace(workspace_dir)
    print_checkpoint_validation(issues)
    if issues:
        return False

    task_memory_file = workspace_dir / ".mini_agent" / "task_memory.json"
    artifact_verification = verify_task_artifacts(TaskMemoryStore(str(task_memory_file)).load(), workspace_dir)
    drift_count = (
        artifact_verification["changed"]
        + artifact_verification["missing"]
        + artifact_verification["outside_workspace"]
    )
    if drift_count:
        print_panel(
            "工作记忆漂移",
            [
                f"{label('changed')} {artifact_verification['changed']}",
                f"{label('missing')} {artifact_verification['missing']}",
                f"{label('outside')} {artifact_verification['outside_workspace']}",
                dim("恢复会继续；记录的 hash 用于审计，不代表当前文件状态。"),
            ],
            accent=Colors.BRIGHT_YELLOW,
        )

    restored_messages = validation["valid_messages"]
    if not restored_messages:
        print_panel("Checkpoint", ["最新 checkpoint 无法恢复。"], accent=Colors.BRIGHT_YELLOW)
        return False

    agent.restore_messages(restored_messages)
    restore_status = checkpoint_store.get_restore_status()
    lines = [f"{label('状态')} {status_badge('已恢复最新 checkpoint', Colors.BRIGHT_GREEN)}"]
    if restore_status:
        lines.append(dim(restore_status))
    print_panel("Checkpoint", lines, accent=Colors.BRIGHT_GREEN)
    return True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Mini Agent - AI assistant with file tools and MCP support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mini-agent                              # Use current directory as workspace
  mini-agent --workspace /path/to/dir     # Use specific workspace directory
  mini-agent log                          # Show log directory and recent files
  mini-agent log agent_run_xxx.log        # Read a specific log file
        """,
    )
    parser.add_argument(
        "--workspace",
        "-w",
        type=str,
        default=None,
        help="Workspace directory (default: current directory)",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default=None,
        help="Execute a task non-interactively and exit",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="mini-agent 0.1.0",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # log subcommand
    log_parser = subparsers.add_parser("log", help="Show log directory or read log files")
    log_parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="Log filename to read (optional, shows directory if omitted)",
    )

    return parser.parse_args()


def notify_tool_status(level: str, message: str) -> None:
    """Render tool registry status messages for the CLI."""

    if level == "success":
        print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")
    elif level == "warning":
        print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")
    elif level == "detail":
        print(f"{Colors.DIM}  {message}{Colors.RESET}")
    else:
        print(f"{Colors.BRIGHT_CYAN}{message}{Colors.RESET}")


async def initialize_base_tools(config: Config):
    """Initialize base tools (independent of workspace)

    These tools are loaded from package configuration and don't depend on workspace.
    Note: File tools are now workspace-dependent and initialized in add_workspace_tools()

    Args:
        config: Configuration object

    Returns:
        Tuple of (list of tools, skill loader if skills enabled)
    """

    result = await ToolRegistry(config, notify_tool_status).build_base_tools()
    print()  # Empty line separator
    return result.tools, result.skill_loader


def add_workspace_tools(tools: List[Tool], config: Config, workspace_dir: Path, llm_client: LLMClient):
    """Add workspace-dependent tools

    These tools need to know the workspace directory.

    Args:
        tools: Existing tools list to add to
        config: Configuration object
        workspace_dir: Workspace directory path
    """
    ToolRegistry(config, notify_tool_status).add_workspace_tools(tools, workspace_dir, llm_client=llm_client)


async def _quiet_cleanup():
    """Clean up runtime resources, suppressing noisy async teardown tracebacks."""
    # Silence the asyncgen finalization noise that anyio/mcp emits when
    # stdio_client's task group is torn down across tasks.  The handler is
    # intentionally NOT restored: asyncgen finalization happens during
    # asyncio.run() shutdown (after run_agent returns), so restoring the
    # handler here would still let the noise through.  Since this runs
    # right before process exit, swallowing late exceptions is safe.
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(lambda _loop, _ctx: None)
    try:
        from mini_agent.tools.bash_tool import BackgroundShellManager

        await BackgroundShellManager.terminate_all()
    except Exception:
        pass
    try:
        from mini_agent.tools.mcp_loader import cleanup_mcp_connections

        await cleanup_mcp_connections()
    except Exception:
        pass


async def run_agent(workspace_dir: Path, task: str = None):
    """Run Agent in interactive or non-interactive mode.

    Args:
        workspace_dir: Workspace directory path
        task: If provided, execute this task and exit (non-interactive mode)
    """
    session_start = datetime.now()

    # 1. Load configuration from package directory
    config_path = Config.get_default_config_path()

    if not config_path.exists():
        user_config_dir = Path.home() / ".mini-agent" / "config"
        print_panel(
            "配置文件缺失",
            [
                f"{label('搜索路径')} mini_agent/config/config.yaml",
                f"{label('')} ~/.mini-agent/config/config.yaml",
                f"{label('')} <package>/config/config.yaml",
                "",
                f"{label('手动创建')} {user_config_dir / 'config.yaml'}",
                dim("字段示例请参考 README_CN.md，然后填入 API Key 与模型配置。"),
            ],
            accent=Colors.BRIGHT_RED,
        )
        return

    try:
        config = Config.from_yaml(config_path)
    except FileNotFoundError:
        print_panel("配置错误", [f"{label('未找到')} {config_path}"], accent=Colors.BRIGHT_RED)
        return
    except ValueError as e:
        print_panel("配置错误", [f"{label('错误')} {e}", dim("请检查 YAML 格式与字段。")], accent=Colors.BRIGHT_RED)
        return
    except Exception as e:
        print_panel("配置错误", [f"{label('加载失败')} {e}"], accent=Colors.BRIGHT_RED)
        return

    # 2. Initialize LLM client
    from mini_agent.retry import RetryConfig as RetryConfigBase

    # Convert configuration format
    retry_config = RetryConfigBase(
        enabled=config.llm.retry.enabled,
        max_retries=config.llm.retry.max_retries,
        initial_delay=config.llm.retry.initial_delay,
        max_delay=config.llm.retry.max_delay,
        exponential_base=config.llm.retry.exponential_base,
        retryable_exceptions=(Exception,),
    )

    # Create retry callback function to display retry information in terminal
    def on_retry(exception: Exception, attempt: int):
        """Retry callback function to display retry information"""
        next_delay = retry_config.calculate_delay(attempt - 1)
        print_panel(
            "LLM 重试",
            [
                f"{label('第几次')} {attempt}",
                f"{label('错误')} {exception}",
                f"{label('等待')} {next_delay:.1f}s 后进入第 {attempt + 1} 次尝试",
            ],
            accent=Colors.BRIGHT_YELLOW,
        )

    # Convert provider string to LLMProvider enum
    provider = LLMProvider.ANTHROPIC if config.llm.provider.lower() == "anthropic" else LLMProvider.OPENAI
    openai_prompt_cache_key = config.llm.openai_prompt_cache_key
    openai_prompt_cache_retention = config.llm.openai_prompt_cache_retention
    if provider == LLMProvider.OPENAI:
        if not openai_prompt_cache_key:
            workspace_digest = hashlib.sha256(str(workspace_dir.resolve()).encode("utf-8")).hexdigest()[:16]
            openai_prompt_cache_key = f"mini-agent:{config.llm.model}:{workspace_digest}"
        if not openai_prompt_cache_retention:
            openai_prompt_cache_retention = "24h"

    llm_client = LLMClient(
        api_key=config.llm.api_key,
        provider=provider,
        api_base=config.llm.api_base,
        model=config.llm.model,
        retry_config=retry_config if config.llm.retry.enabled else None,
        openai_prompt_cache_key=openai_prompt_cache_key,
        openai_prompt_cache_retention=openai_prompt_cache_retention,
        disable_thinking=config.llm.disable_thinking,
        enable_reasoning_split=config.llm.enable_reasoning_split,
        preserve_thinking=config.llm.preserve_thinking,
    )
    if provider == LLMProvider.OPENAI:
        print_panel(
            "OpenAI Prompt Cache",
            [
                f"{label('状态')} {status_badge('已启用', Colors.BRIGHT_GREEN)}",
                f"{label('key')} {openai_prompt_cache_key}",
                f"{label('retention')} {openai_prompt_cache_retention}",
            ],
            accent=Colors.BRIGHT_GREEN,
        )

    # Set retry callback
    if config.llm.retry.enabled:
        llm_client.retry_callback = on_retry
        print(f"{Colors.GREEN}✓ LLM retry enabled: max {config.llm.retry.max_retries} retries{Colors.RESET}")

    # 3. Initialize base tools (independent of workspace)
    tools, skill_loader = await initialize_base_tools(config)

    # 4. Add workspace-dependent tools
    add_workspace_tools(tools, config, workspace_dir, llm_client)

    # 4.5 Create task memory hook for automatic recording (independent of model tool calls)
    task_memory_hook = None
    task_memory_resume_accepted = False
    checkpoint_store = CheckpointStore(workspace_dir / ".mini_agent" / "checkpoints")
    if config.tools.enable_task_memory:
        task_memory_file = workspace_dir / ".mini_agent" / "task_memory.json"
        episode_memory_file = workspace_dir / ".mini_agent" / "episodes.jsonl"
        task_memory_hook = TaskMemoryHook(
            memory_file=str(task_memory_file),
            workspace_dir=str(workspace_dir),
            episode_memory_file=str(episode_memory_file),
            auto_start=True,
            auto_finish=True,
        )
        # Show resume info on startup if an active task exists
        resume_summary = task_memory_hook.get_resume_summary()
        if resume_summary and not task:
            print_panel(
                "上次任务记忆",
                [dim(line) for line in resume_summary.splitlines()],
                accent=Colors.BRIGHT_YELLOW,
            )
            answer = (await asyncio.to_thread(input, "是否恢复任务记忆？[y/N]: ")).strip().lower()
            if answer in {"y", "yes"}:
                task_memory_resume_accepted = True
            else:
                task_memory_hook.abandon_task(reason="User declined active task memory resume on startup")

    confirmation_prompt_active = threading.Event()

    async def confirm_bash_tool_call(
        tool_name: str,
        arguments: dict,
        decision: CommandSecurityDecision,
    ) -> bool:
        """Ask the user to approve a medium-risk bash command."""

        command = str(arguments.get("command", ""))
        run_in_background = bool(arguments.get("run_in_background", False))

        print_panel(
            "Bash 执行确认",
            [
                f"{label('工具')} {tool_name}",
                f"{label('命令')} {command}",
                f"{label('后台运行')} {run_in_background}",
                f"{label('风险等级')} {decision.risk_level}",
                f"{label('命中规则')} {', '.join(decision.matched_rules) or 'none'}",
            ],
            accent=Colors.BRIGHT_YELLOW,
        )

        confirmation_prompt_active.set()
        try:
            answer = await asyncio.to_thread(input, "允许执行这条命令？[y/N]: ")
        finally:
            confirmation_prompt_active.clear()

        return answer.strip().lower() in {"y", "yes"}

    # 5. Load System Prompt (with priority search)
    system_prompt_path = Config.find_config_file(config.agent.system_prompt_path)
    if system_prompt_path and system_prompt_path.exists():
        core_system_prompt = system_prompt_path.read_text(encoding="utf-8")
        system_prompt = core_system_prompt
        print(f"{Colors.GREEN}✓ System prompt loaded: {system_prompt_path}{Colors.RESET}")
    else:
        core_system_prompt = "You are Mini-Agent, an intelligent assistant that can help users complete various tasks."
        system_prompt = core_system_prompt
        print(f"{Colors.YELLOW}⚠ System prompt not found, using default{Colors.RESET}")

    # 6. Build layered system prompt
    system_prompt = SystemPromptBuilder(
        core_prompt=system_prompt,
        workspace_dir=workspace_dir,
        skill_loader=skill_loader,
    ).build()
    if skill_loader and getattr(skill_loader, "loaded_skills", None):
        print(f"{Colors.GREEN}✓ Added {len(skill_loader.loaded_skills)} skills metadata to system prompt{Colors.RESET}")
    if MarkdownMemoryStore(workspace_dir / ".memory").load():
        print(f"{Colors.GREEN}✓ Added long-term memory layer to system prompt{Colors.RESET}")

    # 7. Create Agent
    agent = Agent(
        llm_client=llm_client,
        system_prompt=system_prompt,
        tools=tools,
        max_steps=config.agent.max_steps,
        workspace_dir=str(workspace_dir),
        token_limit=config.agent.token_limit,
        core_system_prompt=core_system_prompt,
        skill_loader=skill_loader,
        request_context_limit=config.agent.request_context_limit,
        context_layer_budgets=config.agent.context_layer_budgets.to_prompt_layer_budgets(),
        tool_confirmation_callback=(
            confirm_bash_tool_call if config.tools.enable_bash_confirmation and not task else None
        ),
        task_memory_hook=task_memory_hook,
        checkpoint_store=checkpoint_store,
        token_pricing=config.llm.token_pricing,
        preserve_thinking=config.llm.preserve_thinking,
        show_thinking=config.llm.show_thinking,
        log_thinking=config.llm.log_thinking,
    )

    if not task:
        checkpoint_summary = checkpoint_store.get_resume_summary()
        if checkpoint_summary:
            print_resume_checkpoint_summary(checkpoint_summary)
            answer = (await asyncio.to_thread(input, "是否恢复最新 checkpoint？[y/N]: ")).strip().lower()
            if answer in {"y", "yes"}:
                try_restore_checkpoint(agent, checkpoint_store, workspace_dir)
            elif task_memory_hook is not None and not task_memory_resume_accepted:
                task_memory_hook.abandon_task(reason="User declined checkpoint resume on startup")

    # 8. Display welcome information
    if not task:
        print_banner()
        print_session_info(agent, workspace_dir, config.llm.model)

    # 8.5 Non-interactive mode: execute task and exit
    if task:
        print_panel("任务执行", [f"{label('模式')} 非交互", f"{label('状态')} Agent 开始执行"], accent=Colors.BRIGHT_BLUE)
        agent.add_user_message(task)
        try:
            await agent.run()
        except Exception as e:
            print_panel("执行错误", [f"{label('错误')} {e}"], accent=Colors.BRIGHT_RED)
        finally:
            print_stats(agent, session_start)

        # Cleanup MCP connections
        await _quiet_cleanup()
        return

    # 9. Setup prompt_toolkit session
    # Command completer
    command_completer = WordCompleter(
        [
            "/help",
            "/clear",
            "/history",
            "/resume",
            "/task",
            "/memory",
            "/memory review",
            "/memory delete",
            "/stats",
            "/log",
            "/exit",
            "/quit",
            "/q",
        ],
        ignore_case=True,
        sentence=True,
    )

    # Custom style for prompt
    prompt_style = Style.from_dict(
        {
            "prompt": "#00d7ff bold",
            "separator": "#666666",
            "hint": "#888888",
        }
    )

    # Custom key bindings
    kb = KeyBindings()

    @kb.add("c-u")  # Ctrl+U: Clear current line
    def _(event):
        """Clear the current input line"""
        event.current_buffer.reset()

    @kb.add("c-l")  # Ctrl+L: Clear screen (optional bonus)
    def _(event):
        """Clear the screen"""
        event.app.renderer.clear()

    @kb.add("c-j")  # Ctrl+J (对应 Ctrl+Enter)
    def _(event):
        """Insert a newline"""
        event.current_buffer.insert_text("\n")

    # Create prompt session with history and auto-suggest
    # Use FileHistory for persistent history across sessions (stored in user's home directory)
    history_file = Path.home() / ".mini-agent" / ".history"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=command_completer,
        style=prompt_style,
        key_bindings=kb,
    )

    # 10. Interactive loop
    while True:
        try:
            # Get user input using prompt_toolkit
            user_input = await session.prompt_async(
                [
                    ("class:prompt", "你"),
                    ("", " › "),
                ],
                multiline=False,
                enable_history_search=True,
            )
            user_input = user_input.strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()

                if command in ["/exit", "/quit", "/q"]:
                    print_panel("退出", ["会话结束，下面是本次统计。"], accent=Colors.BRIGHT_YELLOW)
                    print_stats(agent, session_start)
                    break

                elif command == "/help":
                    print_help()
                    continue

                elif command == "/clear":
                    # Clear message history but keep system prompt
                    old_count = len(agent.messages)
                    agent.messages = [agent.messages[0]]  # Keep only system message
                    print_panel(
                        "清空会话",
                        [
                            f"{label('已清理')} {old_count - 1} 条消息",
                            f"{label('保留')} system prompt",
                        ],
                        accent=Colors.BRIGHT_GREEN,
                    )
                    continue

                elif command == "/history":
                    print_panel(
                        "历史消息",
                        [f"{label('当前数量')} {len(agent.messages)} 条"],
                        accent=Colors.BRIGHT_CYAN,
                    )
                    continue

                elif command == "/resume":
                    summary = checkpoint_store.get_resume_summary()
                    if not summary:
                        print_panel("Checkpoint", ["当前没有可恢复的 checkpoint。"], accent=Colors.BRIGHT_YELLOW)
                        continue
                    print_resume_checkpoint_summary(summary)
                    try_restore_checkpoint(agent, checkpoint_store, workspace_dir)
                    continue

                elif command == "/task":
                    print_current_task(workspace_dir)
                    continue

                elif command == "/memory":
                    print_memory_status(workspace_dir)
                    continue

                elif command == "/memory review":
                    print_memory_review(workspace_dir)
                    continue

                elif command.startswith("/memory delete "):
                    memory_id = user_input.split(maxsplit=2)[2].strip()
                    delete_memory(workspace_dir, memory_id)
                    continue

                elif command == "/stats":
                    print_stats(agent, session_start)
                    continue

                elif command == "/log" or command.startswith("/log "):
                    # Parse /log command
                    parts = user_input.split(maxsplit=1)
                    if len(parts) == 1:
                        # /log - show log directory
                        show_log_directory(open_file_manager=True)
                    else:
                        # /log <filename> - read specific log file
                        filename = parts[1].strip("\"'")
                        read_log_file(filename)
                    continue

                else:
                    print_panel(
                        "未知命令",
                        [f"{label('输入')} {user_input}", dim("输入 /help 查看可用命令。")],
                        accent=Colors.BRIGHT_RED,
                    )
                    continue

            # Normal conversation - exit check
            if user_input.lower() in ["exit", "quit", "q"]:
                print_panel("退出", ["会话结束，下面是本次统计。"], accent=Colors.BRIGHT_YELLOW)
                print_stats(agent, session_start)
                break

            # Run Agent with Esc cancellation support
            print(f"\n{Colors.BRIGHT_BLUE}Agent{Colors.RESET} {Colors.DIM}› 正在思考，按 Esc 可中断{Colors.RESET}\n")
            message_checkpoint = len(agent.messages)
            agent.add_user_message(user_input)

            # Create cancellation event
            cancel_event = asyncio.Event()
            agent.cancel_event = cancel_event

            # Esc key listener thread
            esc_listener_stop = threading.Event()
            esc_cancelled = [False]  # Mutable container for thread access

            def esc_key_listener():
                """Listen for Esc key in a separate thread."""
                if platform.system() == "Windows":
                    try:
                        import msvcrt

                        while not esc_listener_stop.is_set():
                            if confirmation_prompt_active.is_set():
                                esc_listener_stop.wait(0.05)
                                continue
                            if msvcrt.kbhit():
                                char = msvcrt.getch()
                                if char == b"\x1b":  # Esc
                                    print(f"\n{Colors.BRIGHT_YELLOW}⏹ 已收到 Esc，正在中断...{Colors.RESET}")
                                    esc_cancelled[0] = True
                                    cancel_event.set()
                                    break
                            esc_listener_stop.wait(0.05)
                    except Exception:
                        pass
                    return

                # Unix/macOS
                try:
                    import select
                    import termios
                    import tty

                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)

                    try:
                        tty.setcbreak(fd)
                        while not esc_listener_stop.is_set():
                            if confirmation_prompt_active.is_set():
                                esc_listener_stop.wait(0.05)
                                continue
                            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                            if rlist:
                                char = sys.stdin.read(1)
                                if char == "\x1b":  # Esc
                                    print(f"\n{Colors.BRIGHT_YELLOW}⏹ 已收到 Esc，正在中断...{Colors.RESET}")
                                    esc_cancelled[0] = True
                                    cancel_event.set()
                                    break
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass

            # Start Esc listener thread
            esc_thread = threading.Thread(target=esc_key_listener, daemon=True)
            esc_thread.start()

            # Run agent with periodic cancellation check
            try:
                agent_task = asyncio.create_task(agent.run())

                # Poll for cancellation while agent runs
                while not agent_task.done():
                    if esc_cancelled[0]:
                        cancel_event.set()
                    await asyncio.sleep(0.1)

                # Get result
                agent_task.result()
                if not agent.last_run_completed:
                    agent.truncate_messages(message_checkpoint)
                    print(f"{Colors.DIM}   已回滚未完成轮次；下一轮会从上一个完整上下文继续。{Colors.RESET}")

            except asyncio.CancelledError:
                print_panel("执行中断", ["Agent 执行已取消。"], accent=Colors.BRIGHT_YELLOW)
                agent.truncate_messages(message_checkpoint)
            finally:
                agent.cancel_event = None
                esc_listener_stop.set()
                esc_thread.join(timeout=0.2)

            # Visual separation
            print()
            print_rule()
            print()

        except KeyboardInterrupt:
            print_panel("退出", ["收到 Ctrl+C，正在结束会话。"], accent=Colors.BRIGHT_YELLOW)
            print_stats(agent, session_start)
            break

        except Exception as e:
            print_panel("运行错误", [f"{label('错误')} {e}"], accent=Colors.BRIGHT_RED)
            print_rule()

    # 11. Cleanup MCP connections
    await _quiet_cleanup()


def main():
    """Main entry point for CLI"""
    configure_output_encoding()

    # Parse command line arguments
    args = parse_args()

    # Handle log subcommand
    if args.command == "log":
        if args.filename:
            read_log_file(args.filename)
        else:
            show_log_directory(open_file_manager=True)
        return

    # Determine workspace directory
    # Expand ~ to user home directory for portability
    if args.workspace:
        workspace_dir = Path(args.workspace).expanduser().absolute()
    else:
        # Use current working directory
        workspace_dir = Path.cwd()

    # Ensure workspace directory exists
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Run the agent (config always loaded from package directory)
    asyncio.run(run_agent(workspace_dir, task=args.task))


if __name__ == "__main__":
    main()
