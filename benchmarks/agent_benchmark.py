"""Deterministic benchmark runner for Mini Agent harness behavior.

This benchmark intentionally uses a scripted LLM so the result measures the
agent harness itself: prompt assembly, tool execution, safety policy, token
accounting, and stop conditions. Model-quality benchmarks can reuse the same
report shape later with a real LLM client.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable

from mini_agent.agent import Agent
from mini_agent.checkpoint import CheckpointStore
from mini_agent.config import Config
from mini_agent.evals import (
    EvalCandidate,
    EvalExecution,
    EvalRunReport,
    EvalSQLiteStore,
    EvalSuite,
    EvalTask,
    run_eval_suite,
    with_eval_metrics,
)
from mini_agent.llm import LLMClient
from mini_agent.observability import SQLiteTraceStore, StoreTraceRecorder, TraceRecorder
from mini_agent.schema import FunctionCall, LLMResponse, Message, TokenUsage, ToolCall
from mini_agent.schema import LLMProvider
from mini_agent.summarizer import is_context_snip_message, is_harness_summary_message
from mini_agent.tools import BashTool, EditTool, ReadTool, WriteTool
from mini_agent.tools.task_memory_tool import TaskMemoryHook


@dataclass
class ScriptedResponse:
    content: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 100
    completion_tokens: int = 20

    def to_llm_response(self, call_index: int) -> LLMResponse:
        usage = TokenUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
            cached_tokens=0,
            cache_write_tokens=0,
        )
        if not self.tool_name:
            return LLMResponse(
                content=self.content,
                tool_calls=None,
                finish_reason="stop",
                usage=usage,
            )

        return LLMResponse(
            content=self.content,
            tool_calls=[
                ToolCall(
                    id=f"call_{call_index}",
                    type="function",
                    function=FunctionCall(name=self.tool_name, arguments=self.arguments),
                )
            ],
            finish_reason="tool_calls",
            usage=usage,
        )


class ScriptedLLM:
    """Minimal LLM client that returns predefined responses."""

    def __init__(self, responses: list[ScriptedResponse]):
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def generate(self, messages: list[Message], tools: list[Any] | None = None) -> LLMResponse:
        self.calls.append(
            {
                "message_count": len(messages),
                "tool_names": [tool.name for tool in tools or []],
                "last_role": messages[-1].role if messages else None,
            }
        )
        index = len(self.calls) - 1
        if index >= len(self.responses):
            return LLMResponse(
                content="script exhausted",
                tool_calls=None,
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            )
        return self.responses[index].to_llm_response(index + 1)


@dataclass
class BenchmarkCase:
    name: str
    description: str
    task: str
    responses: list[ScriptedResponse]
    files: dict[str, str] = field(default_factory=dict)
    expect_output_contains: list[str] = field(default_factory=list)
    expect_files: dict[str, str] = field(default_factory=dict)
    expect_tool_messages_contain: list[str] = field(default_factory=list)
    expect_completed: bool = True
    max_steps: int = 6
    token_limit: int = 10000
    seed_messages: list[Message] = field(default_factory=list)
    enable_checkpoint: bool = False
    enable_task_memory: bool = False
    custom_check: Callable[[Agent, Path, str], dict[str, bool]] | None = None


def _prompt_compression_check(agent: Agent, workspace: Path, output: str) -> dict[str, bool]:
    del workspace, output
    summary_messages = [message for message in agent.messages if is_harness_summary_message(message)]
    snip_messages = [message for message in agent.messages if is_context_snip_message(message)]
    return {
        "has_compression_marker": bool(summary_messages or snip_messages),
        "kept_current_task": any(
            message.role == "user" and "当前问题必须保留" in str(message.content)
            for message in agent.messages
        ),
    }


def _checkpoint_check(agent: Agent, workspace: Path, output: str) -> dict[str, bool]:
    del agent, output
    latest = workspace / ".mini_agent" / "checkpoints" / "latest.json"
    if not latest.exists():
        return {"checkpoint_latest_exists": False, "checkpoint_reason_completed": False}
    data = json.loads(latest.read_text(encoding="utf-8"))
    return {
        "checkpoint_latest_exists": True,
        "checkpoint_reason_completed": data.get("reason") == "completed",
    }


def _task_memory_check(agent: Agent, workspace: Path, output: str) -> dict[str, bool]:
    del agent, output
    task_file = workspace / ".mini_agent" / "task_memory.json"
    episode_file = workspace / ".mini_agent" / "episodes.json"
    if not task_file.exists() or not episode_file.exists():
        return {
            "task_memory_exists": task_file.exists(),
            "episode_memory_exists": episode_file.exists(),
            "task_finished": False,
            "artifact_recorded": False,
        }

    task_data = json.loads(task_file.read_text(encoding="utf-8"))
    episode_data = json.loads(episode_file.read_text(encoding="utf-8"))
    tasks = task_data.get("tasks", [])
    latest_task = tasks[-1] if tasks else {}
    artifacts = latest_task.get("artifacts", [])
    episodes = episode_data.get("episodes", []) if isinstance(episode_data, dict) else []
    return {
        "task_memory_exists": True,
        "episode_memory_exists": True,
        "task_finished": latest_task.get("status") == "completed",
        "artifact_recorded": any(item.get("tool") == "write_file" for item in artifacts if isinstance(item, dict)),
        "episode_recorded": bool(episodes),
    }


def default_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            name="direct_answer",
            description="LLM can finish without tools.",
            task="回答项目定位",
            responses=[
                ScriptedResponse(
                    content="Mini Agent 是一个本地 CLI Agent Harness。",
                    prompt_tokens=320,
                    completion_tokens=18,
                )
            ],
            expect_output_contains=["CLI Agent Harness"],
        ),
        BenchmarkCase(
            name="read_file",
            description="Agent preserves tool call/result chain for file reading.",
            task="读取 notes.txt 并总结",
            files={"notes.txt": "workspace 是 Agent 的本地工作边界。\n"},
            responses=[
                ScriptedResponse(tool_name="read_file", arguments={"path": "notes.txt"}),
                ScriptedResponse(content="notes.txt 说明 workspace 是本地工作边界。"),
            ],
            expect_output_contains=["workspace", "工作边界"],
            expect_tool_messages_contain=["workspace 是 Agent"],
        ),
        BenchmarkCase(
            name="write_file",
            description="Agent can create an artifact inside workspace.",
            task="创建 result.md",
            responses=[
                ScriptedResponse(
                    tool_name="write_file",
                    arguments={"path": "result.md", "content": "# Benchmark\n\npassed\n"},
                ),
                ScriptedResponse(content="result.md 已创建。"),
            ],
            expect_output_contains=["已创建"],
            expect_files={"result.md": "# Benchmark\n\npassed\n"},
        ),
        BenchmarkCase(
            name="bash_policy_blocks_dangerous_command",
            description="Bash policy blocks destructive commands before execution.",
            task="尝试执行危险命令",
            responses=[
                ScriptedResponse(tool_name="bash", arguments={"command": "rm -rf ./important"}),
                ScriptedResponse(content="危险命令已被安全策略拦截。"),
            ],
            expect_output_contains=["拦截"],
            expect_tool_messages_contain=["Command blocked by security policy"],
        ),
        BenchmarkCase(
            name="max_steps_guard",
            description="Agent exits cleanly when the model keeps requesting tools.",
            task="模型一直请求工具时要停止",
            responses=[
                ScriptedResponse(tool_name="read_file", arguments={"path": "missing.txt"}),
                ScriptedResponse(tool_name="read_file", arguments={"path": "missing.txt"}),
                ScriptedResponse(tool_name="read_file", arguments={"path": "missing.txt"}),
            ],
            expect_output_contains=["couldn't be completed"],
            expect_completed=False,
            max_steps=3,
        ),
        BenchmarkCase(
            name="prompt_compression_keeps_current_task",
            description="Long history is compacted while the current task stays available.",
            task="当前问题必须保留：请回答压缩是否按层处理。",
            seed_messages=[
                Message(role="user", content="旧任务：分析 Agent harness。"),
                Message(role="assistant", content="已完成大量分析。" + " 历史细节" * 600),
            ],
            responses=[
                ScriptedResponse(content="压缩按层处理，并且当前问题仍然保留。"),
            ],
            expect_output_contains=["压缩按层处理"],
            token_limit=700,
            custom_check=_prompt_compression_check,
        ),
        BenchmarkCase(
            name="unknown_tool_is_reported_cleanly",
            description="Runtime reports unknown tool calls as tool errors instead of crashing.",
            task="调用不存在的工具",
            responses=[
                ScriptedResponse(tool_name="missing_tool", arguments={"value": "x"}),
                ScriptedResponse(content="未知工具被干净地报告。"),
            ],
            expect_output_contains=["未知工具"],
            expect_tool_messages_contain=["Unknown tool: missing_tool"],
        ),
        BenchmarkCase(
            name="checkpoint_saved_during_run",
            description="Checkpoint store records a completed run.",
            task="验证 checkpoint",
            responses=[ScriptedResponse(content="checkpoint 已保存。")],
            expect_output_contains=["checkpoint"],
            enable_checkpoint=True,
            custom_check=_checkpoint_check,
        ),
        BenchmarkCase(
            name="subagent_disabled_by_default",
            description="The task tool is unavailable unless subagent is explicitly enabled.",
            task="尝试启动 subagent",
            responses=[
                ScriptedResponse(
                    tool_name="task",
                    arguments={"description": "分析", "prompt": "独立分析项目"},
                ),
                ScriptedResponse(content="subagent 默认关闭，因此 task 工具不可用。"),
            ],
            expect_output_contains=["默认关闭"],
            expect_tool_messages_contain=["Unknown tool: task"],
        ),
        BenchmarkCase(
            name="task_memory_records_completion",
            description="Task memory records tool side effects and completion episodes.",
            task="写入文件并完成任务记忆",
            responses=[
                ScriptedResponse(
                    tool_name="write_file",
                    arguments={"path": "memory-result.md", "content": "memory benchmark\n"},
                ),
                ScriptedResponse(content="任务完成，已记录记忆。"),
            ],
            expect_output_contains=["任务完成"],
            expect_files={"memory-result.md": "memory benchmark\n"},
            enable_task_memory=True,
            custom_check=_task_memory_check,
        ),
    ]


def _write_fixture_files(workspace: Path, files: dict[str, str]) -> None:
    for relative_path, content in files.items():
        path = workspace / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _read_file_if_exists(workspace: Path, relative_path: str) -> str:
    path = workspace / relative_path
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def _benchmark_case_expected_status(case: BenchmarkCase) -> str:
    return "completed" if case.expect_completed else "max_steps"


def _benchmark_case_status(case: BenchmarkCase, checks: dict[str, bool]) -> str:
    if not all(checks.values()):
        return "failed"
    return _benchmark_case_expected_status(case)


def _benchmark_suite() -> EvalSuite:
    return EvalSuite(
        suite_id="mini-agent-harness",
        name="Mini Agent Harness",
        version="deterministic",
        tasks=[
            EvalTask(
                task_id=case.name,
                prompt=case.task,
                description=case.description,
                expected_output_contains=case.expect_output_contains,
                expected_files=case.expect_files,
                expected_tool_evidence_contains=case.expect_tool_messages_contain,
                expected_status=_benchmark_case_expected_status(case),
                metadata={"legacy_case": case.name},
            )
            for case in default_cases()
        ],
    )


async def run_case(case: BenchmarkCase, trace_recorder: TraceRecorder | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"mini-agent-bench-{case.name}-") as tmp:
        workspace = Path(tmp)
        _write_fixture_files(workspace, case.files)

        llm = ScriptedLLM(case.responses)
        tools = [
            ReadTool(workspace_dir=str(workspace)),
            WriteTool(workspace_dir=str(workspace)),
            EditTool(workspace_dir=str(workspace)),
            BashTool(workspace_dir=str(workspace)),
        ]
        agent = Agent(
            llm_client=llm,  # type: ignore[arg-type]
            system_prompt="你是 Mini Agent benchmark runner。",
            tools=tools,
            workspace_dir=str(workspace),
            max_steps=case.max_steps,
            token_limit=case.token_limit,
            checkpoint_store=(
                CheckpointStore(workspace / ".mini_agent" / "checkpoints")
                if case.enable_checkpoint
                else None
            ),
            task_memory_hook=(
                TaskMemoryHook(
                    memory_file=str(workspace / ".mini_agent" / "task_memory.json"),
                    workspace_dir=str(workspace),
                    episode_memory_file=str(workspace / ".mini_agent" / "episodes.json"),
                )
                if case.enable_task_memory
                else None
            ),
            trace_recorder=trace_recorder,
        )
        agent.messages.extend(case.seed_messages)
        agent.add_user_message(case.task)

        started = perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            output = await agent.run()
        elapsed_ms = round((perf_counter() - started) * 1000, 2)

        tool_messages = [message.content for message in agent.messages if message.role == "tool"]
        file_checks = {
            path: (workspace / path).exists() and (workspace / path).read_text(encoding="utf-8") == expected
            for path, expected in case.expect_files.items()
        }
        checks = {
            "output_contains": _contains_all(output, case.expect_output_contains),
            "tool_messages_contain": all(
                any(needle in message for message in tool_messages)
                for needle in case.expect_tool_messages_contain
            ),
            "files": all(file_checks.values()) if file_checks else True,
            "completion_state": agent.last_run_completed is case.expect_completed,
        }
        if case.custom_check is not None:
            checks.update(case.custom_check(agent, workspace, output))
        status = _benchmark_case_status(case, checks)

        return {
            "name": case.name,
            "description": case.description,
            "passed": all(checks.values()),
            "checks": checks,
            "status": status,
            "agent_run_id": agent.run_id,
            "workspace_files": {
                path: _read_file_if_exists(workspace, path)
                for path in case.expect_files
            },
            "tool_evidence": tool_messages,
            "elapsed_ms": elapsed_ms,
            "llm_calls": len(llm.calls),
            "tool_messages": len(tool_messages),
            "message_count": len(agent.messages),
            "tokens": {
                "prompt": agent.cumulative_prompt_tokens,
                "completion": agent.cumulative_completion_tokens,
                "total": agent.cumulative_total_tokens,
                "cached": agent.cumulative_cached_tokens,
            },
            "output": output,
        }


async def run_eval_benchmark(
    eval_run_id: str = "mini-agent-harness-deterministic",
    db_path: str | Path | None = None,
    trace_db_path: str | Path | None = None,
    eval_db_path: str | Path | None = None,
) -> EvalRunReport:
    cases = {case.name: case for case in default_cases()}
    suite = _benchmark_suite()
    candidate = EvalCandidate(candidate_id="scripted-harness", model="scripted", label="Scripted Harness")
    trace_path = trace_db_path or db_path
    eval_path = eval_db_path or db_path
    trace_recorder = StoreTraceRecorder(SQLiteTraceStore(trace_path)) if trace_path is not None else None

    async def run_candidate(_: EvalCandidate, task: EvalTask) -> EvalExecution:
        result = await run_case(cases[task.task_id], trace_recorder=trace_recorder)
        tokens = result["tokens"]
        return EvalExecution(
            output=result["output"],
            status=result["status"],
            agent_run_id=result["agent_run_id"],
            workspace_files=result["workspace_files"],
            tool_evidence=result["tool_evidence"],
            duration_ms=result["elapsed_ms"],
            prompt_tokens=tokens["prompt"],
            completion_tokens=tokens["completion"],
            total_tokens=tokens["total"],
            metadata={
                "legacy_checks": result["checks"],
                "legacy_passed": result["passed"],
                "description": result["description"],
                "llm_calls": result["llm_calls"],
                "tool_messages": result["tool_messages"],
                "message_count": result["message_count"],
            },
        )

    report = with_eval_metrics(await run_eval_suite(eval_run_id, suite, [candidate], run_candidate))
    if eval_path is not None:
        EvalSQLiteStore(eval_path).save_report(report)
    return report


def _legacy_report_from_eval(report: EvalRunReport) -> dict[str, Any]:
    results = [
        {
            "name": result.task_id,
            "description": result.metadata.get("description", ""),
            "passed": result.passed,
            "checks": result.metadata.get("legacy_checks", {}),
            "status": result.status,
            "agent_run_id": result.agent_run_id,
            "elapsed_ms": result.duration_ms,
            "llm_calls": result.metadata.get("llm_calls", 0),
            "tool_messages": result.metadata.get("tool_messages", 0),
            "message_count": result.metadata.get("message_count", 0),
            "tokens": {
                "prompt": result.prompt_tokens,
                "completion": result.completion_tokens,
                "total": result.total_tokens,
                "cached": 0,
            },
            "output": result.output,
        }
        for result in report.results
    ]
    passed = sum(1 for result in results if result["passed"])
    return {
        "suite": report.suite.suite_id,
        "suite_version": report.suite.version,
        "eval_run_id": report.eval_run_id,
        "case_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 4) if results else 0,
        "total_tokens": sum(result["tokens"]["total"] for result in results),
        "total_elapsed_ms": round(sum(result["elapsed_ms"] for result in results), 2),
        "cases": results,
    }


async def run_benchmark(trace_db_path: str | Path | None = None) -> dict[str, Any]:
    return _legacy_report_from_eval(await run_eval_benchmark(trace_db_path=trace_db_path))


@dataclass
class RealBenchmarkCase:
    name: str
    description: str
    task: str
    files: dict[str, str] = field(default_factory=dict)
    expect_output_contains: list[str] = field(default_factory=list)
    expect_files: dict[str, list[str]] = field(default_factory=dict)
    expect_tool_messages_contain: list[str] = field(default_factory=list)
    max_steps: int = 8


@dataclass(frozen=True)
class RealEvalCandidate:
    candidate_id: str
    config: Config
    config_path: Path | None = None
    label: str | None = None


RealCaseRunner = Callable[
    [RealBenchmarkCase, RealEvalCandidate, Path, TraceRecorder | None],
    Awaitable[dict[str, Any]],
]


def load_real_eval_candidates(candidate_specs: list[str]) -> list[RealEvalCandidate]:
    """Load named real-model eval candidates from name=config.yaml specs."""

    candidates: list[RealEvalCandidate] = []
    for spec in candidate_specs:
        if "=" not in spec:
            raise ValueError(f"Candidate must use name=path format: {spec}")
        name, raw_path = spec.split("=", 1)
        candidate_id = name.strip()
        config_path = Path(raw_path.strip()).expanduser()
        if not candidate_id:
            raise ValueError(f"Candidate name is empty: {spec}")
        if not config_path.exists():
            raise FileNotFoundError(f"Candidate config not found: {config_path}")
        config = Config.from_yaml(config_path)
        candidates.append(
            RealEvalCandidate(
                candidate_id=candidate_id,
                config=config,
                config_path=config_path,
                label=candidate_id,
            )
        )
    return candidates


def default_real_cases() -> list[RealBenchmarkCase]:
    return [
        RealBenchmarkCase(
            name="real_direct_architecture_answer",
            description="Real model answers a concise architecture question without needing tools.",
            task=(
                "请用两句话说明 Mini Agent 作为本地 CLI Agent Harness 的核心价值。"
                "回答必须包含：harness、工具、安全。"
            ),
            expect_output_contains=["harness", "工具", "安全"],
            max_steps=4,
        ),
        RealBenchmarkCase(
            name="real_read_and_summarize",
            description="Real model reads a fixture file and summarizes the key idea.",
            task=(
                "请读取 fixture_notes.md，然后用一句话总结它。"
                "回答必须包含：workspace、checkpoint。"
            ),
            files={
                "fixture_notes.md": (
                    "# Notes\n\n"
                    "workspace 是 Agent 的本地工作边界。\n"
                    "checkpoint 用来保存可恢复的对话上下文。\n"
                )
            },
            expect_output_contains=["workspace", "checkpoint"],
            max_steps=6,
        ),
        RealBenchmarkCase(
            name="real_write_report",
            description="Real model creates a small markdown artifact in the workspace.",
            task=(
                "请创建 bench_result.md，内容用 Markdown 写三行："
                "标题 Mini Agent Benchmark，结论 passed，指标 real-model。"
                "完成后简短回复。"
            ),
            expect_files={
                "bench_result.md": ["Mini Agent Benchmark", "passed", "real-model"],
            },
            max_steps=6,
        ),
        RealBenchmarkCase(
            name="real_edit_existing_file",
            description="Real model edits an existing file using the edit tool.",
            task=(
                "请读取 profile.md，然后把里面的 status 从 draft 改成 ready。"
                "完成后回复必须包含：ready。"
            ),
            files={
                "profile.md": "# Profile\n\nname: mini-agent\nstatus: draft\n",
            },
            expect_output_contains=["ready"],
            expect_files={"profile.md": ["status: ready"]},
            max_steps=8,
        ),
        RealBenchmarkCase(
            name="real_multi_file_synthesis",
            description="Real model reads two files and writes a synthesis artifact.",
            task=(
                "请读取 architecture.md 和 risks.md，然后创建 synthesis.md。"
                "synthesis.md 必须包含 harness、checkpoint、Bash policy 三个词。"
                "完成后简短回复。"
            ),
            files={
                "architecture.md": (
                    "# Architecture\n\n"
                    "Mini Agent 的核心是 harness：负责 prompt、工具、状态和执行边界。\n"
                    "checkpoint 用来保存可恢复上下文。\n"
                ),
                "risks.md": (
                    "# Risks\n\n"
                    "Bash policy 用来在执行 shell 前做安全拦截和确认。\n"
                ),
            },
            expect_files={"synthesis.md": ["harness", "checkpoint", "Bash policy"]},
            max_steps=10,
        ),
        RealBenchmarkCase(
            name="real_missing_file_recovery",
            description="Real model handles a missing file gracefully.",
            task=(
                "请尝试读取 missing.md。如果文件不存在，不要创建它，"
                "只回复：missing.md 不存在。"
            ),
            expect_output_contains=["missing.md", "不存在"],
            expect_tool_messages_contain=["File not found: missing.md"],
            max_steps=6,
        ),
        RealBenchmarkCase(
            name="real_bash_low_risk_command",
            description="Real model runs a low-risk shell command through BashTool.",
            task=(
                "请使用 bash 工具执行这个低风险命令：echo benchmark-ok。"
                "然后回复必须包含 benchmark-ok。"
            ),
            expect_output_contains=["benchmark-ok"],
            expect_tool_messages_contain=["benchmark-ok"],
            max_steps=6,
        ),
        RealBenchmarkCase(
            name="real_bash_policy_blocks_dangerous_command",
            description="Real model attempts a dangerous command and the Bash policy blocks it.",
            task=(
                "请使用 bash 工具执行 rm -rf ./important，用来验证安全策略。"
                "如果工具返回被拦截，请回复必须包含：安全策略拦截。"
            ),
            expect_output_contains=["安全策略拦截"],
            expect_tool_messages_contain=["Command blocked by security policy"],
            max_steps=6,
        ),
    ]


def _build_real_llm(config: Config) -> LLMClient:
    return LLMClient(
        api_key=config.llm.api_key,
        provider=LLMProvider(config.llm.provider),
        api_base=config.llm.api_base,
        model=config.llm.model,
        retry_config=config.llm.retry,
        openai_prompt_cache_key=config.llm.openai_prompt_cache_key,
        openai_prompt_cache_retention=config.llm.openai_prompt_cache_retention,
        disable_thinking=config.llm.disable_thinking,
        enable_reasoning_split=config.llm.enable_reasoning_split,
        preserve_thinking=config.llm.preserve_thinking,
    )


def _real_benchmark_suite(cases: list[RealBenchmarkCase] | None = None) -> EvalSuite:
    selected_cases = cases or default_real_cases()
    return EvalSuite(
        suite_id="mini-agent-real-model",
        name="Mini Agent Real Model",
        version="real",
        tasks=[
            EvalTask(
                task_id=case.name,
                prompt=case.task,
                description=case.description,
                expected_output_contains=case.expect_output_contains,
                expected_files=case.expect_files,
                expected_tool_evidence_contains=case.expect_tool_messages_contain,
                expected_status="completed",
                metadata={"legacy_case": case.name},
            )
            for case in selected_cases
        ],
    )


def _expected_files_as_needles(expected_files: dict[str, str | list[str]]) -> dict[str, list[str]]:
    return {
        path: [needles] if isinstance(needles, str) else list(needles)
        for path, needles in expected_files.items()
    }


def _task_max_steps(task: EvalTask, default: int = 8) -> int:
    raw_value = task.metadata.get("max_steps", default)
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _real_cases_from_suite(suite: EvalSuite) -> list[RealBenchmarkCase]:
    return [
        RealBenchmarkCase(
            name=task.task_id,
            description=task.description,
            task=task.prompt,
            expect_output_contains=list(task.expected_output_contains),
            expect_files=_expected_files_as_needles(task.expected_files),
            expect_tool_messages_contain=list(task.expected_tool_evidence_contains),
            max_steps=_task_max_steps(task),
        )
        for task in suite.tasks
    ]


def _eval_candidate_from_real(candidate: RealEvalCandidate) -> EvalCandidate:
    return EvalCandidate(
        candidate_id=candidate.candidate_id,
        model=candidate.config.llm.model,
        label=candidate.label or candidate.candidate_id,
        metadata={
            "provider": candidate.config.llm.provider,
            "api_base": candidate.config.llm.api_base,
            "config_path": str(candidate.config_path) if candidate.config_path is not None else None,
        },
    )


async def run_real_case(
    case: RealBenchmarkCase,
    config: Config,
    output_root: Path,
    candidate_id: str = "default",
    trace_recorder: TraceRecorder | None = None,
) -> dict[str, Any]:
    workspace = output_root / "workspaces" / candidate_id / case.name
    workspace.mkdir(parents=True, exist_ok=True)
    _write_fixture_files(workspace, case.files)

    llm = _build_real_llm(config)
    tools = [
        ReadTool(workspace_dir=str(workspace)),
        WriteTool(workspace_dir=str(workspace)),
        EditTool(workspace_dir=str(workspace)),
        BashTool(workspace_dir=str(workspace)),
    ]
    agent = Agent(
        llm_client=llm,
        system_prompt=(
            "你是 Mini Agent 真实模型 benchmark runner。"
            "请只完成用户给出的 benchmark 任务，必要时使用工具，回答保持简洁。"
        ),
        tools=tools,
        workspace_dir=str(workspace),
        max_steps=case.max_steps,
        token_limit=config.agent.token_limit,
        token_pricing=config.llm.token_pricing,
        preserve_thinking=config.llm.preserve_thinking,
        show_thinking=False,
        log_thinking=config.llm.log_thinking,
        trace_recorder=trace_recorder,
    )
    agent.add_user_message(case.task)

    started = perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        output = await agent.run()
    elapsed_ms = round((perf_counter() - started) * 1000, 2)

    tool_messages = [message.content for message in agent.messages if message.role == "tool"]
    file_checks = {
        path: _contains_all(_read_file_if_exists(workspace, path), needles)
        for path, needles in case.expect_files.items()
    }
    output_check = _contains_all(output.lower(), [needle.lower() for needle in case.expect_output_contains])
    checks = {
        "output_contains": output_check,
        "files": all(file_checks.values()) if file_checks else True,
        "tool_messages_contain": all(
            any(needle in message for message in tool_messages)
            for needle in case.expect_tool_messages_contain
        ),
        "completed": agent.last_run_completed,
    }
    status = "completed" if agent.last_run_completed else "failed"

    return {
        "name": case.name,
        "description": case.description,
        "passed": all(checks.values()),
        "checks": checks,
        "status": status,
        "agent_run_id": agent.run_id,
        "workspace_files": {
            path: _read_file_if_exists(workspace, path)
            for path in case.expect_files
        },
        "tool_evidence": tool_messages,
        "elapsed_ms": elapsed_ms,
        "llm_calls": sum(1 for message in agent.messages if message.role == "assistant"),
        "tool_messages": len(tool_messages),
        "message_count": len(agent.messages),
        "tokens": {
            "prompt": agent.cumulative_prompt_tokens,
            "completion": agent.cumulative_completion_tokens,
            "total": agent.cumulative_total_tokens,
            "cached": agent.cumulative_cached_tokens,
            "cache_write": agent.cumulative_cache_write_tokens,
        },
        "cost": agent.cumulative_token_cost.model_dump(mode="json"),
        "output": output,
        "workspace": str(workspace),
    }


async def _run_real_case_for_candidate(
    case: RealBenchmarkCase,
    candidate: RealEvalCandidate,
    output_root: Path,
    trace_recorder: TraceRecorder | None,
) -> dict[str, Any]:
    return await run_real_case(
        case,
        candidate.config,
        output_root,
        candidate_id=candidate.candidate_id,
        trace_recorder=trace_recorder,
    )


async def run_real_eval_benchmark(
    candidates: list[RealEvalCandidate],
    output_root: Path,
    eval_run_id: str = "mini-agent-real-model",
    db_path: str | Path | None = None,
    cases: list[RealBenchmarkCase] | None = None,
    suite: EvalSuite | None = None,
    case_runner: RealCaseRunner | None = None,
) -> EvalRunReport:
    if cases is not None:
        selected_cases = cases
    elif suite is not None:
        selected_cases = _real_cases_from_suite(suite)
    else:
        selected_cases = default_real_cases()
    case_by_id = {case.name: case for case in selected_cases}
    eval_suite = suite or _real_benchmark_suite(selected_cases)
    missing_cases = [task.task_id for task in eval_suite.tasks if task.task_id not in case_by_id]
    if missing_cases:
        raise ValueError(f"Eval suite tasks do not have runnable cases: {', '.join(missing_cases)}")
    eval_candidates = [_eval_candidate_from_real(candidate) for candidate in candidates]
    real_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    trace_recorder = StoreTraceRecorder(SQLiteTraceStore(db_path)) if db_path is not None else None
    runner = case_runner or _run_real_case_for_candidate

    async def run_candidate(candidate: EvalCandidate, task: EvalTask) -> EvalExecution:
        result = await runner(case_by_id[task.task_id], real_by_id[candidate.candidate_id], output_root, trace_recorder)
        tokens = result["tokens"]
        cost = result.get("cost", {})
        return EvalExecution(
            output=result["output"],
            status=result["status"],
            agent_run_id=result["agent_run_id"],
            workspace_files=result["workspace_files"],
            tool_evidence=result["tool_evidence"],
            duration_ms=result["elapsed_ms"],
            prompt_tokens=tokens["prompt"],
            completion_tokens=tokens["completion"],
            total_tokens=tokens["total"],
            total_cost=cost.get("total_cost", 0.0),
            currency=cost.get("currency", "USD"),
            metadata={
                "legacy_checks": result["checks"],
                "legacy_passed": result["passed"],
                "description": result["description"],
                "llm_calls": result["llm_calls"],
                "tool_messages": result["tool_messages"],
                "message_count": result["message_count"],
                "workspace": result.get("workspace"),
            },
        )

    report = with_eval_metrics(await run_eval_suite(eval_run_id, eval_suite, eval_candidates, run_candidate))
    if db_path is not None:
        EvalSQLiteStore(db_path).save_report(report)
    return report


async def run_real_benchmark(output_root: Path) -> dict[str, Any]:
    config = Config.load()
    results = [await run_real_case(case, config, output_root) for case in default_real_cases()]
    passed = sum(1 for result in results if result["passed"])
    total_cost = round(sum(result["cost"]["total_cost"] for result in results), 8)
    return {
        "suite": "mini-agent-real-model",
        "provider": config.llm.provider,
        "model": config.llm.model,
        "case_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 4) if results else 0,
        "total_tokens": sum(result["tokens"]["total"] for result in results),
        "total_cached_tokens": sum(result["tokens"]["cached"] for result in results),
        "total_cache_write_tokens": sum(result["tokens"]["cache_write"] for result in results),
        "total_cost": total_cost,
        "currency": config.llm.token_pricing.currency,
        "total_elapsed_ms": round(sum(result["elapsed_ms"] for result in results), 2),
        "cases": results,
    }


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"# {report['suite']}",
        "",
        f"- Model: `{report.get('provider', 'scripted')}` / `{report.get('model', 'scripted')}`",
        f"- Passed: {report['passed']}/{report['case_count']} ({report['pass_rate'] * 100:.1f}%)",
        f"- Total tokens: {report.get('total_tokens', 0)}",
        f"- Cached tokens: {report.get('total_cached_tokens', 0)}",
        f"- Cache write tokens: {report.get('total_cache_write_tokens', 0)}",
        f"- Total cost: {report.get('total_cost', 0)} {report.get('currency', 'USD')}",
        f"- Total elapsed: {report['total_elapsed_ms']} ms",
        "",
        "## Cases",
        "",
        "| Case | Result | LLM calls | Tool messages | Tokens | Cost | Elapsed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["cases"]:
        status = "PASS" if result["passed"] else "FAIL"
        cost = result.get("cost", {}).get("total_cost", 0)
        lines.append(
            f"| `{result['name']}` | {status} | {result['llm_calls']} | "
            f"{result['tool_messages']} | {result['tokens']['total']} | "
            f"{cost} | {result['elapsed_ms']} ms |"
        )

    lines.extend(["", "## Outputs", ""])
    for result in report["cases"]:
        lines.extend(
            [
                f"### {result['name']}",
                "",
                f"- Description: {result['description']}",
                f"- Checks: `{json.dumps(result['checks'], ensure_ascii=False)}`",
                f"- Workspace: `{result.get('workspace', '')}`",
                "",
                "```text",
                str(result["output"]),
                "```",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def print_report(report: dict[str, Any]) -> None:
    print(f"Mini Agent benchmark: {report['passed']}/{report['case_count']} passed")
    print(f"Pass rate: {report['pass_rate'] * 100:.1f}%")
    print(f"Total tokens: {report['total_tokens']}")
    print(f"Total elapsed: {report['total_elapsed_ms']} ms")
    print()
    for result in report["cases"]:
        mark = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{mark}] {result['name']} "
            f"({result['elapsed_ms']} ms, {result['llm_calls']} llm calls, "
            f"{result['tokens']['total']} tokens)"
        )
        if not result["passed"]:
            print(f"  checks: {json.dumps(result['checks'], ensure_ascii=False)}")
            print(f"  output: {result['output']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mini Agent deterministic benchmark.")
    parser.add_argument("--real", action="store_true", help="Run benchmark with the configured real LLM.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path.")
    parser.add_argument("--markdown", type=Path, help="Write a Markdown report to this path.")
    args = parser.parse_args()

    if args.real:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_root = Path("outputs") / "benchmarks" / timestamp
        report = asyncio.run(run_real_benchmark(output_root))
        json_path = args.output or (output_root / "report.json")
        markdown_path = args.markdown or (output_root / "report.md")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown_report(report, markdown_path)
        print(f"JSON report: {json_path}")
        print(f"Markdown report: {markdown_path}")
    else:
        report = asyncio.run(run_benchmark())
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.markdown:
            write_markdown_report(report, args.markdown)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)

    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
