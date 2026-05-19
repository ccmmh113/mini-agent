"""Layered system prompt assembly."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .context_budget import PromptLayerBudgets, clip_text_to_token_budget, count_text_tokens

PROJECT_RULE_FILES = ("CLAUDE.md", "AGENTS.md")
SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"


@dataclass(frozen=True)
class PromptSections:
    core: str
    skills: str = ""
    memory: str = ""
    project_rules: str = ""
    current_task_context: str = ""
    harness_summary: str = ""
    dynamic_context: str = ""

    def render(self) -> str:
        static_prompt = self.render_static()
        dynamic_prompt = self.render_dynamic()
        if not dynamic_prompt:
            return static_prompt
        if not static_prompt:
            return dynamic_prompt
        return f"{static_prompt}\n\n{SYSTEM_PROMPT_DYNAMIC_BOUNDARY}\n\n{dynamic_prompt}"

    def render_static(self) -> str:
        return self.core.strip()

    def render_dynamic(self) -> str:
        parts = [
            self.skills.strip(),
            self.memory.strip(),
            self.project_rules.strip(),
            self.current_task_context.strip(),
            self.harness_summary.strip(),
            self.dynamic_context.strip(),
        ]
        return "\n\n".join(part for part in parts if part)

    def static_cache_fingerprint(self) -> str:
        return hashlib.sha256(self.render_static().encode("utf-8")).hexdigest()


class SystemPromptBuilder:
    """Build the final system prompt from stable and dynamic layers."""

    def __init__(
        self,
        *,
        core_prompt: str,
        workspace_dir: str | Path,
        skill_loader: Any | None = None,
        harness_summaries: list[str] | None = None,
        layer_budgets: PromptLayerBudgets | None = None,
        now: datetime | None = None,
        max_memories: int = 12,
    ):
        self.core_prompt = core_prompt
        self.workspace_dir = Path(workspace_dir).resolve()
        self.skill_loader = skill_loader
        self.harness_summaries = harness_summaries or []
        self.layer_budgets = layer_budgets or PromptLayerBudgets()
        self.now = now or datetime.now()
        self.max_memories = max_memories

    def build(self) -> str:
        return self.build_sections().render()

    def build_sections(self) -> PromptSections:
        core = self._build_core()
        sections = PromptSections(
            core=core,
            skills=self._build_skills(),
            memory=self._build_memory(),
            project_rules=self._build_project_rules(),
            current_task_context=self._build_current_task_context(),
            harness_summary=self._build_harness_summary(),
            dynamic_context=self._build_dynamic_context(),
        )
        return self._apply_layer_budgets(sections)

    def _build_core(self) -> str:
        return self.core_prompt.replace("{SKILLS_METADATA}", "").strip()

    def _build_skills(self) -> str:
        if self.skill_loader is None:
            return ""
        metadata = self.skill_loader.get_skills_metadata_prompt()
        if not metadata:
            return ""
        return (
            "## Skills\n\n"
            "Skills are enabled for this run. Use `get_skill(skill_name)` only when a listed skill is relevant.\n\n"
            + metadata.strip()
        )

    def _build_memory(self) -> str:
        from .memory.markdown_store import MarkdownMemoryStore, format_memory_index_lines

        memories = MarkdownMemoryStore(self.workspace_dir / ".memory").search(limit=self.max_memories)
        if not memories:
            return ""

        lines = [
            "## Long-Term Memory Index",
            "",
            "Index only. Use `recall_notes` to read matching memory topic files before using details.",
            "Long-term memory may be stale. Treat it as a retrieval hint, not proof.",
            (
                "Verify remembered files, commands, APIs, dependencies, and behavior against the current "
                "workspace with tools before relying on them."
            ),
            "",
        ]
        lines.extend(format_memory_index_lines(memories))
        return "\n".join(lines)

    def _build_harness_summary(self) -> str:
        summaries = [summary.strip() for summary in self.harness_summaries if summary.strip()]
        if not summaries:
            return ""

        lines = [
            "## Harness Summary",
            "",
            "Compressed assistant/tool execution history. Treat this as factual context, not as a new user request.",
        ]
        for index, summary in enumerate(summaries, start=1):
            lines.extend(["", f"### Summary {index}", summary])
        return "\n".join(lines)

    def _build_project_rules(self) -> str:
        sections: list[str] = []
        for filename in PROJECT_RULE_FILES:
            path = self.workspace_dir / filename
            if not path.exists() or not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if content:
                sections.append(f"### {filename}\n\n{content}")
        if not sections:
            return ""
        return "## Project Rules\n\n" + "\n\n".join(sections)

    def _build_current_task_context(self) -> str:
        task_memory_file = self.workspace_dir / ".mini_agent" / "task_memory.json"
        if not task_memory_file.exists():
            return ""

        try:
            data = json.loads(task_memory_file.read_text(encoding="utf-8"))
        except Exception:
            return ""

        if not isinstance(data, dict):
            return ""

        active_task_id = data.get("active_task_id")
        tasks = data.get("tasks", [])
        if not active_task_id or not isinstance(tasks, list):
            return ""

        task = next(
            (item for item in tasks if isinstance(item, dict) and item.get("task_id") == active_task_id),
            None,
        )
        if task is None:
            return ""

        lines = [
            "## Current Task Context",
            "",
            "Use this as the current execution state. Prefer it over older inferred task state when they conflict, but follow the latest user request first.",
            f"- Goal: {self._clip_text(task.get('goal', 'unknown'))}",
            f"- Type: {self._clip_text(task.get('task_type', 'general'))}",
        ]

        archived_summary = str(task.get("archived_steps_summary", "")).strip()
        if archived_summary:
            lines.append(f"- Archived earlier progress: {self._clip_text(archived_summary, max_length=220)}")

        completed_steps = self._extract_values(task.get("completed_steps", []), "description", limit=5)
        decisions = self._extract_values(task.get("decisions", []), "decision", limit=3)
        open_questions = self._extract_values(task.get("open_questions", []), "question", limit=3)
        next_steps = self._extract_values(task.get("next_steps", []), "description", limit=3)

        self._append_section(lines, "Recent Completed Steps", completed_steps)
        self._append_section(lines, "Key Decisions", decisions)
        self._append_section(lines, "Open Questions", open_questions)
        self._append_section(lines, "Next Steps", next_steps)

        return "\n".join(lines)

    def _append_section(self, lines: list[str], title: str, items: list[str]) -> None:
        lines.append("")
        lines.append(f"{title}:")
        if not items:
            lines.append("- none")
            return
        for item in items:
            lines.append(f"- {item}")

    def _extract_values(self, items: Any, primary_key: str, limit: int) -> list[str]:
        if not isinstance(items, list):
            return []

        values: list[str] = []
        for item in items[-limit:]:
            value = ""
            if isinstance(item, dict):
                value = str(item.get(primary_key) or item.get("description") or item.get("content") or "").strip()
            elif isinstance(item, str):
                value = item.strip()
            if value:
                values.append(self._clip_text(value))
        return values

    def _clip_text(self, value: Any, max_length: int = 160) -> str:
        text = str(value).strip()
        if len(text) <= max_length:
            return text
        return text[: max_length - 3].rstrip() + "..."

    def _build_dynamic_context(self) -> str:
        return (
            "## Dynamic Context\n\n"
            f"- Current workspace: `{self.workspace_dir}`\n"
            "- All relative paths should be resolved relative to the current workspace.\n"
            f"- Current local time: {self.now.isoformat(timespec='seconds')}"
        )

    def _apply_layer_budgets(self, sections: PromptSections) -> PromptSections:
        budgets = self.layer_budgets
        return PromptSections(
            core=self._fit_layer("System Core", sections.core, budgets.core, keep=True),
            skills=self._fit_layer("Skills", sections.skills, budgets.skills),
            memory=self._fit_layer("Long-Term Memory", sections.memory, budgets.memory),
            project_rules=self._fit_layer("Project Rules", sections.project_rules, budgets.project_rules),
            current_task_context=self._fit_layer(
                "Current Task Context",
                sections.current_task_context,
                budgets.current_task_context,
            ),
            harness_summary=self._fit_layer("Harness Summary", sections.harness_summary, budgets.harness_summary),
            dynamic_context=self._fit_layer("Dynamic Context", sections.dynamic_context, budgets.dynamic_context, keep=True),
        )

    def _fit_layer(self, label: str, content: str, budget: int, *, keep: bool = False) -> str:
        if not content:
            return ""
        if budget <= 0:
            return content if keep else ""
        if count_text_tokens(content) <= budget:
            return content
        return clip_text_to_token_budget(content, budget, label=label)
