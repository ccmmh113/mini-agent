"""Tests for layered system prompt assembly."""

from datetime import datetime
from pathlib import Path

from mini_agent.context_budget import PromptLayerBudgets
from mini_agent.memory.markdown_store import MarkdownMemoryStore
from mini_agent.prompt_builder import SYSTEM_PROMPT_DYNAMIC_BOUNDARY, SystemPromptBuilder


class FakeSkillLoader:
    loaded_skills = ["python"]

    def get_skills_metadata_prompt(self) -> str:
        return "- python: Python workflow guidance"


def test_system_prompt_builder_renders_layers(tmp_path: Path):
    MarkdownMemoryStore(tmp_path / ".memory").save_memory(
        content="Full durable note: the user prefers concise implementation summaries with detailed examples kept out of startup context.",
        memory_type="user",
        name="concise-summaries",
        description="User prefers concise implementation summaries",
    )
    (tmp_path / "AGENTS.md").write_text("Use focused tests before broad tests.", encoding="utf-8")

    prompt = SystemPromptBuilder(
        core_prompt="Core instructions\n\n{SKILLS_METADATA}",
        workspace_dir=tmp_path,
        skill_loader=FakeSkillLoader(),
        now=datetime(2026, 5, 16, 10, 30, 0),
    ).build()

    assert "Core instructions" in prompt
    assert "{SKILLS_METADATA}" not in prompt
    assert "## Skills" in prompt
    assert "- python: Python workflow guidance" in prompt
    assert "## Long-Term Memory Index" in prompt
    assert "Use `recall_notes`" in prompt
    assert "Verify remembered files, commands, APIs, dependencies, and behavior" in prompt
    assert "- [concise-summaries](concise-summaries.md) `user`" in prompt
    assert "Full durable note:" not in prompt
    assert "## Project Rules" in prompt
    assert "Use focused tests before broad tests." in prompt
    assert "## Current Task Context" not in prompt
    assert "## Dynamic Context" in prompt
    assert f"Current workspace: `{tmp_path.resolve()}`" in prompt
    assert "2026-05-16T10:30:00" in prompt
    assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in prompt
    assert prompt.index("Core instructions") < prompt.index(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
    assert prompt.index(SYSTEM_PROMPT_DYNAMIC_BOUNDARY) < prompt.index("## Skills")


def test_system_prompt_builder_omits_empty_optional_layers(tmp_path: Path):
    prompt = SystemPromptBuilder(
        core_prompt="Core only",
        workspace_dir=tmp_path,
        now=datetime(2026, 5, 16, 10, 30, 0),
    ).build()

    assert "Core only" in prompt
    assert "## Skills" not in prompt
    assert "## Long-Term Memory" not in prompt
    assert "## Project Rules" not in prompt
    assert "## Current Task Context" not in prompt
    assert "## Dynamic Context" in prompt


def test_system_prompt_builder_includes_harness_summary(tmp_path: Path):
    prompt = SystemPromptBuilder(
        core_prompt="Core only",
        workspace_dir=tmp_path,
        harness_summaries=["Read the codebase and found the prompt boundary."],
        now=datetime(2026, 5, 16, 10, 30, 0),
    ).build()

    assert "## Harness Summary" in prompt
    assert "Compressed assistant/tool execution history" in prompt
    assert "Read the codebase and found the prompt boundary." in prompt


def test_system_prompt_builder_applies_layer_budgets(tmp_path: Path):
    (tmp_path / "AGENTS.md").write_text("\n".join(f"rule-{index}: keep focused" for index in range(300)), encoding="utf-8")

    prompt = SystemPromptBuilder(
        core_prompt="Core only",
        workspace_dir=tmp_path,
        layer_budgets=PromptLayerBudgets(project_rules=80),
        now=datetime(2026, 5, 16, 10, 30, 0),
    ).build()

    assert "## Project Rules" in prompt
    assert "Project Rules compressed to fit 80 token budget" in prompt
    assert "rule-0: keep focused" in prompt


def test_system_prompt_builder_includes_current_task_context(tmp_path: Path):
    task_memory_dir = tmp_path / ".mini_agent"
    task_memory_dir.mkdir(parents=True, exist_ok=True)
    (task_memory_dir / "task_memory.json").write_text(
        """
{
  "active_task_id": "task-1",
  "tasks": [
    {
      "task_id": "task-1",
      "goal": "Refactor memory pipeline",
      "task_type": "coding",
      "status": "active",
      "completed_steps": [
        {"description": "Read note_tool.py", "timestamp": "2026-05-16T10:00:00"},
        {"description": "Removed model-callable task memory tools", "timestamp": "2026-05-16T10:10:00"}
      ],
      "decisions": [
        {"decision": "Task memory is runtime-managed only", "reason": "Avoid duplicate writes", "timestamp": "2026-05-16T10:20:00"}
      ],
      "artifacts": [],
      "open_questions": [
        {"question": "How should current task context be injected?", "timestamp": "2026-05-16T10:25:00"}
      ],
      "next_steps": [
        {"description": "Add a minimal current task context layer", "timestamp": "2026-05-16T10:30:00"}
      ],
      "created_at": "2026-05-16T10:00:00",
      "updated_at": "2026-05-16T10:30:00"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    prompt = SystemPromptBuilder(
        core_prompt="Core only",
        workspace_dir=tmp_path,
        now=datetime(2026, 5, 16, 10, 30, 0),
    ).build()

    assert "## Current Task Context" in prompt
    assert "- Goal: Refactor memory pipeline" in prompt
    assert "- Type: coding" in prompt
    assert "- Archived earlier progress:" not in prompt
    assert "Recent Completed Steps:" in prompt
    assert "- Read note_tool.py" in prompt
    assert "- Removed model-callable task memory tools" in prompt
    assert "Key Decisions:" in prompt
    assert "- Task memory is runtime-managed only" in prompt
    assert "Open Questions:" in prompt
    assert "- How should current task context be injected?" in prompt
    assert "Next Steps:" in prompt
    assert "- Add a minimal current task context layer" in prompt


def test_system_prompt_builder_includes_archived_task_summary(tmp_path: Path):
    task_memory_dir = tmp_path / ".mini_agent"
    task_memory_dir.mkdir(parents=True, exist_ok=True)
    (task_memory_dir / "task_memory.json").write_text(
        """
{
  "active_task_id": "task-1",
  "tasks": [
    {
      "task_id": "task-1",
      "goal": "Long task",
      "task_type": "coding",
      "status": "active",
      "completed_steps": [],
      "decisions": [],
      "artifacts": [],
      "open_questions": [],
      "next_steps": [],
      "archived_steps_summary": "[2026-05-16T10:30:00] Archived 35 earlier completed steps: alpha; beta; gamma",
      "created_at": "2026-05-16T10:00:00",
      "updated_at": "2026-05-16T10:30:00"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    prompt = SystemPromptBuilder(
        core_prompt="Core only",
        workspace_dir=tmp_path,
        now=datetime(2026, 5, 16, 10, 30, 0),
    ).build()

    assert "- Archived earlier progress: [2026-05-16T10:30:00] Archived 35 earlier completed steps: alpha; beta; gamma" in prompt


def test_prompt_sections_expose_static_and_dynamic_parts(tmp_path: Path):
    sections = SystemPromptBuilder(
        core_prompt="Core only",
        workspace_dir=tmp_path,
        skill_loader=FakeSkillLoader(),
        now=datetime(2026, 5, 16, 10, 30, 0),
    ).build_sections()

    assert sections.render_static() == "Core only"
    assert "## Skills" in sections.render_dynamic()
    assert "## Dynamic Context" in sections.render_dynamic()
    assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in sections.render()
    assert sections.static_cache_fingerprint() == "342187d847e7a69e84f355c95f4e9bcf8594242444690c80a77cfea7bafb1b24"
