"""Lightweight Markdown-backed long-term memory store."""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


MEMORY_TYPES = {"user", "feedback", "project", "reference"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return slug[:64] or f"memory-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text.strip()

    end = text.find("\n---", 4)
    if end < 0:
        return {}, text.strip()

    raw_meta = text[4:end].strip()
    body = text[end + 4 :].strip()
    metadata: dict[str, str] = {}
    for line in raw_meta.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata, body


def _format_frontmatter(metadata: dict[str, Any]) -> str:
    lines = ["---"]
    for key in ("name", "description", "type", "created_at", "updated_at"):
        value = metadata.get(key)
        if value is not None:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


@dataclass(frozen=True)
class MarkdownMemory:
    name: str
    description: str
    type: str
    content: str
    path: Path
    created_at: str = ""
    updated_at: str = ""


class MarkdownMemoryStore:
    """Store durable memories as one Markdown file per memory.

    The layout intentionally matches a local-project workflow:
    .memory/
      MEMORY.md      # index only
      prefer_tabs.md # one memory document
    """

    def __init__(self, memory_dir: str | Path):
        self.memory_dir = Path(memory_dir)
        self.index_file = self.memory_dir / "MEMORY.md"

    def load(self) -> list[MarkdownMemory]:
        if not self.memory_dir.exists():
            return []

        memories: list[MarkdownMemory] = []
        for path in sorted(self.memory_dir.glob("*.md")):
            if path.name == "MEMORY.md":
                continue
            try:
                metadata, content = _parse_frontmatter(path.read_text(encoding="utf-8"))
            except OSError:
                continue

            memory_type = metadata.get("type", "project")
            if memory_type not in MEMORY_TYPES:
                memory_type = "project"
            fallback_description = content.splitlines()[0][:120] if content else path.stem
            memories.append(
                MarkdownMemory(
                    name=metadata.get("name") or path.stem,
                    description=metadata.get("description") or fallback_description,
                    type=memory_type,
                    content=content,
                    path=path,
                    created_at=metadata.get("created_at", ""),
                    updated_at=metadata.get("updated_at", ""),
                )
            )
        return memories

    def save_memory(
        self,
        *,
        content: str,
        memory_type: str = "project",
        name: str | None = None,
        description: str | None = None,
    ) -> MarkdownMemory:
        content = content.strip()
        if not content:
            raise ValueError("Memory content cannot be empty")

        memory_type = memory_type if memory_type in MEMORY_TYPES else "project"
        final_name = _slugify(name or description or content[:80])
        path = self.memory_dir / f"{final_name}.md"
        suffix = 2
        while path.exists():
            path = self.memory_dir / f"{final_name}-{suffix}.md"
            suffix += 1
        final_name = path.stem

        timestamp = _now()
        metadata = {
            "name": final_name,
            "description": (description or content.splitlines()[0])[:160],
            "type": memory_type,
            "created_at": timestamp,
            "updated_at": timestamp,
        }

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{_format_frontmatter(metadata)}\n\n{content}\n", encoding="utf-8")
        self.write_index()
        return MarkdownMemory(
            name=final_name,
            description=str(metadata["description"]),
            type=memory_type,
            content=content,
            path=path,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def delete(self, name: str) -> bool:
        path = self.memory_dir / f"{name}.md"
        if not path.exists() or path.name == "MEMORY.md":
            return False
        path.unlink()
        self.write_index()
        return True

    def write_index(self) -> None:
        memories = self.load()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Memory Index",
            "",
            "This file is an index only. Each memory lives in its own Markdown file.",
            "",
        ]
        if not memories:
            lines.append("No memories recorded yet.")
        else:
            for memory in memories:
                lines.append(f"- [{memory.name}]({memory.path.name}) `{memory.type}` - {memory.description}")
        self.index_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def search(
        self,
        *,
        query: str | None = None,
        memory_type: str | None = None,
        limit: int = 10,
    ) -> list[MarkdownMemory]:
        memories = self.load()
        if memory_type:
            memories = [memory for memory in memories if memory.type == memory_type]

        if query:
            terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]+", query)}
            if terms:
                scored: list[tuple[int, str, MarkdownMemory]] = []
                for memory in memories:
                    haystack = f"{memory.name} {memory.description} {memory.content}".lower()
                    score = sum(1 for term in terms if term in haystack)
                    if score > 0:
                        scored.append((score, memory.updated_at or memory.created_at, memory))
                scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
                memories = [memory for _, _, memory in scored]
            else:
                memories = []
        else:
            memories.sort(key=lambda memory: memory.updated_at or memory.created_at, reverse=True)

        return memories[: max(1, limit)]


def format_markdown_memories(title: str, memories: list[MarkdownMemory]) -> str:
    lines = [title]
    for index, memory in enumerate(memories, 1):
        lines.append(f"{index}. [{memory.type}] {memory.name} - {memory.description}")
        lines.append(f"   {memory.content}")
    return "\n".join(lines)
