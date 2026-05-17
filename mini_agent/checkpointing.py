"""Checkpoint coordination for agent runs."""

from __future__ import annotations

from pathlib import Path

from .checkpoint import CheckpointStore
from .schema import Message


class CheckpointCoordinator:
    """Persist recovery snapshots at named agent loop boundaries."""

    def __init__(self, checkpoint_store: CheckpointStore | None):
        self.checkpoint_store = checkpoint_store

    def save(
        self,
        *,
        step: int,
        reason: str,
        messages: list[Message],
        workspace_dir: Path,
        available_tools: list[str],
    ) -> None:
        if self.checkpoint_store is None:
            return

        self.checkpoint_store.save(
            step=step,
            reason=reason,
            messages=messages,
            workspace_dir=workspace_dir,
            available_tools=available_tools,
        )
