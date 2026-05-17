"""Checkpoint storage for lightweight agent recovery."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .schema import Message


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class CheckpointStore:
    """Persist lightweight conversation checkpoints to the workspace."""

    def __init__(self, checkpoint_dir: str | Path, max_history: int = 20):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.history_dir = self.checkpoint_dir / "history"
        self.latest_file = self.checkpoint_dir / "latest.json"
        self.max_history = max_history

    def save(
        self,
        *,
        step: int,
        reason: str,
        messages: list[Message],
        workspace_dir: str | Path,
        available_tools: list[str],
    ) -> dict[str, Any]:
        payload = {
            "version": 1,
            "created_at": _now(),
            "step": step,
            "reason": reason,
            "workspace_dir": str(Path(workspace_dir).resolve()),
            "messages": [message.model_dump(mode="json") for message in messages],
            "tool_state": {
                "available_tools": available_tools,
            },
        }

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        serialized = json.dumps(payload, indent=2, ensure_ascii=False)
        self.latest_file.write_text(serialized, encoding="utf-8")

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        history_file = self.history_dir / f"{timestamp}-step{step + 1}-{reason}.json"
        history_file.write_text(serialized, encoding="utf-8")
        self._trim_history()
        return payload

    def load_latest(self) -> dict[str, Any] | None:
        if not self.latest_file.exists():
            return None
        try:
            data = json.loads(self.latest_file.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def load_latest_messages(self) -> list[Message]:
        validation = self.validate_messages()
        return validation["valid_messages"]

    def validate_messages(self) -> dict[str, Any]:
        data = self.load_latest()
        if not data:
            return {
                "total": 0,
                "valid": 0,
                "dropped": 0,
                "drop_ratio": 0.0,
                "valid_messages": [],
            }

        raw_messages = data.get("messages", [])
        if not isinstance(raw_messages, list):
            return {
                "total": 0,
                "valid": 0,
                "dropped": 0,
                "drop_ratio": 0.0,
                "valid_messages": [],
            }

        messages: list[Message] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            try:
                messages.append(Message.model_validate(item))
            except Exception:
                continue
        total = len(raw_messages)
        valid = len(messages)
        dropped = total - valid
        drop_ratio = (dropped / total) if total else 0.0
        return {
            "total": total,
            "valid": valid,
            "dropped": dropped,
            "drop_ratio": drop_ratio,
            "valid_messages": messages,
        }

    def get_resume_summary(self) -> str | None:
        data = self.load_latest()
        if not data:
            return None

        messages = data.get("messages", [])
        user_goal = "unknown"
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content", "")
                    if isinstance(content, str):
                        first_line = content.strip().split("\n")[0].strip()
                        if first_line:
                            user_goal = first_line[:120]
                    break

        reason = data.get("reason", "unknown")
        step = data.get("step", 0)
        created_at = data.get("created_at", "unknown")
        message_count = len(messages) if isinstance(messages, list) else 0
        return (
            f"Checkpoint ({reason})\n"
            f"  Goal: {user_goal}\n"
            f"  Step: {step + 1}\n"
            f"  Messages: {message_count}\n"
            f"  Saved: {created_at}"
        )

    def get_restore_status(self) -> str | None:
        data = self.load_latest()
        if not data:
            return None
        reason = data.get("reason", "unknown")
        step = data.get("step", 0)
        created_at = data.get("created_at", "unknown")
        return f"Restored to step {step + 1} from checkpoint reason '{reason}' saved at {created_at}."

    def validate_for_workspace(self, workspace_dir: str | Path) -> list[str]:
        data = self.load_latest()
        if not data:
            return ["Checkpoint file is missing or invalid."]

        issues: list[str] = []
        if data.get("version") != 1:
            issues.append(f"Unsupported checkpoint version: {data.get('version')}")

        checkpoint_workspace = str(data.get("workspace_dir", ""))
        current_workspace = str(Path(workspace_dir).resolve())
        if checkpoint_workspace and checkpoint_workspace != current_workspace:
            issues.append(f"Checkpoint workspace mismatch: {checkpoint_workspace} != {current_workspace}")

        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            issues.append("Checkpoint does not contain valid messages.")
            return issues

        validation = self.validate_messages()
        if validation["valid"] == 0:
            issues.append("Checkpoint does not contain any schema-valid messages.")
            return issues

        if validation["dropped"] > 0:
            issues.append(
                "Checkpoint contains invalid messages: "
                f"total={validation['total']}, valid={validation['valid']}, dropped={validation['dropped']}"
            )

        first_message = validation["valid_messages"][0]
        if first_message.role != "system":
            issues.append("Checkpoint restore rejected: first valid message is not a system message.")

        if validation["drop_ratio"] > 0.2:
            issues.append(
                "Checkpoint restore rejected: invalid message drop ratio is too high "
                f"({validation['drop_ratio']:.0%})."
            )

        return issues

    def _trim_history(self) -> None:
        files = sorted(self.history_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        for file_path in files[self.max_history :]:
            try:
                file_path.unlink()
            except OSError:
                continue
