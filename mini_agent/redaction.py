"""Utilities for removing secrets from logs and tool-visible data."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

REDACTION = "[REDACTED]"

SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "cookie",
    "credential",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "token",
)

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), REDACTION),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"), REDACTION),
    (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), REDACTION),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"), f"Bearer {REDACTION}"),
    (
        re.compile(
            r"(?i)(\b(?:api[_-]?key|token|secret|password|authorization|cookie|credential|access[_-]?key|refresh[_-]?token|private[_-]?key)\b\s*[:=]\s*)([^\s,;]+)"
        ),
        rf"\1{REDACTION}",
    ),
    (
        re.compile(
            r"(?i)([\"'](?:api[_-]?key|token|secret|password|authorization|cookie|credential|access[_-]?key|refresh[_-]?token|private[_-]?key)[\"']\s*:\s*[\"'])([^\"']+)([\"'])"
        ),
        rf"\1{REDACTION}\3",
    ),
]


def is_sensitive_key(key: str) -> bool:
    """Return whether a mapping key likely names secret-bearing data."""

    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def redact_text(text: str) -> str:
    """Redact known secret patterns from free-form text."""

    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def redact_data(value: Any) -> Any:
    """Recursively redact secrets while preserving the input shape."""

    if isinstance(value, str):
        return redact_text(value)

    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if is_sensitive_key(str(key)):
                redacted[key] = REDACTION
            else:
                redacted[key] = redact_data(item)
        return redacted

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [redact_data(item) for item in value]

    return value


def redact_tool_result(result: Any) -> Any:
    """Return a redacted copy of a tool result, including subclass fields."""

    updates: dict[str, Any] = {}
    for field_name in type(result).model_fields:
        value = getattr(result, field_name)
        updates[field_name] = redact_data(value)
    return result.model_copy(update=updates)
