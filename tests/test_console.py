"""Tests for console rendering."""

from __future__ import annotations

import io
import sys

from mini_agent.console import AgentConsoleRenderer


def test_console_renderer_degrades_unencodable_symbols(monkeypatch, tmp_path):
    """Console output should not crash on terminals with narrow encodings."""

    raw_output = io.BytesIO()
    gbk_stdout = io.TextIOWrapper(raw_output, encoding="gbk", errors="strict")
    monkeypatch.setattr(sys, "stdout", gbk_stdout)

    AgentConsoleRenderer().log_file(tmp_path / "agent.log")

    gbk_stdout.flush()
    output = raw_output.getvalue().decode("gbk")
    assert "Log file:" in output
