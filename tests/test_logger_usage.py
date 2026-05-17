"""Tests for logger usage persistence."""

from mini_agent.logger import AgentLogger
from mini_agent.schema import TokenUsage


def test_logger_writes_usage_into_response_log(tmp_path, monkeypatch):
    logger = AgentLogger()
    monkeypatch.setattr(logger, "log_dir", tmp_path)
    logger.start_new_run()

    logger.log_response(
        content="done",
        finish_reason="stop",
        usage=TokenUsage(
            prompt_tokens=1200,
            completion_tokens=80,
            total_tokens=1280,
            cached_tokens=900,
            cache_write_tokens=200,
        ),
    )

    log_text = logger.get_log_file_path().read_text(encoding="utf-8")
    assert '"prompt_tokens": 1200' in log_text
    assert '"completion_tokens": 80' in log_text
    assert '"total_tokens": 1280' in log_text
    assert '"cached_tokens": 900' in log_text
    assert '"cache_write_tokens": 200' in log_text
