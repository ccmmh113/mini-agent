"""Tests for OpenAI prompt cache configuration and usage mapping."""

from pathlib import Path
from types import SimpleNamespace

from mini_agent.config import Config
from mini_agent.llm.openai_client import OpenAIClient
from mini_agent.retry import RetryConfig
from mini_agent.schema import FunctionCall, Message, ToolCall


def test_config_loads_openai_prompt_cache_settings(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
api_key: test-key
provider: openai
api_base: https://api.openai.com/v1
model: gpt-4o
openai_prompt_cache_key: project:workspace
openai_prompt_cache_retention: 24h
""".strip(),
        encoding="utf-8",
    )

    config = Config.from_yaml(config_file)

    assert config.llm.openai_prompt_cache_key == "project:workspace"
    assert config.llm.openai_prompt_cache_retention == "24h"
    assert config.llm.disable_thinking is False
    assert config.llm.enable_reasoning_split is False
    assert config.llm.preserve_thinking is False
    assert config.llm.show_thinking is False
    assert config.llm.log_thinking is False


def test_config_loads_token_pricing(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
api_key: test-key
provider: openai
api_base: https://api.openai.com/v1
model: gpt-4o
token_pricing:
  input_per_1m: 2.5
  output_per_1m: 10
  cache_read_per_1m: 1.25
  cache_write_per_1m: 3.75
  currency: USD
""".strip(),
        encoding="utf-8",
    )

    config = Config.from_yaml(config_file)

    assert config.llm.token_pricing.input_per_1m == 2.5
    assert config.llm.token_pricing.output_per_1m == 10
    assert config.llm.token_pricing.cache_read_per_1m == 1.25
    assert config.llm.token_pricing.cache_write_per_1m == 3.75
    assert config.llm.token_pricing.currency == "USD"


def test_config_loads_subagent_settings(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
api_key: test-key
provider: openai
api_base: https://api.openai.com/v1
model: gpt-4o
tools:
  enable_subagent: true
subagent:
  max_steps: 7
  token_limit: 5000
  request_context_limit: 5
  allowed_tools:
    - read_file
    - recall_notes
  allow_nested_subagent: false
""".strip(),
        encoding="utf-8",
    )

    config = Config.from_yaml(config_file)

    assert config.tools.enable_subagent is True
    assert config.subagent.max_steps == 7
    assert config.subagent.token_limit == 5000
    assert config.subagent.request_context_limit == 5
    assert config.subagent.allowed_tools == ["read_file", "recall_notes"]
    assert config.subagent.allow_nested_subagent is False


async def test_openai_client_maps_cached_tokens_from_usage():
    client = OpenAIClient(
        api_key="test-key",
        api_base="https://api.openai.com/v1",
        model="gpt-4o",
        retry_config=RetryConfig(enabled=False),
        prompt_cache_key="project:workspace",
        prompt_cache_retention="24h",
    )

    captured_params = {}

    async def fake_create(**params):
        captured_params.update(params)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="cached reply", reasoning_content=None, tool_calls=None)
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1200,
                completion_tokens=80,
                total_tokens=1280,
                prompt_tokens_details=SimpleNamespace(cached_tokens=900),
            ),
        )

    client.client.chat.completions.create = fake_create
    response = await client.generate(messages=[Message(role="user", content="Hello")], tools=None)

    assert captured_params["prompt_cache_key"] == "project:workspace"
    assert captured_params["prompt_cache_retention"] == "24h"
    assert "extra_body" not in captured_params
    assert response.usage is not None
    assert response.usage.prompt_tokens == 1200
    assert response.usage.completion_tokens == 80
    assert response.usage.total_tokens == 1280
    assert response.usage.cached_tokens == 900


async def test_openai_client_does_not_replay_thinking_by_default():
    client = OpenAIClient(
        api_key="test-key",
        api_base="https://api.openai.com/v1",
        model="gpt-4o",
        retry_config=RetryConfig(enabled=False),
    )

    captured_params = {}

    async def fake_create(**params):
        captured_params.update(params)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="reply", tool_calls=None))],
            usage=None,
        )

    client.client.chat.completions.create = fake_create
    await client.generate(messages=[Message(role="assistant", content="prior", thinking="hidden")], tools=None)

    assert captured_params["messages"] == [{"role": "assistant", "content": "prior"}]


async def test_openai_client_can_enable_reasoning_split_and_replay_thinking():
    client = OpenAIClient(
        api_key="test-key",
        api_base="https://api.openai.com/v1",
        model="gpt-4o",
        retry_config=RetryConfig(enabled=False),
        enable_reasoning_split=True,
        preserve_thinking=True,
    )

    captured_params = {}

    async def fake_create(**params):
        captured_params.update(params)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="reply", tool_calls=None))],
            usage=None,
        )

    client.client.chat.completions.create = fake_create
    await client.generate(messages=[Message(role="assistant", content="prior", thinking="hidden")], tools=None)

    assert captured_params["extra_body"] == {"reasoning_split": True}
    assert captured_params["messages"] == [{"role": "assistant", "content": "prior", "reasoning_content": "hidden"}]


async def test_openai_client_can_disable_thinking():
    client = OpenAIClient(
        api_key="test-key",
        api_base="https://api.openai.com/v1",
        model="gpt-4o",
        retry_config=RetryConfig(enabled=False),
        disable_thinking=True,
    )

    captured_params = {}

    async def fake_create(**params):
        captured_params.update(params)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="reply", tool_calls=None))],
            usage=None,
        )

    client.client.chat.completions.create = fake_create
    await client.generate(messages=[Message(role="user", content="hi")], tools=None)

    assert captured_params["extra_body"] == {"thinking": {"type": "disabled"}}


async def test_openai_client_does_not_replay_tool_call_reasoning_by_default():
    client = OpenAIClient(
        api_key="test-key",
        api_base="https://api.openai.com/v1",
        model="gpt-4o",
        retry_config=RetryConfig(enabled=False),
    )

    captured_params = {}

    async def fake_create(**params):
        captured_params.update(params)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="reply", tool_calls=None))],
            usage=None,
        )

    client.client.chat.completions.create = fake_create
    await client.generate(
        messages=[
            Message(
                role="assistant",
                content="",
                thinking="hidden",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        type="function",
                        function=FunctionCall(name="write_file", arguments={"path": "x", "content": "y"}),
                    )
                ],
            )
        ],
        tools=None,
    )

    assert captured_params["messages"] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": "{\"path\": \"x\", \"content\": \"y\"}"},
                }
            ],
        }
    ]
