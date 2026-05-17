import pytest

from mini_agent.schema import TokenPricing, TokenUsage
from mini_agent.token_accounting import estimate_token_cost, uncached_prompt_tokens


def test_uncached_prompt_tokens_excludes_cache_read_and_write_tokens():
    usage = TokenUsage(
        prompt_tokens=1_500,
        completion_tokens=200,
        total_tokens=1_700,
        cached_tokens=900,
        cache_write_tokens=100,
    )

    assert uncached_prompt_tokens(usage) == 500


def test_estimate_token_cost_uses_per_million_rates():
    usage = TokenUsage(
        prompt_tokens=1_500,
        completion_tokens=200,
        total_tokens=1_700,
        cached_tokens=900,
        cache_write_tokens=100,
    )
    pricing = TokenPricing(
        input_per_1m=2.0,
        output_per_1m=8.0,
        cache_read_per_1m=1.0,
        cache_write_per_1m=3.0,
        currency="USD",
    )

    cost = estimate_token_cost(usage, pricing)

    assert cost is not None
    assert cost.input_cost == 0.001
    assert cost.output_cost == 0.0016
    assert cost.cache_read_cost == 0.0009
    assert cost.cache_write_cost == 0.0003
    assert cost.total_cost == pytest.approx(0.0038)


def test_estimate_token_cost_returns_none_without_configured_rates():
    usage = TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120)

    assert estimate_token_cost(usage, TokenPricing()) is None
