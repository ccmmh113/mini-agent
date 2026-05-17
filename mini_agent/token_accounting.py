"""Token usage and local cost accounting helpers."""

from __future__ import annotations

from .schema import TokenCost, TokenPricing, TokenUsage


def uncached_prompt_tokens(usage: TokenUsage) -> int:
    """Return prompt tokens billed at normal input price."""

    return max(usage.prompt_tokens - usage.cached_tokens - usage.cache_write_tokens, 0)


def estimate_token_cost(usage: TokenUsage, pricing: TokenPricing | None) -> TokenCost | None:
    """Estimate cost from API-reported usage and configured per-million rates."""

    if pricing is None or not pricing.configured:
        return None

    input_cost = uncached_prompt_tokens(usage) * pricing.input_per_1m / 1_000_000
    output_cost = usage.completion_tokens * pricing.output_per_1m / 1_000_000
    cache_read_cost = usage.cached_tokens * pricing.cache_read_per_1m / 1_000_000
    cache_write_cost = usage.cache_write_tokens * pricing.cache_write_per_1m / 1_000_000
    total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost

    return TokenCost(
        input_cost=input_cost,
        output_cost=output_cost,
        cache_read_cost=cache_read_cost,
        cache_write_cost=cache_write_cost,
        total_cost=total_cost,
        currency=pricing.currency,
    )
