"""LLM client wrapper that supports multiple providers.

This module provides a unified interface for different LLM providers
(Anthropic and OpenAI) through a single LLMClient class.
"""

import logging

from ..retry import RetryConfig
from ..schema import LLMProvider, LLMResponse, Message
from .anthropic_client import AnthropicClient
from .base import LLMClientBase
from .openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM Client wrapper supporting multiple providers."""

    # MiniMax API domains that need automatic suffix handling
    MINIMAX_DOMAINS = ("api.minimax.io", "api.minimaxi.com")

    def __init__(
        self,
        api_key: str,
        provider: LLMProvider = LLMProvider.OPENAI,
        api_base: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        retry_config: RetryConfig | None = None,
        openai_prompt_cache_key: str | None = None,
        openai_prompt_cache_retention: str | None = None,
        disable_thinking: bool = False,
        enable_reasoning_split: bool = False,
        preserve_thinking: bool = False,
    ):
        """Initialize LLM client with specified provider.

        Args:
            api_key: API key for authentication
            provider: LLM provider (anthropic or openai)
            api_base: Base URL for the API (default: https://api.minimaxi.com)
                     For MiniMax API, suffix is auto-appended based on provider.
                     For third-party APIs (e.g., https://api.siliconflow.cn/v1), used as-is.
            model: Model name to use
            retry_config: Optional retry configuration
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.retry_config = retry_config or RetryConfig()
        self.preserve_thinking = preserve_thinking

        # Normalize api_base (remove trailing slash)
        api_base = api_base.rstrip("/")

        # Check if this is a MiniMax API endpoint
        is_minimax = any(domain in api_base for domain in self.MINIMAX_DOMAINS)

        if is_minimax:
            # For MiniMax API, ensure correct suffix based on provider
            # Strip any existing suffix first
            api_base = api_base.replace("/anthropic", "").replace("/v1", "")
            if provider == LLMProvider.ANTHROPIC:
                full_api_base = f"{api_base}/anthropic"
            elif provider == LLMProvider.OPENAI:
                full_api_base = f"{api_base}/v1"
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        else:
            # For third-party APIs, use api_base as-is
            full_api_base = api_base

        self.api_base = full_api_base

        # Instantiate the appropriate client
        self._client: LLMClientBase
        if provider == LLMProvider.ANTHROPIC:
            self._client = AnthropicClient(
                api_key=api_key,
                api_base=full_api_base,
                model=model,
                retry_config=retry_config,
                preserve_thinking=preserve_thinking,
            )
        elif provider == LLMProvider.OPENAI:
            self._client = OpenAIClient(
                api_key=api_key,
                api_base=full_api_base,
                model=model,
                retry_config=retry_config,
                prompt_cache_key=openai_prompt_cache_key,
                prompt_cache_retention=openai_prompt_cache_retention,
                disable_thinking=disable_thinking,
                enable_reasoning_split=enable_reasoning_split,
                preserve_thinking=preserve_thinking,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info("Initialized LLM client with provider: %s, api_base: %s", provider, full_api_base)

    @property
    def retry_callback(self):
        """Get retry callback.Q"""
        return self._client.retry_callback

    @retry_callback.setter
    def retry_callback(self, value):
        """Set retry callback."""
        self._client.retry_callback = value

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
    ) -> LLMResponse:
        """Generate response from LLM.

        Args:
            messages: List of conversation messages
            tools: Optional list of Tool objects or dicts

        Returns:
            LLMResponse containing the generated content
        """
        return await self._client.generate(messages, tools)
