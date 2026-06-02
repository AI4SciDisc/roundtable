"""Anthropic Claude LLM client."""

from __future__ import annotations

from typing import Any

from roundtable.llm.base import LLMClient, LLMResponse


class AnthropicClient(LLMClient):
    """LLM client for Anthropic's Claude models.

    Args:
        model: Model identifier (e.g. "claude-sonnet-4-6", "claude-opus-4-6").
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        **kwargs: Additional default kwargs passed to messages.create.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package is required. Install with: "
                "pip install roundtable-policy[anthropic]"
            )
        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self._client = anthropic.Anthropic(**client_kwargs)
        self._async_client = anthropic.AsyncAnthropic(**client_kwargs)

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        merged = {**self.default_kwargs, **kwargs}
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **merged,
        }
        if system:
            create_kwargs["system"] = system

        response = self._client.messages.create(**create_kwargs)
        content = response.content[0].text if response.content else ""
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return LLMResponse(content=content, model=self.model, usage=usage, raw=response)

    async def agenerate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        merged = {**self.default_kwargs, **kwargs}
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **merged,
        }
        if system:
            create_kwargs["system"] = system

        response = await self._async_client.messages.create(**create_kwargs)
        content = response.content[0].text if response.content else ""
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return LLMResponse(content=content, model=self.model, usage=usage, raw=response)
