"""OpenAI-compatible LLM client."""

from __future__ import annotations

from typing import Any

from roundtable.llm.base import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    """LLM client for OpenAI and OpenAI-compatible APIs (e.g. vLLM, Together).

    Args:
        model: Model identifier (e.g. "gpt-4o", "gpt-4.1").
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        base_url: Optional custom base URL for compatible APIs.
        **kwargs: Additional default kwargs passed to chat completions.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI package is required. Install with: "
                "pip install roundtable-policy[openai]"
            )
        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**client_kwargs)
        self._async_client = openai.AsyncOpenAI(**client_kwargs)

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        merged = {**self.default_kwargs, **kwargs}
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **merged,
        )
        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
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
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        merged = {**self.default_kwargs, **kwargs}
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **merged,
        )
        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return LLMResponse(content=content, model=self.model, usage=usage, raw=response)
