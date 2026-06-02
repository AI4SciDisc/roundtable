"""Google Gemini LLM client (via the google-genai SDK)."""

from __future__ import annotations

from typing import Any

from roundtable.llm.base import LLMClient, LLMResponse


class GoogleClient(LLMClient):
    """LLM client for Google's Gemini models.

    Args:
        model: Model identifier (e.g. "gemini-2.5-pro", "gemini-2.5-flash").
        api_key: Google API key. If None, the SDK reads GOOGLE_API_KEY /
            GEMINI_API_KEY from the environment.
        **kwargs: Additional default config fields for generate_content.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package is required. Install with: "
                "pip install roundtable-policy[google]"
            )
        self._client = (
            genai.Client(api_key=api_key) if api_key is not None else genai.Client()
        )

    def _config(
        self, system: str, temperature: float, max_tokens: int, extra: dict[str, Any]
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            **self.default_kwargs,
            **extra,
        }
        if system:
            config["system_instruction"] = system
        return config

    @staticmethod
    def _usage(response: Any) -> dict[str, int]:
        meta = getattr(response, "usage_metadata", None)
        if not meta:
            return {}
        return {
            "prompt_tokens": getattr(meta, "prompt_token_count", 0) or 0,
            "completion_tokens": getattr(meta, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(meta, "total_token_count", 0) or 0,
        }

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._config(system, temperature, max_tokens, kwargs),
        )
        return LLMResponse(
            content=response.text or "",
            model=self.model,
            usage=self._usage(response),
            raw=response,
        )

    async def agenerate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._config(system, temperature, max_tokens, kwargs),
        )
        return LLMResponse(
            content=response.text or "",
            model=self.model,
            usage=self._usage(response),
            raw=response,
        )
