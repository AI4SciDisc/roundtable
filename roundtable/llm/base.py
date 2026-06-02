"""Abstract LLM client interface for provider-agnostic model access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Standard response wrapper from any LLM provider.

    Attributes:
        content: The generated text.
        model: The model identifier used.
        usage: Token usage stats (prompt_tokens, completion_tokens, etc.).
        raw: The raw provider-specific response object.
    """

    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class LLMClient(ABC):
    """Abstract interface for LLM API clients.

    Implement this to add support for a new LLM provider (e.g. OpenAI,
    Anthropic, local vLLM, etc.). The Roundtable agents use this interface
    to generate text, keeping the framework provider-agnostic.
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.default_kwargs = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user message / main prompt.
            system: Optional system message.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Provider-specific overrides.

        Returns:
            An LLMResponse with the generated text.
        """
        ...

    async def agenerate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async version of generate. Defaults to sync call."""
        return self.generate(
            prompt, system=system, temperature=temperature,
            max_tokens=max_tokens, **kwargs,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
