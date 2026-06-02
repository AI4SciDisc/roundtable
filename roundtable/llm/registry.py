"""Model registry for config-driven LLM client creation."""

from __future__ import annotations

from typing import Any

from roundtable.llm.base import LLMClient

# Maps provider name -> (module_path, class_name).
# "openai" doubles as the client for any OpenAI-compatible HTTP API
# (vLLM, Together, Groq, DeepSeek, Ollama, Azure, ...) via the base_url arg.
_PROVIDERS: dict[str, tuple[str, str]] = {
    "openai": ("roundtable.llm.openai", "OpenAIClient"),
    "openai_compatible": ("roundtable.llm.openai", "OpenAIClient"),
    "anthropic": ("roundtable.llm.anthropic", "AnthropicClient"),
    "google": ("roundtable.llm.google", "GoogleClient"),
    "gemini": ("roundtable.llm.google", "GoogleClient"),
}


def register_provider(name: str, module_path: str, class_name: str) -> None:
    """Register a custom LLM provider.

    Args:
        name: Short provider name for use in configs (e.g. "vllm").
        module_path: Dotted module path containing the client class.
        class_name: Name of the LLMClient subclass.

    Example::

        register_provider("vllm", "my_package.vllm_client", "VLLMClient")
    """
    _PROVIDERS[name] = (module_path, class_name)


def create_client(provider: str, model: str, **kwargs: Any) -> LLMClient:
    """Create an LLM client from a provider name and model.

    Args:
        provider: Provider name (e.g. "openai", "anthropic").
        model: Model identifier.
        **kwargs: Provider-specific arguments (api_key, base_url, etc.).

    Returns:
        An initialized LLMClient instance.

    Raises:
        ValueError: If the provider is not registered.
    """
    if provider not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS.keys()))
        raise ValueError(
            f"Unknown provider {provider!r}. Available: {available}. "
            f"Use register_provider() to add custom providers."
        )
    module_path, class_name = _PROVIDERS[provider]
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(model=model, **kwargs)
