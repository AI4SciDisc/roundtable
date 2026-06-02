"""LLM-backed player agent implementation."""

from __future__ import annotations

from typing import Any

from roundtable.agents.base import PlayerAgent
from roundtable.core.types import AgentResponse, TaskInput
from roundtable.llm.base import LLMClient
from roundtable.prompts import DEFAULT_PLAYER_TEMPLATE, render


class LLMPlayerAgent(PlayerAgent):
    """Player agent backed by an LLM client.

    Args:
        agent_id: Unique identifier for this player.
        client: An LLMClient instance to use for generation.
        system_prompt: System prompt (a user-supplied hyperparameter).
        prompt_template: User message template. Available placeholders:
            ``{query}``, ``{domain}``, plus any ``task.metadata`` keys.
        temperature: Sampling temperature for generation.
        max_tokens: Maximum tokens to generate.
    """

    def __init__(
        self,
        agent_id: str,
        client: LLMClient,
        system_prompt: str = "",
        prompt_template: str = DEFAULT_PLAYER_TEMPLATE,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(agent_id)
        self.client = client
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _format_prompt(self, task: TaskInput) -> str:
        return render(
            self.prompt_template,
            query=task.query,
            domain=task.domain,
            **task.metadata,
        )

    def propose(self, task: TaskInput) -> AgentResponse:
        response = self.client.generate(
            prompt=self._format_prompt(task),
            system=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return AgentResponse(
            agent_id=self.agent_id,
            content=response.content,
            metadata={"model": response.model, "usage": response.usage},
        )

    async def apropose(self, task: TaskInput) -> AgentResponse:
        response = await self.client.agenerate(
            prompt=self._format_prompt(task),
            system=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return AgentResponse(
            agent_id=self.agent_id,
            content=response.content,
            metadata={"model": response.model, "usage": response.usage},
        )


def create_player_from_config(config: dict[str, Any]) -> LLMPlayerAgent:
    """Create a player agent from a configuration dict.

    Recognised keys: ``agent_id``, ``provider``, ``model`` (required);
    ``system_prompt``, ``prompt_template``, ``temperature``, ``max_tokens``,
    ``api_key``, ``base_url`` (optional).
    """
    from roundtable.llm.registry import create_client

    client_kwargs: dict[str, Any] = {}
    if "api_key" in config:
        client_kwargs["api_key"] = config["api_key"]
    if "base_url" in config:
        client_kwargs["base_url"] = config["base_url"]

    client = create_client(
        provider=config["provider"],
        model=config["model"],
        **client_kwargs,
    )
    return LLMPlayerAgent(
        agent_id=config["agent_id"],
        client=client,
        system_prompt=config.get("system_prompt", ""),
        prompt_template=config.get("prompt_template", DEFAULT_PLAYER_TEMPLATE),
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 4096),
    )
