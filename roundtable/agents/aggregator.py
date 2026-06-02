"""LLM-backed aggregator (fusion) agent implementation."""

from __future__ import annotations

from typing import Any

from roundtable.agents.base import AggregatorAgent
from roundtable.core.types import AgentResponse, ConsensusResponse, TaskInput
from roundtable.llm.base import LLMClient
from roundtable.prompts import (
    DEFAULT_AGGREGATOR_TEMPLATE,
    format_responses,
    render,
)


class LLMAggregatorAgent(AggregatorAgent):
    """Aggregator agent backed by an LLM client.

    Synthesizes a consensus response from multiple player responses,
    conditioned on the confidence-weight table.

    Args:
        client: An LLMClient instance to use for synthesis.
        system_prompt: System prompt (a user-supplied hyperparameter).
        prompt_template: User message template. Available placeholders:
            ``{query}``, ``{responses}`` (the serialized candidate answers),
            ``{confidence}`` (the confidence-weight table), plus any
            ``task.metadata`` keys.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens for the consensus response.
    """

    def __init__(
        self,
        client: LLMClient,
        system_prompt: str = "",
        prompt_template: str = DEFAULT_AGGREGATOR_TEMPLATE,
        temperature: float = 0.5,
        max_tokens: int = 4096,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _format_prompt(
        self,
        task: TaskInput,
        responses: list[AgentResponse],
        confidence_context: str,
    ) -> str:
        return render(
            self.prompt_template,
            query=task.query,
            responses=format_responses([(r.agent_id, r.content) for r in responses]),
            confidence=confidence_context,
            **task.metadata,
        )

    def aggregate(
        self,
        task: TaskInput,
        responses: list[AgentResponse],
        confidence_context: str,
    ) -> ConsensusResponse:
        llm_response = self.client.generate(
            prompt=self._format_prompt(task, responses, confidence_context),
            system=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return ConsensusResponse(
            content=llm_response.content,
            source_agent_ids=[r.agent_id for r in responses],
            metadata={"model": llm_response.model, "usage": llm_response.usage},
        )

    async def aaggregate(
        self,
        task: TaskInput,
        responses: list[AgentResponse],
        confidence_context: str,
    ) -> ConsensusResponse:
        llm_response = await self.client.agenerate(
            prompt=self._format_prompt(task, responses, confidence_context),
            system=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return ConsensusResponse(
            content=llm_response.content,
            source_agent_ids=[r.agent_id for r in responses],
            metadata={"model": llm_response.model, "usage": llm_response.usage},
        )

    def __repr__(self) -> str:
        return f"LLMAggregatorAgent(model={self.client.model!r})"


def create_aggregator_from_config(config: dict[str, Any]) -> LLMAggregatorAgent:
    """Create an aggregator agent from a configuration dict.

    Recognised keys: ``provider``, ``model`` (required); ``system_prompt``,
    ``prompt_template``, ``temperature``, ``max_tokens``, ``api_key``,
    ``base_url`` (optional).
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
    return LLMAggregatorAgent(
        client=client,
        system_prompt=config.get("system_prompt", ""),
        prompt_template=config.get("prompt_template", DEFAULT_AGGREGATOR_TEMPLATE),
        temperature=config.get("temperature", 0.5),
        max_tokens=config.get("max_tokens", 4096),
    )
