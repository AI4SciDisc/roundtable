"""LLM-backed grader agent implementation."""

from __future__ import annotations

import re
from typing import Any, Callable

from roundtable.agents.base import GraderAgent
from roundtable.core.types import AgentResponse, EvaluationResult, TaskInput
from roundtable.llm.base import LLMClient
from roundtable.prompts import DEFAULT_GRADER_TEMPLATE, render

# A parser maps the grader LLM's raw text + ids + dimensions to a result.
GradeParser = Callable[[str, str, str, "list[str] | None"], EvaluationResult]


def default_parse_grade(
    text: str,
    grader_id: str,
    agent_id: str,
    dimensions: list[str] | None = None,
) -> EvaluationResult:
    """Default mechanical parser for grader output.

    Extracts an overall ``Score:`` in [-100, 100], an ``Uncertainty:`` in
    [0, 50], an optional ``Rationale:``, and optional per-dimension scores.
    Your grader prompt should instruct the model to emit these fields; swap
    in a custom parser via ``LLMGraderAgent(parser=...)`` for other formats.
    """
    score_match = re.search(r"(?:Overall\s+)?Score:\s*([+-]?\d+)", text)
    score = float(score_match.group(1)) if score_match else 0.0
    score = max(-100.0, min(100.0, score))

    unc_match = re.search(r"(?:Overall\s+)?Uncertainty:\s*(\d+)", text)
    uncertainty = float(unc_match.group(1)) if unc_match else 25.0
    uncertainty = max(0.0, min(50.0, uncertainty))

    rat_match = re.search(r"Rationale:\s*(.+)", text, re.DOTALL)
    rationale = rat_match.group(1).strip() if rat_match else ""

    dim_scores: dict[str, float] = {}
    if dimensions:
        for dim in dimensions:
            pat = re.compile(
                rf"{re.escape(dim)}.*?Score\s*=\s*([+-]?\d+)", re.IGNORECASE
            )
            m = pat.search(text)
            if m:
                dim_scores[dim] = max(-100.0, min(100.0, float(m.group(1))))

    return EvaluationResult(
        grader_id=grader_id,
        agent_id=agent_id,
        score=score,
        uncertainty=uncertainty,
        rationale=rationale,
        dimension_scores=dim_scores,
    )


class LLMGraderAgent(GraderAgent):
    """Grader agent backed by an LLM client.

    Args:
        grader_id: Unique identifier for this grader.
        client: An LLMClient instance to use for evaluation.
        dimensions: Optional rubric dimensions for multi-dimensional scoring;
            exposed to the template as ``{dimensions}`` and passed to the parser.
        system_prompt: System prompt (a user-supplied hyperparameter).
        prompt_template: User message template. Available placeholders:
            ``{query}``, ``{response}``, ``{reference}``, ``{dimensions}``,
            plus any ``task.metadata`` keys. The template should instruct the
            model to emit whatever format ``parser`` expects.
        parser: Callable that turns the model's text into an EvaluationResult.
            Defaults to :func:`default_parse_grade`.
        temperature: Sampling temperature (lower = more deterministic grading).
        max_tokens: Maximum tokens for the evaluation response.
    """

    def __init__(
        self,
        grader_id: str,
        client: LLMClient,
        dimensions: list[str] | None = None,
        system_prompt: str = "",
        prompt_template: str = DEFAULT_GRADER_TEMPLATE,
        parser: GradeParser = default_parse_grade,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        super().__init__(grader_id)
        self.client = client
        self.dimensions = dimensions
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.parser = parser
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _format_prompt(
        self,
        task: TaskInput,
        response: AgentResponse,
        reference: str | None,
    ) -> str:
        return render(
            self.prompt_template,
            query=task.query,
            response=response.content,
            reference=reference or task.reference_answer or "",
            dimensions=", ".join(self.dimensions) if self.dimensions else "",
            **task.metadata,
        )

    def evaluate(
        self,
        task: TaskInput,
        response: AgentResponse,
        reference: str | None = None,
    ) -> EvaluationResult:
        llm_response = self.client.generate(
            prompt=self._format_prompt(task, response, reference),
            system=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self.parser(
            llm_response.content, self.grader_id, response.agent_id, self.dimensions
        )

    async def aevaluate(
        self,
        task: TaskInput,
        response: AgentResponse,
        reference: str | None = None,
    ) -> EvaluationResult:
        llm_response = await self.client.agenerate(
            prompt=self._format_prompt(task, response, reference),
            system=self.system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return self.parser(
            llm_response.content, self.grader_id, response.agent_id, self.dimensions
        )


def create_grader_from_config(config: dict[str, Any]) -> LLMGraderAgent:
    """Create a grader agent from a configuration dict.

    Recognised keys: ``grader_id``, ``provider``, ``model`` (required);
    ``dimensions``, ``system_prompt``, ``prompt_template``, ``temperature``,
    ``max_tokens``, ``api_key``, ``base_url`` (optional).
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
    return LLMGraderAgent(
        grader_id=config["grader_id"],
        client=client,
        dimensions=config.get("dimensions"),
        system_prompt=config.get("system_prompt", ""),
        prompt_template=config.get("prompt_template", DEFAULT_GRADER_TEMPLATE),
        temperature=config.get("temperature", 0.3),
        max_tokens=config.get("max_tokens", 2048),
    )
