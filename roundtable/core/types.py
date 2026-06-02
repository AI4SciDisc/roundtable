"""Core data types for Roundtable Policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskInput:
    """A single task/query to be processed by the roundtable.

    Attributes:
        query: The question or prompt text.
        task_id: Unique identifier for this task.
        domain: Optional domain label (e.g. "physics", "biology").
        reference_answer: Ground-truth answer if available (for ScienceEval-style tasks).
        metadata: Arbitrary additional data (rubric, subtask type, etc.).
    """

    query: str
    task_id: str = ""
    domain: str = ""
    reference_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """A response produced by a player agent.

    Attributes:
        agent_id: Identifier of the player agent that produced this response.
        content: The generated text response.
        metadata: Optional extra data (e.g. model name, token counts).
    """

    agent_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Evaluation output from a single grader for a single player's response.

    Attributes:
        grader_id: Identifier of the grader agent.
        agent_id: Identifier of the player agent being evaluated.
        score: Quality score in [-100, 100].
        uncertainty: Calibrated 95% confidence interval half-width.
        rationale: Optional textual justification.
        dimension_scores: Per-rubric-dimension scores (for ScienceNarrative).
    """

    grader_id: str
    agent_id: str
    score: float
    uncertainty: float = 0.0
    rationale: str = ""
    dimension_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class AggregatedEvaluation:
    """Committee-averaged evaluation for a single player agent in one round.

    Produced by averaging EvaluationResults across all graders.

    Attributes:
        agent_id: The player agent being evaluated.
        mean_score: Average score across graders.
        mean_uncertainty: Average uncertainty across graders.
        dimension_scores: Averaged per-dimension scores.
        num_graders: How many graders contributed.
    """

    agent_id: str
    mean_score: float
    mean_uncertainty: float
    dimension_scores: dict[str, float] = field(default_factory=dict)
    num_graders: int = 0


@dataclass
class ConsensusResponse:
    """The fused consensus response produced by the aggregator.

    Attributes:
        content: The synthesized response text.
        source_agent_ids: IDs of player agents whose responses were aggregated.
        metadata: Extra data (e.g. which weights were applied).
    """

    content: str
    source_agent_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundResult:
    """Complete result of one roundtable round.

    Attributes:
        task: The input task for this round.
        player_responses: Responses from all player agents.
        consensus: The fused consensus response.
        evaluations: Per-player aggregated evaluations.
        round_index: The round number (0-indexed).
    """

    task: TaskInput
    player_responses: list[AgentResponse]
    consensus: ConsensusResponse
    evaluations: list[AggregatedEvaluation]
    round_index: int = 0
