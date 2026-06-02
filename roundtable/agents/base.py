"""Abstract base classes for all Roundtable Policy agent roles."""

from __future__ import annotations

from abc import ABC, abstractmethod

from roundtable.core.types import (
    AgentResponse,
    ConsensusResponse,
    EvaluationResult,
    TaskInput,
)


class PlayerAgent(ABC):
    """Base class for player (proposer) agents.

    A player agent independently generates a candidate response to a given task.
    In the paper this corresponds to A_i^P.

    Subclass this to wrap any LLM or custom reasoning engine.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    @abstractmethod
    def propose(self, task: TaskInput) -> AgentResponse:
        """Generate a candidate response for the given task.

        Args:
            task: The input query to respond to.

        Returns:
            An AgentResponse with the generated content.
        """
        ...

    async def apropose(self, task: TaskInput) -> AgentResponse:
        """Async version of propose. Defaults to calling the sync version."""
        return self.propose(task)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_id={self.agent_id!r})"


class GraderAgent(ABC):
    """Base class for grader (evaluator) agents.

    A grader agent evaluates a player's response, producing a score in [-100, 100]
    and an uncertainty estimate (95% confidence interval). In the paper this
    corresponds to A_l^G.

    Subclass this to implement custom evaluation logic or LLM-based grading.
    """

    def __init__(self, grader_id: str) -> None:
        self.grader_id = grader_id

    @abstractmethod
    def evaluate(
        self,
        task: TaskInput,
        response: AgentResponse,
        reference: str | None = None,
    ) -> EvaluationResult:
        """Evaluate a single player response.

        Args:
            task: The original task/query.
            response: The player agent's response to evaluate.
            reference: Optional ground-truth answer for reference-based grading.

        Returns:
            An EvaluationResult with score and uncertainty.
        """
        ...

    async def aevaluate(
        self,
        task: TaskInput,
        response: AgentResponse,
        reference: str | None = None,
    ) -> EvaluationResult:
        """Async version of evaluate."""
        return self.evaluate(task, response, reference)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(grader_id={self.grader_id!r})"


class AggregatorAgent(ABC):
    """Base class for the fusion (aggregator) agent.

    The aggregator synthesizes a consensus response from all player responses,
    conditioned on the confidence-weight table. In the paper this corresponds to A^F.

    Subclass this to implement custom aggregation strategies.
    """

    @abstractmethod
    def aggregate(
        self,
        task: TaskInput,
        responses: list[AgentResponse],
        confidence_context: str,
    ) -> ConsensusResponse:
        """Fuse multiple player responses into a single consensus response.

        Args:
            task: The original task/query.
            responses: All player agent responses for this round.
            confidence_context: A formatted string representation of the
                confidence-weight table, to be included in the aggregation prompt.

        Returns:
            A ConsensusResponse with the synthesized answer.
        """
        ...

    async def aaggregate(
        self,
        task: TaskInput,
        responses: list[AgentResponse],
        confidence_context: str,
    ) -> ConsensusResponse:
        """Async version of aggregate."""
        return self.aggregate(task, responses, confidence_context)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
