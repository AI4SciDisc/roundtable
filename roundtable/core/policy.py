"""RoundtablePolicy: the main orchestrator implementing Algorithm 1."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from roundtable.agents.base import AggregatorAgent, GraderAgent, PlayerAgent
from roundtable.core.confidence_table import ConfidenceWeightTable
from roundtable.core.types import (
    AggregatedEvaluation,
    AgentResponse,
    EvaluationResult,
    RoundResult,
    TaskInput,
)

logger = logging.getLogger(__name__)


class RoundtablePolicy:
    """Roundtable Policy: confidence-weighted consensus aggregation.

    Implements Algorithm 1 from the paper. Orchestrates the three phases:
      1. Proposition: player agents independently generate candidate responses.
      2. Inference: aggregator fuses responses conditioned on confidence-weight table.
      3. Backward: grader committee evaluates responses and updates the table.

    Args:
        players: List of player (proposer) agents.
        graders: List of grader (evaluator) agents.
        aggregator: The fusion/aggregator agent.
        dimensions: Subtask or rubric dimension names for the confidence table.
        confidence_table: Optional pre-existing confidence-weight table.
            If None, a new one is initialized from the player/dimension lists.
        ema_decay: Optional EMA decay factor for table updates. If None, uses
            the 1/t schedule from Algorithm 1.
        inference_mode: If True, skip the backward (evaluation) phase and
            use the pre-constructed confidence table as-is.
    """

    def __init__(
        self,
        players: list[PlayerAgent],
        graders: list[GraderAgent],
        aggregator: AggregatorAgent,
        dimensions: list[str] | None = None,
        confidence_table: ConfidenceWeightTable | None = None,
        ema_decay: float | None = None,
        inference_mode: bool = False,
    ) -> None:
        self.players = players
        self.graders = graders
        self.aggregator = aggregator
        self.inference_mode = inference_mode
        self._results: list[RoundResult] = []

        agent_ids = [p.agent_id for p in players]
        dims = dimensions or ["default"]

        if confidence_table is not None:
            self.confidence_table = confidence_table
        else:
            self.confidence_table = ConfidenceWeightTable(ema_decay=ema_decay)
            self.confidence_table.initialize(agent_ids, dims)

    # ------------------------------------------------------------------
    # Phase 1: Proposition
    # ------------------------------------------------------------------

    def _proposition(self, task: TaskInput) -> list[AgentResponse]:
        """Run all player agents to produce candidate responses."""
        responses: list[AgentResponse] = []
        for player in self.players:
            logger.info("Player %s proposing...", player.agent_id)
            resp = player.propose(task)
            responses.append(resp)
        return responses

    async def _aproposition(self, task: TaskInput) -> list[AgentResponse]:
        """Async proposition: run all players concurrently."""
        coros = [player.apropose(task) for player in self.players]
        return list(await asyncio.gather(*coros))

    # ------------------------------------------------------------------
    # Phase 2: Inference (Consensus Aggregation)
    # ------------------------------------------------------------------

    def _inference(
        self,
        task: TaskInput,
        responses: list[AgentResponse],
    ) -> str:
        """Aggregate responses using the confidence-weight table."""
        confidence_context = self.confidence_table.format_for_prompt()
        logger.info("Aggregating %d responses...", len(responses))
        consensus = self.aggregator.aggregate(task, responses, confidence_context)
        return consensus.content

    async def _ainference(
        self,
        task: TaskInput,
        responses: list[AgentResponse],
    ) -> str:
        confidence_context = self.confidence_table.format_for_prompt()
        consensus = await self.aggregator.aaggregate(
            task, responses, confidence_context,
        )
        return consensus.content

    # ------------------------------------------------------------------
    # Phase 3: Backward (Evaluation + Table Update)
    # ------------------------------------------------------------------

    def _backward(
        self,
        task: TaskInput,
        responses: list[AgentResponse],
    ) -> list[AggregatedEvaluation]:
        """Evaluate all player responses and update the confidence table."""
        # Collect evaluations from all graders for all players
        all_evals: list[EvaluationResult] = []
        for grader in self.graders:
            for response in responses:
                logger.info(
                    "Grader %s evaluating player %s...",
                    grader.grader_id, response.agent_id,
                )
                ev = grader.evaluate(task, response, task.reference_answer)
                all_evals.append(ev)

        return self._process_evaluations(all_evals)

    async def _abackward(
        self,
        task: TaskInput,
        responses: list[AgentResponse],
    ) -> list[AggregatedEvaluation]:
        """Async backward: evaluate all responses concurrently."""
        coros = [
            grader.aevaluate(task, response, task.reference_answer)
            for grader in self.graders
            for response in responses
        ]
        all_evals = list(await asyncio.gather(*coros))
        return self._process_evaluations(all_evals)

    def _process_evaluations(
        self,
        all_evals: list[EvaluationResult],
    ) -> list[AggregatedEvaluation]:
        """Average evaluations across graders and update the confidence table."""
        # Group evaluations by agent_id
        by_agent: dict[str, list[EvaluationResult]] = defaultdict(list)
        for ev in all_evals:
            by_agent[ev.agent_id].append(ev)

        aggregated: list[AggregatedEvaluation] = []
        for agent_id, evals in by_agent.items():
            n = len(evals)
            mean_score = sum(e.score for e in evals) / n
            mean_unc = sum(e.uncertainty for e in evals) / n

            # Average dimension scores across graders
            dim_scores: dict[str, list[float]] = defaultdict(list)
            for e in evals:
                for dim, val in e.dimension_scores.items():
                    dim_scores[dim].append(val)
            avg_dim_scores = {
                dim: sum(vals) / len(vals) for dim, vals in dim_scores.items()
            }

            agg = AggregatedEvaluation(
                agent_id=agent_id,
                mean_score=mean_score,
                mean_uncertainty=mean_unc,
                dimension_scores=avg_dim_scores,
                num_graders=n,
            )
            aggregated.append(agg)

            # Update confidence-weight table
            # If we have per-dimension scores, use those; otherwise use overall
            if avg_dim_scores:
                unc_map = {dim: mean_unc for dim in avg_dim_scores}
                self.confidence_table.update(agent_id, avg_dim_scores, unc_map)
            else:
                # Map overall score to all dimensions
                score_map = {
                    d: mean_score for d in self.confidence_table.dimensions
                }
                unc_map = {d: mean_unc for d in self.confidence_table.dimensions}
                self.confidence_table.update(agent_id, score_map, unc_map)

        self.confidence_table.increment_round()
        return aggregated

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run_round(self, task: TaskInput) -> RoundResult:
        """Execute one full round of the Roundtable Policy.

        This runs Algorithm 1 for a single task:
          1. Proposition: all players generate responses.
          2. Inference: aggregator fuses with confidence weighting.
          3. Backward: graders evaluate and update the table
             (skipped if inference_mode=True).

        Args:
            task: The input task to process.

        Returns:
            A RoundResult containing all outputs from this round.
        """
        round_idx = len(self._results)
        logger.info("=== Round %d: %s ===", round_idx, task.task_id or "task")

        # Phase 1: Proposition
        responses = self._proposition(task)

        # Phase 2: Inference
        confidence_context = self.confidence_table.format_for_prompt()
        consensus = self.aggregator.aggregate(task, responses, confidence_context)

        # Phase 3: Backward (unless inference-only mode)
        evaluations: list[AggregatedEvaluation] = []
        if not self.inference_mode:
            evaluations = self._backward(task, responses)

        result = RoundResult(
            task=task,
            player_responses=responses,
            consensus=consensus,
            evaluations=evaluations,
            round_index=round_idx,
        )
        self._results.append(result)
        return result

    async def arun_round(self, task: TaskInput) -> RoundResult:
        """Async version of run_round — runs phases with concurrency."""
        round_idx = len(self._results)
        logger.info("=== Round %d: %s ===", round_idx, task.task_id or "task")

        # Phase 1: Proposition (concurrent)
        responses = await self._aproposition(task)

        # Phase 2: Inference
        confidence_context = self.confidence_table.format_for_prompt()
        consensus = await self.aggregator.aaggregate(
            task, responses, confidence_context,
        )

        # Phase 3: Backward (concurrent, unless inference-only)
        evaluations: list[AggregatedEvaluation] = []
        if not self.inference_mode:
            evaluations = await self._abackward(task, responses)

        result = RoundResult(
            task=task,
            player_responses=responses,
            consensus=consensus,
            evaluations=evaluations,
            round_index=round_idx,
        )
        self._results.append(result)
        return result

    def run(self, tasks: list[TaskInput]) -> list[RoundResult]:
        """Run the full Roundtable Policy over a sequence of tasks.

        This is the top-level entry for Algorithm 1: iterates over tasks,
        running proposition → inference → backward for each.

        Args:
            tasks: The sequence of tasks to process.

        Returns:
            List of RoundResults, one per task.
        """
        results: list[RoundResult] = []
        for task in tasks:
            result = self.run_round(task)
            results.append(result)
        return results

    async def arun(self, tasks: list[TaskInput]) -> list[RoundResult]:
        """Async version of run."""
        results: list[RoundResult] = []
        for task in tasks:
            result = await self.arun_round(task)
            results.append(result)
        return results

    @property
    def results(self) -> list[RoundResult]:
        """All results from rounds executed so far."""
        return list(self._results)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RoundtablePolicy:
        """Create a RoundtablePolicy from a configuration dict.

        See configs/example.yaml for the expected structure.

        Args:
            config: Full configuration dictionary.

        Returns:
            An initialized RoundtablePolicy ready to run.
        """
        from roundtable.agents.aggregator import create_aggregator_from_config
        from roundtable.agents.grader import create_grader_from_config
        from roundtable.agents.player import create_player_from_config

        players = [
            create_player_from_config(pc) for pc in config["players"]
        ]
        graders = [
            create_grader_from_config(gc) for gc in config["graders"]
        ]
        aggregator = create_aggregator_from_config(config["aggregator"])
        dimensions = config.get("dimensions", ["default"])

        # Load pre-existing confidence table if specified
        table = None
        table_path = config.get("confidence_table_path")
        if table_path:
            table = ConfidenceWeightTable.load(table_path)

        return cls(
            players=players,
            graders=graders,
            aggregator=aggregator,
            dimensions=dimensions,
            confidence_table=table,
            ema_decay=config.get("ema_decay"),
            inference_mode=config.get("inference_mode", False),
        )
