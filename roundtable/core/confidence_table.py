"""Confidence-weight table: structured long-term memory of agent reliability."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ConfidenceWeightTable:
    """Confidence-weight table theta from the paper.

    Stores per-player, per-dimension cumulative (score, uncertainty) tuples.
    This serves as a structured memory of each player's historical reliability.

    The table is a matrix of shape (num_players, num_dimensions) where each
    entry holds cumulative (score_sum, uncertainty_sum) values.

    Attributes:
        agent_ids: Ordered list of player agent identifiers.
        dimensions: Ordered list of dimension/subtask names.
        scores: Cumulative score matrix [num_players x num_dimensions].
        uncertainties: Cumulative uncertainty matrix [num_players x num_dimensions].
        round_count: Number of rounds incorporated so far.
        ema_decay: Exponential moving average decay factor (1 - decay gives
            weight to new observations). Corresponds to the (1 - 1/t) factor
            in Algorithm 1.
    """

    agent_ids: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    scores: np.ndarray | None = None
    uncertainties: np.ndarray | None = None
    round_count: int = 0
    ema_decay: float | None = None

    def __post_init__(self) -> None:
        if self.agent_ids and self.dimensions:
            n_agents = len(self.agent_ids)
            n_dims = len(self.dimensions)
            if self.scores is None:
                self.scores = np.zeros((n_agents, n_dims), dtype=np.float64)
            if self.uncertainties is None:
                self.uncertainties = np.zeros((n_agents, n_dims), dtype=np.float64)

    def initialize(self, agent_ids: list[str], dimensions: list[str]) -> None:
        """Initialize (or reset) the table for a given set of agents and dimensions.

        Args:
            agent_ids: Player agent identifiers.
            dimensions: Subtask or rubric dimension names.
        """
        self.agent_ids = list(agent_ids)
        self.dimensions = list(dimensions)
        n_agents = len(agent_ids)
        n_dims = len(dimensions)
        self.scores = np.zeros((n_agents, n_dims), dtype=np.float64)
        self.uncertainties = np.zeros((n_agents, n_dims), dtype=np.float64)
        self.round_count = 0

    def update(
        self,
        agent_id: str,
        dimension_scores: dict[str, float],
        dimension_uncertainties: dict[str, float],
    ) -> None:
        """Update the table with new evaluation results for one player.

        Implements the EMA update from Algorithm 1, line 8:
            theta <- (1 - 1/t) * theta + (1/t) * (s, u)

        If ema_decay is set, uses that instead of 1/t.

        Args:
            agent_id: The player agent to update.
            dimension_scores: Mapping dimension_name -> score for this round.
            dimension_uncertainties: Mapping dimension_name -> uncertainty.
        """
        if self.scores is None or self.uncertainties is None:
            raise RuntimeError("Table not initialized. Call initialize() first.")

        agent_idx = self.agent_ids.index(agent_id)
        self.round_count = max(self.round_count, 1)
        t = self.round_count + 1

        if self.ema_decay is not None:
            alpha = 1.0 - self.ema_decay
        else:
            alpha = 1.0 / t

        for dim_name, score in dimension_scores.items():
            if dim_name not in self.dimensions:
                continue
            dim_idx = self.dimensions.index(dim_name)
            self.scores[agent_idx, dim_idx] = (
                (1.0 - alpha) * self.scores[agent_idx, dim_idx] + alpha * score
            )
            unc = dimension_uncertainties.get(dim_name, 0.0)
            self.uncertainties[agent_idx, dim_idx] = (
                (1.0 - alpha) * self.uncertainties[agent_idx, dim_idx] + alpha * unc
            )

    def increment_round(self) -> None:
        """Advance the round counter after all agents have been updated."""
        self.round_count += 1

    def get_agent_profile(self, agent_id: str) -> dict[str, dict[str, float]]:
        """Get the reliability profile for a single agent.

        Returns:
            Dict mapping dimension -> {"score": ..., "uncertainty": ...}.
        """
        if self.scores is None or self.uncertainties is None:
            return {}
        agent_idx = self.agent_ids.index(agent_id)
        profile: dict[str, dict[str, float]] = {}
        for dim_idx, dim_name in enumerate(self.dimensions):
            profile[dim_name] = {
                "score": float(self.scores[agent_idx, dim_idx]),
                "uncertainty": float(self.uncertainties[agent_idx, dim_idx]),
            }
        return profile

    def format_for_prompt(self) -> str:
        """Format the confidence-weight table as a readable string for LLM prompts.

        This is injected into the aggregator prompt so it can condition on
        historical reliability when fusing responses.

        Returns:
            A formatted string representation of the table.
        """
        if self.scores is None or self.round_count == 0:
            return "No historical performance data available yet."

        lines = ["Agent Historical Performance (confidence-weight table):"]
        lines.append(f"Rounds completed: {self.round_count}")
        lines.append("")

        for agent_idx, agent_id in enumerate(self.agent_ids):
            lines.append(f"  Agent {agent_id}:")
            for dim_idx, dim_name in enumerate(self.dimensions):
                score = self.scores[agent_idx, dim_idx]
                unc = self.uncertainties[agent_idx, dim_idx]
                lines.append(
                    f"    {dim_name}: score={score:+.2f}, uncertainty={unc:.2f}"
                )
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save the table to a JSON file.

        Args:
            path: Output file path.
        """
        data: dict[str, Any] = {
            "agent_ids": self.agent_ids,
            "dimensions": self.dimensions,
            "round_count": self.round_count,
            "ema_decay": self.ema_decay,
        }
        if self.scores is not None:
            data["scores"] = self.scores.tolist()
        if self.uncertainties is not None:
            data["uncertainties"] = self.uncertainties.tolist()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ConfidenceWeightTable:
        """Load a table from a JSON file.

        Args:
            path: Input file path.

        Returns:
            A restored ConfidenceWeightTable instance.
        """
        data = json.loads(Path(path).read_text())
        table = cls(
            agent_ids=data["agent_ids"],
            dimensions=data["dimensions"],
            round_count=data["round_count"],
            ema_decay=data.get("ema_decay"),
        )
        if "scores" in data:
            table.scores = np.array(data["scores"], dtype=np.float64)
        if "uncertainties" in data:
            table.uncertainties = np.array(data["uncertainties"], dtype=np.float64)
        return table
