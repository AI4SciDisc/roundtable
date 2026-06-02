"""
Roundtable Policy: Confidence-Weighted Consensus Aggregation for Multi-Agent Reasoning.

An inference-time multi-agent framework that improves reasoning by modeling agent
reliability, maintaining a structured long-term confidence-weight table, and
performing confidence-weighted consensus aggregation. Provider-agnostic, and all
prompts are user-supplied hyperparameters.
"""

from roundtable.agents.aggregator import LLMAggregatorAgent
from roundtable.agents.base import AggregatorAgent, GraderAgent, PlayerAgent
from roundtable.agents.grader import LLMGraderAgent
from roundtable.agents.player import LLMPlayerAgent
from roundtable.core.confidence_table import ConfidenceWeightTable
from roundtable.core.policy import RoundtablePolicy
from roundtable.core.types import (
    AgentResponse,
    ConsensusResponse,
    EvaluationResult,
    RoundResult,
    TaskInput,
)
from roundtable.llm.registry import create_client, register_provider

__version__ = "0.1.0"

__all__ = [
    "RoundtablePolicy",
    "ConfidenceWeightTable",
    "TaskInput",
    "AgentResponse",
    "EvaluationResult",
    "ConsensusResponse",
    "RoundResult",
    "PlayerAgent",
    "GraderAgent",
    "AggregatorAgent",
    "LLMPlayerAgent",
    "LLMGraderAgent",
    "LLMAggregatorAgent",
    "create_client",
    "register_provider",
]
