"""Agent roles: players (proposers), graders (evaluators), aggregator (fusion)."""

from roundtable.agents.aggregator import LLMAggregatorAgent
from roundtable.agents.base import AggregatorAgent, GraderAgent, PlayerAgent
from roundtable.agents.grader import LLMGraderAgent, default_parse_grade
from roundtable.agents.player import LLMPlayerAgent

__all__ = [
    "PlayerAgent",
    "GraderAgent",
    "AggregatorAgent",
    "LLMPlayerAgent",
    "LLMGraderAgent",
    "LLMAggregatorAgent",
    "default_parse_grade",
]
