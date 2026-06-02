"""Mechanical prompt templating.

This module deliberately contains **no** task or role instructions. System
prompts and prompt templates are user-supplied hyperparameters — see the
``system_prompt`` / ``prompt_template`` arguments on the agent classes. The
helpers here only interpolate user templates with runtime data and serialize
structured objects (candidate responses, the confidence-weight table) into
text. The defaults below are minimal placeholders, not prompts.

Template placeholders available to each role (any unused placeholder is left
empty, and arbitrary ``task.metadata`` keys are also exposed):

* player      — ``{query}``, ``{domain}``
* aggregator  — ``{query}``, ``{responses}``, ``{confidence}``
* grader      — ``{query}``, ``{response}``, ``{reference}``, ``{dimensions}``
"""

from __future__ import annotations

from typing import Any

# Minimal mechanical defaults: they only position the data, they do not
# instruct the model. Override them with your own prompts.
DEFAULT_PLAYER_TEMPLATE = "{query}"
DEFAULT_AGGREGATOR_TEMPLATE = "{query}\n\n{responses}\n\n{confidence}"
DEFAULT_GRADER_TEMPLATE = "{query}\n\n{response}"


class _SafeDict(dict):
    """dict that renders missing ``{placeholders}`` as empty strings."""

    def __missing__(self, key: str) -> str:  # noqa: D105
        return ""


def render(template: str, **fields: Any) -> str:
    """Fill ``template`` with ``fields`` using str.format semantics.

    Missing placeholders render as empty strings instead of raising, so a
    user template may reference any subset of the available fields.
    """
    return template.format_map(_SafeDict(fields))


def format_responses(responses: list[tuple[str, str]]) -> str:
    """Serialize ``(agent_id, content)`` pairs into a labelled text block."""
    blocks = [f"[{agent_id}]\n{content}" for agent_id, content in responses]
    return "\n\n".join(blocks)
