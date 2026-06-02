"""Configuration management for Roundtable Policy experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load experiment configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    return yaml.safe_load(Path(path).read_text())


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate a configuration dict and return a list of errors (empty = valid).

    Args:
        config: The configuration to validate.

    Returns:
        List of error message strings. Empty if valid.
    """
    errors: list[str] = []

    if "players" not in config:
        errors.append("Missing 'players' section")
    elif not isinstance(config["players"], list) or len(config["players"]) == 0:
        errors.append("'players' must be a non-empty list")
    else:
        for i, p in enumerate(config["players"]):
            if "agent_id" not in p:
                errors.append(f"Player {i}: missing 'agent_id'")
            if "provider" not in p:
                errors.append(f"Player {i}: missing 'provider'")
            if "model" not in p:
                errors.append(f"Player {i}: missing 'model'")

    if "graders" not in config:
        errors.append("Missing 'graders' section")
    elif not isinstance(config["graders"], list) or len(config["graders"]) == 0:
        errors.append("'graders' must be a non-empty list")
    else:
        for i, g in enumerate(config["graders"]):
            if "grader_id" not in g:
                errors.append(f"Grader {i}: missing 'grader_id'")
            if "provider" not in g:
                errors.append(f"Grader {i}: missing 'provider'")
            if "model" not in g:
                errors.append(f"Grader {i}: missing 'model'")

    if "aggregator" not in config:
        errors.append("Missing 'aggregator' section")
    else:
        agg = config["aggregator"]
        if "provider" not in agg:
            errors.append("Aggregator: missing 'provider'")
        if "model" not in agg:
            errors.append("Aggregator: missing 'model'")

    return errors
