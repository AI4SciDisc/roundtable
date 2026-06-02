"""CLI entry point for running Roundtable Policy experiments."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from roundtable.config import load_config, validate_config
from roundtable.core.policy import RoundtablePolicy
from roundtable.tasks.json_dataset import JsonDataset
from roundtable.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="roundtable",
        description="Roundtable Policy: Confidence-Weighted Consensus Aggregation",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- run command ---
    run_parser = subparsers.add_parser("run", help="Run a roundtable experiment")
    run_parser.add_argument(
        "--config", "-c", required=True, help="Path to YAML config file"
    )
    run_parser.add_argument(
        "--data", "-d", required=True, help="Path to task data (JSON)"
    )
    run_parser.add_argument(
        "--output", "-o", default="results.json", help="Output results path"
    )
    run_parser.add_argument(
        "--dimensions", nargs="*", help="Evaluation dimensions"
    )
    run_parser.add_argument(
        "--max-tasks", type=int, default=None, help="Max tasks to run"
    )
    run_parser.add_argument(
        "--log-level", default="INFO", help="Logging level"
    )
    run_parser.add_argument(
        "--save-table", default=None,
        help="Path to save the final confidence-weight table",
    )

    # --- validate command ---
    val_parser = subparsers.add_parser("validate", help="Validate a config file")
    val_parser.add_argument("config", help="Path to YAML config file")

    args = parser.parse_args(argv)

    if args.command == "validate":
        _cmd_validate(args)
    elif args.command == "run":
        _cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_validate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    errors = validate_config(config)
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("Configuration is valid.")


def _cmd_run(args: argparse.Namespace) -> None:
    setup_logging(args.log_level)

    # Load and validate config
    config = load_config(args.config)
    errors = validate_config(config)
    if errors:
        logger.error("Configuration errors: %s", errors)
        sys.exit(1)

    # Set dimensions from args or config
    if args.dimensions:
        config["dimensions"] = args.dimensions

    # Build policy
    policy = RoundtablePolicy.from_config(config)

    # Load tasks
    dims = config.get("dimensions", ["default"])
    dataset = JsonDataset(args.data, dimensions=dims)
    tasks = list(dataset)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    logger.info(
        "Starting experiment: %d players, %d graders, %d tasks",
        len(policy.players), len(policy.graders), len(tasks),
    )

    # Run
    results = policy.run(tasks)

    # Save results
    output_data = []
    for r in results:
        output_data.append({
            "task_id": r.task.task_id,
            "query": r.task.query,
            "consensus": r.consensus.content,
            "player_responses": [
                {"agent_id": pr.agent_id, "content": pr.content}
                for pr in r.player_responses
            ],
            "evaluations": [
                {
                    "agent_id": ev.agent_id,
                    "mean_score": ev.mean_score,
                    "mean_uncertainty": ev.mean_uncertainty,
                    "dimension_scores": ev.dimension_scores,
                }
                for ev in r.evaluations
            ],
            "round_index": r.round_index,
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_data, indent=2))
    logger.info("Results saved to %s", out_path)

    # Optionally save confidence table
    if args.save_table:
        policy.confidence_table.save(args.save_table)
        logger.info("Confidence table saved to %s", args.save_table)

    # Print summary
    print(f"\nExperiment complete: {len(results)} rounds")
    print(f"Results: {out_path}")
    if policy.confidence_table.round_count > 0:
        print(f"\nFinal confidence-weight table:")
        print(policy.confidence_table.format_for_prompt())


if __name__ == "__main__":
    main()
