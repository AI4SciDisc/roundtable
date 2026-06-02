"""
Example: Run Roundtable Policy from a YAML configuration file.

Usage:
    export OPENAI_API_KEY="your-key"
    python examples/from_config.py --config configs/example_openai.yaml --data data.json
"""

import argparse
import json
from pathlib import Path

from roundtable.config import load_config, validate_config
from roundtable.core.policy import RoundtablePolicy
from roundtable.tasks.json_dataset import JsonDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--data", "-d", required=True)
    parser.add_argument("--max-tasks", type=int, default=5)
    parser.add_argument("--output", "-o", default="results.json")
    args = parser.parse_args()

    # Load & validate config
    config = load_config(args.config)
    errors = validate_config(config)
    if errors:
        print("Config errors:", errors)
        return

    # Build policy from config
    policy = RoundtablePolicy.from_config(config)

    # Load tasks
    dataset = JsonDataset(args.data, dimensions=config.get("dimensions"))
    tasks = list(dataset)[: args.max_tasks]

    print(f"Running {len(tasks)} tasks with {len(policy.players)} players...")
    results = policy.run(tasks)

    # Save results
    output = []
    for r in results:
        output.append({
            "task_id": r.task.task_id,
            "consensus": r.consensus.content,
            "evaluations": [
                {"agent_id": e.agent_id, "score": e.mean_score}
                for e in r.evaluations
            ],
        })
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
