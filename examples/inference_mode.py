"""
Example: Inference-only mode with a pre-trained confidence-weight table.

After training the table over multiple rounds (via full run), you can
skip the backward (evaluation) phase and use the table directly for
faster inference. This corresponds to Section 2.4 of the paper.

Usage:
    python examples/inference_mode.py
"""

from roundtable import RoundtablePolicy, TaskInput, ConfidenceWeightTable
from roundtable.agents.player import LLMPlayerAgent
from roundtable.agents.aggregator import LLMAggregatorAgent
from roundtable.llm.openai import OpenAIClient


def main():
    # Load a pre-trained confidence-weight table
    # (saved from a previous training run)
    table = ConfidenceWeightTable.load("confidence_table.json")
    print("Loaded confidence table:")
    print(table.format_for_prompt())

    # Set up agents (no graders needed in inference mode)
    client = OpenAIClient(model="gpt-4o")
    players = [
        LLMPlayerAgent("player_1", client),
        LLMPlayerAgent("player_2", OpenAIClient(model="gpt-4o-mini")),
    ]
    aggregator = LLMAggregatorAgent(client)

    # Create policy in inference mode — skips evaluation phase
    policy = RoundtablePolicy(
        players=players,
        graders=[],  # No graders needed
        aggregator=aggregator,
        confidence_table=table,
        inference_mode=True,
    )

    # Run on new tasks
    task = TaskInput(
        query="Explain the relationship between entropy and free energy.",
        task_id="inference_1",
    )
    result = policy.run_round(task)
    print(f"\nConsensus: {result.consensus.content}")


if __name__ == "__main__":
    main()
