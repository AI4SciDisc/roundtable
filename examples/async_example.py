"""
Example: Run Roundtable Policy with async concurrency.

Async mode runs all player proposals concurrently and all grader
evaluations concurrently, significantly reducing wall-clock time
when using API-based LLMs.

Usage:
    export OPENAI_API_KEY="your-key"
    python examples/async_example.py
"""

import asyncio

from roundtable import RoundtablePolicy, TaskInput
from roundtable.agents.player import LLMPlayerAgent
from roundtable.agents.grader import LLMGraderAgent
from roundtable.agents.aggregator import LLMAggregatorAgent
from roundtable.llm.openai import OpenAIClient


async def main():
    client = OpenAIClient(model="gpt-4o-mini")

    players = [
        LLMPlayerAgent(f"player_{i}", client, temperature=0.5 + i * 0.2)
        for i in range(4)
    ]
    graders = [
        LLMGraderAgent(f"grader_{i}", client, temperature=0.3)
        for i in range(2)
    ]
    aggregator = LLMAggregatorAgent(client)

    policy = RoundtablePolicy(
        players=players,
        graders=graders,
        aggregator=aggregator,
        dimensions=["default"],
    )

    tasks = [
        TaskInput(query="What causes superconductivity?", task_id="t1"),
        TaskInput(query="Explain the Mpemba effect.", task_id="t2"),
    ]

    # Async run — players and graders operate concurrently within each round
    results = await policy.arun(tasks)

    for r in results:
        print(f"\nTask: {r.task.query}")
        print(f"Consensus: {r.consensus.content[:300]}")


if __name__ == "__main__":
    asyncio.run(main())
