"""
Quickstart: Run Roundtable Policy.

Prompts are hyperparameters — you supply every system prompt and template.
The strings below are illustrative; replace them with your own.

Usage:
    export OPENAI_API_KEY="your-key"
    python examples/quickstart.py
"""

from roundtable import (
    LLMAggregatorAgent,
    LLMGraderAgent,
    LLMPlayerAgent,
    RoundtablePolicy,
    TaskInput,
)
from roundtable.llm.openai import OpenAIClient


def main():
    strong = OpenAIClient(model="gpt-4o")
    fast = OpenAIClient(model="gpt-4o-mini")

    # --- Players: you choose the system prompt and how the query is framed ---
    player_system = "You are a careful reasoner. Answer the question directly."
    players = [
        LLMPlayerAgent("player_1", strong, system_prompt=player_system,
                       prompt_template="{query}", temperature=0.7),
        LLMPlayerAgent("player_2", fast, system_prompt=player_system,
                       prompt_template="{query}", temperature=0.9),
    ]

    # --- Graders: the template must elicit the format the parser expects ---
    grader_system = "You are a strict evaluator of answer quality."
    grader_template = (
        "Question:\n{query}\n\nResponse:\n{response}\n\n"
        "Reply with exactly:\n"
        "Score: <integer -100..100>\n"
        "Uncertainty: <integer 0..50>\n"
        "Rationale: <one sentence>"
    )
    graders = [
        LLMGraderAgent("grader_1", strong, system_prompt=grader_system,
                       prompt_template=grader_template),
        LLMGraderAgent("grader_2", fast, system_prompt=grader_system,
                       prompt_template=grader_template),
    ]

    # --- Aggregator: {confidence} injects the running reliability table ---
    aggregator = LLMAggregatorAgent(
        strong,
        system_prompt="Synthesize the best single answer from the candidates.",
        prompt_template=(
            "Question:\n{query}\n\nCandidates:\n{responses}\n\n"
            "Reliability so far:\n{confidence}"
        ),
    )

    policy = RoundtablePolicy(
        players=players, graders=graders, aggregator=aggregator,
        dimensions=["reasoning", "factuality"],
    )

    tasks = [
        TaskInput(query="Why do nucleation rates increase at greater undercooling?",
                  task_id="task_1"),
        TaskInput(query="What is the typical carrier mobility in silicon?",
                  task_id="task_2"),
    ]

    for result in policy.run(tasks):
        print(f"\n{'=' * 60}\nTask: {result.task.query[:70]}...")
        print(f"\nConsensus:\n{result.consensus.content[:500]}")
        for ev in result.evaluations:
            print(f"  {ev.agent_id}: score={ev.mean_score:+.1f} "
                  f"uncertainty={ev.mean_uncertainty:.1f}")

    print(f"\n{'=' * 60}\nFinal confidence-weight table:")
    print(policy.confidence_table.format_for_prompt())
    policy.confidence_table.save("confidence_table.json")


if __name__ == "__main__":
    main()
