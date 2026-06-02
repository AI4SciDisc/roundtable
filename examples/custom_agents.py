"""
Example: Using custom agent implementations with Roundtable Policy.

Shows how to subclass the base agents for non-LLM or specialized use cases.
"""

from roundtable import RoundtablePolicy, TaskInput
from roundtable.agents.base import PlayerAgent, GraderAgent, AggregatorAgent
from roundtable.core.types import AgentResponse, ConsensusResponse, EvaluationResult


# --- Custom player that uses a local model or API ---
class MyCustomPlayer(PlayerAgent):
    """Example: wrap your own model or reasoning engine."""

    def __init__(self, agent_id: str, specialty: str):
        super().__init__(agent_id)
        self.specialty = specialty

    def propose(self, task):
        # Replace this with your actual model call
        content = (
            f"[{self.agent_id} | specialty={self.specialty}] "
            f"My answer to '{task.query[:50]}...' is: ..."
        )
        return AgentResponse(
            agent_id=self.agent_id,
            content=content,
            metadata={"specialty": self.specialty},
        )


# --- Custom grader with domain-specific rubrics ---
class MyCustomGrader(GraderAgent):
    """Example: rule-based or model-based grading."""

    def evaluate(self, task, response, reference=None):
        # Replace with your evaluation logic
        score = 50.0  # placeholder
        uncertainty = 10.0
        return EvaluationResult(
            grader_id=self.grader_id,
            agent_id=response.agent_id,
            score=score,
            uncertainty=uncertainty,
            rationale="Custom evaluation logic",
        )


# --- Custom aggregator ---
class MyCustomAggregator(AggregatorAgent):
    """Example: weighted average or retrieval-augmented aggregation."""

    def aggregate(self, task, responses, confidence_context):
        # Simple concatenation example — replace with your fusion strategy
        combined = "\n\n".join(
            f"[{r.agent_id}]: {r.content}" for r in responses
        )
        content = f"Aggregated from {len(responses)} agents:\n{combined}"
        return ConsensusResponse(
            content=content,
            source_agent_ids=[r.agent_id for r in responses],
        )


def main():
    players = [
        MyCustomPlayer("physics_expert", specialty="physics"),
        MyCustomPlayer("bio_expert", specialty="biology"),
        MyCustomPlayer("math_expert", specialty="math"),
    ]
    graders = [MyCustomGrader("grader_1"), MyCustomGrader("grader_2")]
    aggregator = MyCustomAggregator()

    policy = RoundtablePolicy(
        players=players,
        graders=graders,
        aggregator=aggregator,
        dimensions=["physics", "biology", "math"],
    )

    tasks = [
        TaskInput(query="Explain entropy in thermodynamics", task_id="t1"),
        TaskInput(query="How does CRISPR-Cas9 work?", task_id="t2"),
    ]

    results = policy.run(tasks)

    for r in results:
        print(f"\nTask: {r.task.query}")
        print(f"Consensus: {r.consensus.content[:200]}")
        for ev in r.evaluations:
            print(f"  {ev.agent_id}: {ev.mean_score:+.1f}")

    print("\n" + policy.confidence_table.format_for_prompt())


if __name__ == "__main__":
    main()
