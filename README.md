# Roundtable Policy

A general, provider-agnostic framework for **confidence-weighted consensus aggregation** in multi-agent reasoning ([paper](https://arxiv.org/abs/2509.16839)). Each round: players answer independently → an aggregator fuses them, weighted by a running reliability table θ → graders score the answers and update θ.

All prompts are **your** hyperparameters — the framework ships none.

## Install

```bash
pip install -e ".[openai]"        # or [anthropic], [google], [all]
```

## Use

```python
from roundtable import (
    RoundtablePolicy, TaskInput,
    LLMPlayerAgent, LLMGraderAgent, LLMAggregatorAgent,
)
from roundtable.llm.openai import OpenAIClient

client = OpenAIClient(model="gpt-4o")

players = [
    LLMPlayerAgent("p1", client, system_prompt="...", prompt_template="{query}"),
    LLMPlayerAgent("p2", client, system_prompt="...", prompt_template="{query}"),
]
graders = [
    LLMGraderAgent("g1", client, system_prompt="...", prompt_template=(
        "{query}\n\n{response}\n\n"
        "Score: <-100..100>\nUncertainty: <0..50>\nRationale: <one line>"
    )),
]
aggregator = LLMAggregatorAgent(client, system_prompt="...",
    prompt_template="{query}\n\n{responses}\n\n{confidence}")

policy = RoundtablePolicy(players, graders, aggregator,
                          dimensions=["reasoning", "factuality"])

results = policy.run([TaskInput(query="...")])
print(results[0].consensus.content)
```

**Template placeholders** (filled at runtime; anything you omit is left blank):

| Role | Placeholders |
|---|---|
| player | `{query}` `{domain}` |
| grader | `{query}` `{response}` `{reference}` `{dimensions}` |
| aggregator | `{query}` `{responses}` `{confidence}` |

The default grader parser reads `Score:` / `Uncertainty:` / `Rationale:` — pass a custom `parser=` for other formats. Use `policy.arun(...)` for concurrency.

## Providers

`openai`, `openai_compatible` (vLLM/Together/Groq/DeepSeek/Ollama via `base_url`), `anthropic`, `google` (alias `gemini`). Add your own:

```python
from roundtable import register_provider
register_provider("myapi", "my_pkg.client", "MyClient")  # subclass of LLMClient
```

## Config-driven

```bash
roundtable validate configs/example_openai.yaml
roundtable run -c configs/example_openai.yaml -d data.json -o results.json
```

See [`configs/`](configs/) and [`examples/`](examples/). Bring your own agents by subclassing `PlayerAgent` / `GraderAgent` / `AggregatorAgent`.

## Citation

```bibtex
@article{yao2026roundtable,
  title={Roundtable Policy: Confidence-Weighted-Consensus Aggregation Improves Multi-Agent-System Reasoning},
  author={Yao, Yu and Dong, Jiayi and Yang, Yang and Li, Ju and Du, Yilun},
  year={2026}
}
```
