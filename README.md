# EquivCache

EquivCache is a portfolio-grade implementation of semantic caching for LLM agent calls.
The project thesis is simple: cosine similarity is a useful candidate selector, but it is
not semantic equivalence. The production version should use a learned equivalence
classifier to decide cache hits.

This first implementation deliberately ships the baseline first:

- async `EquivCache.get_or_call(...)`
- SQLite persistence
- embedding provider protocol
- FAISS-backed vector search when `equivcache[ml]` is installed
- dependency-light in-memory search for tests and smoke demos
- measured cosine-baseline benchmark fixtures for customer-support prompts

The classifier fields are present in the public result shape, but `classifier_score` is
`None` in this milestone. That keeps the API ready for the learned judge without pretending
the classifier has been trained.

## Install

```bash
pip install equivcache
pip install "equivcache[ml]"
```

For local development:

```bash
uv sync --all-extras
uv run pytest
uv run ruff check
uv run pyright
```

## Minimal Example

```python
from equivcache import EquivCache

cache = EquivCache(domain="customer-support")

async def call_llm(prompt: str) -> str:
    return "fresh LLM response"

result = await cache.get_or_call(
    prompt="Summarize this customer's latest billing ticket",
    llm_function=call_llm,
)

print(result.response)
print(result.was_cache_hit)
```

## Baseline Benchmark

The customer-support benchmark is intentionally designed to show the baseline failure
mode: prompts about the same object can require opposite actions.

```bash
uv run python benchmarks/cosine_baseline_customer_support.py
```

The numbers from that benchmark are the only performance claims this implementation makes.
