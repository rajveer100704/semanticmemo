# Quickstart

EquivCache is an async-first semantic cache for LLM agent calls. The first implementation
ships the baseline cache layer: embeddings, top-k retrieval, SQLite persistence, and a
cosine threshold decision. The learned equivalence classifier is the next milestone and
will replace the threshold as the actual cache-hit decision.

Install the core package:

```bash
pip install equivcache
```

Install real embedding and vector-search dependencies:

```bash
pip install "equivcache[ml]"
```

Minimal use:

```python
from equivcache import EquivCache

cache = EquivCache(domain="customer-support")

async def call_llm(prompt: str) -> str:
    return "fresh response from your provider"

result = await cache.get_or_call(prompt="Summarize this ticket", llm_function=call_llm)
print(result.response)
print(result.was_cache_hit)
```
