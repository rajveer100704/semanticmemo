"""Customer-support cosine-baseline benchmark.

This fixture intentionally uses a simplistic domain embedding that emphasizes objects
like "refund" and "subscription" while muting actions like "approve" and "deny". That
makes the false-positive failure mode easy to reproduce without external services.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from equivcache import CacheConfig, EquivCache
from equivcache.types import FloatVector


class CustomerSupportFixtureEmbeddingProvider:
    dim = 8

    _features = {
        "refund": 0,
        "refunds": 0,
        "billing": 1,
        "invoice": 1,
        "subscription": 2,
        "cancel": 2,
        "cancellation": 2,
        "shipping": 3,
        "delivery": 3,
        "account": 4,
        "login": 4,
        "escalate": 5,
        "escalation": 5,
        "angry": 6,
        "upset": 6,
    }

    def embed(self, text: str) -> FloatVector:
        vector = np.zeros(self.dim, dtype=np.float32)
        for raw in text.lower().replace("?", " ").replace(".", " ").split():
            token = raw.strip(",;:")
            index = self._features.get(token, 7)
            vector[index] += 1.0
        return vector


SCENARIOS = [
    {
        "name": "same_refund_summary",
        "seed_prompt": "Summarize the customer's refund request for the support agent.",
        "seed_response": "The customer is asking about a refund request.",
        "query_prompt": "Give the support agent a summary of this refund request.",
        "expected_equivalent": True,
    },
    {
        "name": "approve_vs_deny_refund",
        "seed_prompt": "Approve the customer's refund request and write the response.",
        "seed_response": "Your refund has been approved.",
        "query_prompt": "Deny the customer's refund request and write the response.",
        "expected_equivalent": False,
    },
    {
        "name": "cancel_vs_retain_subscription",
        "seed_prompt": "Cancel the customer's subscription immediately.",
        "seed_response": "Your subscription has been cancelled.",
        "query_prompt": "Convince the customer not to cancel their subscription.",
        "expected_equivalent": False,
    },
    {
        "name": "same_shipping_update",
        "seed_prompt": "Summarize the customer's shipping delay issue.",
        "seed_response": "The customer is waiting on a delayed delivery.",
        "query_prompt": "Summarize this delayed delivery support ticket.",
        "expected_equivalent": True,
    },
]


async def run_scenario(scenario: dict[str, object], db_path: Path) -> dict[str, object]:
    cache = EquivCache(
        domain="customer-support",
        config=CacheConfig(
            db_path=db_path,
            embedding_dim=CustomerSupportFixtureEmbeddingProvider.dim,
            cosine_threshold=0.80,
            estimated_llm_cost_usd="0.002",
        ),
        embedding_provider=CustomerSupportFixtureEmbeddingProvider(),
        use_faiss=False,
    )

    async def call_llm(prompt: str) -> str:
        if prompt == scenario["seed_prompt"]:
            return str(scenario["seed_response"])
        return f"fresh response for: {prompt}"

    await cache.get_or_call(prompt=str(scenario["seed_prompt"]), llm_function=call_llm)
    result = await cache.get_or_call(prompt=str(scenario["query_prompt"]), llm_function=call_llm)
    cache.close()
    false_positive = result.was_cache_hit and not bool(scenario["expected_equivalent"])
    return {
        "name": scenario["name"],
        "expected_equivalent": scenario["expected_equivalent"],
        "was_cache_hit": result.was_cache_hit,
        "similarity_score": result.similarity_score,
        "false_positive": false_positive,
    }


async def main() -> None:
    results = []
    with TemporaryDirectory() as temp_dir:
        for index, scenario in enumerate(SCENARIOS):
            results.append(await run_scenario(scenario, Path(temp_dir) / f"{index}.db"))

    hits = sum(1 for result in results if result["was_cache_hit"])
    false_positives = sum(1 for result in results if result["false_positive"])
    summary = {
        "scenarios": len(results),
        "hits": hits,
        "hit_rate": hits / len(results),
        "false_positives": false_positives,
        "results": results,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
