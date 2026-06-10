"""Use the bundled pretrained classifier to block false-positive cache hits.

This is the opt-in, zero-training path. ``ClassifierConfig.bundled()`` loads the
pretrained checkpoint shipped inside the package; cosine search then only
selects candidates, and the learned classifier makes the final cache decision.

Requires the optional ML dependencies:

    pip install "SemanticMemo[ml]"

Run with:
    uv run python examples/bundled_classifier.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from semanticmemo import CacheConfig, ClassifierConfig, SemanticMemo


async def main() -> None:
    with TemporaryDirectory() as temp_dir:
        cache = SemanticMemo(
            domain="customer-support",
            config=CacheConfig(
                db_path=Path(temp_dir) / "semanticmemo.db",
                estimated_llm_cost_usd="0.002",
            ),
            classifier=ClassifierConfig.bundled(),
        )

        calls = 0

        async def call_llm(prompt: str) -> str:
            nonlocal calls
            calls += 1
            return f"fresh response for: {prompt}"

        # Prime the cache with one approval prompt.
        await cache.get_or_call(
            prompt="Draft a reply approving the customer's refund request",
            llm_function=call_llm,
        )

        # A genuine paraphrase should reuse the cached response.
        paraphrase = await cache.get_or_call(
            prompt="Write a message to the customer approving their refund request",
            llm_function=call_llm,
        )

        # The opposite action is ~0.88 cosine-similar but must NOT hit the cache.
        opposite = await cache.get_or_call(
            prompt="Draft a reply denying the customer's refund request",
            llm_function=call_llm,
        )

        print(
            {
                "paraphrase_was_hit": paraphrase.was_cache_hit,
                "paraphrase_classifier_score": paraphrase.classifier_score,
            }
        )
        print(
            {
                "opposite_was_hit": opposite.was_cache_hit,
                "opposite_classifier_score": opposite.classifier_score,
            }
        )
        print({"total_llm_calls": calls})
        cache.close()


if __name__ == "__main__":
    asyncio.run(main())
