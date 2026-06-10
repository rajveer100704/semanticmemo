"""Compose SemanticMemo with orchflow: cache the LLM call inside each pipeline step.

orchflow (https://pypi.org/project/orchflow/) orchestrates multi-step agent
pipelines. SemanticMemo wraps the LLM call inside each step, so when a pipeline runs
again over a repeated input the step is served from cache instead of paying for
another call -- and the savings compound across every step of the pipeline.

orchflow is an optional companion library; it is NOT a dependency of semanticmemo.
Install it to run this example:

    pip install orchflow

Run:
    uv run python examples/with_orchflow.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from semanticmemo import CacheConfig, SemanticMemo


async def main() -> None:
    try:
        from orchflow import Flow, StepContext, step
    except ImportError:
        print("This example needs the optional 'orchflow' package.")
        print("Install it with:  pip install orchflow")
        raise SystemExit(0) from None

    with TemporaryDirectory() as temp_dir:
        cache = SemanticMemo(
            domain="research-pipeline",
            config=CacheConfig(
                db_path=Path(temp_dir) / "semanticmemo.db",
                estimated_llm_cost_usd="0.01",
            ),
        )
        llm_calls = 0

        async def call_llm(prompt: str) -> str:
            nonlocal llm_calls
            llm_calls += 1
            return f"output for: {prompt}"

        @step
        async def research(topic: str, context: StepContext) -> str:
            result = await cache.get_or_call(
                prompt=f"Research the key facts about {topic}",
                llm_function=call_llm,
            )
            return result.response

        @step
        async def summarize(facts: str, context: StepContext) -> str:
            result = await cache.get_or_call(
                prompt=f"Write a short summary of: {facts}",
                llm_function=call_llm,
            )
            return result.response

        flow = Flow([research, summarize])
        # The middle topic repeats, so its whole two-step pipeline is cached.
        topics = ["vector databases", "vector databases", "semantic caching"]
        for topic in topics:
            outcome = await flow.run(topic)
            print(f"  {topic!r:<26} -> {outcome.output!r}")

        print(f"\nLLM calls made: {llm_calls} (across {len(topics)} two-step runs)")
        print(f"Estimated cost saved by caching: ${cache.stats().total_cost_saved_usd}")
        cache.close()


if __name__ == "__main__":
    asyncio.run(main())
