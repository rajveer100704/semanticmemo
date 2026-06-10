"""Compose SemanticMemo with guardloop: cache LLM calls under a hard budget cap.

guardloop (https://pypi.org/project/guardloop/) runs an agent under budget caps
(cost, tokens, time, tool calls). SemanticMemo sits inside the agent: on a cache hit
it returns immediately without an LLM call, so that turn consumes none of the
budget. The two are complementary -- caching lowers spend, the cap bounds the
worst case.

guardloop is an optional companion library; it is NOT a dependency of semanticmemo.
Install it to run this example:

    pip install guardloop

Run:
    uv run python examples/with_guardloop.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from semanticmemo import CacheConfig, SemanticMemo


async def main() -> None:
    try:
        from guardloop import BudgetConfig, GuardLoop, RunContext
    except ImportError:
        print("This example needs the optional 'guardloop' package.")
        print("Install it with:  pip install guardloop")
        raise SystemExit(0) from None

    with TemporaryDirectory() as temp_dir:
        cache = SemanticMemo(
            domain="support-agent",
            config=CacheConfig(
                db_path=Path(temp_dir) / "semanticmemo.db",
                estimated_llm_cost_usd="0.01",
            ),
        )
        llm_calls = 0

        async def call_llm(prompt: str) -> str:
            # In production this would call ctx.openai / ctx.anthropic -- the
            # guardloop-wrapped client that meters spend against the budget.
            # A SemanticMemo cache hit skips this call entirely.
            nonlocal llm_calls
            llm_calls += 1
            return f"answer for: {prompt}"

        runtime = GuardLoop(
            budget=BudgetConfig(
                cost_limit_usd="0.10",
                token_limit=10_000,
                time_limit_seconds=60,
                tool_call_limit=20,
            ),
        )

        async def agent(ctx: RunContext, prompt: str) -> str:
            # SemanticMemo runs first; a cache hit returns without touching the
            # budget-metered LLM client.
            result = await cache.get_or_call(prompt=prompt, llm_function=call_llm)
            return result.response

        prompts = [
            "Summarize the customer's billing issue",
            "Summarize the customer's billing issue",
            "Explain the refund policy",
        ]
        for prompt in prompts:
            outcome = await runtime.run(agent, prompt)
            print(f"  {prompt!r}\n    -> {outcome}")

        print(f"\nLLM calls actually made: {llm_calls} of {len(prompts)} turns")
        print(f"Estimated cost saved by caching: ${cache.stats().total_cost_saved_usd}")
        cache.close()


if __name__ == "__main__":
    asyncio.run(main())
