"""Verify a SmartMemo-cached agent with agenteval.

agenteval (https://pypi.org/project/agenteval-py/) runs an agent many times and
checks its pass rate. A semantic cache should make a repeated query return the
same cached answer on every run -- this file uses agenteval to assert that the
cached agent stays consistent.

agenteval-py is an optional companion library; it is NOT a dependency of
smartmemo. Install it to run this file:

    pip install agenteval-py

This file is an agenteval test, not a plain script. Run it with agenteval:

    agenteval examples/with_agenteval.py

Running it directly with ``python`` only prints these instructions.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from smartmemo import CacheConfig, SmartMemo

try:
    import agenteval
    from agenteval import Tracer

    _HAVE_AGENTEVAL = True
except ImportError:  # pragma: no cover - optional companion library
    _HAVE_AGENTEVAL = False


if _HAVE_AGENTEVAL:

    @agenteval.test(n=20, threshold=0.95)
    async def test_cached_agent_is_consistent(tracer: Tracer) -> None:
        """A primed SmartMemo cache must return the same answer every run."""

        with TemporaryDirectory() as temp_dir:
            cache = SmartMemo(
                domain="faq-agent",
                config=CacheConfig(db_path=Path(temp_dir) / "smartmemo.db"),
            )

            async def call_llm(prompt: str) -> str:
                return "Our refund window is 30 days from purchase."

            question = "What is the refund window?"
            # Prime the cache, then the agent should serve the cached answer.
            await cache.get_or_call(prompt=question, llm_function=call_llm)

            async with tracer.run(input=question) as run:
                result = await cache.get_or_call(prompt=question, llm_function=call_llm)
                run.set_output(result.response)

            (tracer.assert_that().response_contains("30 days").no_errors().check())
            cache.close()


if __name__ == "__main__":
    print(__doc__)
    if not _HAVE_AGENTEVAL:
        print("agenteval-py is not installed. Install it with:  pip install agenteval-py")
