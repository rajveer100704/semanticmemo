"""Metrics collection and cost calculation services."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semanticmemo.models import CacheConfig


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string using a 4-character-per-token heuristic.

    This provides a zero-dependency, high-speed fallback that correlates closely
    with actual BPE tokenizations.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


class MetricsCollector:
    """Collects token and cost savings for SemanticMemo queries."""

    def __init__(self, config: CacheConfig) -> None:
        self.config = config

    def calculate_cost(self, prompt: str, response: str, model: str | None = None) -> Decimal:
        """Calculate estimated cost in USD for a given prompt and response.

        Leverages the dynamic `model_costs` mappings in CacheConfig, falling back to
        static defaults if no match is found.
        """
        prompt_tokens = estimate_tokens(prompt)
        response_tokens = estimate_tokens(response)

        # Retrieve default cost mapping
        costs = self.config.model_costs.get(
            "default",
            {"input": Decimal("0.0015"), "output": Decimal("0.002")},
        )

        if model:
            model_lower = model.lower()
            matched_key = None
            # Find the most specific model key match
            for key in self.config.model_costs:
                if key.lower() in model_lower or model_lower in key.lower():
                    matched_key = key
                    break
            if matched_key:
                costs = self.config.model_costs[matched_key]

        input_cost = (Decimal(prompt_tokens) / Decimal("1000")) * costs.get(
            "input", Decimal("0.0015")
        )
        output_cost = (Decimal(response_tokens) / Decimal("1000")) * costs.get(
            "output", Decimal("0.002")
        )
        return input_cost + output_cost
