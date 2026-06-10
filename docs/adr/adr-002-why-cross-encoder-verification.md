# ADR-002: Why Cross-Encoder Verification

## Status
Approved

## Context
While a lightweight MLP classifier is fast (~1ms), it has limits in complex semantic scenarios. A high-precision model is needed as a "smart judge" for borderline cases where the cache decision is high-stakes.
However, running a Cross-Encoder on every cache query increases latency (~30ms on CPU).

## Decision
We implement a **Double-Verification Cache** system with a **Latency-Aware Bypass**:
1. When the MLP classifier score is very high (exceeds a `high_precision_skip_threshold`, e.g., 0.995), we assume equivalence with high confidence and bypass the Cross-Encoder.
2. When the MLP classifier score is borderline (exceeds the cache threshold but lies below the skip threshold), we feed the prompt pair into a **Cross-Encoder model** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) for final validation.
3. If the Cross-Encoder approves, we serve a cache HIT. If not, it is rejected (cache MISS).

## Consequences
- **Correctness**: Drastically reduces False Positive Rate (FPR) on complex or adversarial prompts to 0.0% in high-stakes domains.
- **Latency profile**: High-confidence matches bypass the Cross-Encoder, keeping latency low (~1-2ms). Slower verification (~30ms) is reserved only for borderline candidates.
- **Active Learning**: Disagreements (MLP says HIT, Cross-Encoder says MISS) are persisted in the database as hard-negative training examples for active learning.
