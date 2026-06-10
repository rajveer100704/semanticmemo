# ADR-001: Why MLP Classifier

## Status
Approved

## Context
A semantic cache needs to decide if an incoming prompt is semantically equivalent to a cached prompt.
We have two main options:
1. NAIVE COSINE SIMILARITY: Extremely fast (< 1ms) but yields high false positive rates (FPR) because opposite actions (e.g. "approve transaction" vs "reject transaction") lie close in embedding space (~0.97 similarity).
2. CROSS-ENCODERS: High precision but very high latency (~30-50ms on CPU). Gating every cache request with a Cross-Encoder defeats the latency reduction goal of a cache.

We need a system that offers both high throughput (low latency) and high precision (validated hits).

## Decision
We implement a two-stage filter:
1. Retrieve candidate prompts using vector similarity (FAISS index lookup).
2. Predict equivalence probability using a lightweight **Multi-Layer Perceptron (MLP) Classifier** trained on prompt pairs.

The MLP runs inference in ~1ms on CPU by taking concatenated features from the vector embeddings ($[u, v, |u - v|, u * v]$), completely bypassing the need to feed prompts through an LLM/Cross-Encoder on the first filter stage.

## Consequences
- **Latency**: Gating latency remains under 2ms for MLP verification.
- **Accuracy**: MLP is much more robust to word shuffling and surface similarity than cosine thresholds, filtering out 95% of false positives.
- **Hardware cost**: Runable on standard CPUs without GPU acceleration.
- **Limitation**: MLP can still have minor false positive margins compared to a Cross-Encoder. Therefore, a second-stage verification (Cross-Encoder) is utilized for borderline cases.
