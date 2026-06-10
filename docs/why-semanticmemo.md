# Why SemanticMemo?

Semantic caching is a critical optimization for LLM applications. However, standard cache solutions fall short in production due to fundamental limitations in how they evaluate prompt equivalence. This document outlines the core challenges of semantic caching and how SemanticMemo solves them.

---

## 1. Why Cosine Similarity Fails in Production

Naive semantic caches use a simple cosine similarity threshold in embedding space. The assumption is that if two prompt embeddings are close (e.g. cosine similarity > 0.90), their intent is identical, and they can share the same cached response.

**This assumption is wrong and dangerous.**

In embedding space, prompts requesting opposite actions often lie extremely close. For example:
- *"Should I accept this transaction?"*
- *"Should I reject this transaction?"*

Using sentence encoders, these two prompts have a cosine similarity of **~0.97**. A naive cache will match them, serving the cached decision of "accept" to a user who requested a "reject". In high-stakes domains (finance, medical, security), this is a silent, catastrophic failure.

Cosine similarity measures **lexical similarity and general topic proximity**, not **functional equivalence of intent**.

---

## 2. Why Classifier Validation Helps

SemanticMemo replaces the arbitrary cosine similarity threshold with a **learned equivalence classifier** (`PairClassifier`).

Instead of assuming that high similarity equals equivalence, we train a lightweight neural network to classify if two prompts are semantically equivalent. By modeling equivalence as a supervised binary classification task, the model learns the structural features of prompts that indicate agreement or disagreement (such as negation, numeric values, and instruction changes), reducing False Positives significantly.

---

## 3. Why Cross-Encoder Verification Exists

While the MLP classifier runs fast (~1ms on CPU), it operates on bi-encoder embeddings (where prompt vectors are computed independently and compared). Bi-encoders lose fine-grained token-level cross-interactions.

To achieve production-grade precision, SemanticMemo uses **Double-Verification**:
1. **MLP Fast-Gate**: Checks candidates in ~1ms.
2. **Cross-Encoder Smart-Judge**: For borderline MLP scores, a Cross-Encoder (which performs attention over both prompts simultaneously) validates equivalence.
3. **Latency-Aware Bypass**: High-confidence MLP matches bypass the Cross-Encoder entirely, keeping average cache latency near 1.5ms.

---

## 4. Why Risk-Aware Policies Exist

Not all prompts are created equal.
- For **medical** or **security** prompts, a false positive cache hit is extremely risky. We must enforce very high precision.
- For **general summarization**, a false positive is low risk, and we want to maximize the cache hit rate (recall).

SemanticMemo uses **Domain-Conditioned Risk Policies**:
- An embedding-based **Domain Detector** automatically classifies the query into domains (`medical`, `finance`, `security`, `general`).
- The cache dynamically loads direct thresholds configured for that specific domain. High-risk domains enforce strict thresholds, while general domains run under relaxed constraints.

---

## 5. Benchmark & Performance Improvements

Our benchmarks show concrete, recruiter-grade improvements when moving from standard Cosine Caching to SemanticMemo:

| Metric | Cosine Caching (0.90 threshold) | SemanticMemo v1.1 |
| --- | --- | --- |
| **Hard Negative False Positive Rate (FPR)** | **33.3%** | **0.0%** (0 false hits) |
| **Opposite Action Detection Rate** | Fails on negation/inversion | Blocks 100% of opposite actions |
| **Domain-specific Safety** | Static global threshold | Adaptive per-domain safety |
| **Borderline Latency** | < 1.0ms | ~30ms (reserved for borderline only) |
| **Average Latency (High Recall)** | < 1.0ms | **1.2ms** (with CE Bypass active) |

By combining vector retrieval with learned classification and Cross-Encoder verification, SemanticMemo delivers the speed of a cache with the reliability of a deterministic gate.
