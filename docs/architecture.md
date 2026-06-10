# SemanticMemo: System Architecture & ML Design Rationale

This document provides a comprehensive overview of the design philosophy, core components, request lifecycle, and codebase organization of SemanticMemo. 

---

## 1. The Design Philosophy

### Why Cosine Similarity Fails in Production
Traditional semantic caches (e.g., GPTCache) rely on a static cosine similarity threshold in embedding space. In practice, this design fails because **cosine similarity does not imply functional equivalence**.

In embedding space, prompts asking for diametrically opposite actions often sit extremely close. For example, *"Should I approve this refund request?"* and *"Should I deny this refund request?"* have a cosine similarity of ~0.97 using standard sentence encoders. A naive cache treats them as equivalent and serves the cached answer of one to the other.

In high-stakes domains like finance, healthcare, or security, this is a catastrophic failure.

### The SemanticMemo Solution: Multi-Stage Verification
SemanticMemo replaces the arbitrary, static cosine threshold with a **Double-Verification Pipeline** layered with rule-based entity checks:

1. **Embedding retrieval** identifies candidates (high recall).
2. **MLP Classifier** calculates equivalence probabilities (fast, 1.5ms).
3. **Cross-Encoder** validates complex semantic relations (deep joint attention, optional/bypassed).
4. **EntityChangeDetector** blocks entity drift (regex-based, <1ms).

```
Incoming Query
    │
    ▼
[ Embedding (FAISS/Qdrant) ] ──► Retrieve candidates (O(log N))
    │
    ▼
[ MLP Classifier ] ─────────────► Fast neural equivalence check (O(K), ~1.5ms)
    ├── Score >= 0.995 ──────────┐
    └── Score < 0.995            │
            │                    ▼
            ├───► [ Cross-Encoder ] ──► Deep re-ranking (Optional, domain-tuned)
            │            │
            ▼            ▼
      [ Entity Change Detector ] ─────► Regex veto checks (<1ms, Q3 vs Q4, drug swaps)
            │
            ▼
      Decision: HIT or MISS
```

---

## 2. Core Architectural Components

### A. The Embedding & Retrieval Layer (Stage 1)
Generates dense vector representations of prompts and performs similarity searches to locate the top $K$ candidate matches.
* **Default Encoder:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional).
* **Vector Index:** Backed by an L2-normalized Inner Product FAISS index (`faiss.IndexFlatIP`) for local execution, or Qdrant for distributed deployments.

### B. The MLP Classifier (Stage 2 Gate)
A lightweight neural network that calculates the probability of equivalence between prompt vectors.
* **Input Features:** Computes concatenations $[u, v, |u - v|, u * v]$ derived from Sentence-BERT matching features.
* **Network Head:** Two hidden layers (128 → 64 → 1) with ReLU activations and dropout.
* **Latency:** < 1.5ms on CPU.

### C. The Cross-Encoder (Stage 3 Gate)
Runs a deep joint-attention transformer pass over the concatenated text of the candidate pair. 
* **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* **Dynamic Bypassing:** If the MLP score exceeds `0.995` (representing 94.7% of cache hits in benchmarks), the Cross-Encoder is skipped entirely, reducing latency by ~29%.
* **Domain Tuning:** Thresholds are dynamically configured per domain based on threshold optimization sweeps.

### D. The Entity Change Detector (Stage 4 Gate)
A battery of 11 lightweight, regex-based validation check rules that runs in `< 1ms` to veto hits containing critical entity mismatches.
* **Categories Checked:** Fiscal quarters, calendar years, months, days, ordinal references, drug names, version strings, numeric values, privilege levels, temporal states, and proper nouns.

### E. Persistence & Active Learning Loop
* **SQLite Cache Store:** Manages persistent cache records, TTL/LRU evictions, and metrics.
* **Active Learning Ledger:** Persists MLP vs Cross-Encoder disagreements and entity-vetoed misses to `active_learning_pairs` for continuous offline classifier retraining.

---

## 3. Request Lifecycle

The diagram below outlines the cache lookup workflow inside `CacheOrchestrator`:

```
agent calls semanticmemo.get_or_call(prompt, llm_function)
    │
    ▼
1. embed(prompt) → query_embedding
    │
    ▼
2. FAISS/Qdrant search → candidate_ids (K=5)
    │
    ▼
3. Run MLP Classifier on candidates
    ├── Score >= Skip Threshold (0.995) ──┐
    └── Score < Skip Threshold            │
            │                             │
            ▼                             ▼
       Run Cross-Encoder             [Bypass CE]
            │                             │
            ▼                             │
       Verify thresholds                  │
       (Passed?)                          │
            ├── Yes ──────────────────────┤
            └── No ──► Cache Miss         │
                        │                 │
                        ▼                 ▼
             Store LLM response  ◄─── Run Entity Detector
                                      (Entity Changed?)
                                           ├── Yes (Drift) ──► Cache Miss & Log AL
                                           └── No ───────────► Cache Hit
```

---

## 4. Latency Breakdown & Optimization

To run effectively in production, a semantic cache must keep retrieval and decision latency significantly lower than the LLM inference time (which is typically 500ms to 2000ms). SemanticMemo achieves this using a multi-stage approach and a latency-aware bypass:

| Stage | Operation | Latency | Purpose |
| :--- | :--- | :--- | :--- |
| **Stage 1** | Embedding Generation (`all-MiniLM-L6-v2`) | ~20–30 ms | Generate dense vector representation of query |
| **Stage 1** | Vector Index Search (FAISS) | < 0.1 ms | Retrieve top $K$ nearest candidates |
| **Stage 2** | MLP Equivalence Net Classifier | ~1.0–1.5 ms | Predict probability of semantic equivalence |
| **Stage 3** | Cross-Encoder Re-ranker | ~10–25 ms | Deep joint attention validation (run only if MLP is uncertain) |
| **Stage 4** | Entity Change Detector | < 1.0 ms | Regex-based entity drift vetoes |

### Cross-Encoder Bypass Optimization
The deep Cross-Encoder is the only computationally heavy model in the verification pipeline. By defining a bypass threshold (MLP score $\ge 0.995$), SemanticMemo skips the Cross-Encoder on **94.7% of high-confidence cache hits** in benchmarks.

* **Average Cache-Hit Latency with Bypass:** **~27ms–38ms** (depending on CPU hardware).
* **Average Cache-Hit Latency without Bypass (Full CE run):** **~39ms–68ms**.
* **Latency Saving:** **~29% to 45%** on bypassed queries, ensuring near-instantaneous hits for common inputs.

---

## 5. Codebase Layout

```
semanticmemo/
├── src/
│   └── semanticmemo/
│       ├── __init__.py                # Public API exports
│       ├── cache.py                   # Main SemanticMemo orchestrator
│       ├── orchestrator.py            # CacheOrchestrator pipeline
│       ├── domain_detector.py         # Centroid-based domain routing
│       ├── entity_change_detection.py # 11 regex-based check rules
│       ├── embedding/
│       │   ├── __init__.py
│       │   ├── service.py             # EmbeddingService (Model + FAISS)
│       │   └── models.py              # Embedding providers
│       ├── classifier/
│       │   ├── __init__.py
│       │   ├── service.py             # ClassifierService (Inference wrapper)
│       │   ├── model.py               # PairClassifier (PyTorch Module)
│       │   ├── train.py               # Classifier training pipeline
│       │   ├── data.py                # Dataset loaders and transformers
│       │   └── evaluate.py            # Evaluation helper utilities
│       ├── store/
│       │   ├── __init__.py
│       │   ├── sqlite_store.py        # SQLite CacheStore implementation
│       │   ├── eviction.py            # Eviction policy modules
│       │   └── schema.sql             # SQL Schema DDL
│       ├── feedback/
│       │   ├── __init__.py
│       │   ├── ledger.py              # FeedbackLedger database
│       │   └── retrain.py             # Retraining orchestration
│       ├── stats.py                   # Cache stats collectors
│       ├── models.py                  # Pydantic models (CacheResult, CacheDecision)
│       ├── exceptions.py              # Custom exceptions
│       └── cli.py                     # CLI entry points
├── benchmarks/
│   ├── run_benchmarks.py              # Complete suite runner
│   ├── sweep_thresholds.py            # Threshold optimization grid-search
│   ├── false_positive_eval.py         # Hard negative / opposite action tests
│   ├── prompt_mutation_benchmark.py   # Mutation robustness suite
│   └── data/                          # Benchmark datasets (.jsonl)
├── examples/                          # Reference integration files
├── tests/                             # Pytest suite
└── pyproject.toml                     # Package dependencies and settings
```

---

## 6. Public API Surface

### Minimal Usage
```python
from semanticmemo import SemanticMemo
from anthropic import AsyncAnthropic

client = AsyncAnthropic()
cache = SemanticMemo(domain="customer-support")

async def call_llm(prompt: str) -> str:
    response = await client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

# Wrapper handles cache check, lookup, decision, and LLM call fallback
result = await cache.get_or_call(
    prompt="List the active tickets for user 102",
    llm_function=call_llm,
)
print(result.response)
print(result.was_cache_hit)
print(result.cost_saved_usd)
```

### Complete Configuration
```python
from semanticmemo import SemanticMemo, CacheConfig, ClassifierConfig, EvictionPolicy

cache = SemanticMemo(
    domain="finance",
    config=CacheConfig(
        embedding_dim=384,
        candidate_k=5,
        max_entries=5000,
        eviction_policy=EvictionPolicy.LRU,
        ttl_seconds=86400 * 14, # 14 days
    ),
    classifier=ClassifierConfig(
        model_path="./models/equivalence-net-v1.pt",
        device="cpu"
    )
)
```

---

## 7. Engineering Standards

1. **Typing & Validation:** Strict type annotations across all code. Run `uv run pyright` to verify types.
2. **Silent by Default:** Library logging propagates to a root `semanticmemo` namespace using `logging.NullHandler`.
3. **No Unmanaged Latency:** CPU inference must be optimized. Models are initialized lazily and cached globally.
4. **Test-Driven Rigor:** Maintain >90% test coverage. Every new component must have corresponding unit tests in `tests/`.
