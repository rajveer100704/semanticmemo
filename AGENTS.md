# AGENTS.md — SemanticMemo Project Guide

> **Author & Lead Engineer:** Rajveer Singh Saggu  
> **Repository:** [rajveer100704/semanticmemo](https://github.com/rajveer100704/semanticmemo)  
> This guide is the single source of truth for the SemanticMemo project. It documents the ML rationale, systems architecture, and engineering principles behind this learned semantic caching layer.

---

## The Philosophy & Vision

### Why Cosine Similarity Fails in Production
Semantic caching is essential for scaling LLM applications. However, existing industry solutions (like GPTCache) rely on a naive cosine similarity threshold in embedding space. In my work, I realized a foundational truth:

**Cosine similarity is not equivalence.**

In embedding space, prompts asking for opposite actions can sit extremely close. For example, *"Should I accept this meeting request?"* and *"Should I reject this meeting request?"* have a cosine similarity of ~0.97 using standard sentence encoders. A naive cache treats them as equivalent and serves the cached answer of one to the other.

In critical domains like finance, healthcare, or security, this is not just a cheap cache hit — it is a silent, catastrophic bug. It is a false-positive that returns wrong data to an agent, prompting incorrect actions.

### The SemanticMemo Solution
SemanticMemo replaces the arbitrary, static cosine threshold with a **learned equivalence classifier** fine-tuned on domain-specific prompt traces. By modeling semantic equivalence as a supervised matching problem, it distinguishes intent from surface similarity.

**The Vision:**
> Providing a highly reliable, domain-adapted semantic cache that teams can confidently run in production. Cache hit rates of 60-80% with classifier-validated correctness, preventing silent bugs and keeping LLM operating costs predictable.

---

## The Problem in Concrete Terms

When building real-world AI agents, developers are forced into two compromises:
- **No Caching:** Every query hits the LLM API. High latency, high API costs, and linear scaling with traffic.
- **Naive Cosine Caching:** Fast and cheap, but high false-positive rates (FPR). The agent receives cached responses to different prompts, leading to incorrect actions that are notoriously hard to debug.

SemanticMemo introduces a third option: **Double-Verification Caching**. It combines fast vector retrieval with a high-precision, low-latency classifier to ensure only functionally equivalent prompts trigger cache hits.

---

## The Mental Model: A Multi-Stage Verification Pipeline

SemanticMemo works like a layered security checkpoint:
1. **Metal Detector (High Recall, Low Latency):** Uses fast vector similarity (FAISS / Qdrant) to quickly retrieve the top $K$ candidates ($O(\log N)$).
2. **First Gate (MLP Classifier):** Evaluates candidate pairs via a fast (~1ms) classifier ($O(K)$). If the score is extremely high (e.g. $\ge 0.995$), it bypasses the next gate.
3. **Second Gate (Cross-Encoder):** Runs a deep joint-attention transformer pass on candidate pairs where the MLP is uncertain, providing high-precision validation.
4. **Final Gate (Entity Change Detector):** Runs a zero-latency (<1ms) battery of 11 regex detectors to veto any matches that contain critical entity drift (e.g. quarter, drug, year, numeric, or proper noun changes).

This structure ensures both sub-millisecond retrieval scaling and maximum correctness in production.

---

## Core Architecture

The codebase is structured around five core concepts:

### 1. The Embedding Layer (Fast Filter)
Embeds incoming queries and queries the vector index (FAISS or Qdrant) for the top $K$ candidate vectors.
- **Default Encoder:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional).
- **Index:** `faiss.IndexFlatIP` (normalized inner product for cosine similarity).

### 2. The Equivalence Classifier (Stage 1 MLP)
A lightweight neural network computing equivalence probabilities on concatenated features ($[u, v, |u - v|, u * v]$).
- **Architecture:** MLP head with 2 hidden layers (128 → 64 → 1) and ReLU activation.
- **Inference Latency:** < 1.5ms on CPU.

### 3. The Cross-Encoder (Stage 2 Re-ranker)
A transformer-based cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) that parses joint-attention representations.
- **Domain Tuning:** Thresholds are dynamically conditioned per domain (Medical, Finance, Security, Customer Support).
- **Latency Optimization:** Bypassed when the MLP score $\ge$ skip threshold (triggering on ~94.7% of hits).

### 4. The Entity Change Detector (Stage 3 Veto)
A battery of 11 lightweight, regex-based check rules validating key entities (e.g. Q3 vs Q4, Tesla vs Apple).
- **Latency:** < 1ms on CPU, no ML inference.
- **Configurability:** Fully toggleable per detector or system-wide.

### 5. The Cache Store & Active Learning Ledger (Persistence)
SQLite database managing cache storage (LRU/TTL eviction) and active learning logging.
- **Active Learning:** Logs MLP/CE disagreements and entity-vetoed misses to `active_learning_pairs` as training data for future classifier retraining.

---

## Complete Request Lifecycle

The diagram below outlines the cache lookup workflow:

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

## Public API Surface

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

## Internal Code Structure

```
semanticmemo/
├── src/
│   └── semanticmemo/
│       ├── __init__.py                # Public API exports
│       ├── cache.py                   # Main SemanticMemo orchestrator
│       ├── orchestrator.py            # CacheOrchestrator pipeline
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

## Development History & Roadmap

### Version 1.0.0 (Production Stable)
- **Double Verification Caching:** FAISS + MLP Classifier + Cross-Encoder verification.
- **Domain Auto-Detection:** Automatically routes queries using domain centroid distance.
- **Risk-Aware Policies:** Allows stricter classification thresholds on sensitive domains.
- **Performance Optimizations:** Global model caching and CPU-based warmup profiles.
- **Full Benchmarks:** Hard negative FPR reduced from 33.3% to 0% with minimal latency overhead.

### Future Enhancements
- **pgvector & Qdrant Backends:** Support for external vector databases in distributed setups.
- **Online Continual Learning:** Background training worker updating the classifier weights asynchronously.

---

## Engineering Standards

1. **Typing & Validation:** Strict type annotations across all code. Run `uv run pyright` to verify types.
2. **Silent by Default:** Library logging propagates to a root `semanticmemo` namespace using `logging.NullHandler`.
3. **No Unmanaged Latency:** CPU inference must be optimized. Models are initialized lazily and cached globally.
4. **Test-Driven Rigor:** Maintain >90% test coverage. Every new component must have corresponding unit tests in `tests/`.
