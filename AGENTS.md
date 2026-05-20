# AGENTS.md — SmartMemo Project Guide

> This document is the single source of truth for the SmartMemo project.
> Read this fully before writing any code. Every architectural and ML decision is explained here.

---

## What is SmartMemo?

SmartMemo is a **semantic caching layer for LLM agent calls** that uses a *learned* equivalence classifier instead of a fixed cosine-similarity threshold.

It sits between an agent and its LLM provider. When the agent makes an LLM call, SmartMemo checks if a "semantically equivalent" call has been made before. If yes, it returns the cached response and saves the cost of an actual LLM call. If no, it lets the call through and stores the result for next time.

The crucial difference from existing semantic caches (GPTCache, etc.): SmartMemo does not trust cosine similarity to decide equivalence. It uses a small ML classifier — finetuned on real agent traces from the specific domain — to make that decision. This eliminates the false-positive failures that have kept production teams from adopting semantic caching.

**One-line pitch:**
> "Semantic caching for LLM agents that actually works in production — a learned equivalence classifier replaces the naive cosine threshold that causes false-positive failures in GPTCache and similar tools."

---

## The Foundational Truth

Before any code, internalize this:

**Cosine similarity is not equivalence.**

Two prompts can be 97% similar in embedding space and ask for opposite things. "Should I accept this meeting?" and "Should I reject this meeting?" have near-identical embeddings but require opposite responses. Naive semantic caching cannot tell them apart.

This is not a tunable-threshold problem. There is no threshold that works across domains, tasks, or prompt styles. Whatever threshold you pick, in some domain the false positive rate is unacceptable, and in another the cache hit rate is too low to matter.

The solution is not a better threshold. The solution is **learning what equivalence means**, per domain, from data.

That is what SmartMemo is.

---

## The Problem in Concrete Terms

Production agent teams currently choose between two bad options:

**Option A: No semantic caching.**
Every LLM call hits the API. Cost scales linearly with traffic. A high-traffic agent burns thousands of dollars a day on near-duplicate calls.

**Option B: Naive semantic caching (GPTCache et al).**
Cache hits look great on paper (40-60% hit rate). In production, false positives cause subtle agent bugs. Medical agents give wrong answers. Customer support agents return responses for the wrong customer's question. Worse, these failures are *hard to debug* — the cache appears to work, the agent doesn't know it got a wrong response.

Teams adopt Option B in development, hit a false-positive incident, rip it out, go back to Option A. The cycle repeats across the industry.

**SmartMemo exists because Option C is missing.** Cache hit rates of 60-80% with classifier-validated correctness, measured against a real evaluation suite. That's the gap.

---

## The Mental Model: A Two-Stage Filter

Think of SmartMemo as airport security. You don't search every passenger thoroughly — that would take forever. You also don't let everyone through unscreened — that's unsafe. Instead, you have two stages:

1. **Metal detector (fast, dumb, high-recall):** flag anyone metallic. Quick to run, catches most candidates, but lots of false alarms.
2. **Manual screening (slow, smart, high-precision):** examine each flagged passenger carefully. Decide if they're actually a threat.

SmartMemo works the same way:

1. **Embedding similarity (fast filter):** find the top K most similar cached prompts. Quick, O(log N) with a vector index, but doesn't actually decide cache hits.
2. **Learned classifier (smart judge):** for each candidate, run a small trained model that outputs equivalence probability. This is the actual cache-hit decision.

The first stage gives you tractability. The second stage gives you correctness. Neither alone is enough.

---

## The Four Core Concepts

Everything in SmartMemo fits into one of these. Master them — these are what you'll defend in interviews.

---

### Concept 1: The Embedding Layer (Fast Filter)

**What it is:** A vector store of embeddings for every cached prompt. New queries get embedded; similarity search finds the top K candidates.

**Why it exists:** Without it, you'd run the classifier against every cached prompt for every new call. That's O(N) classifier calls per query — unusable at scale. Embedding filtering reduces this to O(K) where K is small (5-10).

**Implementation choices:**
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dims, fast, free) for v1. Configurable later.
- **Vector store:** FAISS for local/single-node v1. pgvector or Qdrant for production.
- **Similarity metric:** cosine similarity.
- **K (number of candidates):** default 5. Configurable.

**Critical principle:** **embedding similarity is a candidate selector, not a decision maker.** The threshold here should be *permissive* (e.g., top-K with no minimum score) — false negatives at this stage are unrecoverable, false positives just give the classifier more candidates to reject. Lean toward recall.

---

### Concept 2: The Equivalence Classifier (Smart Judge)

**What it is:** A small neural network that takes two prompts (or their embeddings) and outputs a single number: the probability that they would produce essentially the same useful response.

**Why it exists:** This is the ML core of SmartMemo. It replaces the naive cosine threshold that breaks production semantic caches. The classifier learns what equivalence actually means for a domain, from data.

**Architecture (v1):**

```
prompt_A ──┐
           ├──> [shared encoder] ──> embedding_A ──┐
prompt_B ──┘                                       │
                                                   ├──> [concat | diff | hadamard] ──> [MLP] ──> sigmoid(p)
prompt_A ──┐                                       │
           ├──> [shared encoder] ──> embedding_B ──┘
prompt_B ──┘
```

- **Encoder:** `sentence-transformers/all-MiniLM-L6-v2` (shared weights, frozen initially, optionally finetuned later)
- **Pair representation:** concatenation of `[emb_A, emb_B, |emb_A - emb_B|, emb_A * emb_B]` (this is the standard "matching" representation from Sentence-BERT)
- **MLP head:** 2 hidden layers (128 → 64 → 1), ReLU activations, dropout 0.1
- **Output:** sigmoid → probability in [0, 1]
- **Loss:** binary cross-entropy with class weighting if dataset is imbalanced

**Training data sources (this is the hard part — see "ML Strategy" section below for details):**
- Self-supervised paraphrasing for positive examples
- Random pairing within and across domains for negative examples
- LLM-as-judge labeling for higher-quality examples
- Replay-based labeling (call LLM twice, compare responses) for ground truth

**Inference target:** classifier must run in under 20ms per pair on CPU. Otherwise it eats the latency savings of caching. This constrains model size — keep it small.

**Decision threshold:** configurable per deployment. Default 0.85. Higher = stricter (fewer cache hits, higher correctness). Lower = looser (more hits, more false positives). The whole point is that users can tune this per their domain's tolerance.

---

### Concept 3: The Cache Store (Memory)

**What it is:** Storage for cached prompts, their responses, and metadata.

**Why it exists:** Obvious — you need somewhere to store the cache. The non-obvious parts are eviction, consistency, and metadata.

**Schema (v1, SQLite + FAISS):**

```
table: cache_entries
  - id: UUID
  - prompt: TEXT
  - prompt_embedding: BLOB (float32[384])
  - response: TEXT
  - model: TEXT
  - created_at: TIMESTAMP
  - last_hit_at: TIMESTAMP
  - hit_count: INT
  - feedback_negative_count: INT
  - feedback_positive_count: INT
  - metadata_json: TEXT

faiss_index: cache_entries.prompt_embedding ↔ cache_entries.id
```

**Eviction policy (default):** LRU with a configurable size cap. When cache exceeds max entries, evict entries with oldest `last_hit_at`. Configurable: TTL-based eviction, size-based, hybrid.

**Critical principle:** **the cache must support negative feedback.** When a cached response was wrong, that entry's `feedback_negative_count` increments. Entries with high negative feedback get evicted preferentially, and become negative training examples for future classifier retraining.

---

### Concept 4: The Feedback Loop (Learning System)

**What it is:** A mechanism for the system to learn from its mistakes over time. Bad cache hits become training data for the next classifier retrain.

**Why it exists:** Without this, SmartMemo is a static classifier that gets *worse* over time as the domain drifts. With this, it gets *better* — adapting to the specific patterns of the agent it's deployed for.

**Feedback sources:**
- **Explicit:** application calls `cache.report_bad_hit(query_id, reason=...)` after detecting a problem (e.g., agent got a thumbs-down from user)
- **Implicit:** if the agent makes the same call again immediately after a cache hit (within 30 seconds), it likely regenerated because the cache hit was unhelpful. Auto-flag as negative.
- **Verifier-based:** if the deployment uses agenteval or a verifier in the agent loop, verifier failures on cached responses are auto-flagged.

**Retraining pipeline:**
1. Accumulate feedback events into a labeled dataset (positive pairs from successful hits, negative pairs from bad hits)
2. On a schedule (nightly, weekly, or threshold-triggered), retrain the classifier on the updated dataset
3. Evaluate the new classifier on a held-out test set. If it's better, deploy. If worse, keep the old one.
4. Version classifiers — keep the last N for rollback.

**v1 scope:** ship the feedback collection and storage. Make retraining manual via CLI: `smartmemo retrain`. Automated retraining is v0.2.

---

## How the Four Concepts Compose

Here is a complete cache lookup, end to end:

```
agent calls smartmemo.get_or_call(prompt, llm_function)
    │
    ▼
1. embed(prompt) → query_embedding [Concept 1]
    │
    ▼
2. FAISS.search(query_embedding, k=5) → candidate_ids [Concept 1]
    │
    ▼
3. for each candidate:
     run classifier(query_embedding, candidate.embedding) → probability [Concept 2]
   pick best candidate above threshold
    │
    ▼
4. if cache hit:
     return candidate.response
     candidate.hit_count += 1
     candidate.last_hit_at = now
   else:
     response = await llm_function(prompt) [actual LLM call]
     store(prompt, embedding, response, model) → cache [Concept 3]
     return response
    │
    ▼
5. (later) application calls smartmemo.report_bad_hit(query_id) → feedback ledger [Concept 4]
```

The agent doesn't need to know any of this exists. SmartMemo is a wrapper around the LLM call — same input, same output, just sometimes much cheaper.

---

## Public API Surface

```python
from smartmemo import (
    SmartMemo,
    CacheConfig,
    ClassifierConfig,
    CacheStore,
    EvictionPolicy,
    CacheResult,
    CacheStats,
)
```

Minimal usage:
```python
from smartmemo import SmartMemo
from anthropic import AsyncAnthropic

client = AsyncAnthropic()
cache = SmartMemo(domain="customer-support")

async def call_llm(prompt: str) -> str:
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

# Drop-in wrapper. Returns either cached response or fresh LLM response.
result = await cache.get_or_call(
    prompt="Summarize this user's recent tickets",
    llm_function=call_llm,
)
print(result.response)
print(result.was_cache_hit)        # bool
print(result.cost_saved_usd)        # Decimal, 0 if miss
```

Full usage:
```python
cache = SmartMemo(
    domain="customer-support",
    config=CacheConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        candidate_k=5,
        classifier_threshold=0.85,
        max_entries=10_000,
        eviction_policy=EvictionPolicy.LRU,
        ttl_seconds=86_400 * 7,        # 7 days
    ),
    classifier=ClassifierConfig(
        model_path="./models/classifier-v3.pt",  # optional pretrained
        device="cpu",
    ),
)

result = await cache.get_or_call(prompt="...", llm_function=call_llm)

# Feedback
if user_gave_thumbs_down:
    await cache.report_bad_hit(result.query_id, reason="user rejected")

# Stats
stats = cache.stats()
print(stats.hit_rate)
print(stats.total_cost_saved_usd)
print(stats.false_positive_estimate)
```

---

## Architecture Internals

```
smartmemo.get_or_call(prompt, llm_function)
    │
    ▼
CacheOrchestrator
    │
    ├── EmbeddingService       # embed prompts, manage FAISS index
    ├── ClassifierService      # run pair classifier, batched
    ├── CacheStore             # SQLite + FAISS persistence
    ├── FeedbackLedger         # accumulate labels for retraining
    └── StatsCollector         # hit rate, cost saved, FP estimate
```

### Key internal classes

**`EmbeddingService`** — owns the embedding model and the FAISS index.
```
EmbeddingService(model_name, dim):
  - model: SentenceTransformer
  - index: faiss.IndexFlatIP
  - id_map: dict[int, UUID]
  - embed(text) -> np.ndarray
  - search(query_emb, k) -> list[(uuid, score)]
  - add(uuid, emb)
  - remove(uuid)
```

**`ClassifierService`** — wraps the trained equivalence classifier.
```
ClassifierService(model_path, device):
  - model: PairClassifier (torch.nn.Module)
  - predict(emb_a, emb_b) -> float
  - predict_batch(pairs) -> list[float]
  - reload(new_model_path)
```

**`PairClassifier`** — the actual neural network.
```
class PairClassifier(nn.Module):
    def __init__(self, embed_dim=384):
        self.pair_proj = nn.Linear(embed_dim * 4, 128)
        self.mlp = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    def forward(self, emb_a, emb_b):
        # build [a, b, |a-b|, a*b] -> project -> mlp -> sigmoid
        ...
```

**`CacheStore`** — persistence layer.
```
CacheStore(db_path):
  - get(uuid) -> CacheEntry
  - add(prompt, embedding, response, model, metadata) -> uuid
  - update_hit(uuid)
  - increment_bad_feedback(uuid)
  - evict(policy, max_entries)
  - all_entries() -> Iterator[CacheEntry]
```

**`FeedbackLedger`** — append-only log of labeled events for retraining.
```
FeedbackLedger(db_path):
  - record_hit(query_id, cached_id, classifier_score)
  - record_bad_hit(query_id, cached_id, reason)
  - record_miss(query_id, prompt_embedding)
  - export_training_dataset() -> Dataset
```

**`CacheResult`** — what `get_or_call()` returns.
```
CacheResult:
  - response: str
  - was_cache_hit: bool
  - query_id: UUID
  - cache_entry_id: UUID | None
  - classifier_score: float | None
  - cost_saved_usd: Decimal
  - latency_ms: float
```

---

## ML Strategy (The Hard Part)

This is what separates SmartMemo from a toy. Read carefully.

### The training data problem

To train the equivalence classifier you need pairs of prompts labeled as equivalent (1) or not (0). Where do they come from?

**Source 1: Self-supervised augmentation (for v1 bootstrap)**
- Take a corpus of agent prompts
- Generate paraphrases using a strong LLM (Claude or GPT-4): "rewrite this prompt 5 different ways preserving intent" → each pair labeled 1 (equivalent)
- Random pairs of prompts from the corpus → labeled 0 (not equivalent)
- This gives you tens of thousands of pairs cheaply. Quality is okay but not great.

**Source 2: LLM-as-judge labeling (for higher quality)**
- For ambiguous pairs (cosine similarity 0.7-0.95), ask a strong LLM: "would these two prompts produce essentially the same useful response?"
- LLM outputs label + reasoning
- This is more expensive but produces high-quality labels for the hard cases that matter most

**Source 3: Replay-based ground truth (the gold standard)**
- For a set of prompt pairs, actually call the LLM twice
- Use a strong LLM to compare the two responses for semantic equivalence
- If responses are equivalent → prompts are functionally equivalent (label 1)
- Most expensive but produces the most defensible labels

**Source 4: Feedback loop (for continual improvement)**
- Once SmartMemo is deployed, every reported bad hit becomes a labeled negative example
- Every successful hit (no bad feedback within a time window) becomes a labeled positive
- This is how the classifier improves *for the specific domain over time*

For v1, mix sources 1 and 2: bootstrap with self-supervised, refine the hard cases with LLM-as-judge. Reserve source 3 for the test set (smaller but high-quality). Source 4 comes online once people start using it.

### Evaluation methodology

**Offline metrics (on held-out test set):**
- **Precision:** of pairs the classifier predicts as equivalent, what fraction actually are? **This is the most important metric — it directly correlates with false positives in production.**
- **Recall:** of pairs that actually are equivalent, what fraction does the classifier catch? This correlates with cache hit rate.
- **F1, AUC-ROC:** standard summary metrics.
- **Calibration:** does a predicted probability of 0.9 actually mean 90% of those pairs are equivalent? Use reliability diagrams. Miscalibration here would make the threshold meaningless.

**Online metrics (on production traffic):**
- **Cache hit rate:** fraction of queries that result in a cache hit
- **Effective cost saved:** $ of LLM calls avoided
- **False positive rate (estimated):** fraction of cache hits that get bad feedback within N minutes
- **Latency impact:** median and p99 latency with vs without cache

**Critical evaluation principle:** **precision matters more than recall for SmartMemo.** A false negative just makes one extra LLM call (small cost). A false positive returns wrong data to the agent (potentially big cost — user trust, incorrect actions). Tune threshold to favor precision unless the domain explicitly tolerates errors.

### Cold start

On day one of a new deployment, the classifier has no domain-specific training. Options:
- **Default to a generic pretrained classifier** trained on a mixed-domain dataset (ship one of these with the library)
- **High threshold during cold start** (e.g., 0.95) → low hit rate but high correctness
- **Gradually lower threshold** as feedback data accumulates and classifier is retrained

For v1, ship a generic pretrained classifier. Document the cold-start tradeoff. Provide a CLI command to retrain on local data.

---

## Project Structure

```
smartmemo/
├── src/
│   └── smartmemo/
│       ├── __init__.py                # public API
│       ├── cache.py                   # SmartMemo main class
│       ├── orchestrator.py            # CacheOrchestrator
│       ├── embedding/
│       │   ├── __init__.py
│       │   ├── service.py             # EmbeddingService
│       │   └── models.py              # supported embedding models
│       ├── classifier/
│       │   ├── __init__.py
│       │   ├── service.py             # ClassifierService
│       │   ├── model.py               # PairClassifier nn.Module
│       │   ├── train.py               # training loop
│       │   ├── data.py                # dataset construction
│       │   └── evaluate.py            # eval metrics
│       ├── store/
│       │   ├── __init__.py
│       │   ├── sqlite_store.py        # CacheStore implementation
│       │   ├── eviction.py            # eviction policies
│       │   └── schema.sql
│       ├── feedback/
│       │   ├── __init__.py
│       │   ├── ledger.py              # FeedbackLedger
│       │   └── retrain.py             # retraining trigger
│       ├── stats.py                   # StatsCollector
│       ├── models.py                  # Pydantic data models (CacheResult, etc)
│       ├── exceptions.py
│       └── cli.py                     # `smartmemo retrain`, `smartmemo stats`
├── examples/
│   ├── basic_usage.py                 # wrap an LLM call, see hit rate
│   ├── with_agentruntime.py           # SmartMemo + AgentRuntime cost demo
│   ├── with_orchflow.py               # SmartMemo in a multi-step pipeline
│   ├── train_custom_classifier.py     # finetune on your domain
│   └── feedback_loop.py               # reporting bad hits, retraining
├── benchmarks/
│   ├── compare_to_gptcache.py         # head-to-head vs GPTCache
│   ├── false_positive_eval.py         # the killer demo: medical pairs
│   └── cost_savings.py                # measured savings on a real workload
├── tests/
│   ├── test_embedding.py
│   ├── test_classifier.py
│   ├── test_store.py
│   ├── test_orchestrator.py
│   ├── test_feedback.py
│   └── test_integration.py
├── models/
│   └── classifier-v2.pt               # shipped pretrained classifier
├── data/
│   └── README.md                      # how to get training data
├── docs/
│   ├── quickstart.md
│   ├── concepts.md
│   ├── ml/
│   │   ├── how-the-classifier-works.md
│   │   ├── training-your-own.md
│   │   └── evaluation.md
│   ├── api-reference.md
│   └── integration/
│       ├── with-agentruntime.md
│       ├── with-orchflow.md
│       └── production-deployment.md
├── pyproject.toml
├── README.md
└── CLAUDE.md                          # this file
```

---

## Implementation Order (Build Phases)

This is the longest of your four projects. Plan **6-8 weeks** for a solid v1. The ML training phases especially need patience.

> **Build status — v0.2.0.** Phases 1–4 are complete; Phase 6 is largely done (the
> package and docs ship; the blog/social content does not). The bundled classifier is
> **`classifier-v2`**, retrained across nine domains. Phase 5's sibling-library
> integrations were evaluated and **not adopted**: `guardloop`, `agenteval-py`, and
> `orchflow` are pre-1.0, single-maintainer, and orthogonal to caching, so they ship
> only as optional, lazy-imported `examples/with_*.py` with no dependency added. The
> planned GPTCache head-to-head benchmark was dropped — GPTCache's runtime
> dependency-install design did not run cleanly. The phase plan below is kept as the
> original design record; `CHANGELOG.md` is the authoritative record of what shipped.

### Phase 1 — Embedding Pipeline & Naive Cache (Week 1-2)
Get a working semantic cache without the classifier — i.e., GPTCache-equivalent. This is the baseline you'll later beat.

1. `embedding/service.py` — load `all-MiniLM-L6-v2`, FAISS index, add/search/remove
2. `store/sqlite_store.py` — schema, CRUD, eviction
3. `models.py` — `CacheResult`, `CacheConfig`, `CacheEntry`
4. `orchestrator.py` — `get_or_call()` using cosine threshold (no classifier yet)
5. `cache.py` — `SmartMemo` public class
6. `examples/basic_usage.py` — wrap a Claude/OpenAI call, run 100 similar queries, see hit rate
7. Tests for embedding, store, basic cache flow

**Done when:** you have a working semantic cache with cosine threshold. Measure baseline hit rate and false positive rate on a test set. **You will beat this baseline with the classifier.**

### Phase 2 — Training Data & Classifier (Week 3-4)
Build the ML core.

1. `classifier/data.py` — dataset construction
   - Self-supervised paraphrase generation (use Claude/GPT to paraphrase a corpus)
   - Negative example sampling
   - Test set construction (manually labeled or LLM-as-judge labeled)
2. `classifier/model.py` — `PairClassifier` nn.Module
3. `classifier/train.py` — training loop, checkpoints, logging
4. `classifier/evaluate.py` — precision/recall/F1/AUC/calibration metrics
5. Train a v1 classifier on ~10k pairs, evaluate it. Iterate until you beat cosine baseline on precision.
6. Save the trained model. Ship it in `models/classifier-v2.pt`.
7. Tests for training loop (smoke test), eval metrics

**Done when:** trained classifier exists; precision on test set is at least 10 points higher than cosine threshold at same recall. Document the gap.

### Phase 3 — Classifier Integration (Week 5)
Wire the classifier into the cache flow.

1. `classifier/service.py` — `ClassifierService` with model loading, batched inference
2. Update `orchestrator.py` — after FAISS candidate search, run classifier, decide hit/miss
3. `examples/basic_usage.py` v2 — show same workload with and without classifier
4. Benchmark: cache hit rate and false positive rate vs. baseline
5. Tests for classifier integration, threshold behavior

**Done when:** SmartMemo with classifier shows higher precision than cosine baseline on the same workload. RunResult includes `classifier_score`.

### Phase 4 — Feedback Loop (Week 6)
Add the learning system.

1. `feedback/ledger.py` — append-only event log
2. `cache.report_bad_hit()`, `cache.report_good_hit()` public methods
3. Implicit feedback detection: same prompt re-issued within N seconds after a hit → auto bad-flag
4. `feedback/retrain.py` — CLI command `smartmemo retrain` reads feedback, retrains classifier, evaluates, deploys if better
5. `examples/feedback_loop.py` — simulate user feedback, retrain, show improvement
6. Tests for ledger, retrain flow

**Done when:** you can simulate a workload, report bad hits, retrain, and demonstrate the classifier got better on those specific cases.

### Phase 5 — Integration with Related Agent Tooling (Week 7)
This phase demonstrates how SmartCache composes with adjacent agent infrastructure.

1. `examples/with_agentruntime.py` — wrap LLM calls inside AgentRuntime, show cost meter diff
2. `examples/with_orchflow.py` — multi-step pipeline, show compounded savings
3. `benchmarks/compare_to_gptcache.py` — head-to-head, same workload, show your false-positive rate is dramatically lower
4. `benchmarks/false_positive_eval.py` — the killer demo (medical/legal/finance prompts where cosine fails badly and classifier catches it)
5. Document the killer demos in README with screenshots

**Done when:** you can run a single command that demonstrates SmartMemo + AgentRuntime + orchflow producing measurably cheaper runs than the same agent without caching.

### Phase 6 — Polish & Ship (Week 8)
1. Full documentation site (quickstart, concepts, ML internals, API)
2. README with GIFs/screenshots — focus on the false-positive demo and the cost-savings demo
3. PyPI publish: `pip install smartmemo`
4. Blog post: "Why naive semantic caching fails in production (and what to do instead)"
5. LinkedIn series — one post per concept (embedding filter, classifier, feedback loop)
6. A talk-style write-up for the technical depth: "Training an equivalence classifier for LLM caches"

---

## Key Design Decisions

**1. Two-stage filtering, always.** Embedding similarity narrows; classifier decides. Don't try to replace the embedding stage with the classifier (too slow). Don't try to skip the classifier and tune the threshold (the entire point of the project).

**2. Precision over recall.** False positives are the failure mode that matters. A cache miss costs one LLM call. A false-positive hit returns wrong data to the agent. Default thresholds favor precision.

**3. Domain-specific by design.** SmartMemo instances are tied to a domain. The classifier is finetuned per domain. Don't try to build "one classifier for all use cases" — that's exactly what cosine similarity already is, badly.

**4. Feedback is a first-class citizen.** The library has APIs and ledger storage for feedback from day one. You can't bolt this on later — it's the difference between SmartMemo and "a classifier that sits in front of a cache."

**5. Ship a useful pretrained classifier.** Cold start matters. New users should get value on day one. Train a generic classifier on diverse data, ship it, document its limitations.

**6. CPU-friendly inference.** The classifier must run fast on CPU. If users need a GPU to use the cache, adoption drops to zero. Keep the model small.

**7. Standard tooling.** PyTorch for the classifier, sentence-transformers for embeddings, FAISS for vector search, SQLite for storage. No exotic choices. Senior engineers should be able to read your code and immediately understand the stack.

**8. Composability with your other libraries.** SmartMemo wraps LLM calls. AgentRuntime wraps agents. Orchflow orchestrates steps. Agenteval tests outputs. The four compose. Examples demonstrate this explicitly.

---

## Tech Stack

- **Python 3.11+**
- **PyTorch** — classifier model and training
- **sentence-transformers** — embedding model (`all-MiniLM-L6-v2`)
- **FAISS** (`faiss-cpu`) — vector similarity search
- **SQLite** — cache and feedback persistence (via `sqlite3` stdlib)
- **Pydantic v2** — data models
- **anthropic** and **openai** — for LLM calls in examples and training data generation
- **numpy** — embedding manipulation
- **pytest, pytest-asyncio, ruff, pyright** — dev tooling
- **uv** — package management

Optional:
- **pgvector** or **qdrant-client** — production vector store backends (post-v1)

---

## Environment Variables

```bash
# For examples and training data generation
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional config
SMARTMEMO_DB_PATH=./smartmemo.db
SMARTMEMO_MODEL_PATH=./models/classifier-v2.pt
SMARTMEMO_LOG_LEVEL=INFO
SMARTMEMO_DEVICE=cpu                  # or "cuda" for GPU inference
```

---

## Common Interview Questions & First-Principles Answers

Memorize these — they're the project in interview form.

**Q: Why not just use GPTCache?**
A: GPTCache uses a fixed cosine similarity threshold. Cosine similarity isn't equivalence — "should I accept this meeting" and "should I reject this meeting" are 97% similar by cosine but require opposite responses. The fixed threshold causes false-positive cache hits in production. SmartMemo replaces the threshold with a learned classifier that's finetuned per domain, eliminating the false-positive failure mode.

**Q: How is the classifier trained?**
A: Pairs of prompts labeled as equivalent or not. Positive examples from paraphrasing (use an LLM to rewrite prompts while preserving intent). Negative examples from random pairing. Hard cases (mid-range cosine similarity) get LLM-as-judge labels for quality. Once deployed, user feedback on bad cache hits produces additional negative examples for continual retraining.

**Q: Why a two-stage filter instead of running the classifier directly?**
A: Running the classifier against every cached prompt would be O(N) per query — unusable at scale. Embedding similarity provides O(log N) candidate selection via vector index, then the classifier runs against only K candidates (typically 5-10). This makes the system tractable while keeping decision quality high.

**Q: How do you evaluate it?**
A: Offline: precision, recall, F1, AUC-ROC, and calibration on a labeled test set. The precision metric is the most important — it directly corresponds to false positive rate in production. Online: cache hit rate, cost saved, latency impact, and estimated false positive rate via implicit feedback (regenerations after cache hits).

**Q: What happens on cold start when there's no domain data?**
A: Ship a generic pretrained classifier trained on diverse cross-domain data. Default to a higher threshold (0.95) during cold start for higher precision. Provide a CLI to retrain on the user's domain once enough feedback or labeled data is available.

**Q: Why precision over recall?**
A: A false negative (missed cache hit) costs one extra LLM call — small money. A false positive (wrong cache hit) returns wrong data to the agent — potentially big user-trust cost, possibly safety-relevant. The asymmetry of error costs argues for precision-first tuning. We document this and let users adjust.

**Q: How does this compose with the rest of your stack?**
A: AgentRuntime tracks cost — SmartMemo demonstrably reduces it. Agenteval tests agent behavior — we use it to validate that SmartMemo doesn't introduce regressions. Orchflow orchestrates multi-agent pipelines — SmartMemo wraps the LLM calls in each step, with savings compounding across steps.

**Q: Is this really ML, or is it just engineering?**
A: It's both. The engineering — vector indexing, caching, feedback loops — is necessary infrastructure. The ML — training a pair classifier with proper dataset construction, evaluation, calibration, and continual learning — is the core contribution. Without the ML, this is GPTCache. The ML is what makes it production-trustworthy.

---

## What Success Looks Like

A developer should be able to:

1. `pip install smartmemo`
2. Wrap their existing LLM call site with `smartmemo.get_or_call()`
3. Run their workload and see cost savings within an hour
4. Trust the cache because the classifier blocks the false positives that would break GPTCache
5. Optionally provide feedback and retrain to make the cache better for their domain over time

In under 30 lines of code. Without GPU. Without rewriting their agent.

That's the bar.

---

## The One-Paragraph Pitch

*"Semantic caches for LLM agents have a fatal flaw — cosine similarity treats 'should I accept?' and 'should I reject?' as equivalent, causing false-positive hits that return wrong responses in production. SmartMemo replaces the naive threshold with a small learned classifier, finetuned per domain on real agent traces, that decides equivalence based on intent rather than surface similarity. The result is 60-80% cache hit rates with measurable correctness guarantees — the missing piece between 'GPTCache exists' and 'agent teams actually trust it in production.'"*
