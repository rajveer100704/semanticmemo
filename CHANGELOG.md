# Changelog

## 1.2.0

Upgrade release featuring zero-latency entity drift protection, domain-conditioned thresholds, active learning builder, and safety-framed benchmarking.

**New features & upgrades:**

- **Entity Drift Protection** — The `EntityChangeDetector` runs 11 fast, regex-based validation checks in `< 1ms` to intercept critical entity changes (e.g. Q3 vs Q4, Tesla vs Apple, 100mg vs 200mg) that neural models and cross-encoders often miss.
- **Domain-Conditioned Thresholds** — Replaces generic risk tiers with domain-specific MLP and Cross-Encoder thresholds (Medical: 0.995/0.97, Finance: 0.990/0.95, Security: 0.997/0.98, Customer Support: 0.90/0.85) to optimize precision/recall trade-offs.
- **Active Learning Dataset Builder** — Automatically logs MLP vs Cross-Encoder disagreements to the `active_learning_pairs` database table to build hard-negative datasets for continuous retraining.
- **Safety-Framed Benchmarking** — Rewrote the semantic drift benchmark to compare three systems (Cosine, SM v1, SM v2) on Safety Accuracy (correct blocks), Dangerous False Positive Rate (DFP), and Reuse Accuracy (valid hits).
- **Comprehensive Tests** — Added 33 unit tests under `tests/test_entity_change_detection.py` validating each regex detector and disabled configurations.

## 1.0.0

First stable release. Published to PyPI as `semanticmemo` (internal import name unchanged: `import semanticmemo`).

This release promotes SemanticMemo from a cosine-threshold semantic cache to a four-stage
verification pipeline with measured, reproducible false-positive reduction across four
high-stakes production domains.

**Architecture additions:**

- **Double Verification pipeline** — MLP classifier (stage 3) + Cross-Encoder (stage 4) replace the
  single cosine threshold. The MLP is fast (~1ms); the Cross-Encoder is high-precision (~3–8ms) and
  is only invoked when the MLP is uncertain.
- **Opposite-action veto** — rule-based pre-filter that blocks antonym pairs (approve/deny,
  buy/sell, increase/decrease, enable/disable, allow/block, grant/revoke, etc.) before any ML
  stage runs. Catches the canonical hard-negative failure case with zero latency overhead.
- **Latency-aware MLP bypass** — when the MLP score exceeds `high_precision_skip_threshold`
  (default 0.995), the Cross-Encoder is skipped entirely. Measured bypass rate: **94.7%** of
  cache hits; average latency with bypass: **~27ms** vs **~39ms** without.
- **Domain auto-detection** — `DomainDetector` routes each query to its risk tier using embedding
  centroids, enabling per-domain threshold policies without explicit domain tagging by the caller.
- **Risk-aware policy engine** — `RiskPolicy` maps domains to `RiskTier.HIGH` or `RiskTier.LOW`
  and applies the corresponding MLP/CE thresholds. Ships with sensible defaults:
  medical/finance/security/legal → HIGH (MLP=0.99, CE=0.95); general/customer-support → LOW (MLP=0.90, CE=0.85).
- **Explainable `CacheDecision`** — every `CacheResult.decision` records the decision reason,
  risk tier, per-stage scores, and per-stage latency. Available on both hits and misses.

**Performance and profiling:**

- **Global `_MODEL_CACHE`** in `CrossEncoderService` — the Cross-Encoder is loaded once and
  shared across all `SemanticMemo` instances in the same process. Eliminates the 1–5 second
  cold-start penalty on the first inference.
- **Five new latency profiling fields** on `CacheResult` and `CacheDecision`:
  `embedding_latency_ms`, `retrieval_latency_ms`, `mlp_latency_ms`, `cross_encoder_latency_ms`,
  `total_latency_ms`. Enables per-stage bottleneck diagnosis in production.

**Benchmark suite expansion:**

- `benchmarks/run_benchmarks.py` — full 4-method × 4-domain comparison matrix (Cosine Baseline,
  MLP Classifier, Double Verification, SemanticMemo). Includes per-stage latency breakout table,
  hard-negative stress test, and Cross-Encoder bypass statistics. Output: `comparison_matrix.json`.
- `benchmarks/sweep_thresholds.py` — 40 MLP thresholds × 30 CE thresholds = 1,200 configurations
  per domain, with per-domain FPR constraints (Security < 5%, Finance < 10%). Output:
  `threshold_report.md` with optimal per-domain config and production recommendation.
- `benchmarks/data/hard_negatives.jsonl` — 12 curated opposite-action pairs across finance,
  security, medical, and software domains.
- All four domain datasets expanded to 20 pairs each (10 positive + 10 hard-negative),
  replacing the previous 5–10 pair sets.
- Model warmup in both benchmark scripts eliminates download/load time from measured latency.

**Key measured results:**

| Metric | Cosine Baseline | SemanticMemo |
| :--- | :---: | :---: |
| Hard-Negative FPR | 33.3% | **0.0%** |
| Finance FPR | 20% | **0%** |
| Security FPR | 20% | **0%** |
| Medical FPR | 30% | **10%** |
| Classifier Precision (gold set, equal recall) | 0.527 | **0.829** |
| CE Bypass Rate | — | **94.7%** |
| Avg Cache-Hit Latency | — | **~27ms** |

**Documentation:**

- `docs/results.md` — comprehensive 9-section benchmark results document: architecture diagram,
  full comparison matrix, hard-negative stress test results, latency breakdown, bypass rate
  analysis, threshold sweep results, cost savings model, and classifier gold-set evaluation.
- `README.md` rewritten for v1.0.0: problem statement, architecture mermaid diagram, all
  benchmark tables, full quickstart with production config, project structure, and roadmap.

**Packaging:**

- PyPI package renamed to `semanticmemo`. Internal import name unchanged: `import semanticmemo`.
- `Development Status` classifier upgraded from Beta to `5 - Production/Stable`.
- Added `Topic :: Scientific/Engineering :: Artificial Intelligence` classifier.
- `__version__ = "1.0.0"` exported from `semanticmemo.__init__`.

## 0.3.0

A hardening release: no new product features, but the existing ones become safer to run
in production. Every change is backward compatible; new configuration is opt-in and off
by default.

- Add structured logging under the `semanticmemo` logger namespace. The library is silent by
  default (it attaches only a `NullHandler`); applications opt in by configuring the
  `semanticmemo` logger.
- Open the SQLite store in WAL mode with a 5-second busy timeout, and serialize writes
  with an internal re-entrant lock so one store instance is safe to use from multiple
  threads of a process. The concurrency guarantees are documented honestly — it is not a
  distributed cache.
- Bug fix: enable `PRAGMA foreign_keys`, so the schema's declared `ON DELETE CASCADE`
  foreign keys now actually fire. Deleting or evicting a cache entry previously left
  orphan `lookup_records` and `feedback_events` rows behind; it now cleans them up.
- `SemanticMemo` supports `async with` for guaranteed cleanup; `close()` is unchanged.
- Add opt-in LLM-call retries: `CacheConfig(retry=RetryConfig(...))` retries transient
  failures of `llm_function` with bounded exponential backoff. Off by default; only the
  cache-miss path is retried. Adds the `RetryConfig` model and the `LLMCallError`
  exception (raised only when retries are enabled and exhausted).
- CI now tests Python 3.11 through 3.14; the PyPI publish workflow runs the test suite
  before building and releasing.
- Add `CONTRIBUTING.md`.

## 0.2.0

- Retrain the bundled classifier as `equivalence-net-v1`: trained on 16,576 labeled pairs
  across nine domains (adds medical, legal, and finance, plus negation-by-insertion hard
  negatives). It beats the cosine baseline by +30 precision points at equal recall on the
  gold set and replaces `classifier-v1` as the shipped checkpoint.
- Add opt-in implicit-feedback detection: with `CacheConfig(implicit_feedback=...)`,
  re-issuing the same prompt within a window after a cache hit auto-records a bad-hit
  event. Off by default; explicit feedback always takes precedence.
- Add the `ImplicitFeedbackConfig` model and `CacheResult.implicit_bad_hit_recorded`.
- `get_or_call` now trims surrounding whitespace from the prompt, so the cache key is
  whitespace-insensitive — prompts differing only by leading or trailing whitespace no
  longer create duplicate cache entries.
- Add `benchmarks/false_positive_eval.py` and `benchmarks/data/high_stakes_pairs.jsonl`:
  a high-stakes medical/legal/finance opposite-action evaluation.
- Add optional integration examples (`examples/with_orchflow.py`, `with_guardloop.py`,
  `with_agenteval.py`) composing SemanticMemo with companion libraries; none are
  dependencies of semanticmemo.
- Add `docs/feedback.md` covering explicit and implicit feedback and retraining.

## 0.1.0

- Ship `classifier-v1`, a pretrained generic equivalence classifier, inside the package.
- Add `ClassifierConfig.bundled()` for opt-in, zero-training classifier-gated caching.
- Add `scripts/generate_training_data.py`: a local-LLM (Ollama) paraphrase and templated
  hard-negative training-data pipeline, with a committed 10,800-pair dataset.
- Add a hand-curated 84-pair equivalence gold test set under `data/gold/`.
- Add `scripts/train_classifier_v1.py` and `benchmarks/classifier_vs_cosine.py`:
  `classifier-v1` beats the cosine baseline by +32 precision points at equal recall on
  the gold set.
- Document that the optional `[ml]` dependencies (PyTorch, FAISS, SentenceTransformers)
  are required to import semanticmemo.

## 0.0.4

- Add `semanticmemo retrain` for manual feedback-to-checkpoint retraining.
- Add `semanticmemo.feedback.retrain_from_feedback(...)` with auditable retrain reports.
- Support optional seed data, validation gates, and checkpoint promotion when gates pass.
- Document the manual feedback loop from collection through deliberate classifier promotion.

## 0.0.3

- Add durable lookup records for cache hits so feedback can be exported later.
- Add durable good/bad feedback events tied to cache-hit query IDs.
- Add `SemanticMemo.export_feedback_pairs(...)` and `semanticmemo export-feedback`.
- Export feedback-derived JSONL compatible with the classifier training pipeline.

## 0.0.2

- Add optional classifier-gated cache decisions through `SemanticMemo(..., classifier=...)`.
- Populate `CacheResult.classifier_score` when the classifier evaluates cache candidates.
- Preserve cosine-threshold behavior when no classifier checkpoint is configured.
- Document classifier-enabled usage and the current cold-start boundary.

## 0.0.1

Initial SemanticMemo release.

- Async semantic-cache facade with SQLite persistence.
- Dependency-light hash embeddings for tests and smoke demos.
- Optional SentenceTransformers and FAISS support through the `ml` extra.
- Customer-support cosine-baseline benchmark fixture.
- Pair-classifier model, dataset, training, evaluation, and checkpoint inference utilities.
- CLI commands for cache stats, classifier training, and classifier evaluation.
- GitHub Actions CI and PyPI trusted publishing workflow.


