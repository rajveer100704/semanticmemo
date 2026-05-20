# SmartMemo

SmartMemo is a semantic memory and caching layer for LLM agent calls. Its core thesis is
simple: cosine similarity is a useful candidate selector, but it is not semantic
equivalence. SmartMemo uses embedding search to find likely cache candidates, then uses a
learned equivalence classifier to decide whether a cached response is safe to reuse.

As of `0.2.0`, SmartMemo **ships a pretrained classifier**, so that decision works out of
the box — no training required.

- async `SmartMemo.get_or_call(...)`
- a bundled pretrained equivalence classifier (`classifier-v2`), opt-in with one line
- SQLite persistence
- embedding provider protocol with SentenceTransformers embeddings and FAISS vector search
- a reproducible local-LLM training-data pipeline and a hand-curated gold test set
- classifier training, evaluation, checkpoint inference, and classifier-gated cache hits
- explicit and opt-in implicit feedback capture, durable export, and gated retraining

Without a classifier, SmartMemo decides cache hits with a cosine threshold — the measured
baseline. With the bundled classifier, cosine search becomes the candidate selector and
the learned classifier makes the final cache-hit decision.

## Install

SmartMemo's embedding and classifier stack depends on PyTorch, FAISS, and
SentenceTransformers, so install the `ml` extra:

```bash
pip install "smartmemo[ml]"
```

For local development:

```bash
uv sync --all-extras
uv run pytest
uv run ruff check
uv run pyright
```

## Minimal Example

```python
from smartmemo import ClassifierConfig, SmartMemo

cache = SmartMemo(
    domain="customer-support",
    classifier=ClassifierConfig.bundled(),
)

async def call_llm(prompt: str) -> str:
    return "fresh LLM response"

result = await cache.get_or_call(
    prompt="Summarize this customer's latest billing ticket",
    llm_function=call_llm,
)

print(result.response)
print(result.was_cache_hit)
print(result.classifier_score)
```

## The Bundled Classifier

`classifier-v2` is a generic, cross-domain equivalence classifier shipped inside the
package at `smartmemo/_models/classifier-v2.pt`. It is a small MLP over
`all-MiniLM-L6-v2` embeddings, trained on 16,576 labeled prompt pairs across nine
domains, built by a local LLM paraphraser (positives) and templated
same-object/opposite-action swaps including negation (hard negatives). The whole
pipeline is `scripts/generate_training_data.py`.

Measured on a hand-curated gold set of 84 held-out prompt pairs (31 equivalent, 53 not).
The set deliberately includes opposite-action pairs — the case a fixed cosine threshold
gets wrong:

| Decision method                   | Precision | Recall | F1   |
|------------------------------------|-----------|--------|------|
| Cosine baseline (at equal recall)  | 0.53      | 0.94   | 0.67 |
| `classifier-v2` (threshold 0.95)   | 0.83      | 0.94   | 0.88 |

That is **+30 precision points at equal recall**: on this gold set the cosine baseline
makes 26 false-positive cache hits where `classifier-v2` makes 6. The full, auditable
model card — including the in-distribution validation metrics — is
`smartmemo/_models/classifier-v2.report.json`.

`classifier-v2` is still a generic cold-start model. On out-of-distribution, adversarial
prompts it beats the cosine baseline but is not infallible — see the high-stakes
benchmark below. It is bound to the `all-MiniLM-L6-v2` embedding space (384 dimensions),
and per-domain accuracy improves with the feedback-driven retraining loop below.

## Benchmarks

```bash
uv run python benchmarks/cosine_baseline_customer_support.py
uv run python benchmarks/classifier_vs_cosine.py
uv run python benchmarks/false_positive_eval.py
```

The first benchmark shows the cosine baseline's false-positive failure mode on
customer-support prompts. The second scores the bundled classifier against the cosine
baseline on the gold set and writes `benchmarks/results/classifier_vs_cosine.json`.

The third runs a small, hand-authored set of high-stakes medical/legal/finance
opposite-action prompts. On that adversarial set the cosine baseline wrongly serves 8 of
16 opposite-action pairs from cache; `classifier-v2` wrongly serves 6 — better than
cosine, but a reminder that a generic classifier is not infallible on out-of-distribution
prompts and that domain retraining still matters. GPTCache and similar semantic caches
decide hits by embedding similarity, so the cosine baseline here represents that class of
tool.

## Training Your Own Classifier

SmartMemo includes a trainable pair classifier over prompt embeddings. To reproduce the
shipped model from the committed dataset:

```bash
uv run python scripts/train_classifier.py
```

To train on your own JSONL prompt pairs:

```bash
uv run smartmemo train-classifier \
  --data data/fixtures/customer_support_pairs.jsonl \
  --out models/classifier-custom.pt \
  --domain customer-support \
  --epochs 5
```

Then point SmartMemo at the checkpoint:

```python
from smartmemo import ClassifierConfig, SmartMemo

cache = SmartMemo(
    domain="customer-support",
    classifier=ClassifierConfig(model_path="models/classifier-custom.pt"),
)
```

## Feedback Export

SmartMemo records cache-hit lookups so explicit feedback can become training data:

```python
result = await cache.get_or_call(
    prompt="Approve the customer's refund request",
    llm_function=call_llm,
)

if result.was_cache_hit and user_rejected_answer:
    await cache.report_bad_hit(result.query_id, reason="wrong refund decision")

written = cache.export_feedback_pairs("data/feedback_pairs.jsonl")
print(written)
```

The exported JSONL uses the same prompt-pair shape accepted by `smartmemo train-classifier`.

## Implicit Feedback

Users rarely file explicit feedback, but they do re-ask a question when the answer was
unhelpful. Implicit feedback — opt-in, off by default — treats re-issuing the *same*
prompt shortly after a cache hit as a signal that the earlier hit was bad, and records it
automatically:

```python
from smartmemo import CacheConfig, ImplicitFeedbackConfig, SmartMemo

cache = SmartMemo(
    domain="customer-support",
    config=CacheConfig(
        implicit_feedback=ImplicitFeedbackConfig(window_seconds=30.0),
    ),
)
```

When a re-issue is detected, `CacheResult.implicit_bad_hit_recorded` is `True` and an
auto-labeled bad-hit event is recorded (told apart from explicit feedback by its
metadata). Matching is exact — a re-phrased re-issue is not detected — and explicit
feedback always takes precedence. See `docs/feedback.md`.

## Manual Retraining

Use `smartmemo retrain` to turn durable feedback into a candidate classifier checkpoint:

```bash
uv run smartmemo --db-path .smartmemo/cache.db retrain \
  --out models/classifier-candidate.pt \
  --validation-data data/validation_pairs.jsonl \
  --seed-data data/fixtures/customer_support_pairs.jsonl \
  --domain customer-support \
  --min-precision 0.95 \
  --promote-to models/classifier-active.pt
```

The command always trains a candidate and writes an auditable
`<checkpoint>.report.json`. Promotion only copies the candidate to `--promote-to` when the
validation gates pass. SmartMemo does not run background retraining or automatically reload
classifiers at runtime.

## Release

Version `0.2.0` is configured for PyPI as `smartmemo`. The repository publishes through
GitHub Actions trusted publishing from `.github/workflows/publish-pypi.yml` with the
`pypi` environment.

```bash
git tag v0.2.0
git push origin v0.2.0
```

That tag builds the source distribution and wheel, then uploads them to PyPI.
