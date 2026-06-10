# Quickstart

SemanticMemo is an async-first semantic cache for LLM agent calls. Embedding search finds
candidate cache entries; a learned equivalence classifier decides whether a candidate is
genuinely safe to reuse. That classifier is what separates SemanticMemo from a fixed cosine
threshold: "approve this refund" and "deny this refund" are highly cosine-similar but must
never share a cache entry.

## Install

SemanticMemo's embedding and classifier stack depends on PyTorch, FAISS, and
SentenceTransformers, so install the `ml` extra:

```bash
pip install "semanticmemo[ml]"
```

## Minimal use

```python
from semanticmemo import SemanticMemo

async def call_llm(prompt: str) -> str:
    return "fresh response from your provider"

async with SemanticMemo(domain="customer-support") as cache:
    result = await cache.get_or_call(prompt="Summarize this ticket", llm_function=call_llm)
    print(result.response)
    print(result.was_cache_hit)
```

`async with` closes the underlying store when the block exits, even on error. If you
manage the instance's lifetime yourself instead, call `cache.close()` when done.

Without a classifier, SemanticMemo decides cache hits with a cosine-similarity threshold —
the measured baseline the classifier is built to beat.

## Classifier-gated caching (recommended)

SemanticMemo ships a pretrained generic equivalence classifier. Turn it on with one line:

```python
from semanticmemo import ClassifierConfig, SemanticMemo

cache = SemanticMemo(
    domain="customer-support",
    classifier=ClassifierConfig.bundled(),
)
```

Cosine search now only selects candidates; the learned classifier makes the final
cache-hit decision, and `CacheResult.classifier_score` is populated. The bundled model
is a generic cold-start classifier — accuracy on your own traffic improves with the
feedback-driven retraining loop below. See `docs/ml/how-the-classifier-works.md`.

## Feedback export

```python
result = await cache.get_or_call(prompt="Summarize this ticket", llm_function=call_llm)

if result.was_cache_hit:
    await cache.report_bad_hit(result.query_id, reason="user rejected cached answer")

cache.export_feedback_pairs("data/feedback_pairs.jsonl")
```

## Manual retraining

```bash
uv run semanticmemo --db-path .semanticmemo/cache.db retrain \
  --out models/classifier-candidate.pt \
  --validation-data data/validation_pairs.jsonl \
  --seed-data data/fixtures/customer_support_pairs.jsonl \
  --domain customer-support \
  --min-precision 0.95 \
  --promote-to models/classifier-active.pt
```

The retrain command trains a candidate checkpoint and writes a report next to it. It only
promotes the checkpoint when validation gates pass; runtime classifier loading remains an
explicit `ClassifierConfig(model_path=...)` choice.

## Retrying transient LLM failures

LLM provider calls can fail transiently. Retries are opt-in and off by default — pass a
`RetryConfig` to retry the cache-miss call with bounded exponential backoff:

```python
from semanticmemo import CacheConfig, RetryConfig, SemanticMemo

cache = SemanticMemo(
    domain="customer-support",
    config=CacheConfig(retry=RetryConfig(max_attempts=3, initial_backoff_seconds=0.5)),
)
```

Only the cache-miss path is retried — a cache hit never calls the LLM. By default any
exception triggers a retry; narrow it with `RetryConfig(retry_on=(ConnectionError,
TimeoutError))`. When all attempts are exhausted, SemanticMemo raises
`semanticmemo.exceptions.LLMCallError`, chaining the last failure as its cause. With `retry`
left unset, the LLM call behaves exactly as before: one attempt, exceptions raised
unchanged.

## Logging

SemanticMemo logs under the `semanticmemo` logger namespace and is silent by default (it
attaches only a `NullHandler`). Opt in from your application:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("semanticmemo").setLevel(logging.DEBUG)
```


