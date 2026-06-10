# Feedback and Retraining

SemanticMemo optimizes for precision: a false negative costs one extra LLM call, but
a false positive returns the wrong cached answer. The feedback loop exists to turn
those wrong hits into training signal, so the classifier improves on the prompts
your application actually sees.

There are two ways feedback is recorded — explicit and implicit — and both feed
the same export and retraining path.

## Explicit feedback

Every cache hit returns a `query_id`. Pass it back to record a verdict on that
specific hit:

```python
result = await cache.get_or_call(prompt="Approve the refund", llm_function=call_llm)

if result.was_cache_hit and answer_was_wrong:
    await cache.report_bad_hit(result.query_id, reason="opposite action")
elif result.was_cache_hit and answer_was_right:
    await cache.report_good_hit(result.query_id)
```

`report_bad_hit` and `report_good_hit` write durable feedback events tied to the
cache-hit lookup. They return `False` if the `query_id` is unknown.

## Implicit feedback (opt-in)

Added in `0.2.0`. Users rarely file explicit feedback, but they do re-ask a
question when the answer was unhelpful. Implicit feedback treats that re-issue as
a signal: when the *same* prompt is sent again within a short window after a cache
hit, the earlier hit is auto-recorded as a bad hit.

It is **off by default**. Enable it through `CacheConfig`:

```python
from semanticmemo import CacheConfig, ImplicitFeedbackConfig, SemanticMemo

cache = SemanticMemo(
    domain="customer-support",
    config=CacheConfig(
        implicit_feedback=ImplicitFeedbackConfig(window_seconds=30.0),
    ),
)
```

When a re-issue is detected, `CacheResult.implicit_bad_hit_recorded` is `True` on
the call that triggered it, and the auto-recorded event carries
`reason="implicit:re-issued"` with `metadata.auto_detected = true`, so implicit
feedback can be told apart from explicit feedback downstream.

What it deliberately does **not** do:

- It matches the prompt **exactly** (after trimming surrounding whitespace). A
  re-phrased re-issue is not detected. Matching by embedding similarity would
  reintroduce the false-positive failure mode SemanticMemo exists to avoid.
- An earlier hit that already has feedback — explicit or implicit — is never
  flagged again, so explicit feedback always takes precedence.
- It is best-effort: if the earlier hit's cache entry was evicted, there is
  nothing left to flag.

## Exporting feedback

Recorded feedback exports to the same JSONL prompt-pair shape the trainer accepts:

```python
written = cache.export_feedback_pairs("data/feedback_pairs.jsonl")
```

or from the CLI:

```bash
uv run semanticmemo --db-path .semanticmemo/cache.db export-feedback --out data/feedback_pairs.jsonl
```

## Retraining

`semanticmemo retrain` turns exported feedback into a candidate classifier
checkpoint, evaluates it against a validation set, and only promotes it when the
validation gates pass:

```bash
uv run semanticmemo --db-path .semanticmemo/cache.db retrain \
  --out models/classifier-candidate.pt \
  --validation-data data/validation_pairs.jsonl \
  --seed-data data/fixtures/customer_support_pairs.jsonl \
  --domain customer-support \
  --min-precision 0.95 \
  --promote-to models/classifier-active.pt
```

Retraining always writes an auditable `<checkpoint>.report.json`. Promotion to
`--promote-to` happens only when the gates pass. SemanticMemo never retrains in the
background or swaps classifiers at runtime — promotion is a deliberate, explicit
step. A classifier is only worth enabling if it is validated to beat the cosine
baseline; that is exactly what the gates check.


