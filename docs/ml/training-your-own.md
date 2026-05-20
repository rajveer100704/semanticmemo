# Training Your Own Classifier

SmartMemo already ships a pretrained classifier (`ClassifierConfig.bundled()`). Train your
own when you want a domain-specialized model or want to reproduce the shipped one.

## Reproducing the bundled classifier

The shipped `classifier-v2` is fully reproducible from committed data:

```bash
# Optional: regenerate the dataset from the prompt corpus (requires a local Ollama model)
python scripts/generate_training_data.py

# Train and package the classifier from the committed dataset
uv run python scripts/train_classifier.py
```

`train_classifier.py` writes the checkpoint to `src/smartmemo/_models/classifier-v2.pt`
and an auditable model card next to it (gold-set and high-stakes metrics included).

## Dataset format

Training data is JSONL prompt pairs. Each line must include `prompt_a`, `prompt_b`, and
`label`. Optional fields such as `domain`, `source`, and `split` make it easier to filter
experiments.

```json
{"prompt_a":"Approve this refund.","prompt_b":"Accept this refund request.","label":1,"domain":"customer-support","split":"train"}
{"prompt_a":"Approve this refund.","prompt_b":"Deny this refund request.","label":0,"domain":"customer-support","split":"train"}
```

Run a local training job:

```bash
uv run smartmemo train-classifier \
  --data data/fixtures/customer_support_pairs.jsonl \
  --out models/classifier-custom.pt \
  --domain customer-support \
  --epochs 5
```

By default, the CLI uses `sentence-transformers/all-MiniLM-L6-v2`, so the `ml` extra is
required. For dependency-light smoke runs, use deterministic hash embeddings:

```bash
uv run smartmemo train-classifier \
  --data data/fixtures/customer_support_pairs.jsonl \
  --out models/classifier-smoke.pt \
  --embedding-provider hash \
  --embedding-dim 64 \
  --epochs 2
```

The smoke command proves the pipeline works. It is not a quality benchmark.

## Training From Feedback

SmartMemo can export explicit cache-hit feedback as JSONL pairs:

```bash
uv run smartmemo export-feedback \
  --out data/feedback_pairs.jsonl \
  --split train
```

Bad-hit feedback is exported as label `0`; good-hit feedback is exported as label `1`.
The query prompt is paired with the cached prompt that was reused. Cached responses are not
duplicated in the feedback export.

The exported file uses the same format as other classifier datasets, so it can be passed
directly to `smartmemo train-classifier`.

## Manual Retraining From Feedback

For an auditable feedback loop, use `smartmemo retrain`:

```bash
uv run smartmemo --db-path .smartmemo/cache.db retrain \
  --out models/classifier-candidate.pt \
  --validation-data data/validation_pairs.jsonl \
  --seed-data data/fixtures/customer_support_pairs.jsonl \
  --domain customer-support \
  --threshold 0.85 \
  --min-precision 0.95 \
  --min-recall 0.0 \
  --promote-to models/classifier-active.pt
```

The command loads feedback-derived pairs from the cache database, appends optional seed
training records, trains a candidate checkpoint, evaluates it on held-out validation data,
and writes `<checkpoint>.report.json`. If `--promote-to` is provided, the candidate is
copied to that path only when the configured gates pass.

This is intentionally an operator-controlled workflow. SmartMemo does not run background
training, auto-promote failed candidates, or hot-reload classifiers in running processes.
