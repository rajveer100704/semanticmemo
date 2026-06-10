# Evaluation

SemanticMemo optimizes for precision before recall. A false negative costs one extra LLM
call. A false positive returns the wrong cached answer, which is the production failure
mode this project is built to avoid.

Evaluate a checkpoint with:

```bash
uv run semanticmemo eval-classifier \
  --data data/fixtures/customer_support_pairs.jsonl \
  --model models/equivalence-net-v1.pt \
  --domain customer-support \
  --split test
```

The evaluator reports precision, recall, F1, accuracy, ROC AUC when both classes are
present, expected calibration error, and the confusion-matrix counts.

## The gold set and the cosine baseline

The trustworthy benchmark for the shipped classifier is `data/gold/equivalence_gold.jsonl`
— 84 hand-curated prompt pairs that are never used for training. Compare the classifier
against the cosine baseline on it with:

```bash
uv run python benchmarks/classifier_vs_cosine.py
```

Both methods are scored on the same embeddings, and precision is compared *at equal
recall* — a fair comparison, since either method can trade recall for precision by moving
its threshold. The acceptance gate for `equivalence-net-v1` is precision at least 10 points
above the cosine baseline at equal recall; it currently clears that by +30 points. Results
are written to `benchmarks/results/classifier_vs_cosine.json`, and the shipped model card
is `semanticmemo/_models/equivalence-net-v1.report.json`.

## The high-stakes benchmark

`benchmarks/false_positive_eval.py` runs a small, hand-authored set of medical, legal,
and finance opposite-action prompt pairs — the kind of confusion that is genuinely
dangerous to get wrong. It is deliberately adversarial and partly out of the training
distribution. On it, the cosine baseline wrongly serves 8 of 16 opposite-action pairs
from cache and `equivalence-net-v1` wrongly serves 6: better than cosine, but an honest
reminder that a generic classifier is not infallible and that domain retraining matters.
The set is illustrative, not a production traffic sample.


