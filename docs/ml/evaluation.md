# Evaluation

EquivCache optimizes for precision before recall. A false negative costs one extra LLM
call. A false positive returns the wrong cached answer, which is the production failure
mode this project is built to avoid.

Evaluate a checkpoint with:

```bash
uv run equivcache eval-classifier \
  --data data/fixtures/customer_support_pairs.jsonl \
  --model models/classifier-v1.pt \
  --domain customer-support \
  --split test
```

The evaluator reports precision, recall, F1, accuracy, ROC AUC when both classes are
present, expected calibration error, and the confusion-matrix counts. Early seed datasets
are for pipeline validation only. Portfolio claims should use a larger held-out test set
and should compare against the measured cosine baseline.
