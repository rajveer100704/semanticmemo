# Data

This directory holds the corpora, generated datasets, and evaluation fixtures used to
build and measure the SmartMemo equivalence classifier.

- `corpus/` — hand-authored source material. `base_prompts.jsonl` is a diverse set of
  agent prompts across nine domains; `action_templates.jsonl` defines same-object,
  opposite-action pairs (including negation) used to mint guaranteed-correct hard
  negatives.
- `training/` — the generated labeled prompt-pair dataset (`pairs_v2.train.jsonl`,
  `pairs_v2.validation.jsonl`) plus a `manifest.json` recording how it was produced.
  Regenerate it with `python scripts/generate_training_data.py` (requires a local
  Ollama model). The committed dataset is the source of truth, so retraining the
  classifier does not require Ollama.
- `gold/` — `equivalence_gold.jsonl`, a hand-curated test set of 84 prompt pairs. This
  is the trustworthy benchmark: it is never used for training, only for measuring the
  classifier against the cosine baseline.
- `fixtures/` — the original small customer-support seed fixture, kept for smoke checks.
