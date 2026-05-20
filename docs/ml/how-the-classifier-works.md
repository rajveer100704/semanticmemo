# How The Classifier Works

SmartMemo uses embedding search to find candidate cache entries, but the equivalence
classifier makes the final decision about whether a candidate is actually safe to reuse.
This matters because cosine similarity is only a neighborhood signal. Prompt pairs can
have very similar embeddings while asking for opposite actions.

The classifier is intentionally small. It receives two embedding vectors and builds the
standard matching representation:

```text
[embedding_a, embedding_b, abs(embedding_a - embedding_b), embedding_a * embedding_b]
```

That vector goes through a compact MLP (`embed_dim * 4 -> 128 -> 64 -> 1`) and returns a
probability in `[0, 1]`. A higher probability means the model believes both prompts
should produce the same useful response. The model implementation lives in
`src/smartmemo/classifier/model.py`.

## The bundled classifier (`classifier-v2`)

SmartMemo ships a pretrained checkpoint inside the package at
`smartmemo/_models/classifier-v2.pt`. Load it with `ClassifierConfig.bundled()`.

`classifier-v2` is a **generic cold-start model**. Its training data was built locally:
a hand-authored prompt corpus across nine domains is expanded into 16,576 labeled pairs
by a local LLM (paraphrases for positives) and by templated same-object/opposite-action
swaps, including negation (guaranteed-correct hard negatives). The full pipeline is
`scripts/generate_training_data.py`; the auditable model card is
`smartmemo/_models/classifier-v2.report.json`.

Two limitations are worth knowing:

- It is bound to the `all-MiniLM-L6-v2` embedding space (384 dimensions). Use the same
  embedding model when the bundled classifier is active.
- Being generic, it trades per-domain peak accuracy for broad coverage. Per-domain
  accuracy improves once you collect feedback and run `smartmemo retrain` — see
  `docs/ml/training-your-own.md`.

At runtime, `SmartMemo(..., classifier=ClassifierConfig.bundled())` (or any
`ClassifierConfig(model_path=...)`) loads a checkpoint and uses classifier scores instead
of cosine thresholding for cache-hit decisions.
