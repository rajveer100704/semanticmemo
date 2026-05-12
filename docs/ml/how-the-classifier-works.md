# How The Classifier Works

EquivCache uses embedding search to find candidate cache entries, but the classifier is
the component that should eventually decide whether a candidate is actually equivalent.
This matters because cosine similarity is only a neighborhood signal. Prompt pairs can
have very similar embeddings while asking for opposite actions.

The Phase 2 classifier is intentionally small. It receives two embedding vectors and
builds the standard matching representation:

```text
[embedding_a, embedding_b, abs(embedding_a - embedding_b), embedding_a * embedding_b]
```

That vector goes through a compact MLP and returns a probability in `[0, 1]`. A higher
probability means the model believes both prompts should produce the same useful response.

The implementation lives in `src/equivcache/classifier/model.py`. It is not wired into
`EquivCache.get_or_call()` yet. That integration belongs to the next phase, after the
training and evaluation path is stable.
