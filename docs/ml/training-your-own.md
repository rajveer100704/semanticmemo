# Training Your Own Classifier

Training data is JSONL prompt pairs. Each line must include `prompt_a`, `prompt_b`, and
`label`. Optional fields such as `domain`, `source`, and `split` make it easier to filter
experiments.

```json
{"prompt_a":"Approve this refund.","prompt_b":"Accept this refund request.","label":1,"domain":"customer-support","split":"train"}
{"prompt_a":"Approve this refund.","prompt_b":"Deny this refund request.","label":0,"domain":"customer-support","split":"train"}
```

Run a local training job:

```bash
uv run equivcache train-classifier \
  --data data/fixtures/customer_support_pairs.jsonl \
  --out models/classifier-v1.pt \
  --domain customer-support \
  --epochs 5
```

By default, the CLI uses `sentence-transformers/all-MiniLM-L6-v2`, so the `ml` extra is
required. For dependency-light smoke runs, use deterministic hash embeddings:

```bash
uv run equivcache train-classifier \
  --data data/fixtures/customer_support_pairs.jsonl \
  --out models/classifier-smoke.pt \
  --embedding-provider hash \
  --embedding-dim 64 \
  --epochs 2
```

The smoke command proves the pipeline works. It is not a quality benchmark.
