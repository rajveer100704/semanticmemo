# Production Deployment Guide

Deploying SemanticMemo in a production environment requires consideration of concurrency, latency, storage, and reliability. This guide details recommended patterns and configurations.

---

## 1. Deployment Architectures

### Pattern A: Single-Node (Embedded)
Ideal for single-instance web servers, edge functions, or worker nodes.
- **Store**: Embedded SQLite database (`db_path` located on a local fast SSD).
- **Index**: Local FAISS vector index (in-memory, rebuilt/updated on startup).
- **Inference**: Models run on CPU with thread-local pooling.

### Pattern B: Distributed Vector / Local Cache (Recommended for Scale)
For multi-node cluster deployments (e.g., Kubernetes), using a centralized vector database alongside local classifiers.
- **Store & Index**: Use **Qdrant** as the external vector database. Configure `vector_store_type="qdrant"` in `CacheConfig` with a centralized cluster URL.
- **Inference**: Run the MLP and Cross-Encoder locally on each application node to avoid network round-trips for the classifier stage.

---

## 2. Performance Tuning

### CPU Warmup Profile
PyTorch model initialization and the first inference call can introduce a 100-300ms latency spike due to model download, compilation, and cache allocation.
- **Solution**: Always perform a warmup check on application boot:
  ```python
  cache = SemanticMemo(domain="finance", classifier=ClassifierConfig.bundled())
  # Warmup call
  await cache.get_or_call(
      prompt="warmup",
      llm_function=lambda p: asyncio.sleep(0)
  )
  ```

### Latency-Aware Bypassing
Configure the `high_precision_skip_threshold` (default `0.995`). Matches exceeding this score bypass the second-stage Cross-Encoder, saving 30ms of latency per hit.

---

## 3. Active Learning Pipeline

Disagreements between the MLP classifier and Cross-Encoder are saved to the local database. To constantly improve your MLP's precision:
1. Export the captured disagreements periodically:
   ```bash
   semanticmemo export-active-learning --out active_learning_data.jsonl
   ```
2. Retrain the classifier using the exported data:
   ```bash
   semanticmemo retrain --out models/equivalence-net-v2.pt --validation-data tests/data/validation.jsonl
   ```
3. Update your cache configuration to point to the new PT file.

---

## 4. Reliability & Circuit Breaker Designs

When connecting SemanticMemo to your LLM pipelines, ensure a failure of the cache or model inference does not crash your user-facing applications.
- **Fallback**: Wrap cache requests in a `try-except` block to fall back to direct LLM calls if the database is locked or model inference fails:
  ```python
  try:
      result = await cache.get_or_call(prompt=prompt, llm_function=call_llm)
      response = result.response
  except Exception:
      # Fallback to direct LLM call
      response = await call_llm(prompt)
  ```
