# ADR-003: Why Domain-Conditioned Risk Policies

## Status
Approved

## Context
Cache queries belong to different domains. A global, static threshold for semantic cache reuse is inadequate:
- Medical and Security queries require extremely high precision; serving a wrong cached response can be dangerous (high-risk).
- General Summarization or Customer Support queries can accept a slightly lower threshold to maximize cache hit rates (low-risk).
We need a way to dynamically categorize incoming queries and apply risk-aware policies without requiring the user to manually tag every query.

## Decision
We implement a **Domain-Conditioned Risk Policy** system:
1. **Dynamic Centroid Clustering**: We calculate domain centroids from representative prompts using pre-trained sentence embeddings.
2. **Domain Detection**: The `DomainDetector` computes the cosine similarity between the incoming query embedding and each domain centroid to determine the domain (e.g. `medical`, `finance`, `security`, `general`).
3. **Adaptive Thresholds**: Each domain maps directly to custom MLP and Cross-Encoder thresholds. High-risk domains (like `medical` or `security`) enforce strict thresholds (e.g. MLP=0.995, Cross-Encoder=0.97) while low-risk domains use lower thresholds (e.g., MLP=0.90, Cross-Encoder=0.85) to maximize recall.

## Consequences
- **Safety**: Sensitive domains are automatically safeguarded against wrong cache reuse.
- **Efficiency**: General queries maintain a high cache hit rate.
- **Zero-Config Developer UX**: Developers do not need to manually classify prompts at runtime; the domain detector handles routing implicitly.
