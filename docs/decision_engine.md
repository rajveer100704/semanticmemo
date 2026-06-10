# SemanticMemo Decision Engine

This document outlines the math, algorithms, and decision logic that govern cache hits in SemanticMemo.

---

## Double Verification Algorithm

Every cache lookup in SemanticMemo follows a multi-stage validation pipeline:

1. **Candidate Selection**: Retrieval of the top-$K$ candidates based on cosine similarity:
   $$\text{sim}(e_q, e_c) = \cos(e_q, e_c)$$
2. **First-Stage Filter (MLP)**: Estimates equivalence probability using a feed-forward neural network over the matching representation:
   $$P_{\text{MLP}}(q, c) = \sigma(\text{MLP}([e_q; e_c; |e_q - e_c|; e_q \odot e_c]))$$
3. **Bypass Check**: If $P_{\text{MLP}}$ exceeds the bypass threshold ($T_{\text{skip}}$), we assume a high-confidence match and skip the Cross-Encoder:
   $$\text{If } P_{\text{MLP}} \ge T_{\text{skip}} \implies \text{HIT (bypass Cross-Encoder)}$$
4. **Second-Stage Verification (Cross-Encoder)**: Jointly encodes both prompts to catch subtle differences (negations, numbers, opposite actions):
   $$P_{\text{CE}}(q, c) = \sigma(\text{CrossEncoder}(q, c))$$
5. **Decision Logic**:
   * If both classifier scores meet the required thresholds (governed by the active risk policy), it is a **HIT**.
   * Otherwise, it is a **MISS**.

---

## Domain Auto-Detection

SemanticMemo automatically assigns a prompt to one of the following domains: `medical`, `finance`, `legal`, `security`, or `general`.

### Centroid-Based Classification
We maintain a pre-defined set of anchor prompt centroids representing each domain. When a prompt is received:
1. Generate prompt embedding $e_q$.
2. Calculate cosine similarity against each domain centroid vector $C_d$:
   $$\text{similarity}_d = \frac{e_q \cdot C_d}{\|e_q\| \|C_d\|}$$
3. Assign the prompt to the domain with the highest similarity.
4. If the highest similarity falls below a minimum confidence threshold (e.g. 0.5), default to the `general` domain.

> [!NOTE]
> The `DomainDetector` is designed as a pluggable module. While the default implementation uses centroid-based cosine distance for speed and zero-training simplicity, the interface supports plug-in supervised classification models (e.g., small neural classifiers or LLM-based classifiers) for higher classification accuracy under complex prompts.

### Domain Centroid Representation
The centroids are computed by averaging embeddings of representative prompts:
* **Medical**: Prompts containing diagnostic, pharmaceutical, or clinical instruction terminology.
* **Finance**: Prompts containing stock transactions, account status, auditing, or banking decisions.
* **Legal**: Prompts containing contract enforcement, settlements, NDAs, or regulatory advice.
* **Security**: Prompts containing system permissions, access controls, authenticator validation, or firewall rules.
* **General**: Default fall-back domain.

---

## Risk-Aware Policies

Based on the detected domain, the system applies one of two Risk Tiers:

| Risk Tier | Assigned Domains | MLP Threshold ($T_{\text{MLP}}$) | Cross-Encoder Threshold ($T_{\text{CE}}$) |
| :--- | :--- | :--- | :--- |
| **LOW** | `general`, `translation`, `summarization`, `classification` | `0.90` | `0.85` |
| **HIGH** | `medical`, `finance`, `legal`, `security` | `0.99` | `0.95` |

### Explicit Overrides
If explicit thresholds are provided in the `ClassifierConfig` or `CrossEncoderConfig`, they override the values defined by the risk policy.

---

## Latency-Aware Bypassing

To prevent the performance bottleneck of running a Cross-Encoder forward pass on every request, we introduce an adaptive gateway:
* The default skip threshold is set to $T_{\text{skip}} = 0.995$.
* When $P_{\text{MLP}} \ge 0.995$, the MLP classifier is sufficiently confident that the prompts are semantically equivalent.
* Running the Cross-Encoder in this state is redundant and only adds unnecessary latency (saving ~15-30ms).
* Under high-risk scenarios, users can increase $T_{\text{skip}}$ to `1.0` (effectively disabling the bypass) to ensure the Cross-Encoder validates every candidate.


