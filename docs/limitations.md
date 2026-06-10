# SemanticMemo: Limitations and Future Work

As SemanticMemo evolves from a naive embedding cache into a multi-layered production-grade semantic caching pipeline, understanding its bounds and failure modes is critical for senior engineers and ML teams.

This document serves as an honest, benchmark-driven analysis of what semantic drift categories SemanticMemo currently solves, where it faces limitations, and our roadmap for addressing these gaps.

---

## The Verification Pipeline at a Glance

```
Incoming Query
    │
    ▼
1. Embedding (FAISS search) ───► High-Recall Retrieval (O(log N))
    │
    ▼
2. MLP Classifier ─────────────► Learned Similarity Scoring (O(K))
    │
    ▼
3. Cross-Encoder ──────────────► Deep Attention Re-ranking (Domain-Conditioned)
    │
    ▼
4. EntityChangeDetector ───────► Regex Drift Gates (<1ms, rule-based)
    │
    ▼
Decision: HIT or MISS
```

---

## Solved: Entity Drift (Rule-Based Gates)

Semantic drift involving specific, identifiable entity changes is **fully resolved** in SemanticMemo v1.2.0 via the `EntityChangeDetector`. This gate executes 11 high-performance, regex-based validation checks in `< 1ms` to intercept differences that neural models often overlook.

| Category | Example Target | Example Query | Drift Status |
| :--- | :--- | :--- | :--- |
| **Quarter Drift** | `Q3` vs `Q4` | "Get Q3 earnings" vs "Get Q4 earnings" | **Solved** (Blocked as `MISS`) |
| **Drug Drift** | `ibuprofen` vs `acetaminophen` | "Dosage for ibuprofen" vs "Dosage for acetaminophen" | **Solved** (Blocked as `MISS`) |
| **Year Drift** | `2023` vs `2024` | "Tax rules for 2023" vs "Tax rules for 2024" | **Solved** (Blocked as `MISS`) |
| **Numeric Drift** | `100mg` vs `200mg`, `top 5` vs `top 10` | "Top 5 stocks" vs "Top 10 stocks" | **Solved** (Blocked as `MISS`) |
| **Privilege Drift** | `my` vs `administrator` | "Reset my password" vs "Reset administrator password" | **Solved** (Blocked as `MISS`) |
| **Temporal Drift** | `current` vs `historical` | "Current stock price" vs "Historical stock price" | **Solved** (Blocked as `MISS`) |
| **Proper Nouns** | `Apple` vs `Tesla` | "Analyze Apple's revenue" vs "Analyze Tesla's revenue" | **Solved** (Blocked as `MISS`) |
| **Version Drift** | `v1.2` vs `v1.3` | "Install v1.2" vs "Install v1.3" | **Solved** (Blocked as `MISS`) |
| **Months/Days** | `January` vs `March` | "Report for January" vs "Report for March" | **Solved** (Blocked as `MISS`) |
| **Ordinal Drift** | `first` vs `second` | "Read the first chapter" vs "Read the second chapter" | **Solved** (Blocked as `MISS`) |

---

## Partially Solved or Unsolved

Neural embeddings, MLP classifiers, and Cross-encoders are trained to group queries by *semantic intent*. While this is powerful, it creates gaps when queries share the same core intent but differ in **scope, audience, or styling**.

### 1. Scope and Formatting Drift
* **The Problem:** The user requests the same information but in a different format or length.
  * *Example:* "Summarize this article" vs. "Summarize this article in 3 bullet points" or "Summarize this article in markdown".
  * *Current Behavior:* The MLP and Cross-Encoder see identical context and task verbs, resulting in high similarity. If the numbers do not trigger a numeric check (e.g., "markdown" vs "plain text"), this is classified as a `HIT`.
  * *Why it's a limitation:* Serving a plain text cache to a markdown request breaks the application's UI formatting expectations.

### 2. Audience and Persona Drift
* **The Problem:** The query asks for the same core answer but tailors it to a different target demographic.
  * *Example:* "Explain quantum computing to a college student" vs. "Explain quantum computing to a 5-year-old".
  * *Current Behavior:* The systems score highly on similarity due to "quantum computing" and "explain". If the age difference doesn't flag a standard detector, it might be served as a `HIT`.
  * *Why it's a limitation:* Serving a dense, academic explanation to a child fails the prompt's structural constraint.

### 3. Instruction Drift
* **The Problem:** Small instruction modifiers that change constraints without altering main entities.
  * *Example:* "Suggest 5 recipes without nuts" vs. "Suggest 5 recipes without dairy".
  * *Current Behavior:* The MLP/Cross-Encoder correctly penalizes differences if trained on negatives, but if they are out-of-domain words, the similarity may still surpass the general domain thresholds.

---

## Future Work and Roadmap

To address these limitations without introducing unmanageable ML inference latency, the following architectural upgrades are proposed:

### 1. Scope & Style Guard
* **Concept:** A lightweight post-retrieval parser that compares the output format constraints (e.g., `markdown`, `json`, `csv`, `bullet points`) between the source query and target cache entry.
* **Mechanism:** Syntax-matching regex or a micro-classifier trained specifically to extract target formatting from prompts.

### 2. Task Intent Classifier
* **Concept:** A small, fast classifier trained on a taxonomy of prompt tasks (e.g., `Summarization`, `Extraction`, `QA`, `Generation`, `Translation`).
* **Mechanism:** Cache hits are only permitted if both queries map to the exact same task intent class, preventing cross-intent reuse.

### 3. Active Learning for Scope Drift
* **Concept:** Leveraging the v1.1 Active Learning Dataset Builder to capture scope drift failures.
* **Mechanism:** When a developer flags a bad hit due to formatting drift (e.g., missing markdown), that pair is recorded as a hard negative, and the MLP is retrained to learn that formatting-specific tokens (e.g., `markdown`, `json`) have high weights in equivalence scoring.
