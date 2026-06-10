# SemanticMemo Benchmark Methodology

This document outlines the evaluation framework, metrics, and dataset designs used to measure SemanticMemo cache efficiency.

---

## Evaluation Metrics

To evaluate and compare caching models, the benchmark suite computes the following metrics:

1. **Precision**: Measures correctness. What fraction of predicted cache hits are actually semantically equivalent?
   $$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$
   *Crucial for production systems to avoid serving incorrect responses.*
2. **Recall**: Measures cache hit potential. What fraction of equivalent prompt pairs did the cache correctly identify?
   $$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$
   *Directly correlates with overall cache hit rate.*
3. **F1-Score**: Harmonic mean of Precision and Recall.
4. **Latency (ms)**: Measures cache verification overhead compared to raw LLM calls.
5. **False Positive Rate (FPR)**: Rate at which dangerous opposite-action prompts are incorrectly served.
6. **Estimated Cost Saved ($)**: Total dollar amount saved based on saved input/output tokens.

---

## Evaluation Datasets

The evaluation suite utilizes benchmark datasets across four core high-stakes domains:

1. **Customer Support**:
   * Equivalent: "Check if the item has shipped" / "Has my order been sent out yet?"
   * Opposite-action: "Approve user refund" / "Deny user refund"
2. **Finance**:
   * Equivalent: "What is my current bank account balance?" / "Show me my account balance."
   * Opposite-action: "Buy 100 shares of stock X" / "Sell 100 shares of stock X"
3. **Medical**:
   * Equivalent: "What are the common side effects of ibuprofen?" / "List side effects for ibuprofen."
   * Opposite-action: "Increase dosage to 50mg" / "Decrease dosage to 50mg"
4. **Security**:
   * Equivalent: "Grant write permissions to team members" / "Allow writing for team."
   * Opposite-action: "Allow access to administrative panel" / "Block access to administrative panel"

---

## Comparison Models

We score the following decision models to establish a clear progression:

1. **Cosine Baseline**: Naive embedding distance using `all-MiniLM-L6-v2` at a fixed threshold of `0.90`.
2. **MLP Classifier Baseline**: Stage-1 neural network using a fixed threshold of `0.85`.
3. **Double Verification**: Fixed combination of MLP classifier (`0.95`) and Cross-Encoder (defaulting to `cross-encoder/ms-marco-MiniLM-L-6-v2`, with capacity to evaluate alternatives like `cross-encoder/ms-marco-MiniLM-L-12-v2` or `cross-encoder/stsb-roberta-base`) at threshold `0.90` without adaptive policies.
4. **SemanticMemo**: Complete system featuring Double Verification, Domain Auto-Detection, Risk-Aware Policies, and Latency-Aware Bypassing.

---

## Baseline Comparison Matrix

The final report generates a comparative summary in the following format:

| Method | Domain | Precision | Recall | F1 | Latency (ms) | FPR | Cost Saved ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Cosine Baseline | Medical | ... | ... | ... | ... | ... | ... |
| MLP Classifier | Medical | ... | ... | ... | ... | ... | ... |
| Double Verification | Medical | ... | ... | ... | ... | ... | ... |
| SemanticMemo | Medical | ... | ... | ... | ... | ... | ... |


