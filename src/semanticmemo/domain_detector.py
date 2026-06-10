"""Centroid-based domain auto-detection service."""

from __future__ import annotations

import numpy as np

from semanticmemo._logging import get_logger
from semanticmemo.embedding.service import EmbeddingService
from semanticmemo.types import FloatVector

logger = get_logger(__name__)

REPRESENTATIVE_PROMPTS: dict[str, list[str]] = {
    "medical": [
        "What are the common side effects of ibuprofen?",
        "List symptoms and diagnostic criteria for type 2 diabetes.",
        "What is the recommended dosage for pediatric amoxicillin?",
        "Explain the pharmaceutical interactions of aspirin.",
        "List side effects for ibuprofen.",
        "Increase dosage to 50mg",
        "Decrease dosage to 50mg",
    ],
    "finance": [
        "What is my current bank account balance?",
        "Buy 100 shares of Apple stock.",
        "Show me my recent credit card transactions.",
        "Transfer 500 dollars to my savings account.",
        "Show me my account balance.",
        "Buy 100 shares of stock X",
        "Sell 100 shares of stock X",
        "Provide an analysis of Apple's Q3 earnings report.",
        "Show me the quarterly financial earnings report.",
    ],
    "legal": [
        "Review this non-disclosure agreement for liability clauses.",
        "What are the terms of service and termination policies?",
        "Draft a contract amendment for the software license.",
        "What is the legal definition of intellectual property?",
    ],
    "security": [
        "Grant administrator access to user account.",
        "Configure the firewall to block incoming traffic on port 80.",
        "Reset my authentication credentials and security key.",
        "How do I audit user access logs for unauthorized attempts.",
        "Allow access to administrative panel",
        "Grant write permissions to team members",
        "How do I reset my password on this account?",
        "Reset user password.",
    ],
}


class DomainDetector:
    """Classifies prompts into semantic domains using embedding centroids.

    Pluggable design default. Centroids are dynamically built from representative
    prompts in each target domain.
    """

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service
        self.centroids: dict[str, np.ndarray] = {}
        self._compute_centroids()

    def _compute_centroids(self) -> None:
        for domain, prompts in REPRESENTATIVE_PROMPTS.items():
            embeddings = [self.embedding_service.embed(p) for p in prompts]
            centroid = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            self.centroids[domain] = centroid
        logger.debug("Computed centroids for domains: %s", list(self.centroids.keys()))

    def detect(self, query_embedding: FloatVector, confidence_threshold: float = 0.5) -> str:
        """Detect the domain of a prompt based on its embedding.

        Returns one of: 'medical', 'finance', 'legal', 'security', or 'general'.
        """
        q_vec = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        best_domain = "general"
        best_score = -1.0

        for domain, centroid in self.centroids.items():
            score = float(np.dot(q_vec, centroid))
            if score > best_score:
                best_score = score
                best_domain = domain

        if best_score < confidence_threshold:
            logger.debug(
                "Domain confidence %.4f below threshold %.4f, falling back to 'general'",
                best_score,
                confidence_threshold,
            )
            return "general"

        logger.debug("Detected domain '%s' with confidence %.4f", best_domain, best_score)
        return best_domain
