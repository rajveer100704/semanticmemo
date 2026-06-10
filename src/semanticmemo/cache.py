"""Public cache facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from semanticmemo._logging import get_logger
from semanticmemo.classifier import ClassifierService, CrossEncoderService
from semanticmemo.domain_detector import DomainDetector
from semanticmemo.embedding import (
    EmbeddingService,
    FaissVectorIndex,
    InMemoryVectorIndex,
    SentenceTransformerEmbeddingProvider,
)
from semanticmemo.entity_change_detection import EntityChangeConfig
from semanticmemo.exceptions import MissingDependencyError
from semanticmemo.models import CacheConfig, CacheResult, CacheStats, ClassifierConfig
from semanticmemo.orchestrator import CacheOrchestrator
from semanticmemo.store import SQLiteCacheStore
from semanticmemo.types import EmbeddingProvider, LLMFunction

logger = get_logger(__name__)


class SemanticMemo:
    """Async-first semantic cache facade.

    Without a classifier checkpoint, SemanticMemo uses cosine similarity as a
    measured baseline. With a classifier checkpoint, cosine search selects
    candidates and the learned classifier decides cache hits.
    """

    def __init__(
        self,
        *,
        domain: str,
        config: CacheConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        classifier: ClassifierConfig | None = None,
        store: SQLiteCacheStore | None = None,
        use_faiss: bool = True,
        entity_change_config: EntityChangeConfig | None = None,
    ) -> None:
        self.domain = domain
        self.config = config or CacheConfig()
        self.store = store or SQLiteCacheStore(self.config.db_path)
        provider = embedding_provider or SentenceTransformerEmbeddingProvider(
            self.config.embedding_model,
            dim=self.config.embedding_dim,
        )
        if use_faiss:
            try:
                index = FaissVectorIndex(provider.dim)
            except MissingDependencyError as exc:
                logger.warning("faiss unavailable, falling back to InMemoryVectorIndex: %s", exc)
                index = InMemoryVectorIndex(provider.dim)
        else:
            index = InMemoryVectorIndex(provider.dim)
        embedding_service = EmbeddingService(provider, index)
        classifier_service = self._build_classifier_service(classifier)
        cross_encoder_service = self._build_cross_encoder_service()
        domain_detector = self._build_domain_detector(embedding_service)
        self._orchestrator = CacheOrchestrator(
            domain=domain,
            config=self.config,
            store=self.store,
            embedding_service=embedding_service,
            classifier_service=classifier_service,
            cross_encoder_service=cross_encoder_service,
            domain_detector=domain_detector,
            entity_change_config=entity_change_config,
        )

    async def get_or_call(
        self,
        *,
        prompt: str,
        llm_function: LLMFunction,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheResult:
        return await self._orchestrator.get_or_call(
            prompt=prompt,
            llm_function=llm_function,
            model=model,
            metadata=metadata,
        )

    async def report_bad_hit(self, query_id: UUID, reason: str | None = None) -> bool:
        return self._orchestrator.report_bad_hit(query_id, reason=reason)

    async def report_good_hit(self, query_id: UUID) -> bool:
        return self._orchestrator.report_good_hit(query_id)

    def stats(self) -> CacheStats:
        return self._orchestrator.stats()

    def export_feedback_pairs(self, path: Path | str, *, split: str = "train") -> int:
        return self._orchestrator.export_feedback_pairs(str(path), split=split)

    def export_active_learning_pairs(self, path: Path | str, *, split: str = "train") -> int:
        return self.store.export_active_learning_pairs(str(path), split=split)

    def close(self) -> None:
        """Close the underlying store. Safe to call more than once."""
        self.store.close()
        logger.debug("SemanticMemo closed: domain=%s", self.domain)

    async def __aenter__(self) -> SemanticMemo:
        """Enter an ``async with`` block; returns this instance."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit an ``async with`` block, closing the store. Exceptions propagate."""
        self.close()

    def _build_classifier_service(
        self,
        classifier: ClassifierConfig | None,
    ) -> ClassifierService | None:
        if classifier is None or classifier.model_path is None:
            return None
        threshold = (
            classifier.threshold
            if classifier.threshold is not None
            else self.config.classifier_threshold
        )
        return ClassifierService(
            classifier.model_path,
            device=classifier.device,
            threshold=threshold,
        )

    def _build_cross_encoder_service(self) -> CrossEncoderService | None:
        if self.config.cross_encoder is None:
            return None
        return CrossEncoderService(
            model_name=self.config.cross_encoder.model_name,
            device=self.config.cross_encoder.device,
            threshold=self.config.cross_encoder.threshold,
        )

    def _build_domain_detector(self, embedding_service: EmbeddingService) -> DomainDetector | None:
        if self.config.risk_policy is None:
            return None
        return DomainDetector(embedding_service)
