"""Public cache facade."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from semanticmemo._logging import get_logger
from semanticmemo.concurrency import InProcessSingleFlightManager, SingleFlightManager
from semanticmemo.embedding import (
    EmbeddingService,
    FaissVectorIndex,
    InMemoryVectorIndex,
    SentenceTransformerEmbeddingProvider,
)
from semanticmemo.embedding.service import VectorIndex
from semanticmemo.entity_change_detection import EntityChangeConfig
from semanticmemo.exceptions import MissingDependencyError
from semanticmemo.models import CacheConfig, CacheResult, CacheStats, ClassifierConfig
from semanticmemo.orchestrator import CacheOrchestrator
from semanticmemo.store import MetadataStore, SQLiteCacheStore
from semanticmemo.types import EmbeddingProvider, LLMFunction

if TYPE_CHECKING:
    from semanticmemo.classifier import ClassifierService, CrossEncoderService
    from semanticmemo.concurrency import RedisConfig
    from semanticmemo.domain_detector import DomainDetector
    from semanticmemo.embedding.qdrant_index import QdrantConfig
    from semanticmemo.store import PostgresConfig

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
        vector_index: VectorIndex | None = None,
        qdrant_config: QdrantConfig | None = None,
        single_flight_manager: SingleFlightManager | None = None,
        redis_config: RedisConfig | None = None,
        classifier: ClassifierConfig | None = None,
        store: MetadataStore | Any | None = None,
        postgres_config: PostgresConfig | None = None,
        use_faiss: bool = True,
        entity_change_config: EntityChangeConfig | None = None,
    ) -> None:
        self.domain = domain
        self.config = config or CacheConfig()

        if store is not None:
            self.store = store
        elif postgres_config is not None:
            from semanticmemo.store import PostgresCacheStore

            self.store = PostgresCacheStore(config=postgres_config)
        else:
            self.store = SQLiteCacheStore(self.config.db_path)
        provider = embedding_provider or SentenceTransformerEmbeddingProvider(
            self.config.embedding_model,
            dim=self.config.embedding_dim,
        )
        if vector_index is not None:
            index = vector_index
        elif qdrant_config is not None:
            from semanticmemo.embedding.qdrant_index import QdrantVectorIndex

            index = QdrantVectorIndex(provider.dim, config=qdrant_config)
        elif use_faiss:
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

        if single_flight_manager is not None:
            sf_manager = single_flight_manager
        elif redis_config is not None:
            from semanticmemo.concurrency import RedisSingleFlightManager

            sf_manager = RedisSingleFlightManager(config=redis_config)
        else:
            sf_manager = InProcessSingleFlightManager()

        self._orchestrator = CacheOrchestrator(
            domain=domain,
            config=self.config,
            store=self.store,
            embedding_service=embedding_service,
            classifier_service=classifier_service,
            cross_encoder_service=cross_encoder_service,
            domain_detector=domain_detector,
            entity_change_config=entity_change_config,
            single_flight_manager=sf_manager,
        )

    @property
    def authz_service(self):
        return self._orchestrator.authz_service

    @property
    def resource_service(self):
        return self._orchestrator.resource_service

    @property
    def scope_resolver(self):
        return self._orchestrator.scope_resolver

    async def get_or_call(
        self,
        *,
        prompt: str,
        llm_function: LLMFunction,
        user_id: str | None = None,
        tenant_id: str | None = None,
        resource_id: str | None = None,
        prefer_global: bool = False,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheResult:
        """Lookup cached response or execute LLM with RBAC and scoped authorization.

        Note on `tenant_id`: For security, tenant context is always authoritatively
        derived server-side from the authenticated principal (`user.tenant_id`) rather
        than client request arguments. The `tenant_id` parameter is accepted for legacy
        compatibility and default user seeding only.
        """
        return await self._orchestrator.get_or_call(
            prompt=prompt,
            llm_function=llm_function,
            user_id=user_id,
            tenant_id=tenant_id,
            resource_id=resource_id,
            prefer_global=prefer_global,
            model=model,
            metadata=metadata,
        )

    def bootstrap_default_user(
        self,
        user_id: str = "default_user",
        tenant_id: str = "default",
        role: str = "analyst",
    ):
        """Explicit development/testing bootstrap helper to seed an authenticated principal."""
        return self._orchestrator.bootstrap_default_user(
            user_id=user_id,
            tenant_id=tenant_id,
            role=role,
        )

    def register_resource(
        self,
        resource_id: str,
        tenant_id: str = "default",
        initial_version: int = 1,
    ):
        """Register a versioned resource belonging to a tenant."""
        return self.resource_service.register_resource(
            resource_id=resource_id,
            tenant_id=tenant_id,
            initial_version=initial_version,
        )

    def invalidate_resource(self, resource_id: str) -> int:
        """Atomically invalidate a resource by incrementing its version."""
        return self.resource_service.invalidate_resource(resource_id)

    def create_user(
        self,
        user_id: str,
        tenant_id: str = "default",
        role: str = "viewer",
    ):
        """Create a user, associate with a tenant, and assign an initial role."""
        user = self.store.create_user(user_id=user_id, tenant_id=tenant_id)
        self.store.assign_role_to_user(user_id=user_id, role_id=role)
        return user

    def assign_role(self, user_id: str, role_id: str) -> None:
        """Assign an RBAC role to a user."""
        self.store.assign_role_to_user(user_id=user_id, role_id=role_id)

    def remove_role(self, user_id: str, role_id: str) -> None:
        """Remove an RBAC role from a user."""
        self.store.remove_role_from_user(user_id=user_id, role_id=role_id)

    def grant_resource_access(self, resource_id: str, role_id: str, permission_id: str) -> None:
        """Grant a role explicit permission on a resource."""
        self.store.grant_resource_access(
            resource_id=resource_id,
            role_id=role_id,
            permission_id=permission_id,
        )

    def revoke_resource_access(self, resource_id: str, role_id: str, permission_id: str) -> None:
        """Revoke a role's explicit permission on a resource."""
        self.store.revoke_resource_access(
            resource_id=resource_id,
            role_id=role_id,
            permission_id=permission_id,
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
        from semanticmemo.classifier import ClassifierService

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
        from semanticmemo.classifier import CrossEncoderService

        return CrossEncoderService(
            model_name=self.config.cross_encoder.model_name,
            device=self.config.cross_encoder.device,
            threshold=self.config.cross_encoder.threshold,
        )

    def _build_domain_detector(self, embedding_service: EmbeddingService) -> DomainDetector | None:
        if self.config.risk_policy is None:
            return None
        from semanticmemo.domain_detector import DomainDetector

        return DomainDetector(embedding_service)

    @property
    def orchestrator(self) -> CacheOrchestrator:
        """Access underlying CacheOrchestrator instance."""
        return self._orchestrator
