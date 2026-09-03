from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from semanticmemo.embedding.service import normalize
from semanticmemo.exceptions import MissingDependencyError
from semanticmemo.models import CacheScope, RetrievalScope
from semanticmemo.types import FloatVector

if TYPE_CHECKING:
    from semanticmemo.embedding.service import SearchCandidate

logger = logging.getLogger(__name__)

try:
    import qdrant_client
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PayloadSchemaType,
        PointStruct,
        VectorParams,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    qdrant_client = None  # type: ignore[assignment]
    QdrantClient = None  # type: ignore[assignment, misc]


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector backend."""

    url: str | None = None
    api_key: str | None = None
    host: str | None = None
    port: int | None = None
    location: str | None = None  # e.g. ":memory:" for in-memory testing
    path: str | None = None  # e.g. local directory path
    collection_name: str = "semanticmemo_vectors"
    timeout: int | None = 5
    prefer_grpc: bool = False
    https: bool | None = None


def _build_qdrant_filter(retrieval_scope: RetrievalScope) -> Any:
    """Translate a mandatory RetrievalScope into a Qdrant query filter.

    Enforces:
    - RESOURCE: Must match tenant_id, scope=RESOURCE, resource_id,
      resource_version, and pipeline_version. Fails closed if missing.
    - TENANT: Allows (scope=TENANT AND tenant_id=current_tenant) OR (scope=GLOBAL),
      always requiring pipeline_version. RESOURCE entries never match.
    - GLOBAL: Allows only scope=GLOBAL and matching pipeline_version.
    """
    if not QDRANT_AVAILABLE:
        raise MissingDependencyError(
            "qdrant-client is required for Qdrant filtering. "
            "Install with `pip install semanticmemo[qdrant]`."
        )

    scope_enum = (
        retrieval_scope.scope
        if isinstance(retrieval_scope.scope, CacheScope)
        else CacheScope(retrieval_scope.scope)
    )

    if scope_enum == CacheScope.RESOURCE:
        # Require all resource-bound invariants; fail closed if missing
        if not retrieval_scope.resource_id or retrieval_scope.resource_version is None:
            logger.warning(
                "Resource scope query missing resource_id or resource_version; failing closed."
            )
            # Return an impossible filter that matches zero points
            must_conds: list[Any] = [
                FieldCondition(key="scope", match=MatchValue(value="impossible_scope_match")),
            ]
            return Filter(must=must_conds)

        res_must: list[Any] = [
            FieldCondition(key="tenant_id", match=MatchValue(value=retrieval_scope.tenant_id)),
            FieldCondition(key="scope", match=MatchValue(value=CacheScope.RESOURCE.value)),
            FieldCondition(key="resource_id", match=MatchValue(value=retrieval_scope.resource_id)),
            FieldCondition(
                key="resource_version", match=MatchValue(value=retrieval_scope.resource_version)
            ),
            FieldCondition(
                key="pipeline_version", match=MatchValue(value=retrieval_scope.pipeline_version)
            ),
        ]
        return Filter(must=res_must)

    if scope_enum == CacheScope.TENANT:
        # Allow tenant-private entries for this tenant OR global entries.
        # Excludes resource entries and other tenants.
        tenant_must: list[Any] = [
            FieldCondition(
                key="pipeline_version", match=MatchValue(value=retrieval_scope.pipeline_version)
            ),
        ]
        tenant_branch: list[Any] = [
            FieldCondition(key="scope", match=MatchValue(value=CacheScope.TENANT.value)),
            FieldCondition(key="tenant_id", match=MatchValue(value=retrieval_scope.tenant_id)),
        ]
        global_branch: list[Any] = [
            FieldCondition(key="scope", match=MatchValue(value=CacheScope.GLOBAL.value)),
        ]
        should_branches: list[Any] = [
            Filter(must=tenant_branch),
            Filter(must=global_branch),
        ]
        return Filter(must=tenant_must, should=should_branches)

    if scope_enum == CacheScope.GLOBAL:
        # Only global entries matching current pipeline_version
        global_must: list[Any] = [
            FieldCondition(key="scope", match=MatchValue(value=CacheScope.GLOBAL.value)),
            FieldCondition(
                key="pipeline_version", match=MatchValue(value=retrieval_scope.pipeline_version)
            ),
        ]
        return Filter(must=global_must)

    # Unknown scope -> fail closed
    logger.warning("Unrecognized scope %s; failing closed.", scope_enum)
    fallback_must: list[Any] = [
        FieldCondition(key="scope", match=MatchValue(value="impossible_scope_match")),
    ]
    return Filter(must=fallback_must)


class QdrantVectorIndex:
    """Production Qdrant vector backend conforming to the RetrievalScope interface.

    Executes pre-filtering directly inside Qdrant via `query_points(query_filter=...)`
    to guarantee zero candidate contamination across tenants and resources.
    """

    def __init__(
        self,
        dim: int,
        client: Any | None = None,
        config: QdrantConfig | None = None,
        auto_create_collection: bool = True,
    ) -> None:
        if not QDRANT_AVAILABLE or QdrantClient is None:
            raise MissingDependencyError(
                "qdrant-client is required for QdrantVectorIndex. "
                "Install with `pip install semanticmemo[qdrant]` or `pip install qdrant-client`."
            )

        self.dim = dim
        self.config = config or QdrantConfig()
        self.collection_name = self.config.collection_name

        if client is not None:
            self._client = client
        else:
            if self.config.location == ":memory:" or self.config.path == ":memory:":
                self._client = QdrantClient(location=":memory:")
            elif self.config.path:
                self._client = QdrantClient(path=self.config.path)
            elif self.config.url:
                self._client = QdrantClient(
                    url=self.config.url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    prefer_grpc=self.config.prefer_grpc,
                    https=self.config.https,
                )
            elif self.config.host:
                self._client = QdrantClient(
                    host=self.config.host,
                    port=self.config.port or 6333,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    prefer_grpc=self.config.prefer_grpc,
                    https=self.config.https,
                )
            else:
                # Default to in-memory instance for testing
                self._client = QdrantClient(location=":memory:")

        if auto_create_collection:
            self.ensure_collection_and_indexes()

    def ensure_collection_and_indexes(self) -> None:
        """Ensure collection exists with correct dimension, cosine metric, and payload indexes."""
        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            logger.info(
                "Creating Qdrant collection '%s' (dim=%d, distance=COSINE)",
                self.collection_name,
                self.dim,
            )
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

            # Create payload indexes for selective query planning
            fields_to_index = [
                ("tenant_id", PayloadSchemaType.KEYWORD),
                ("scope", PayloadSchemaType.KEYWORD),
                ("resource_id", PayloadSchemaType.KEYWORD),
                ("resource_version", PayloadSchemaType.INTEGER),
                ("pipeline_version", PayloadSchemaType.KEYWORD),
            ]
            for field_name, schema_type in fields_to_index:
                try:
                    import warnings

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=".*Payload indexes have no effect in the local Qdrant.*",
                        )
                        self._client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field_name,
                            field_schema=schema_type,
                        )
                except Exception as exc:
                    logger.debug("Could not create payload index for '%s': %s", field_name, exc)

    def add(
        self,
        entry_id: UUID,
        embedding: FloatVector,
        tenant_id: str = "default",
        scope: CacheScope | str = CacheScope.GLOBAL,
        resource_id: str | None = None,
        resource_version: int | None = None,
        pipeline_version: str = "v2",
    ) -> None:
        """Upsert a vector point with its authorization scope metadata into Qdrant."""
        cscope = CacheScope(scope) if isinstance(scope, str) else scope
        norm_vec = normalize(embedding).tolist()

        payload = {
            "tenant_id": tenant_id,
            "scope": cscope.value,
            "resource_id": resource_id,
            "resource_version": resource_version,
            "pipeline_version": pipeline_version,
        }

        point = PointStruct(
            id=str(entry_id),
            vector=norm_vec,
            payload=payload,
        )

        self._client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    def remove(self, entry_id: UUID) -> None:
        """Delete a vector point by its entry_id."""
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=[str(entry_id)],
        )

    def rebuild(self, items: list[Any]) -> None:
        """Rebuild collection with provided items, preserving scope metadata."""
        # Clear existing collection
        with contextlib.suppress(Exception):
            self._client.delete_collection(self.collection_name)
        self.ensure_collection_and_indexes()

        if not items:
            return

        points: list[PointStruct] = []
        for item in items:
            if isinstance(item, tuple):
                if len(item) == 2:
                    entry_id, embedding = item
                    # Unscoped legacy entry defaults to global
                    points.append(
                        PointStruct(
                            id=str(entry_id),
                            vector=normalize(embedding).tolist(),
                            payload={
                                "tenant_id": "default",
                                "scope": CacheScope.GLOBAL.value,
                                "resource_id": None,
                                "resource_version": None,
                                "pipeline_version": "v2",
                            },
                        )
                    )
                elif len(item) == 7:
                    entry_id, embedding, tenant_id, scope, res_id, res_ver, pipe_ver = item
                    cscope = CacheScope(scope) if isinstance(scope, str) else scope
                    points.append(
                        PointStruct(
                            id=str(entry_id),
                            vector=normalize(embedding).tolist(),
                            payload={
                                "tenant_id": tenant_id,
                                "scope": cscope.value,
                                "resource_id": res_id,
                                "resource_version": res_ver,
                                "pipeline_version": pipe_ver,
                            },
                        )
                    )

        if points:
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search(
        self,
        query_embedding: FloatVector,
        k: int,
        retrieval_scope: RetrievalScope,
    ) -> list[SearchCandidate]:
        """Perform pre-filtered vector retrieval in Qdrant constrained by RetrievalScope.

        Never participates unauthorized or mismatched vectors in similarity scoring.
        """
        from semanticmemo.embedding.service import SearchCandidate

        norm_query = normalize(query_embedding).tolist()
        q_filter = _build_qdrant_filter(retrieval_scope)

        response = self._client.query_points(
            collection_name=self.collection_name,
            query=norm_query,
            query_filter=q_filter,
            limit=k,
            with_payload=False,
        )

        candidates: list[SearchCandidate] = []
        for point in response.points:
            candidates.append(
                SearchCandidate(
                    entry_id=UUID(str(point.id)),
                    score=float(point.score),
                )
            )

        return candidates
