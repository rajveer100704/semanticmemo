from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from semanticmemo.classifier.cross_encoder_service import CrossEncoderService
from semanticmemo.domain_detector import REPRESENTATIVE_PROMPTS, DomainDetector
from semanticmemo.embedding import EmbeddingService
from semanticmemo.models import (
    CacheConfig,
    RiskPolicy,
    RiskTier,
)
from semanticmemo.orchestrator import CacheOrchestrator
from semanticmemo.store import SQLiteCacheStore


def test_domain_detector():
    # Mock embedding service that returns deterministic vectors
    mock_embedding_service = MagicMock(spec=EmbeddingService)

    # Let's mock embed to return a normalized vector matching the domain index
    # medical -> index 0, finance -> index 1, etc.
    domain_to_idx = {"medical": 0, "finance": 1, "legal": 2, "security": 3}

    def mock_embed(text):
        vec = np.zeros(384, dtype=np.float32)
        for domain, idx in domain_to_idx.items():
            if text in REPRESENTATIVE_PROMPTS.get(domain, []):
                vec[idx] = 1.0
                return vec / np.linalg.norm(vec)
        # fallback
        vec[99] = 1.0
        return vec

    mock_embedding_service.embed.side_effect = mock_embed

    detector = DomainDetector(mock_embedding_service)

    # Verify centroids were populated
    assert len(detector.centroids) == 4
    for domain in domain_to_idx:
        assert domain in detector.centroids

    # Test detection of medical query
    medical_vec = np.zeros(384, dtype=np.float32)
    medical_vec[0] = 1.0
    assert detector.detect(medical_vec, confidence_threshold=0.5) == "medical"

    # Test detection of general query
    general_vec = np.zeros(384, dtype=np.float32)
    general_vec[99] = 1.0
    assert detector.detect(general_vec, confidence_threshold=0.5) == "general"


@patch("semanticmemo.classifier.cross_encoder_service.CrossEncoder")
def test_cross_encoder_service(mock_cross_encoder_class):
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([1.386294361])  # sigmoid(1.386...) ~ 0.8
    mock_cross_encoder_class.return_value = mock_ce

    service = CrossEncoderService(model_name="dummy-ce-model", device="cpu", threshold=0.85)

    # Test predict
    score = service.predict("hello", "world")
    assert pytest.approx(score, 0.01) == 0.8

    # Test is_equivalent
    assert not service.is_equivalent("hello", "world")

    # With a lower threshold
    service.threshold = 0.75
    assert service.is_equivalent("hello", "world")


@pytest.mark.anyio
async def test_orchestrator_double_verification_flow():
    # Setup mocks
    mock_store = MagicMock(spec=SQLiteCacheStore)
    mock_embedding = MagicMock(spec=EmbeddingService)
    mock_classifier = MagicMock()
    mock_cross_encoder = MagicMock(spec=CrossEncoderService)
    mock_cross_encoder.threshold = None
    mock_domain_detector = MagicMock(spec=DomainDetector)

    # We mock search to return one candidate
    entry_id = uuid4()
    from semanticmemo.embedding.service import SearchCandidate

    mock_embedding.search.return_value = [SearchCandidate(entry_id=entry_id, score=0.95)]

    # Store gets the entry
    from datetime import datetime

    from semanticmemo.models import CacheEntry

    mock_entry = CacheEntry(
        id=entry_id,
        prompt="cached prompt",
        prompt_embedding=[0.1] * 384,
        response="cached response",
        created_at=datetime.now(),
    )
    mock_store.get.return_value = mock_entry
    mock_store.add.return_value = entry_id

    # Configure risk policy
    risk_policy = RiskPolicy(
        domain_tiers={"medical": RiskTier.HIGH, "general": RiskTier.LOW},
        low_risk_classifier_threshold=0.90,
        low_risk_cross_encoder_threshold=0.85,
        high_risk_classifier_threshold=0.99,
        high_risk_cross_encoder_threshold=0.95,
    )
    config = CacheConfig(risk_policy=risk_policy, high_precision_skip_threshold=0.995)

    orchestrator = CacheOrchestrator(
        domain="medical",
        config=config,
        store=mock_store,
        embedding_service=mock_embedding,
        classifier_service=mock_classifier,
        cross_encoder_service=mock_cross_encoder,
        domain_detector=mock_domain_detector,
    )

    # 1. MLP below threshold -> MISS
    mock_domain_detector.detect.return_value = "medical"  # High risk
    mock_classifier.threshold = None
    mock_classifier.predict_batch.return_value = [0.95]  # threshold is 0.99 for high-risk

    res = await orchestrator.get_or_call(
        prompt="new prompt",
        llm_function=lambda p: "fresh response",
    )
    assert not res.was_cache_hit
    assert res.decision.decision == "miss"
    assert res.decision.reason == "failed_mlp_threshold"

    # 2. MLP above skip threshold -> HIT (bypass CE)
    mock_classifier.predict_batch.return_value = [0.996]  # above skip_threshold 0.995
    mock_cross_encoder.predict.reset_mock()

    res = await orchestrator.get_or_call(
        prompt="new prompt",
        llm_function=lambda p: "fresh response",
    )
    assert res.was_cache_hit
    assert res.decision.decision == "hit"
    assert res.decision.reason == "mlp_bypass"
    mock_cross_encoder.predict.assert_not_called()

    # 3. MLP above threshold but below skip threshold, CE fails -> MISS
    mock_classifier.predict_batch.return_value = [0.991]  # above 0.99, below 0.995
    mock_cross_encoder.predict.return_value = 0.90  # threshold is 0.95 for high-risk

    res = await orchestrator.get_or_call(
        prompt="new prompt",
        llm_function=lambda p: "fresh response",
    )
    assert not res.was_cache_hit
    assert res.decision.decision == "miss"
    assert res.decision.reason == "failed_cross_encoder_threshold"

    # 4. MLP above threshold, CE passes -> HIT
    mock_cross_encoder.predict.return_value = 0.97  # above 0.95

    res = await orchestrator.get_or_call(
        prompt="new prompt",
        llm_function=lambda p: "fresh response",
    )
    assert res.was_cache_hit
    assert res.decision.decision == "hit"
    assert res.decision.reason == "passed_all_thresholds"


@pytest.mark.anyio
async def test_orchestrator_domain_conditioned_thresholds():
    mock_store = MagicMock(spec=SQLiteCacheStore)
    mock_embedding = MagicMock(spec=EmbeddingService)
    mock_classifier = MagicMock()
    mock_cross_encoder = MagicMock(spec=CrossEncoderService)
    mock_cross_encoder.threshold = None
    mock_domain_detector = MagicMock(spec=DomainDetector)

    entry_id = uuid4()
    from semanticmemo.embedding.service import SearchCandidate
    mock_embedding.search.return_value = [SearchCandidate(entry_id=entry_id, score=0.95)]

    from datetime import datetime

    from semanticmemo.models import CacheEntry
    mock_entry = CacheEntry(
        id=entry_id,
        prompt="cached prompt",
        prompt_embedding=[0.1] * 384,
        response="cached response",
        created_at=datetime.now(),
    )
    mock_store.get.return_value = mock_entry
    mock_store.add.return_value = entry_id

    # Configure risk policy with domain_thresholds
    risk_policy = RiskPolicy(
        domain_tiers={"medical": RiskTier.HIGH},
        domain_thresholds={"medical": {"mlp": 0.985, "cross_encoder": 0.93}},
        low_risk_classifier_threshold=0.90,
        low_risk_cross_encoder_threshold=0.85,
        high_risk_classifier_threshold=0.99,
        high_risk_cross_encoder_threshold=0.95,
    )
    config = CacheConfig(risk_policy=risk_policy, high_precision_skip_threshold=0.995)

    orchestrator = CacheOrchestrator(
        domain="medical",
        config=config,
        store=mock_store,
        embedding_service=mock_embedding,
        classifier_service=mock_classifier,
        cross_encoder_service=mock_cross_encoder,
        domain_detector=mock_domain_detector,
    )

    mock_domain_detector.detect.return_value = "medical"
    mock_classifier.threshold = None

    # MLP score is 0.990, which is >= 0.985 (domain mlp threshold), but < 0.99 (high risk default).
    # Since domain_threshold overrides, it should pass MLP.
    mock_classifier.predict_batch.return_value = [0.990]

    # CE score is 0.940, which is >= 0.93 (domain ce threshold), but < 0.95 (high risk default).
    # Since domain_threshold overrides, it should pass CE and result in a HIT.
    mock_cross_encoder.predict.return_value = 0.940

    res = await orchestrator.get_or_call(
        prompt="new prompt",
        llm_function=lambda p: "fresh response",
    )
    assert res.was_cache_hit
    assert res.decision.decision == "hit"
    assert res.decision.reason == "passed_all_thresholds"

    # Now let's test a disagreement scenario to verify active learning logging
    mock_cross_encoder.predict.return_value = 0.920  # below 0.93
    res_disagree = await orchestrator.get_or_call(
        prompt="new prompt disagreement",
        llm_function=lambda p: "fresh response 2",
    )
    assert not res_disagree.was_cache_hit
    assert res_disagree.decision.decision == "miss"
    assert res_disagree.decision.reason == "failed_cross_encoder_threshold"

    # Verify that record_active_learning_pair was called with correct parameters
    mock_store.record_active_learning_pair.assert_called_once_with(
        domain="medical",
        query_prompt="new prompt disagreement",
        cached_prompt="cached prompt",
        similarity_score=pytest.approx(0.95),
        classifier_score=pytest.approx(0.990),
        cross_encoder_score=pytest.approx(0.920),
        label=0,
        source="mlp_ce_disagreement",
    )


@pytest.mark.anyio
async def test_report_bad_hit_records_active_learning():
    mock_store = MagicMock(spec=SQLiteCacheStore)
    mock_embedding = MagicMock(spec=EmbeddingService)
    mock_domain_detector = MagicMock(spec=DomainDetector)

    orchestrator = CacheOrchestrator(
        domain="medical",
        config=CacheConfig(),
        store=mock_store,
        embedding_service=mock_embedding,
        classifier_service=None,
        cross_encoder_service=None,
        domain_detector=mock_domain_detector,
    )

    from datetime import datetime

    from semanticmemo.models import CacheEntry, LookupRecord

    query_id = uuid4()
    entry_id = uuid4()

    mock_lookup = LookupRecord(
        id=query_id,
        domain="medical",
        prompt="bad prompt",
        prompt_embedding=[0.1] * 384,
        cache_entry_id=entry_id,
        similarity_score=0.92,
        classifier_score=0.88,
        cross_encoder_score=0.84,
        created_at=datetime.now(),
    )
    mock_entry = CacheEntry(
        id=entry_id,
        prompt="original prompt",
        prompt_embedding=[0.1] * 384,
        response="some response",
        created_at=datetime.now(),
    )

    mock_store.get_lookup.return_value = mock_lookup
    mock_store.record_feedback.return_value = uuid4()
    mock_store.get.return_value = mock_entry

    success = orchestrator.report_bad_hit(query_id, reason="testing bad hit")
    assert success

    # Verify feedback was stored
    mock_store.record_feedback.assert_called_once_with(
        query_id=query_id,
        label=0,
        reason="testing bad hit",
    )
    # Verify feedback count incremented
    mock_store.increment_bad_feedback.assert_called_once_with(entry_id)
    # Verify active learning pair recorded with source "user_reported_bad_hit"
    mock_store.record_active_learning_pair.assert_called_once_with(
        domain="medical",
        query_prompt="bad prompt",
        cached_prompt="original prompt",
        similarity_score=pytest.approx(0.92),
        classifier_score=pytest.approx(0.88),
        cross_encoder_score=pytest.approx(0.84),
        label=0,
        source="user_reported_bad_hit",
    )

