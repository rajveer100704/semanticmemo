from __future__ import annotations

import semanticmemo


def test_public_api_exports() -> None:
    assert semanticmemo.SemanticMemo is not None
    assert semanticmemo.CacheConfig is not None
    assert semanticmemo.CacheResult is not None
    assert semanticmemo.CacheStats is not None
    assert semanticmemo.CacheEntry is not None
    assert semanticmemo.ClassifierConfig is not None
    assert semanticmemo.EvictionPolicy is not None
    assert semanticmemo.FeedbackEvent is not None
    assert semanticmemo.ImplicitFeedbackConfig is not None
    assert semanticmemo.LookupRecord is not None
    assert semanticmemo.RetryConfig is not None
    assert semanticmemo.RetrainConfig is not None
    assert semanticmemo.RetrainResult is not None
    assert semanticmemo.retrain_from_feedback is not None
