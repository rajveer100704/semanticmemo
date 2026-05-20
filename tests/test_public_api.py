from __future__ import annotations

import smartmemo


def test_public_api_exports() -> None:
    assert smartmemo.SmartMemo is not None
    assert smartmemo.CacheConfig is not None
    assert smartmemo.CacheResult is not None
    assert smartmemo.CacheStats is not None
    assert smartmemo.CacheEntry is not None
    assert smartmemo.ClassifierConfig is not None
    assert smartmemo.EvictionPolicy is not None
    assert smartmemo.FeedbackEvent is not None
    assert smartmemo.ImplicitFeedbackConfig is not None
    assert smartmemo.LookupRecord is not None
    assert smartmemo.RetrainConfig is not None
    assert smartmemo.RetrainResult is not None
    assert smartmemo.retrain_from_feedback is not None
