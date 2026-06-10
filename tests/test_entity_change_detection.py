"""Unit tests for EntityChangeDetector."""

from __future__ import annotations

import pytest

from semanticmemo.entity_change_detection import (
    EntityChangeConfig,
    EntityChangeDetector,
    EntityChangeResult,
)


@pytest.fixture()
def detector() -> EntityChangeDetector:
    return EntityChangeDetector()


# ---------------------------------------------------------------------------
# Quarter detection
# ---------------------------------------------------------------------------


class TestQuarterDetection:
    def test_q3_vs_q4_triggers(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "Provide an analysis of Apple's Q3 earnings report.",
            "Provide an analysis of Apple's Q4 earnings report.",
        )
        assert result.entity_changed
        assert result.detector == "quarter"

    def test_q1_vs_q2_triggers(self, detector: EntityChangeDetector) -> None:
        result = detector.detect("Q1 revenue", "Q2 revenue")
        assert result.entity_changed
        assert result.detector == "quarter"

    def test_same_quarter_no_trigger(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "Provide an analysis of Apple's Q3 earnings.",
            "Give me Apple's Q3 financial results.",
        )
        assert not result.entity_changed

    def test_no_quarter_no_trigger(self, detector: EntityChangeDetector) -> None:
        result = detector.detect("Tell me about renewable energy.", "Tell me about wind power.")
        assert not result.entity_changed


# ---------------------------------------------------------------------------
# Drug / medication entity detection
# ---------------------------------------------------------------------------


class TestDrugDetection:
    def test_ibuprofen_vs_acetaminophen(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "What are the side effects of ibuprofen?",
            "What are the side effects of acetaminophen?",
        )
        assert result.entity_changed
        assert result.detector == "drug"

    def test_same_drug_no_trigger(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "What are the side effects of ibuprofen?",
            "What are the main side effects when taking ibuprofen?",
        )
        assert not result.entity_changed

    def test_aspirin_vs_warfarin(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "Can I take aspirin daily?",
            "Can I take warfarin daily?",
        )
        assert result.entity_changed
        assert result.detector == "drug"


# ---------------------------------------------------------------------------
# Numeric value detection
# ---------------------------------------------------------------------------


class TestNumericDetection:
    def test_different_numbers(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "Summarize the top 5 results.",
            "Summarize the top 10 results.",
        )
        assert result.entity_changed
        assert result.detector == "numeric"

    def test_same_number(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "Show the top 5 results for user 123.",
            "List the top 5 items for user 123.",
        )
        assert not result.entity_changed

    def test_dosage_change(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "What is the recommended dosage of 200mg ibuprofen?",
            "What is the recommended dosage of 400mg ibuprofen?",
        )
        assert result.entity_changed
        assert result.detector in {"numeric", "drug"}


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


class TestVersionDetection:
    def test_version_change(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "What changed in Python 3.11?",
            "What changed in Python 3.12?",
        )
        assert result.entity_changed

    def test_same_version(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "How do I upgrade to v2.0?",
            "What is the best way to upgrade to v2.0?",
        )
        assert not result.entity_changed


# ---------------------------------------------------------------------------
# Privilege level detection
# ---------------------------------------------------------------------------


class TestPrivilegeDetection:
    def test_user_vs_administrator(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "How do I reset my password on this account?",
            "How do I reset the administrator password on this account?",
        )
        assert result.entity_changed
        assert result.detector == "privilege"

    def test_same_privilege_level(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "How do I reset the admin password?",
            "What steps are needed to reset the admin password?",
        )
        assert not result.entity_changed


# ---------------------------------------------------------------------------
# Temporal reference detection
# ---------------------------------------------------------------------------


class TestTemporalDetection:
    def test_current_vs_historical(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "What is the current inflation rate?",
            "What was the historical inflation rate?",
        )
        assert result.entity_changed
        assert result.detector == "temporal"

    def test_same_temporal_ref(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "What is the current inflation rate?",
            "What is the current CPI reading?",
        )
        assert not result.entity_changed

    def test_this_vs_last(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "Summarise this quarter's results.",
            "Summarise last quarter's results.",
        )
        assert result.entity_changed


# ---------------------------------------------------------------------------
# Ordinal detection
# ---------------------------------------------------------------------------


class TestOrdinalDetection:
    def test_first_vs_second(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "What happened in the first phase of the trial?",
            "What happened in the second phase of the trial?",
        )
        assert result.entity_changed
        assert result.detector == "ordinal"

    def test_no_ordinals(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "Tell me about clinical trials.", "Describe clinical trial phases."
        )
        assert not result.entity_changed


# ---------------------------------------------------------------------------
# Month detection
# ---------------------------------------------------------------------------


class TestMonthDetection:
    def test_january_vs_march(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "What were the January sales figures?",
            "What were the March sales figures?",
        )
        assert result.entity_changed
        assert result.detector == "month"

    def test_same_month(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "January was a strong month for tech.",
            "Why did tech perform well in January?",
        )
        assert not result.entity_changed


# ---------------------------------------------------------------------------
# Year detection
# ---------------------------------------------------------------------------


class TestYearDetection:
    def test_year_mismatch(self, detector: EntityChangeDetector) -> None:
        result = detector.detect("2023 earnings report", "2024 earnings report")
        assert result.entity_changed
        assert result.detector == "year"

    def test_same_year(self, detector: EntityChangeDetector) -> None:
        result = detector.detect("2024 Q3 revenue", "Revenue for Q3 2024")
        assert not result.entity_changed


# ---------------------------------------------------------------------------
# Proper noun detection
# ---------------------------------------------------------------------------


class TestProperNounDetection:
    def test_apple_vs_microsoft(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "Provide an analysis of Apple's earnings.",
            "Provide an analysis of Microsoft's earnings.",
        )
        assert result.entity_changed
        assert result.detector == "proper_noun"

    def test_same_company(self, detector: EntityChangeDetector) -> None:
        result = detector.detect(
            "What was Apple's revenue last quarter?",
            "Show me Apple's most recent revenue figures.",
        )
        assert not result.entity_changed


# ---------------------------------------------------------------------------
# Configuration: disabled detectors
# ---------------------------------------------------------------------------


class TestConfigDisabledDetectors:
    def test_disable_quarter_detector(self) -> None:
        config = EntityChangeConfig(disabled_detectors=["quarter"])
        det = EntityChangeDetector(config)
        result = det.detect(
            "Apple Q3 earnings analysis.",
            "Apple Q4 earnings analysis.",
        )
        # Quarter detector disabled — should fall through to proper noun
        # (or not fire at all if no other detector catches it)
        # The important thing: quarter detector did NOT fire
        if result.entity_changed:
            assert result.detector != "quarter"

    def test_disable_numeric_detector(self) -> None:
        config = EntityChangeConfig(disabled_detectors=["numeric"])
        det = EntityChangeDetector(config)
        result = det.detect("Top 5 results.", "Top 10 results.")
        assert not result.entity_changed  # numeric disabled, no other detector fires


# ---------------------------------------------------------------------------
# No-change cases — equivalent paraphrases should never trigger
# ---------------------------------------------------------------------------


class TestNoChangeCases:
    @pytest.mark.parametrize(
        "a, b",
        [
            (
                "Summarize this article on renewable energy.",
                "Summarize the article about renewable energy.",
            ),
            (
                "What are the common side effects of ibuprofen?",
                "What are the main side effects when taking ibuprofen?",
            ),
            (
                "Provide an analysis of Apple's Q3 earnings report.",
                "Give me an analysis of Apple's Q3 financial results.",
            ),
            (
                "How do I reset my password?",
                "What is the procedure for resetting my password?",
            ),
        ],
    )
    def test_equivalent_paraphrases_no_trigger(
        self, a: str, b: str, detector: EntityChangeDetector
    ) -> None:
        result = detector.detect(a, b)
        assert not result.entity_changed, (
            f"False positive for paraphrase pair:\n  A: {a}\n  B: {b}\n  Reason: {result.reason}"
        )


# ---------------------------------------------------------------------------
# EntityChangeResult dataclass
# ---------------------------------------------------------------------------


class TestEntityChangeResult:
    def test_no_change_result(self) -> None:
        result = EntityChangeResult(entity_changed=False)
        assert not result.entity_changed
        assert result.reason == "none"
        assert result.changed_tokens is None
        assert result.detector is None

    def test_change_result(self) -> None:
        result = EntityChangeResult(
            entity_changed=True,
            reason="fiscal quarter mismatch: 'Q3' vs 'Q4'",
            changed_tokens=("Q3", "Q4"),
            detector="quarter",
        )
        assert result.entity_changed
        assert result.changed_tokens == ("Q3", "Q4")
        assert result.detector == "quarter"
