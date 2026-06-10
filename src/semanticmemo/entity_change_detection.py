"""Entity Change Detection for semantic cache safety.

This module implements the ``EntityChangeDetector``, the fourth and final gate
in the SemanticMemo double-verification pipeline:

    Embedding → MLP Classifier → Cross-Encoder → **EntityChangeDetector**

Its sole purpose is to catch cases that the embedding and neural models miss
because they are deliberately insensitive to surface form: prompts where a
critical *entity* (time period, drug name, person, company, numeric value,
version string, etc.) has changed between the cached query and the incoming
query, yet the overall semantic similarity remains extremely high.

Classic examples that motivated this module:
- "Provide an analysis of Apple's **Q3** earnings" vs "…**Q4**…"
- "What are the side effects of **ibuprofen**?" vs "…**acetaminophen**?"
- "Summarise **this quarter's** results" vs "Summarise **last quarter's**"
- "Reset **my** password" vs "Reset the **administrator** password"

Design principles
-----------------
* **No ML inference** – uses only regex and lightweight NLP rules so that it
  adds < 1 ms overhead on CPU.
* **Conservative by design** – false negatives (serving a stale cached answer
  when entity drift is present) are the catastrophic failure; false positives
  (unnecessary LLM calls) are merely wasteful.  When in doubt, return ``True``
  (entity changed).
* **Configurable sensitivity** – ``EntityChangeConfig`` lets callers toggle
  individual detectors or restrict detection to specific entity categories.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityChangeConfig:
    """Tuning knobs for the EntityChangeDetector.

    All detectors are enabled by default.  Disable selectors in
    ``disabled_detectors`` to skip specific checks.

    Set ``enabled=False`` to turn off all entity-change detection entirely,
    useful when running v1 baselines in benchmarks.
    """

    # Master switch: set to False to skip all entity-change checks.
    enabled: bool = True

    # Names of detector methods to skip (e.g. ``["numeric", "version"]``).
    disabled_detectors: Sequence[str] = field(default_factory=tuple)

    # When True, treat *any* ordinal/quarter token mismatch as entity drift.
    strict_time_periods: bool = True

    # When True, trigger on mismatches in named-entity capitalised tokens even
    # when they are not in the pre-built drug / company lists.
    strict_proper_nouns: bool = True


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityChangeResult:
    """Outcome of a single entity-change check.

    Attributes
    ----------
    entity_changed:
        ``True`` if at least one critical entity mismatch was detected.
    reason:
        Human-readable description of the first mismatch found, or ``"none"``
        when no drift was detected.
    changed_tokens:
        Tuple of ``(token_in_query_a, token_in_query_b)`` that triggered the
        detection, if applicable.
    detector:
        Name of the detector method that fired, for observability.
    """

    entity_changed: bool
    reason: str = "none"
    changed_tokens: tuple[str, str] | None = None
    detector: str | None = None


# ---------------------------------------------------------------------------
# Shared regex patterns
# ---------------------------------------------------------------------------

# Fiscal / calendar quarters
_RE_QUARTER = re.compile(r"\b(Q[1-4]|quarter\s+[1-4]|[1-4](?:st|nd|rd|th)\s+quarter)\b", re.I)

# Ordinals (first, second, … tenth; 1st, 2nd, …)
_RE_ORDINAL = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
    r"|[1-9][0-9]*(?:st|nd|rd|th))\b",
    re.I,
)

# Calendar year  (4-digit years between 1900 and 2099)
_RE_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")

# Standalone integers / decimals with optional unit suffix (catches dosage, version, count, amount)
# Matches: 200, 3.5, 200mg, 400mcg, 10ml, 5x
_RE_NUMBER = re.compile(r"\b(\d+(?:\.\d+)?(?:[a-zA-Z]{0,5})?)", re.I)

# Version strings like v1.2, 1.0.3, 2.x
_RE_VERSION = re.compile(r"\b(v?\d+(?:\.\d+){1,3})\b", re.I)

# Date-like tokens: Jan, January, Mon, Monday, etc.
_RE_MONTH = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october"
    r"|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
    re.I,
)
_RE_DAY = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|mon|tue|wed|thu|fri|sat|sun)\b",
    re.I,
)

# Common drug / medication names (non-exhaustive; covers high-stakes cases)
_KNOWN_DRUGS: frozenset[str] = frozenset(
    {
        "ibuprofen",
        "acetaminophen",
        "paracetamol",
        "aspirin",
        "naproxen",
        "amoxicillin",
        "penicillin",
        "metformin",
        "lisinopril",
        "atorvastatin",
        "simvastatin",
        "omeprazole",
        "prednisone",
        "warfarin",
        "clopidogrel",
        "sertraline",
        "fluoxetine",
        "alprazolam",
        "diazepam",
        "lorazepam",
        "insulin",
        "metoprolol",
        "amlodipine",
        "losartan",
        "gabapentin",
        "tramadol",
        "oxycodone",
        "morphine",
        "codeine",
        "hydroxychloroquine",
        "ivermectin",
        "remdesivir",
        "dexamethasone",
    }
)

# Temporal relative references
_TEMPORAL_WORDS: frozenset[str] = frozenset(
    {
        "today",
        "yesterday",
        "tomorrow",
        "now",
        "current",
        "currently",
        "recent",
        "recently",
        "latest",
        "last",
        "previous",
        "prior",
        "next",
        "upcoming",
        "this",
        "past",
        "historical",
        "annual",
        "monthly",
        "weekly",
        "daily",
    }
)

# Privilege / role tokens relevant to security domain
_PRIVILEGE_TOKENS: frozenset[str] = frozenset(
    {
        "administrator",
        "admin",
        "root",
        "superuser",
        "sudo",
        "system",
        "service",
        "privileged",
        "elevated",
        "owner",
        "manager",
        "guest",
        "anonymous",
    }
)


def _token_set(text: str) -> set[str]:
    """Lowercase word tokens from *text*."""
    return set(re.findall(r"[a-zA-Z0-9.']+", text.lower()))


def _proper_noun_tokens(text: str) -> set[str]:
    """Return capitalised tokens that are not sentence-starters.

    Heuristic: a capitalised token *not* at position 0 after a sentence-ending
    punctuation mark is likely a proper noun (name, company, product, etc.).
    """
    # Split into raw words while keeping position info
    words = re.findall(r"[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z])|[A-Z]+", text)
    return {w.lower() for w in words}


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------


class EntityChangeDetector:
    """Detect entity-level drift between two prompt strings.

    Usage::

        detector = EntityChangeDetector()
        result = detector.detect(query_prompt, cached_prompt)
        if result.entity_changed:
            # Force a cache MISS
            ...

    The detector runs a battery of lightweight regex-based checks and returns
    on the *first* mismatch it finds (fail-fast).  The ``reason`` field of the
    returned :class:`EntityChangeResult` identifies which check fired.
    """

    def __init__(self, config: EntityChangeConfig | None = None) -> None:
        self.config = config or EntityChangeConfig()
        self._disabled = set(self.config.disabled_detectors)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, query_prompt: str, cached_prompt: str) -> EntityChangeResult:
        """Return an :class:`EntityChangeResult` for the given pair.

        Parameters
        ----------
        query_prompt:
            The new, incoming prompt.
        cached_prompt:
            The prompt whose response is stored in the cache.
        """
        if not self.config.enabled:
            return EntityChangeResult(entity_changed=False)

        checkers = [
            ("quarter", self._check_quarter),
            ("year", self._check_year),
            ("ordinal", self._check_ordinal),
            ("drug", self._check_drug),
            ("numeric", self._check_numeric),
            ("version", self._check_version),
            ("month", self._check_month),
            ("day", self._check_day),
            ("temporal", self._check_temporal),
            ("privilege", self._check_privilege),
            ("proper_noun", self._check_proper_noun),
        ]
        for name, checker in checkers:
            if name in self._disabled:
                continue
            result = checker(query_prompt, cached_prompt)
            if result.entity_changed:
                return result

        return EntityChangeResult(entity_changed=False)

    # ------------------------------------------------------------------
    # Individual detectors
    # ------------------------------------------------------------------

    def _check_quarter(self, a: str, b: str) -> EntityChangeResult:
        """Detect Q1/Q2/Q3/Q4 or ordinal-quarter mismatches."""
        qa = set(m.group(0).upper() for m in _RE_QUARTER.finditer(a))
        qb = set(m.group(0).upper() for m in _RE_QUARTER.finditer(b))
        if qa and qb and qa != qb:
            ta = next(iter(qa - qb), next(iter(qa)))
            tb = next(iter(qb - qa), next(iter(qb)))
            return EntityChangeResult(
                entity_changed=True,
                reason=f"fiscal quarter mismatch: '{ta}' vs '{tb}'",
                changed_tokens=(ta, tb),
                detector="quarter",
            )
        return EntityChangeResult(entity_changed=False)

    def _check_year(self, a: str, b: str) -> EntityChangeResult:
        """Detect calendar year mismatches (e.g. 2023 vs 2024)."""
        ya = set(m.group(0) for m in _RE_YEAR.finditer(a))
        yb = set(m.group(0) for m in _RE_YEAR.finditer(b))
        if ya and yb and ya != yb:
            ta = next(iter(ya - yb), next(iter(ya)))
            tb = next(iter(yb - ya), next(iter(yb)))
            return EntityChangeResult(
                entity_changed=True,
                reason=f"calendar year mismatch: '{ta}' vs '{tb}'",
                changed_tokens=(ta, tb),
                detector="year",
            )
        return EntityChangeResult(entity_changed=False)

    def _check_ordinal(self, a: str, b: str) -> EntityChangeResult:
        """Detect ordinal token mismatches (first vs second, 1st vs 2nd)."""
        oa = {m.group(0).lower() for m in _RE_ORDINAL.finditer(a)}
        ob = {m.group(0).lower() for m in _RE_ORDINAL.finditer(b)}
        if oa and ob and oa != ob:
            ta = next(iter(oa - ob), next(iter(oa)))
            tb = next(iter(ob - oa), next(iter(ob)))
            return EntityChangeResult(
                entity_changed=True,
                reason=f"ordinal mismatch: '{ta}' vs '{tb}'",
                changed_tokens=(ta, tb),
                detector="ordinal",
            )
        return EntityChangeResult(entity_changed=False)

    def _check_drug(self, a: str, b: str) -> EntityChangeResult:
        """Detect named drug / medication entity swaps."""
        toks_a = _token_set(a)
        toks_b = _token_set(b)
        drugs_a = toks_a & _KNOWN_DRUGS
        drugs_b = toks_b & _KNOWN_DRUGS
        if drugs_a and drugs_b and drugs_a != drugs_b:
            ta = next(iter(drugs_a - drugs_b))
            tb = next(iter(drugs_b - drugs_a))
            return EntityChangeResult(
                entity_changed=True,
                reason=f"drug entity mismatch: '{ta}' vs '{tb}'",
                changed_tokens=(ta, tb),
                detector="drug",
            )
        return EntityChangeResult(entity_changed=False)

    def _check_numeric(self, a: str, b: str) -> EntityChangeResult:
        """Detect changes in standalone numeric values (dosages, counts, amounts).

        Year-like 4-digit numbers are deliberately excluded here because the
        ``year`` detector already handles them with higher precision.
        """

        def _non_year_nums(text: str) -> set[str]:
            return {
                m.group(0)
                for m in _RE_NUMBER.finditer(text)
                if not re.fullmatch(r"(?:19|20)\d{2}", m.group(0))
            }

        na = _non_year_nums(a)
        nb = _non_year_nums(b)
        if na and nb and na != nb:
            ta = next(iter(na - nb), next(iter(na)))
            tb = next(iter(nb - na), next(iter(nb)))
            return EntityChangeResult(
                entity_changed=True,
                reason=f"numeric value mismatch: '{ta}' vs '{tb}'",
                changed_tokens=(ta, tb),
                detector="numeric",
            )
        return EntityChangeResult(entity_changed=False)

    def _check_version(self, a: str, b: str) -> EntityChangeResult:
        """Detect version string changes (v1.2 vs v1.3, 2.0 vs 3.0)."""
        va = {m.group(0).lower() for m in _RE_VERSION.finditer(a)}
        vb = {m.group(0).lower() for m in _RE_VERSION.finditer(b)}
        if va and vb and va != vb:
            ta = next(iter(va - vb), next(iter(va)))
            tb = next(iter(vb - va), next(iter(vb)))
            return EntityChangeResult(
                entity_changed=True,
                reason=f"version mismatch: '{ta}' vs '{tb}'",
                changed_tokens=(ta, tb),
                detector="version",
            )
        return EntityChangeResult(entity_changed=False)

    def _check_month(self, a: str, b: str) -> EntityChangeResult:
        """Detect calendar month changes (January vs March)."""
        ma = {m.group(0).lower() for m in _RE_MONTH.finditer(a)}
        mb = {m.group(0).lower() for m in _RE_MONTH.finditer(b)}
        if ma and mb and ma != mb:
            ta = next(iter(ma - mb), next(iter(ma)))
            tb = next(iter(mb - ma), next(iter(mb)))
            return EntityChangeResult(
                entity_changed=True,
                reason=f"calendar month mismatch: '{ta}' vs '{tb}'",
                changed_tokens=(ta, tb),
                detector="month",
            )
        return EntityChangeResult(entity_changed=False)

    def _check_day(self, a: str, b: str) -> EntityChangeResult:
        """Detect day-of-week changes (Monday vs Friday)."""
        da = {m.group(0).lower() for m in _RE_DAY.finditer(a)}
        db = {m.group(0).lower() for m in _RE_DAY.finditer(b)}
        if da and db and da != db:
            ta = next(iter(da - db), next(iter(da)))
            tb = next(iter(db - da), next(iter(db)))
            return EntityChangeResult(
                entity_changed=True,
                reason=f"day-of-week mismatch: '{ta}' vs '{tb}'",
                changed_tokens=(ta, tb),
                detector="day",
            )
        return EntityChangeResult(entity_changed=False)

    def _check_temporal(self, a: str, b: str) -> EntityChangeResult:
        """Detect temporal reference drift (current vs historical, this vs last)."""
        if not self.config.strict_time_periods:
            return EntityChangeResult(entity_changed=False)

        ta_set = _token_set(a) & _TEMPORAL_WORDS
        tb_set = _token_set(b) & _TEMPORAL_WORDS

        # Only fire when *different* temporal words appear in each prompt —
        # prevents false positives when both share the same temporal reference.
        diff_a = ta_set - tb_set
        diff_b = tb_set - ta_set
        if diff_a and diff_b:
            # Common innocuous pairs that should NOT trigger a miss.
            _harmless_pairs = {
                frozenset({"current", "recent"}),
                frozenset({"recent", "latest"}),
                frozenset({"current", "latest"}),
                frozenset({"last", "recent"}),
                frozenset({"last", "latest"}),
                frozenset({"recent", "most"}),
                frozenset({"current", "most"}),
            }
            pair = frozenset({next(iter(diff_a)), next(iter(diff_b))})
            if pair not in _harmless_pairs:
                ta = next(iter(diff_a))
                tb = next(iter(diff_b))
                return EntityChangeResult(
                    entity_changed=True,
                    reason=f"temporal reference mismatch: '{ta}' vs '{tb}'",
                    changed_tokens=(ta, tb),
                    detector="temporal",
                )
        return EntityChangeResult(entity_changed=False)

    def _check_privilege(self, a: str, b: str) -> EntityChangeResult:
        """Detect privilege-level / role entity swaps (user vs administrator).

        Handles two cases:
        1. Both prompts have privilege tokens but they differ (admin vs root).
        2. One prompt has an explicit privilege token and the other implies a
           regular user context via possessive pronouns (my, our) — this is the
           dangerous escalation pattern: "my password" vs "administrator password".
        """
        pa = _token_set(a) & _PRIVILEGE_TOKENS
        pb = _token_set(b) & _PRIVILEGE_TOKENS

        # Case 1: Both have explicit privilege tokens that differ
        if pa and pb and pa != pb:
            ta = next(iter(pa - pb))
            tb = next(iter(pb - pa))
            return EntityChangeResult(
                entity_changed=True,
                reason=f"privilege/role mismatch: '{ta}' vs '{tb}'",
                changed_tokens=(ta, tb),
                detector="privilege",
            )

        # Case 2: Asymmetric — one side has a privilege token, the other uses
        # possessive pronouns (implying regular user scope). E.g.:
        # "reset MY password" (user-scope) vs "reset the ADMINISTRATOR password"
        _possessives = {"my", "our", "your"}
        tokens_a = _token_set(a)
        tokens_b = _token_set(b)
        has_possessive_a = bool(tokens_a & _possessives)
        has_possessive_b = bool(tokens_b & _possessives)

        if pa and has_possessive_b and not pb:
            # B is regular-user, A is privileged
            ta = next(iter(pa))
            return EntityChangeResult(
                entity_changed=True,
                reason=(
                    f"privilege escalation mismatch: '{ta}' (privileged) vs possessive user scope"
                ),
                changed_tokens=(ta, "user"),
                detector="privilege",
            )
        if pb and has_possessive_a and not pa:
            # A is regular-user, B is privileged
            tb = next(iter(pb))
            return EntityChangeResult(
                entity_changed=True,
                reason=(
                    f"privilege escalation mismatch: possessive user scope vs '{tb}' (privileged)"
                ),
                changed_tokens=("user", tb),
                detector="privilege",
            )
        return EntityChangeResult(entity_changed=False)

    def _check_proper_noun(self, a: str, b: str) -> EntityChangeResult:
        """Detect proper noun / named entity substitutions.

        Uses a capitalisation heuristic: tokens that are Title-cased or
        ALL-CAPS and appear *only* in one of the two prompts are treated as
        potentially distinct entities.  This catches company/person name swaps
        such as "Apple Q3" vs "Microsoft Q3".
        """
        if not self.config.strict_proper_nouns:
            return EntityChangeResult(entity_changed=False)

        pn_a = _proper_noun_tokens(a)
        pn_b = _proper_noun_tokens(b)

        # Filter out very short tokens and common stop words to reduce noise
        _stop = {
            "the",
            "an",
            "a",
            "in",
            "on",
            "at",
            "for",
            "of",
            "to",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "be",
            "i",
            "my",
            "this",
            "that",
            "it",
            "its",
            "by",
            "as",
            "do",
            "can",
            "how",
            "what",
            "give",
            "me",
            "us",
            "list",
            "provide",
            "q",
            "mr",
            "dr",
            "ms",
            "tell",
            "show",
            "get",
            "find",
            "describe",
            "explain",
            "summarize",
            "summarise",
            "analyze",
            "analyse",
            "compare",
            "calculate",
            "generate",
            "write",
            "create",
            "make",
            "help",
            "please",
            "need",
            "want",
            "looking",
        }
        pn_a = {t for t in pn_a if len(t) > 2 and t not in _stop}
        pn_b = {t for t in pn_b if len(t) > 2 and t not in _stop}

        only_a = pn_a - pn_b
        only_b = pn_b - pn_a

        if only_a and only_b:
            ta = next(iter(only_a))
            tb = next(iter(only_b))
            return EntityChangeResult(
                entity_changed=True,
                reason=f"proper noun mismatch: '{ta}' vs '{tb}'",
                changed_tokens=(ta, tb),
                detector="proper_noun",
            )
        return EntityChangeResult(entity_changed=False)
