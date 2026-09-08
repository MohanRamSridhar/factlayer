"""Assigning canonical measure identities at runtime.

Two documents rarely name the same quantity the same way. "Revenue from
services", "service revenue" and "income from services" are one measure written
three ways; "revenue" and "revenue growth" are two measures that share most of
their words. Getting that distinction right is what decides whether the linker
compares the right pairs, so it is worth doing carefully and worth doing
*without* a hand-written list of domain terms -- the brief rules those out, and
a list of macroeconomic measures would not survive the first clinical trial PDF.

The approach here is deliberately not embeddings. Not because embeddings are
wrong, but because they add an API dependency and a second failure mode to buy
an improvement this task does not obviously need: measure names are short,
mostly noun phrases, and drawn from the documents themselves. Three cheap
signals do most of the work:

1. **Rarity weighting.** Tokens are weighted by how rare they are among the
   measures actually seen so far. "gdp" appearing in two names is strong
   evidence; "total" appearing in two names is nearly none. This is computed
   from the corpus at hand, so it adapts to whatever domain shows up.
2. **Derivative markers.** A generic English closed class -- growth, share,
   margin, per -- that changes *what is being measured* rather than describing
   it. "Revenue" and "revenue growth" are never the same measure, regardless of
   how many tokens they share.
3. **Kind agreement.** A percentage and a currency amount are not the same
   measure even when identically worded, which catches "inflation" the index
   versus "inflation" the rate.

Known limitation, stated plainly: this is lexical. Genuine synonyms with no
shared tokens ("turnover" vs "revenue") will not merge. See ``README.md``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from .entities import strip_accents

# Words that carry no measure identity. Generic English plus the handful of
# quantity-neutral nouns that documents sprinkle over every figure.
STOPWORDS = frozenset(
    """
    a an the of in for on at to from by with and or as its their this that these those
    rate value amount figure level number quantum size
    is was are were be been being
    """.split()
)

# Tokens that turn a measure into a *different* measure derived from it.
# Closed class, generic English, no domain knowledge required.
DERIVATIVE_MARKERS = frozenset(
    """
    growth change increase decrease decline rise fall
    expansion contraction reduction addition accretion drawdown depletion
    share proportion percentage margin ratio contribution weight
    per capita average mean median cumulative incremental net gross
    forecast projection target estimate
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Very small, purely morphological. No synonym table: that is where domain
# assumptions sneak in.
_IRREGULAR = {"indices": "index", "analyses": "analysis"}


def _singular(token: str) -> str:
    if token in _IRREGULAR:
        return _IRREGULAR[token]
    if len(token) > 3:
        if token.endswith("ies"):
            return token[:-3] + "y"
        if token.endswith("ses") or token.endswith("xes"):
            return token[:-2]
        if token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
    return token


def tokenise_measure(text: str) -> list[str]:
    """Content tokens of a measure name, normalised for comparison."""
    lowered = strip_accents((text or "").lower())
    tokens = [_singular(t) for t in _TOKEN_RE.findall(lowered)]
    return [t for t in tokens if t and t not in STOPWORDS]


def derivative_markers(tokens: Iterable[str]) -> frozenset[str]:
    return frozenset(t for t in tokens if t in DERIVATIVE_MARKERS)


def slugify_measure(tokens: Iterable[str]) -> str:
    return "_".join(tokens) or "unnamed"


@dataclass
class MeasureCluster:
    """One canonical measure and every surface form that mapped onto it."""

    key: str
    canonical: str
    kind: str
    tokens: frozenset[str]
    markers: frozenset[str]
    surface_forms: Counter = field(default_factory=Counter)

    @property
    def support(self) -> int:
        return sum(self.surface_forms.values())

    def label(self) -> str:
        """The most frequently written form, which reads better than a slug."""
        if not self.surface_forms:
            return self.canonical
        return self.surface_forms.most_common(1)[0][0]


class MeasureRegistry:
    """Assigns ``measure_key`` values, minting new ones as documents arrive.

    The registry is persistent state, not a per-run computation. That is what
    lets a new document be linked against an existing knowledge layer without
    re-deriving keys for everything already stored -- the incremental-ingest
    property the brief asks about.
    """

    def __init__(self, threshold: float = 0.68) -> None:
        self.threshold = threshold
        self.clusters: dict[str, MeasureCluster] = {}
        self._document_frequency: Counter = Counter()
        self._observations = 0

    # -- rarity weighting ---------------------------------------------------

    def _weight(self, token: str) -> float:
        """Inverse document frequency over measures seen so far.

        Smoothed so that an unseen token in an empty registry still scores
        finitely, and floored so a very common token contributes a little
        rather than nothing.
        """
        df = self._document_frequency.get(token, 0)
        return max(0.15, math.log((self._observations + 1) / (df + 1)) + 1.0)

    def _similarity(self, tokens: frozenset[str], cluster: MeasureCluster) -> float:
        """Rarity-weighted Jaccard, with a containment allowance.

        Containment matters because documents abbreviate: "real GDP growth at
        constant prices" and "real GDP growth" should meet, and plain Jaccard
        punishes the longer form for being specific.
        """
        shared = tokens & cluster.tokens
        if not shared:
            return 0.0
        weight_shared = sum(self._weight(t) for t in shared)
        weight_union = sum(self._weight(t) for t in (tokens | cluster.tokens))
        jaccard = weight_shared / weight_union if weight_union else 0.0

        weight_smaller = min(
            sum(self._weight(t) for t in tokens),
            sum(self._weight(t) for t in cluster.tokens),
        )
        containment = weight_shared / weight_smaller if weight_smaller else 0.0
        return max(jaccard, 0.85 * containment)

    # -- assignment ---------------------------------------------------------

    def assign(self, measure_raw: str, kind: str = "other") -> str:
        """Return the canonical key for a measure name, creating it if new."""
        tokens = frozenset(tokenise_measure(measure_raw))
        if not tokens:
            tokens = frozenset({"unnamed"})
        markers = derivative_markers(tokens)

        best: MeasureCluster | None = None
        best_score = 0.0
        for cluster in self.clusters.values():
            # A derived measure never merges with its base, and two differently
            # derived measures never merge with each other.
            if cluster.markers != markers:
                continue
            # Percent and currency readings of the same words are different
            # measures. "unknown"/"other" kinds stay permissive rather than
            # fragmenting the registry on missing metadata.
            if cluster.kind != kind and "other" not in (cluster.kind, kind):
                continue
            score = self._similarity(tokens, cluster)
            if score > best_score:
                best, best_score = cluster, score

        if best is not None and best_score >= self.threshold:
            chosen = best
            # Specific wording is kept; the cluster does not drift wider with
            # every merge, which would eventually swallow unrelated measures.
            chosen.surface_forms[measure_raw.strip()] += 1
        else:
            slug = slugify_measure(sorted(tokens))
            key = f"m_{slug}"[:80]
            suffix = 2
            while key in self.clusters:
                key = f"m_{slug}_{suffix}"[:80]
                suffix += 1
            chosen = MeasureCluster(
                key=key,
                canonical=" ".join(sorted(tokens)),
                kind=kind,
                tokens=tokens,
                markers=markers,
            )
            chosen.surface_forms[measure_raw.strip()] += 1
            self.clusters[key] = chosen

        self._observations += 1
        for token in tokens:
            self._document_frequency[token] += 1
        return chosen.key

    def label_for(self, key: str) -> str:
        cluster = self.clusters.get(key)
        return cluster.label() if cluster else key

    def kind_for(self, key: str) -> str:
        cluster = self.clusters.get(key)
        return cluster.kind if cluster else "other"

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "observations": self._observations,
            "document_frequency": dict(self._document_frequency),
            "clusters": [
                {
                    "key": c.key,
                    "canonical": c.canonical,
                    "kind": c.kind,
                    "tokens": sorted(c.tokens),
                    "markers": sorted(c.markers),
                    "surface_forms": dict(c.surface_forms),
                }
                for c in self.clusters.values()
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MeasureRegistry":
        registry = cls(threshold=payload.get("threshold", 0.68))
        registry._observations = payload.get("observations", 0)
        registry._document_frequency = Counter(payload.get("document_frequency", {}))
        for raw in payload.get("clusters", []):
            registry.clusters[raw["key"]] = MeasureCluster(
                key=raw["key"],
                canonical=raw["canonical"],
                kind=raw.get("kind", "other"),
                tokens=frozenset(raw.get("tokens", [])),
                markers=frozenset(raw.get("markers", [])),
                surface_forms=Counter(raw.get("surface_forms", {})),
            )
        return registry
