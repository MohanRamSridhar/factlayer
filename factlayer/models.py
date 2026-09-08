"""Core data model for the fact knowledge layer.

The design centres on one idea: a bare number is not a fact. "6.4 per cent" only
becomes checkable once you know *what* it measures, *who* it is about, *when* it
applies, *on what basis* it was computed, and *how firm* the figure is. We call
that bundle the :class:`ContextEnvelope`, and it is what makes reconciliation
possible: two figures that disagree are only a contradiction if their envelopes
agree. If the envelopes differ, the disagreement is usually *explained* by the
difference, which is exactly case 3 of the brief.

Nothing here enumerates domain measures. ``measure_key`` is assigned at runtime
by the measure registry, so a document about rainfall creates rainfall measures
without any code change.
"""

from __future__ import annotations

import hashlib
from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Quantities
# --------------------------------------------------------------------------

QuantityKind = Literal[
    "currency", "percent", "count", "ratio", "duration", "index", "temperature", "other"
]


class Quantity(BaseModel):
    """A number together with everything needed to compare it to another number.

    ``canonical_value``/``canonical_unit`` are the comparable form: currency is
    reduced to whole units of its currency (so 8,142 crore INR and 81,415.38
    million INR both land on ~8.14e11 INR and can be compared directly), percents
    stay percents, counts are reduced to a bare count.
    """

    raw: str = Field(description="Verbatim text the number came from, e.g. '₹8,142 Cr'")
    value: float = Field(description="Numeric value as written, e.g. 8142.0")
    unit: str | None = Field(default=None, description="Unit token as written, e.g. 'INR', '%', 'shipments'")
    magnitude: float = Field(default=1.0, description="Scale multiplier implied by words like 'crore'/'Mn'")
    kind: QuantityKind = "other"
    canonical_value: float | None = None
    canonical_unit: str | None = None
    # Some figures are inherently approximate ("about 740 Mn"); tolerance widens.
    approximate: bool = False

    def comparable_to(self, other: Quantity) -> bool:
        return (
            self.canonical_unit is not None
            and self.canonical_unit == other.canonical_unit
            and self.canonical_value is not None
            and other.canonical_value is not None
        )


# --------------------------------------------------------------------------
# Periods
# --------------------------------------------------------------------------

PeriodKind = Literal[
    "fiscal_year",
    "fiscal_quarter",
    "fiscal_half",
    "calendar_year",
    "calendar_quarter",
    "month",
    "instant",
    "range",
    "unknown",
]


class Period(BaseModel):
    """A time window resolved to actual dates.

    Resolving to dates rather than keeping labels is what lets the system tell
    that "Q2 FY25" (an Indian fiscal quarter, Jul-Sep 2024) and "2025Q2" (a
    calendar quarter, Apr-Jun 2025) are different windows despite both reading
    as "Q2" -- a trap that string matching walks straight into.
    """

    raw: str
    kind: PeriodKind = "unknown"
    start: date | None = None
    end: date | None = None
    label: str | None = Field(default=None, description="Canonical label, e.g. 'FY2024-25'")
    # Which month the fiscal year ends in, when a fiscal period was detected.
    fiscal_year_end_month: int | None = None

    def resolved(self) -> bool:
        return self.start is not None and self.end is not None

    def overlaps(self, other: Period) -> bool | None:
        if not (self.resolved() and other.resolved()):
            return None
        return self.start <= other.end and other.start <= self.end  # type: ignore[operator]

    def same_window(self, other: Period) -> bool | None:
        if not (self.resolved() and other.resolved()):
            return None
        return self.start == other.start and self.end == other.end

    def days(self) -> int | None:
        if not self.resolved():
            return None
        return (self.end - self.start).days + 1  # type: ignore[operator]


# --------------------------------------------------------------------------
# Context envelope
# --------------------------------------------------------------------------

EstimateType = Literal[
    "actual",
    "provisional",
    "revised",
    "advance_estimate",
    "projection",
    "target",
    "unknown",
]
"""How firm a figure is. Distinguishing a January advance estimate from a May
provisional estimate is the difference between 'these sources contradict' and
'the number was revised', which is the single most common false contradiction in
statistical documents."""


class ContextEnvelope(BaseModel):
    entity: str = Field(description="Who/what the fact is about, as written")
    entity_key: str = Field(default="", description="Normalised entity identifier")
    period: Period | None = None
    basis: list[str] = Field(
        default_factory=list,
        description=(
            "Qualifiers that change what is being counted: 'consolidated', "
            "'standalone', 'constant prices', 'services only', 'seasonally adjusted'. "
            "Free-form and open-ended by design."
        ),
    )
    estimate_type: EstimateType = "unknown"
    # When a document quotes someone else ("as per the NSO"), the figure's real
    # provenance is that third party, not the document holding it.
    attributed_to: str | None = None
    # Date the document itself was published; drives vintage reasoning.
    as_of: date | None = None
    note: str | None = None


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


class Evidence(BaseModel):
    """A pointer back into the source PDF precise enough to highlight."""

    doc_id: str
    page: int = Field(description="1-based page number within the PDF")
    quote: str = Field(description="Verbatim sentence(s) supporting the fact")
    char_start: int | None = Field(default=None, description="Offset into the page's extracted text")
    char_end: int | None = None
    block_id: str | None = None
    bbox: tuple[float, float, float, float] | None = None

    @property
    def locator(self) -> str:
        return f"{self.doc_id}#p{self.page}"


# --------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------

FactType = Literal["numeric", "state", "event", "attribute"]
"""``numeric`` covers measured quantities. ``state`` covers things that are true
of an entity over a window ("is a director of X") -- these are what make
supersession detectable. ``event`` covers dated occurrences ("resigned on ..."),
``attribute`` covers non-numeric properties ("registered office is at ...")."""


class Fact(BaseModel):
    id: str = ""
    doc_id: str
    fact_type: FactType = "numeric"

    measure_raw: str = Field(description="Measure as the document words it, e.g. 'revenue from services'")
    measure_key: str = Field(default="", description="Canonical measure id, assigned by the registry")

    quantity: Quantity | None = None
    value_text: str | None = Field(
        default=None, description="Value for non-numeric facts, e.g. 'ceased to be a director'"
    )

    context: ContextEnvelope
    evidence: Evidence

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    extractor: str = "unknown"
    # Free-form extras the extractor thought worth keeping. Keeps the schema
    # open so new kinds of facts do not require a migration.
    extra: dict[str, Any] = Field(default_factory=dict)

    def compute_id(self) -> str:
        basis = "|".join(
            [
                self.doc_id,
                str(self.evidence.page),
                self.measure_raw.lower().strip(),
                self.context.entity.lower().strip(),
                (self.context.period.raw if self.context.period else ""),
                str(self.quantity.value if self.quantity else self.value_text),
                ",".join(sorted(self.context.basis)),
            ]
        )
        return "f_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]

    def display_value(self) -> str:
        if self.quantity is not None:
            return self.quantity.raw
        return self.value_text or ""


# --------------------------------------------------------------------------
# Relations
# --------------------------------------------------------------------------


class RelationType(str, Enum):
    CORROBORATES = "corroborates"
    CONTRADICTS = "contradicts"
    RECONCILED = "reconciled"        # differ, but the context difference explains it
    SUPERSEDES = "supersedes"        # later statement replaces an earlier one
    RELATED = "related"              # same measure, not directly comparable
    UNDETERMINED = "undetermined"    # system declined to judge


class ReasonCode(str, Enum):
    """Why the engine reached its verdict. Every relation carries one, so the
    UI can explain itself without paraphrasing a language model."""

    EXACT_MATCH = "exact_match"
    UNIT_CONVERTED_MATCH = "unit_converted_match"
    ROUNDED_MATCH = "rounded_match"
    VALUE_DIVERGENCE = "value_divergence"
    PERIOD_MISMATCH = "period_mismatch"
    PERIOD_LABEL_COLLISION = "period_label_collision"
    BASIS_MISMATCH = "basis_mismatch"
    VINTAGE_REVISION = "vintage_revision"
    ESTIMATE_VS_ACTUAL = "estimate_vs_actual"
    RESIDUAL_EXPLAINED = "residual_explained"
    STATE_SUPERSEDED = "state_superseded"
    SHARED_SENTENCE_AMBIGUITY = "shared_sentence_ambiguity"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    LLM_JUDGEMENT = "llm_judgement"


class Relation(BaseModel):
    id: str = ""
    a_id: str
    b_id: str
    type: RelationType
    reason_code: ReasonCode
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    explanation: str = ""
    # Numeric detail behind the verdict: values compared, deltas, tolerances.
    analysis: dict[str, Any] = Field(default_factory=dict)
    adjudicator: Literal["deterministic", "llm", "hybrid"] = "deterministic"
    cross_document: bool = True

    def compute_id(self) -> str:
        lo, hi = sorted([self.a_id, self.b_id])
        return "r_" + hashlib.sha1(f"{lo}|{hi}".encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


class Document(BaseModel):
    id: str
    filename: str
    title: str | None = None
    publisher: str | None = None
    published: date | None = None
    n_pages: int = 0
    sha256: str = ""
    ingested_at: str = ""
    status: str = "pending"
    stats: dict[str, Any] = Field(default_factory=dict)


class Block(BaseModel):
    """A contiguous run of text on one page, the unit of extraction."""

    id: str
    doc_id: str
    page: int
    text: str
    char_start: int
    char_end: int
    bbox: tuple[float, float, float, float] | None = None
    kind: Literal["prose", "table", "heading", "other"] = "prose"
    # Cheap signal used to decide whether this block is worth an LLM call.
    density: float = 0.0


class ExtractionFailure(BaseModel):
    """Recorded rather than swallowed. Case 4 of the brief asks what went wrong;
    a system that cannot answer that question honestly is not trustworthy."""

    id: str = ""
    doc_id: str
    page: int | None = None
    stage: str
    kind: str
    detail: str
    sample: str | None = None
