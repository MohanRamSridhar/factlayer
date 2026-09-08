"""Tests for the adjudication ladder.

These are the four cases from the brief, expressed as assertions. They run with
no API key and no PDFs, because the judgement layer is deterministic -- which is
the point of keeping the language model out of it. A reviewer can verify the
reasoning without spending a request.
"""

from __future__ import annotations

from datetime import date

import pytest

from factlayer.link.adjudicate import adjudicate, rounding_tolerance
from factlayer.models import (
    ContextEnvelope,
    Evidence,
    Fact,
    Period,
    Quantity,
    ReasonCode,
    RelationType,
)


def make_fact(
    doc_id: str,
    measure: str,
    raw: str,
    value: float,
    *,
    unit: str = "%",
    kind: str = "percent",
    canonical_unit: str | None = "%",
    magnitude: float = 1.0,
    entity: str = "India",
    period: Period | None = None,
    basis: list[str] | None = None,
    estimate_type: str = "unknown",
    as_of: date | None = None,
    fact_type: str = "numeric",
    value_text: str | None = None,
    page: int = 1,
) -> Fact:
    quantity = None
    if fact_type == "numeric":
        quantity = Quantity(
            raw=raw, value=value, unit=unit, magnitude=magnitude, kind=kind,
            canonical_value=value * magnitude, canonical_unit=canonical_unit,
        )
    fact = Fact(
        doc_id=doc_id,
        fact_type=fact_type,
        measure_raw=measure,
        measure_key="m_test",
        quantity=quantity,
        value_text=value_text,
        context=ContextEnvelope(
            entity=entity, entity_key=entity.lower(), period=period,
            basis=basis or [], estimate_type=estimate_type, as_of=as_of,
        ),
        evidence=Evidence(doc_id=doc_id, page=page, quote=f"{measure} was {raw}"),
    )
    fact.id = fact.compute_id()
    return fact


def fy(label: str, start: date, end: date) -> Period:
    return Period(raw=label, kind="fiscal_year", start=start, end=end, label=label)


FY25 = lambda: fy("FY2024-25", date(2024, 4, 1), date(2025, 3, 31))  # noqa: E731
FY24 = lambda: fy("FY2023-24", date(2023, 4, 1), date(2024, 3, 31))  # noqa: E731


# -- Case 1: corroboration -------------------------------------------------


def test_identical_figures_corroborate():
    a = make_fact("survey", "real GDP growth", "6.4 per cent", 6.4, period=FY25())
    b = make_fact("rbi", "growth in real GDP", "6.4%", 6.4, period=FY25())
    rel = adjudicate(a, b)
    assert rel.type is RelationType.CORROBORATES
    assert rel.reason_code is ReasonCode.EXACT_MATCH
    assert rel.cross_document


def test_corroboration_survives_rounding():
    """6.4 does not claim to be 6.400, so 6.37 is agreement, not conflict."""
    a = make_fact("survey", "real GDP growth", "6.4 per cent", 6.4, period=FY25())
    b = make_fact("imf", "real GDP growth", "6.37 per cent", 6.37, period=FY25())
    rel = adjudicate(a, b)
    assert rel.type is RelationType.CORROBORATES
    assert rel.reason_code is ReasonCode.ROUNDED_MATCH


def test_corroboration_across_unit_scales():
    """8,142 crore and 81,415 million are the same money written two ways."""
    a = make_fact(
        "annual", "revenue", "Rs 8,142 Cr", 8142, unit="INR", kind="currency",
        canonical_unit="INR", magnitude=1e7, period=FY24(),
    )
    b = make_fact(
        "deck", "revenue", "INR 81,420 Mn", 81420, unit="INR", kind="currency",
        canonical_unit="INR", magnitude=1e6, period=FY24(),
    )
    rel = adjudicate(a, b)
    assert rel.type is RelationType.CORROBORATES
    assert rel.reason_code is ReasonCode.UNIT_CONVERTED_MATCH


# -- Case 2: genuine contradiction -----------------------------------------


def test_same_envelope_different_values_contradict():
    a = make_fact("survey", "real GDP growth", "6.4 per cent", 6.4, period=FY25())
    b = make_fact("imf", "real GDP growth", "7.2 per cent", 7.2, period=FY25())
    rel = adjudicate(a, b)
    assert rel.type is RelationType.CONTRADICTS
    assert rel.reason_code is ReasonCode.VALUE_DIVERGENCE
    assert rel.analysis["values"]["delta"] == pytest.approx(0.8)


# -- Case 3: apparent contradiction explained by context -------------------


def test_different_periods_are_reconciled_not_contradictory():
    a = make_fact("survey", "real GDP growth", "6.4 per cent", 6.4, period=FY25())
    b = make_fact("rbi", "real GDP growth", "9.2 per cent", 9.2, period=FY24())
    rel = adjudicate(a, b)
    assert rel.type is RelationType.RECONCILED
    assert rel.reason_code is ReasonCode.PERIOD_MISMATCH


def test_period_label_collision_is_called_out():
    """Both say 'Q2' and mean windows a year apart -- the string-matching trap."""
    a = make_fact(
        "indian", "GDP growth", "5.4 per cent", 5.4,
        period=Period(raw="Q2 FY25", kind="fiscal_quarter",
                      start=date(2024, 7, 1), end=date(2024, 9, 30), label="Q2 FY2024-25"),
    )
    b = make_fact(
        "imf", "GDP growth", "7.8 per cent", 7.8,
        period=Period(raw="2025Q2", kind="calendar_quarter",
                      start=date(2025, 4, 1), end=date(2025, 6, 30), label="2025Q2"),
    )
    rel = adjudicate(a, b)
    assert rel.type is RelationType.RECONCILED
    assert rel.reason_code is ReasonCode.PERIOD_LABEL_COLLISION


def test_different_basis_is_reconciled():
    a = make_fact("ar", "revenue", "10.0", 10.0, kind="currency", unit="INR",
                  canonical_unit="INR", period=FY24(), basis=["consolidated"])
    b = make_fact("ar", "revenue", "7.5", 7.5, kind="currency", unit="INR",
                  canonical_unit="INR", period=FY24(), basis=["standalone"])
    rel = adjudicate(a, b)
    assert rel.type is RelationType.RECONCILED
    assert rel.reason_code is ReasonCode.BASIS_MISMATCH


def test_estimate_versus_actual_is_reconciled():
    a = make_fact("survey", "real GDP growth", "6.4 per cent", 6.4,
                  period=FY25(), estimate_type="advance_estimate")
    b = make_fact("rbi", "real GDP growth", "6.5 per cent", 6.5,
                  period=FY25(), estimate_type="actual")
    rel = adjudicate(a, b)
    assert rel.type is RelationType.RECONCILED
    assert rel.reason_code is ReasonCode.ESTIMATE_VS_ACTUAL


def test_vintage_gap_explains_a_revision():
    a = make_fact("early", "real GDP growth", "6.4 per cent", 6.4,
                  period=FY25(), as_of=date(2025, 1, 31))
    b = make_fact("later", "real GDP growth", "6.9 per cent", 6.9,
                  period=FY25(), as_of=date(2025, 8, 30))
    rel = adjudicate(a, b)
    assert rel.type is RelationType.RECONCILED
    assert rel.reason_code is ReasonCode.VINTAGE_REVISION


def test_reconciliation_is_attempted_before_contradiction():
    """The ordering guarantee, stated as a test.

    A pair differing in period *and* value must never come back as a
    contradiction, no matter how large the numeric gap.
    """
    a = make_fact("a", "revenue", "100", 100.0, kind="currency", unit="INR",
                  canonical_unit="INR", period=FY24())
    b = make_fact("b", "revenue", "999", 999.0, kind="currency", unit="INR",
                  canonical_unit="INR", period=FY25())
    rel = adjudicate(a, b)
    assert rel.type is not RelationType.CONTRADICTS


# -- Case 4 support: declining to judge ------------------------------------


def test_unresolved_period_yields_undetermined_not_contradiction():
    a = make_fact("a", "growth", "6.4 per cent", 6.4,
                  period=Period(raw="last year", kind="unknown"))
    b = make_fact("b", "growth", "7.9 per cent", 7.9, period=FY25())
    rel = adjudicate(a, b)
    assert rel.type is RelationType.UNDETERMINED
    assert rel.reason_code is ReasonCode.INSUFFICIENT_CONTEXT


def test_incomparable_units_are_related_not_judged():
    a = make_fact("a", "revenue", "10 per cent", 10.0)
    b = make_fact("b", "revenue", "10 crore", 10.0, unit="INR", kind="currency",
                  canonical_unit="INR")
    rel = adjudicate(a, b)
    assert rel.type is RelationType.RELATED


def test_matching_values_in_different_periods_are_not_corroboration():
    """Two different years both growing 6.4% say nothing about each other."""
    a = make_fact("a", "growth", "6.4 per cent", 6.4, period=FY24())
    b = make_fact("b", "growth", "6.4 per cent", 6.4, period=FY25())
    rel = adjudicate(a, b)
    assert rel.type is RelationType.RELATED
    assert rel.reason_code is ReasonCode.PERIOD_MISMATCH


# -- State facts -----------------------------------------------------------


def test_later_status_supersedes_rather_than_contradicts():
    a = make_fact(
        "prospectus", "board membership", "", 0.0, fact_type="state",
        value_text="is a director of the company", entity="A. Director",
        as_of=date(2022, 5, 1),
    )
    b = make_fact(
        "annual", "board membership", "", 0.0, fact_type="state",
        value_text="ceased to be a director with effect from 30 June 2023",
        entity="A. Director", as_of=date(2024, 8, 1),
    )
    rel = adjudicate(a, b)
    assert rel.type is RelationType.SUPERSEDES
    assert rel.reason_code is ReasonCode.STATE_SUPERSEDED


# -- Tolerance -------------------------------------------------------------


def test_rounding_tolerance_tracks_written_precision():
    coarse = Quantity(raw="6.4 per cent", value=6.4, canonical_value=6.4, canonical_unit="%")
    fine = Quantity(raw="6.437 per cent", value=6.437, canonical_value=6.437, canonical_unit="%")
    assert rounding_tolerance(coarse) == pytest.approx(0.05)
    assert rounding_tolerance(fine) == pytest.approx(0.0005)


def test_a_fact_is_not_compared_to_itself():
    a = make_fact("a", "growth", "6.4 per cent", 6.4, period=FY25())
    assert adjudicate(a, a) is None
