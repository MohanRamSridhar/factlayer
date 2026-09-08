"""Deciding how two facts relate, and being able to say why.

This is the part of the system the brief is actually about. Extraction is a
means; the question is whether two figures that disagree genuinely conflict or
merely describe different things.

The ladder below encodes one rule that matters more than any other:

    **every reconciliation check runs before the contradiction verdict.**

A system that shouts "contradiction" because one document reports a fiscal year
and another a calendar year is worse than useless on real filings, where that
mismatch is the common case rather than the exception. So disagreement is only
ever reported once the engine has failed to explain it.

No language model is consulted here. Verdicts come from comparing resolved
dates, canonical units and context fields, and every verdict carries a
``ReasonCode`` plus the arithmetic behind it. That is what lets the interface
explain a judgement instead of paraphrasing one: the explanation *is* the
computation, not a description of it.
"""

from __future__ import annotations

import re
from typing import Any

from ..models import (
    Fact,
    Period,
    Quantity,
    ReasonCode,
    Relation,
    RelationType,
)

# Relative tolerance for two figures to count as the same number. Deliberately
# tight: documents that mean the same figure usually agree to the digit, and a
# loose tolerance hides real contradictions.
DEFAULT_RELATIVE_TOLERANCE = 0.005
# Figures written with a hedge ("about 740 Mn") get a wider band.
APPROXIMATE_RELATIVE_TOLERANCE = 0.05
# Two publication dates this far apart make a revision plausible.
VINTAGE_GAP_DAYS = 45

_ESTIMATE_FIRMNESS = {
    "advance_estimate": 0,
    "projection": 0,
    "target": 0,
    "provisional": 1,
    "revised": 2,
    "actual": 3,
    "unknown": -1,
}

_DECIMALS_RE = re.compile(r"\d+\.(\d+)")
# Positional markers only -- the tokens that make two different windows *look*
# like the same one. Word boundaries are deliberately not used: the whole point
# is to match "Q2" inside both "Q2 FY25" and "2025Q2", where \b fails on the
# digit side. Broader tokens (fy, year, quarter) are excluded on purpose: two
# labels sharing only "FY" are visibly different and confuse nobody.
_PERIOD_TOKEN_RE = re.compile(r"(?<![a-z])(q[1-4]|h[12])(?![a-z])", re.I)


# --------------------------------------------------------------------------
# Numeric agreement
# --------------------------------------------------------------------------


def rounding_tolerance(quantity: Quantity) -> float:
    """Half of the last written digit, expressed in canonical units.

    "6.4 per cent" does not claim to be 6.400; it claims to be nearer 6.4 than
    6.3 or 6.5. Comparing it to a source reporting 6.37 as though both were
    exact manufactures a contradiction out of rounding, which is one of the
    most common false positives in this problem.
    """
    match = _DECIMALS_RE.search(quantity.raw or "")
    decimals = len(match.group(1)) if match else 0
    ulp = 10.0 ** (-decimals)
    return 0.5 * ulp * (quantity.magnitude or 1.0)


def value_tolerance(a: Quantity, b: Quantity) -> float:
    relative = (
        APPROXIMATE_RELATIVE_TOLERANCE
        if (a.approximate or b.approximate)
        else DEFAULT_RELATIVE_TOLERANCE
    )
    scale = max(abs(a.canonical_value or 0.0), abs(b.canonical_value or 0.0))
    return max(rounding_tolerance(a), rounding_tolerance(b), relative * scale)


def compare_values(a: Quantity, b: Quantity) -> dict[str, Any]:
    va, vb = a.canonical_value, b.canonical_value
    if va is None or vb is None:
        return {"comparable": False}
    delta = abs(va - vb)
    tolerance = value_tolerance(a, b)
    scale = max(abs(va), abs(vb)) or 1.0
    return {
        "comparable": True,
        "a_value": va,
        "b_value": vb,
        "unit": a.canonical_unit,
        "delta": delta,
        "relative_delta": delta / scale,
        "tolerance": tolerance,
        "agree": delta <= tolerance,
        # Distinguishes "identical" from "agrees once rounding is allowed",
        # which is a materially different claim to show a reviewer.
        "exact": delta == 0.0,
        "converted": (a.magnitude or 1.0) != (b.magnitude or 1.0)
        or (a.unit or "") != (b.unit or ""),
    }


# --------------------------------------------------------------------------
# Context comparison
# --------------------------------------------------------------------------


def _normalise_basis(values: list[str]) -> set[str]:
    return {v.strip().lower() for v in values if v and v.strip()}


def compare_periods(a: Period | None, b: Period | None) -> dict[str, Any]:
    """Classify how two time windows relate.

    ``status`` is one of: ``same``, ``different``, ``label_collision``,
    ``unknown``. The collision case is called out separately because it is the
    trap that string matching walks into -- two documents both saying "Q2" and
    meaning windows a year apart.
    """
    if a is None or b is None:
        return {"status": "unknown", "reason": "at least one fact has no period"}

    if a.resolved() and b.resolved():
        if a.same_window(b):
            return {
                "status": "same",
                "window": f"{a.start} to {a.end}",
                "a_label": a.label or a.raw,
                "b_label": b.label or b.raw,
            }
        # Both resolved and different. Did they *look* the same in the text?
        a_tokens = {m.group(0).lower() for m in _PERIOD_TOKEN_RE.finditer(a.raw or "")}
        b_tokens = {m.group(0).lower() for m in _PERIOD_TOKEN_RE.finditer(b.raw or "")}
        collision = bool(a_tokens & b_tokens) and (a.kind != b.kind or not a.overlaps(b))
        return {
            "status": "label_collision" if collision else "different",
            "a_window": f"{a.start} to {a.end}",
            "b_window": f"{b.start} to {b.end}",
            "a_label": a.label or a.raw,
            "b_label": b.label or b.raw,
            "a_kind": a.kind,
            "b_kind": b.kind,
            "overlaps": a.overlaps(b),
            "shared_tokens": sorted(a_tokens & b_tokens),
        }

    # Unresolved: fall back to the written label, which is weak but not nothing.
    if (a.raw or "").strip().lower() == (b.raw or "").strip().lower():
        return {"status": "same", "window": a.raw, "resolved": False}
    return {
        "status": "unknown",
        "reason": "at least one period could not be resolved to dates",
        "a_label": a.raw,
        "b_label": b.raw,
    }


def compare_basis(a: Fact, b: Fact) -> dict[str, Any]:
    basis_a = _normalise_basis(a.context.basis)
    basis_b = _normalise_basis(b.context.basis)
    only_a = sorted(basis_a - basis_b)
    only_b = sorted(basis_b - basis_a)
    return {
        "a_basis": sorted(basis_a),
        "b_basis": sorted(basis_b),
        "only_a": only_a,
        "only_b": only_b,
        "differs": bool(only_a or only_b),
    }


def compare_vintage(a: Fact, b: Fact) -> dict[str, Any]:
    firmness_a = _ESTIMATE_FIRMNESS.get(a.context.estimate_type, -1)
    firmness_b = _ESTIMATE_FIRMNESS.get(b.context.estimate_type, -1)
    gap_days: int | None = None
    later: str | None = None
    if a.context.as_of and b.context.as_of:
        gap_days = abs((a.context.as_of - b.context.as_of).days)
        later = a.id if a.context.as_of > b.context.as_of else b.id
    return {
        "a_estimate_type": a.context.estimate_type,
        "b_estimate_type": b.context.estimate_type,
        "estimate_differs": (
            a.context.estimate_type != b.context.estimate_type
            and -1 not in (firmness_a, firmness_b)
        ),
        "firmer": (
            (a.id if firmness_a > firmness_b else b.id)
            if firmness_a != firmness_b and -1 not in (firmness_a, firmness_b)
            else None
        ),
        "as_of_gap_days": gap_days,
        "later_fact": later,
        "revision_plausible": bool(gap_days is not None and gap_days >= VINTAGE_GAP_DAYS),
    }


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def _relation(
    a: Fact,
    b: Fact,
    rel_type: RelationType,
    reason: ReasonCode,
    confidence: float,
    explanation: str,
    analysis: dict[str, Any],
) -> Relation:
    rel = Relation(
        a_id=a.id,
        b_id=b.id,
        type=rel_type,
        reason_code=reason,
        confidence=round(min(1.0, max(0.0, confidence)), 3),
        explanation=explanation,
        analysis=analysis,
        adjudicator="deterministic",
        cross_document=a.doc_id != b.doc_id,
    )
    rel.id = rel.compute_id()
    return rel


def _describe(fact: Fact) -> str:
    period = fact.context.period
    when = f" for {period.label or period.raw}" if period else ""
    return f"{fact.display_value()}{when}"


def adjudicate(a: Fact, b: Fact) -> Relation | None:
    """Judge one candidate pair. Returns ``None`` if the pair is not worth a row.

    Assumes the caller has already established that the two facts share a
    blocking key (same measure, same entity); this function decides what that
    shared identity actually means.
    """
    if a.id == b.id:
        return None

    if a.fact_type != "numeric" or b.fact_type != "numeric":
        return _adjudicate_non_numeric(a, b)

    if a.quantity is None or b.quantity is None:
        return _relation(
            a, b, RelationType.UNDETERMINED, ReasonCode.INSUFFICIENT_CONTEXT, 0.3,
            "One side carries no parsed quantity, so the two cannot be compared numerically.",
            {},
        )

    values = compare_values(a.quantity, b.quantity)
    periods = compare_periods(a.context.period, b.context.period)
    basis = compare_basis(a, b)
    vintage = compare_vintage(a, b)
    analysis: dict[str, Any] = {
        "values": values, "periods": periods, "basis": basis, "vintage": vintage,
    }

    # Different canonical units means the registry grouped two things that are
    # not the same measurement. Record the link, decline the judgement.
    if not values.get("comparable") or not a.quantity.comparable_to(b.quantity):
        return _relation(
            a, b, RelationType.RELATED, ReasonCode.INSUFFICIENT_CONTEXT, 0.25,
            "Same measure and entity, but the two figures are not in comparable units "
            f"({a.quantity.canonical_unit or 'unknown'} vs {b.quantity.canonical_unit or 'unknown'}).",
            analysis,
        )

    # -- Step 1: do the numbers agree? -------------------------------------
    if values["agree"]:
        return _corroboration(a, b, values, periods, basis, analysis)

    # -- Step 2: the numbers differ. Try to explain it before condemning it.
    return _explain_or_contradict(a, b, values, periods, basis, vintage, analysis)


def _corroboration(
    a: Fact, b: Fact,
    values: dict[str, Any], periods: dict[str, Any], basis: dict[str, Any],
    analysis: dict[str, Any],
) -> Relation:
    """The values match. Decide how strong the corroboration is."""
    # Matching values across *different* windows is coincidence, not agreement:
    # two different years both growing 6.4% say nothing about each other.
    if periods["status"] in ("different", "label_collision"):
        return _relation(
            a, b, RelationType.RELATED, ReasonCode.PERIOD_MISMATCH, 0.4,
            f"Both report {a.display_value()}, but for different periods "
            f"({periods.get('a_label')} vs {periods.get('b_label')}), so the agreement is coincidental.",
            analysis,
        )

    if values["exact"] and not values["converted"]:
        reason, confidence, how = ReasonCode.EXACT_MATCH, 0.95, "identical figures"
    elif values["converted"]:
        reason, confidence, how = (
            ReasonCode.UNIT_CONVERTED_MATCH, 0.9,
            f"the same value once converted to {values['unit']}",
        )
    else:
        reason, confidence, how = (
            ReasonCode.ROUNDED_MATCH, 0.85,
            f"agreement within rounding (difference {values['delta']:.4g}, "
            f"tolerance {values['tolerance']:.4g})",
        )

    note = ""
    if periods["status"] == "unknown":
        confidence -= 0.2
        note = " Period could not be resolved on both sides, so the match is unverified in time."
    if basis["differs"]:
        confidence -= 0.1
        note += (
            f" Note the differing qualifiers ({basis['only_a'] or '-'} vs {basis['only_b'] or '-'})."
        )

    return _relation(
        a, b, RelationType.CORROBORATES, reason, confidence,
        f"{a.doc_id} reports {_describe(a)} and {b.doc_id} reports {_describe(b)}: {how}.{note}",
        analysis,
    )


def _explain_or_contradict(
    a: Fact, b: Fact,
    values: dict[str, Any], periods: dict[str, Any], basis: dict[str, Any],
    vintage: dict[str, Any], analysis: dict[str, Any],
) -> Relation:
    """The values disagree. Walk every explanation before calling it a conflict."""
    gap = (
        f"{a.display_value()} vs {b.display_value()} "
        f"(difference {values['delta']:.4g}, {values['relative_delta']:.1%})"
    )

    # 1. Different time windows explain almost everything.
    if periods["status"] == "label_collision":
        return _relation(
            a, b, RelationType.RECONCILED, ReasonCode.PERIOD_LABEL_COLLISION, 0.85,
            f"{gap}. The periods are written alike ({', '.join(periods['shared_tokens'])}) "
            f"but resolve to different windows: {periods['a_window']} against "
            f"{periods['b_window']}. Different periods, not conflicting figures.",
            analysis,
        )
    if periods["status"] == "different":
        return _relation(
            a, b, RelationType.RECONCILED, ReasonCode.PERIOD_MISMATCH, 0.8,
            f"{gap}. The figures cover different periods "
            f"({periods.get('a_label')}: {periods['a_window']}; "
            f"{periods.get('b_label')}: {periods['b_window']}), so they are not in conflict.",
            analysis,
        )

    # 2. Different qualifiers mean different things are being counted.
    if basis["differs"]:
        return _relation(
            a, b, RelationType.RECONCILED, ReasonCode.BASIS_MISMATCH, 0.75,
            f"{gap}. The two figures are computed on different bases "
            f"({basis['only_a'] or 'none'} vs {basis['only_b'] or 'none'}), "
            "which changes what is being counted.",
            analysis,
        )

    # 3. An estimate and a firmer figure for the same window is a revision.
    if vintage["estimate_differs"]:
        return _relation(
            a, b, RelationType.RECONCILED, ReasonCode.ESTIMATE_VS_ACTUAL, 0.7,
            f"{gap}. One figure is a {vintage['a_estimate_type']} and the other a "
            f"{vintage['b_estimate_type']}; the firmer figure supersedes rather than "
            "contradicts the softer one.",
            analysis,
        )

    # 4. Same window, same basis, but published far enough apart to be a revision.
    if vintage["revision_plausible"]:
        return _relation(
            a, b, RelationType.RECONCILED, ReasonCode.VINTAGE_REVISION, 0.6,
            f"{gap}. The documents were published {vintage['as_of_gap_days']} days apart, "
            "so the later figure is most likely a revision of the earlier one. "
            "Lower confidence: neither document labels it as such.",
            analysis,
        )

    # 5. Both figures came out of one sentence. A sentence almost never asserts
    #    two different values for the same measure over the same period; when it
    #    looks like it does, the extractor has split one statement badly ("rose
    #    from 4.5 to 4.8 per cent") or the registry has merged two neighbouring
    #    measures ("apparel" and "non-apparel" shares). Either way the fault is
    #    ours, not the document's, and reporting a contradiction would blame the
    #    source for our own ambiguity.
    if _same_sentence(a, b):
        return _relation(
            a, b, RelationType.UNDETERMINED, ReasonCode.SHARED_SENTENCE_AMBIGUITY, 0.45,
            f"{gap}. Both figures were extracted from the same sentence, so this is "
            "far more likely to be one statement split badly or two neighbouring "
            "measures merged than a real disagreement. Flagged for review rather "
            "than reported as a contradiction.",
            analysis,
        )

    # 6. Nothing in the context accounts for the gap.
    if periods["status"] == "unknown":
        return _relation(
            a, b, RelationType.UNDETERMINED, ReasonCode.INSUFFICIENT_CONTEXT, 0.4,
            f"{gap}. The figures differ, but at least one period could not be resolved, "
            "so the engine cannot tell whether they describe the same thing.",
            analysis,
        )

    confidence = 0.6 + min(0.35, values["relative_delta"] * 2)
    return _relation(
        a, b, RelationType.CONTRADICTS, ReasonCode.VALUE_DIVERGENCE, confidence,
        f"{gap}. Same measure, same entity, same period ({periods.get('window')}), "
        "same basis and comparable estimate types -- nothing in the surrounding context "
        "accounts for the difference.",
        analysis,
    )


# --------------------------------------------------------------------------
# Non-numeric facts
# --------------------------------------------------------------------------


def _same_sentence(a: Fact, b: Fact) -> bool:
    """Do both facts rest on the same span of source text?"""
    if a.doc_id != b.doc_id or a.evidence.page != b.evidence.page:
        return False
    qa = " ".join((a.evidence.quote or "").split()).lower()
    qb = " ".join((b.evidence.quote or "").split()).lower()
    if not qa or not qb:
        return False
    # Containment as well as equality: extractors often return a short quote and
    # a longer one covering it for two figures in the same clause.
    return qa == qb or qa in qb or qb in qa


def _text_overlap(a: str, b: str) -> float:
    ta = {t for t in re.findall(r"[a-z0-9]+", (a or "").lower()) if len(t) > 2}
    tb = {t for t in re.findall(r"[a-z0-9]+", (b or "").lower()) if len(t) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _adjudicate_non_numeric(a: Fact, b: Fact) -> Relation | None:
    """States and events: the interesting relation is supersession, not equality.

    "X is a director" and "X ceased to be a director" do not contradict each
    other; the second replaces the first. Treating them as a conflict is the
    mistake this branch exists to avoid.
    """
    overlap = _text_overlap(a.value_text or "", b.value_text or "")
    vintage = compare_vintage(a, b)
    analysis = {
        "text_overlap": round(overlap, 3),
        "a_value": a.value_text,
        "b_value": b.value_text,
        "vintage": vintage,
    }

    if overlap >= 0.6:
        return _relation(
            a, b, RelationType.CORROBORATES, ReasonCode.EXACT_MATCH, 0.7,
            f"Both documents assert the same thing about {a.context.entity}: "
            f"{a.value_text!r} and {b.value_text!r}.",
            analysis,
        )

    # Different assertions about the same subject and property. If one document
    # is materially later, the later one is the current state.
    if vintage["as_of_gap_days"] and vintage["revision_plausible"]:
        later, earlier = (a, b) if vintage["later_fact"] == a.id else (b, a)
        return _relation(
            a, b, RelationType.SUPERSEDES, ReasonCode.STATE_SUPERSEDED, 0.7,
            f"{earlier.doc_id} states {earlier.value_text!r}; the later {later.doc_id} "
            f"({vintage['as_of_gap_days']} days on) states {later.value_text!r}. "
            "The later document describes the current state rather than contradicting the earlier one.",
            analysis,
        )

    if overlap < 0.2:
        # Low overlap on the same measure and entity, with no time ordering to
        # resolve it. Flag rather than judge: this is where a reviewer should look.
        return _relation(
            a, b, RelationType.UNDETERMINED, ReasonCode.INSUFFICIENT_CONTEXT, 0.35,
            f"Both documents describe {a.measure_raw} for {a.context.entity} but assert "
            f"different things ({a.value_text!r} vs {b.value_text!r}), with no publication "
            "gap to establish which is current.",
            analysis,
        )
    return None
