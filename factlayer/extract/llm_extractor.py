"""LLM-driven fact extraction with deterministic post-processing.

The split of responsibility is the point of this module. The model reads a page
and says "there is a claim here, it is about X, worded like this, and here is the
sentence it comes from". Everything measurable is then recomputed in Python:
the number is parsed, the unit converted, the period resolved to dates, the
entity keyed, and the quote verified against the actual page text.

That ordering matters. If the model were trusted to normalise, the same figure
could normalise two different ways on two different pages and quietly fail to
match itself. And any fact whose quote cannot be found in the source is dropped
with a recorded reason, so hallucinations cannot enter the knowledge layer.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Iterable

from ..ingest.chunk import ExtractionUnit
from ..ingest.pdf import IngestedDocument
from ..llm import LLMProvider, map_concurrent
from ..models import (
    ContextEnvelope,
    Evidence,
    ExtractionFailure,
    Fact,
    Quantity,
)
from ..normalize.entities import entity_key, is_deictic
from ..normalize.periods import DEFAULT_FY_END_MONTH, parse_period
from ..normalize.units import build_quantity
from .grounding import contains_value, locate_quote
from .prompts import (
    DOC_METADATA_INSTRUCTIONS,
    DOC_METADATA_SYSTEM,
    EXTRACTION_INSTRUCTIONS,
    EXTRACTION_SYSTEM,
)

log = logging.getLogger("factlayer.extract")

VALID_FACT_TYPES = {"numeric", "state", "event", "attribute"}
VALID_ESTIMATE_TYPES = {
    "actual", "provisional", "revised", "advance_estimate",
    "projection", "target", "unknown",
}


class ExtractionResult:
    def __init__(self) -> None:
        self.facts: list[Fact] = []
        self.failures: list[ExtractionFailure] = []

    def extend(self, other: "ExtractionResult") -> None:
        self.facts.extend(other.facts)
        self.failures.extend(other.failures)


def _parse_date(value: Any) -> date | None:
    if not value or not isinstance(value, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def extract_document_metadata(
    doc: IngestedDocument, provider: LLMProvider, model: str
) -> dict[str, Any]:
    """Read bibliographic details and the document's principal entity.

    The principal entity is load-bearing, not decorative: documents talk about
    themselves in the third person ("the Company", "the Bank"), and those
    references have to resolve to a real name before facts from two documents can
    be compared.
    """
    head = "\n\n".join(page.text for page in doc.pages[:3])[:9000]
    user = f"{DOC_METADATA_INSTRUCTIONS}\n\n--- DOCUMENT OPENING ---\n{head}"
    try:
        data = provider.complete_json(
            DOC_METADATA_SYSTEM, user, model=model, max_tokens=900
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("metadata extraction failed for %s: %s", doc.document.filename, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _build_prompt(unit: ExtractionUnit) -> str:
    context = unit.prompt_context()
    parts = [EXTRACTION_INSTRUCTIONS]
    if context:
        parts.append(f"--- PAGE CONTEXT ---\n{context}")
    parts.append(f"--- SOURCE TEXT (page {unit.page}) ---\n{unit.text}")
    return "\n\n".join(parts)


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def _coerce_page(value: Any, unit: ExtractionUnit) -> int:
    """Resolve the page a claim was reported against, defaulting to the unit's."""
    allowed = unit.pages or [unit.page]
    try:
        page = int(str(value).strip())
    except (TypeError, ValueError):
        return unit.page
    return page if page in allowed else unit.page


def _coerce_basis(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    out: list[str] = []
    for item in value:
        text = _coerce_str(item)
        if text:
            out.append(text.lower().strip(" .;"))
    # Order-insensitive comparison later; keep sorted and unique.
    return sorted(set(out))


def build_fact(
    raw: dict[str, Any],
    unit: ExtractionUnit,
    page_texts: dict[int, str],
    doc_id: str,
    principal_entity: str | None,
    published: date | None,
    fy_end_month: int,
    extractor: str,
) -> tuple[Fact | None, ExtractionFailure | None]:
    """Turn one raw model object into a validated, normalised Fact.

    Returns ``(fact, None)`` on success or ``(None, failure)`` with the reason
    the candidate was rejected. Rejections are data, not noise: they are what
    case 4 of the brief asks the system to be honest about.
    """

    def fail(kind: str, detail: str) -> tuple[None, ExtractionFailure]:
        return None, ExtractionFailure(
            doc_id=doc_id,
            page=unit.page,
            stage="extract",
            kind=kind,
            detail=detail,
            sample=str(raw)[:400],
        )

    if not isinstance(raw, dict):
        return fail("malformed_item", "extractor returned a non-object element")

    quote = _coerce_str(raw.get("quote"))
    if not quote:
        return fail("missing_quote", "no quote supplied, so the claim cannot be grounded")

    # Which page did this come from? A batched unit spans several, and the model
    # reports the one it read. An out-of-range answer is not trusted: it falls
    # back to the unit's first page, where the quote almost certainly will not
    # be found, so the claim is rejected rather than filed under a wrong page.
    page = _coerce_page(raw.get("page"), unit)
    page_text = page_texts.get(page, unit.text)

    location = locate_quote(page_text, quote)
    if location is None and len(unit.pages) > 1:
        # Tolerate a misreported page when the quote is genuinely present
        # elsewhere in the batch: the evidence is real, only the label was wrong.
        for candidate in unit.pages:
            if candidate == page:
                continue
            found = locate_quote(page_texts.get(candidate, ""), quote)
            if found is not None:
                page, location = candidate, found
                break
    if location is None:
        return fail(
            "ungrounded_quote",
            "quote could not be located in the source page; the claim was discarded",
        )

    measure = _coerce_str(raw.get("measure"))
    if not measure:
        return fail("missing_measure", "claim has no measure")

    subject = _coerce_str(raw.get("subject")) or ""
    if not subject or is_deictic(subject):
        if principal_entity:
            subject = principal_entity
        elif not subject:
            return fail("missing_subject", "claim has no subject and no document entity to fall back on")

    fact_type = (_coerce_str(raw.get("fact_type")) or "numeric").lower()
    if fact_type not in VALID_FACT_TYPES:
        fact_type = "numeric"

    value_raw = _coerce_str(raw.get("value_raw"))
    value_text = _coerce_str(raw.get("value_text"))

    quantity: Quantity | None = None
    hint_applied = False
    if value_raw:
        if not contains_value(location.text, value_raw):
            return fail(
                "value_not_in_quote",
                f"value {value_raw!r} does not appear in the grounded quote",
            )
        quantity = build_quantity(
            value_raw,
            unit_hint=unit.unit_hint,
            magnitude_hint=unit.magnitude_hint,
        )
        if quantity is None:
            return fail("unparseable_value", f"could not parse a number from {value_raw!r}")
        # Record when a page-level scale note was needed, so a reviewer can see
        # that the magnitude was inferred rather than stated beside the number.
        hint_applied = bool(
            unit.magnitude_hint
            and quantity.magnitude == unit.magnitude_hint
            and str(unit.magnitude_hint) not in value_raw
        )
    elif not value_text:
        return fail("empty_value", "claim carries neither a numeric nor a textual value")

    period_raw = _coerce_str(raw.get("period"))
    period = parse_period(period_raw, fy_end_month=fy_end_month) if period_raw else None
    if period_raw and period is None:
        # Keep the fact but remember the label; an unresolved period simply makes
        # the fact harder to compare, which the engine handles explicitly.
        from ..models import Period

        period = Period(raw=period_raw, kind="unknown")

    estimate_type = (_coerce_str(raw.get("estimate_type")) or "unknown").lower()
    if estimate_type not in VALID_ESTIMATE_TYPES:
        estimate_type = "unknown"

    try:
        confidence = float(raw.get("confidence", 0.6))
    except (TypeError, ValueError):
        confidence = 0.6
    confidence = max(0.0, min(1.0, confidence))
    # A fuzzily-matched quote is weaker evidence than an exact one.
    confidence *= 1.0 if location.method != "fuzzy" else 0.85

    context = ContextEnvelope(
        entity=subject,
        entity_key=entity_key(subject),
        period=period,
        basis=_coerce_basis(raw.get("basis")),
        estimate_type=estimate_type,  # type: ignore[arg-type]
        attributed_to=_coerce_str(raw.get("attributed_to")),
        as_of=published,
    )

    evidence = Evidence(
        doc_id=doc_id,
        page=page,
        quote=location.text.strip(),
        char_start=location.start,
        char_end=location.end,
        block_id=unit.blocks[0].id if unit.blocks else None,
        bbox=unit.blocks[0].bbox if unit.blocks else None,
    )

    fact = Fact(
        doc_id=doc_id,
        fact_type=fact_type,  # type: ignore[arg-type]
        measure_raw=measure,
        quantity=quantity,
        value_text=value_text,
        context=context,
        evidence=evidence,
        confidence=round(confidence, 3),
        extractor=extractor,
        extra={
            "unit_id": unit.id,
            "match_method": location.method,
            "match_score": round(location.score, 3),
            "page_scale_note_applied": hint_applied,
            "page_scale_note": unit.unit_hint,
            "section": unit.heading,
        },
    )
    fact.id = fact.compute_id()
    return fact, None


class LLMExtractor:
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        concurrency: int = 8,
        max_tokens: int = 6000,
    ) -> None:
        self.provider = provider
        self.model = model
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.name = f"llm:{provider.name}:{model}"

    def extract_units(
        self,
        units: list[ExtractionUnit],
        page_texts: dict[int, str],
        doc_id: str,
        principal_entity: str | None,
        published: date | None,
        fy_end_month: int = DEFAULT_FY_END_MONTH,
        progress=None,
    ) -> ExtractionResult:
        result = ExtractionResult()

        def run(unit: ExtractionUnit) -> tuple[ExtractionUnit, Any]:
            payload = self.provider.complete_json(
                EXTRACTION_SYSTEM,
                _build_prompt(unit),
                model=self.model,
                max_tokens=self.max_tokens,
            )
            return unit, payload

        def on_error(unit: ExtractionUnit, exc: Exception) -> None:
            result.failures.append(
                ExtractionFailure(
                    doc_id=doc_id,
                    page=unit.page,
                    stage="extract",
                    kind="llm_call_failed",
                    detail=str(exc)[:400],
                    sample=unit.text[:200],
                )
            )

        outputs = map_concurrent(
            units, run, concurrency=self.concurrency, on_error=on_error, progress=progress
        )

        for output in outputs:
            if output is None:
                continue
            unit, payload = output
            items = payload if isinstance(payload, list) else payload.get("facts", [])
            if not isinstance(items, list):
                result.failures.append(
                    ExtractionFailure(
                        doc_id=doc_id, page=unit.page, stage="extract",
                        kind="unexpected_shape",
                        detail=f"expected a list, got {type(payload).__name__}",
                    )
                )
                continue
            for raw in items:
                fact, failure = build_fact(
                    raw, unit, page_texts, doc_id, principal_entity,
                    published, fy_end_month, self.name,
                )
                if fact is not None:
                    result.facts.append(fact)
                elif failure is not None:
                    result.failures.append(failure)

        return dedupe_facts(result)


def dedupe_facts(result: ExtractionResult) -> ExtractionResult:
    """Collapse identical facts produced by overlapping units.

    Blocks are repeated across unit boundaries on purpose, so the same sentence
    can be extracted twice. Keeping the higher-confidence copy is enough; the
    fact id already encodes everything that makes two facts the same claim.
    """
    best: dict[str, Fact] = {}
    for fact in result.facts:
        existing = best.get(fact.id)
        if existing is None or fact.confidence > existing.confidence:
            best[fact.id] = fact
    merged = ExtractionResult()
    merged.facts = list(best.values())
    merged.failures = result.failures
    return merged
