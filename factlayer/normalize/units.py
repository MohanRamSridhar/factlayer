"""Unit detection and canonicalisation.

The payoff is case 1 of the brief: an earnings deck says "₹8,142 Cr" and the
annual report says "81,415.38" in a table headed "(₹ in Million)". Those are the
same fact. Reducing both to whole rupees makes them directly comparable without
anyone hard-coding that particular pair.

Deliberate non-goal: currency conversion. Converting INR to USD needs an
exchange rate with its own date and source, and silently applying one would
manufacture precision the documents do not contain. Cross-currency pairs are
surfaced as related-but-not-comparable instead.
"""

from __future__ import annotations

import re

from .numbers import find_scale_suffix, parse_number

CURRENCY_SYMBOLS: dict[str, str] = {
    "₹": "INR",
    "rs": "INR",
    "rs.": "INR",
    "inr": "INR",
    "rupees": "INR",
    "rupee": "INR",
    "$": "USD",
    "us$": "USD",
    "usd": "USD",
    "dollars": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
    "¥": "JPY",
    "jpy": "JPY",
    "aed": "AED",
    "sgd": "SGD",
    "cny": "CNY",
    "rmb": "CNY",
}

PERCENT_TOKENS = ("%", "per cent", "percent", "percentage points", "pp", "bps", "basis points")

# Units where a canonical base makes comparison meaningful.
MASS_UNITS: dict[str, float] = {
    "kg": 1.0,
    "kgs": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "ton": 1000.0,
    "tons": 1000.0,
    "tonne": 1000.0,
    "tonnes": 1000.0,
    "mt": 1000.0,
}

DURATION_UNITS: dict[str, float] = {
    "day": 1.0,
    "days": 1.0,
    "week": 7.0,
    "weeks": 7.0,
    "month": 30.436875,
    "months": 30.436875,
    "year": 365.25,
    "years": 365.25,
}

_CURRENCY_PREFIX_RE = re.compile(
    r"(?P<cur>₹|\$|€|£|¥|Rs\.?|INR|USD|US\$|EUR|GBP|JPY|AED|SGD|CNY|RMB)\s*",
    re.IGNORECASE,
)


class UnitReading:
    __slots__ = ("kind", "unit", "canonical_unit", "factor", "is_percent_of_gdp")

    def __init__(
        self,
        kind: str,
        unit: str | None,
        canonical_unit: str | None,
        factor: float = 1.0,
    ) -> None:
        self.kind = kind
        self.unit = unit
        self.canonical_unit = canonical_unit
        self.factor = factor


def singularise(token: str) -> str:
    t = token.lower().strip(" .,;:")
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("ses") or t.endswith("xes"):
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def detect_currency(text_before: str) -> str | None:
    """Find a currency marker in the text immediately preceding a number."""
    window = text_before[-40:]
    matches = list(_CURRENCY_PREFIX_RE.finditer(window))
    if not matches:
        return None
    token = matches[-1].group("cur").lower().rstrip(".")
    return CURRENCY_SYMBOLS.get(token) or CURRENCY_SYMBOLS.get(token + ".")


def detect_percent(text_after: str) -> str | None:
    head = text_after[:24].lower().lstrip()
    if head.startswith("%"):
        return "percent"
    for token in ("per cent", "percent", "percentage point", "bps", "basis point"):
        if head.startswith(token):
            return "basis_points" if token in ("bps", "basis point") else "percent"
    return None


def classify(
    raw_text: str,
    number_start: int,
    number_end: int,
    unit_hint: str | None = None,
) -> UnitReading:
    """Work out what kind of quantity a number in ``raw_text`` is.

    ``unit_hint`` lets a caller pass a unit stated elsewhere -- typically a table
    header like "(₹ in Million)" -- which is how table cells get their units,
    since the cell itself carries no marker.
    """
    before = raw_text[:number_start]
    after = raw_text[number_end:]

    pct = detect_percent(after)
    if pct == "percent":
        return UnitReading("percent", "%", "percent")
    if pct == "basis_points":
        return UnitReading("percent", "bps", "percent", factor=0.01)

    currency = detect_currency(before)
    if currency is None and unit_hint:
        # The hint is usually a whole table header ("(₹ in Million)"), so search
        # inside it rather than treating it as a bare token.
        hint_match = _CURRENCY_PREFIX_RE.search(unit_hint)
        if hint_match:
            token = hint_match.group("cur").lower().rstrip(".")
            currency = CURRENCY_SYMBOLS.get(token) or CURRENCY_SYMBOLS.get(token + ".")
    if currency:
        return UnitReading("currency", currency, currency)

    # Unit noun after the number, e.g. "1.4 Mn Tons", "31 days"
    m = re.match(r"\s*(?:[A-Za-z]{1,12}\.?\s+)?([A-Za-z][A-Za-z\-]{1,20})", after)
    if m:
        noun = singularise(m.group(1))
        if noun in MASS_UNITS:
            return UnitReading("count", noun, "kg", factor=MASS_UNITS[noun])
        if noun in DURATION_UNITS:
            return UnitReading("duration", noun, "days", factor=DURATION_UNITS[noun])

    if unit_hint:
        hint = unit_hint.lower().strip()
        if any(p in hint for p in PERCENT_TOKENS):
            return UnitReading("percent", "%", "percent")

    return UnitReading("other", None, None)


def build_quantity(
    text: str,
    unit_hint: str | None = None,
    magnitude_hint: float | None = None,
):
    """Parse ``text`` into a :class:`~factlayer.models.Quantity`.

    ``magnitude_hint`` covers the table case where the scale lives in a header
    ("₹ in Million") rather than beside the number.
    """
    from ..models import Quantity  # local import to avoid a cycle

    parsed = parse_number(text)
    if parsed is None:
        return None

    scale = find_scale_suffix(text, parsed.end)
    magnitude = scale[0] if scale else (magnitude_hint or 1.0)
    unit_end = parsed.end + (len(scale[1]) + 1 if scale else 0)

    reading = classify(text, parsed.start, unit_end, unit_hint=unit_hint)

    canonical_value = None
    if reading.canonical_unit is not None:
        canonical_value = parsed.value * magnitude * reading.factor
        if reading.canonical_unit == "percent" and reading.unit == "bps":
            canonical_value = parsed.value * 0.01

    return Quantity(
        raw=text.strip()[:120],
        value=parsed.value,
        unit=reading.unit,
        magnitude=magnitude,
        kind=reading.kind,  # type: ignore[arg-type]
        canonical_value=canonical_value,
        canonical_unit=reading.canonical_unit,
        approximate=parsed.approximate,
    )


def magnitude_from_header(text: str) -> float | None:
    """Read a scale declaration such as '(₹ in Million)' or 'Rs. crore'."""
    m = re.search(
        r"\bin\s+(hundred|thousand|lakhs?|lacs?|millions?|crores?|billions?|trillions?)\b",
        text,
        re.IGNORECASE,
    )
    if not m:
        m = re.search(
            r"(?:₹|rs\.?|inr|usd|\$)\s*(?:in\s*)?(thousand|lakhs?|lacs?|millions?|crores?|billions?)\b",
            text,
            re.IGNORECASE,
        )
    if not m:
        return None
    from .numbers import scale_for

    return scale_for(m.group(1))
