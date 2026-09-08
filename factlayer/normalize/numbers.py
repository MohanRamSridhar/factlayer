"""Parsing numbers as documents actually write them.

Financial and statistical PDFs use conventions that trip up naive parsers:
negatives in parentheses, Indian lakh/crore digit grouping, scale words glued to
the digits, and approximation markers. Getting these wrong silently produces
facts that are off by orders of magnitude, so this module is deliberately
conservative: if a token cannot be read confidently it is rejected rather than
guessed at.
"""

from __future__ import annotations

import re

# Scale words -> multiplier. Ordered longest-first when matched so that
# "billion" is not shadowed by "bn".
SCALE_WORDS: dict[str, float] = {
    "hundred": 1e2,
    "thousand": 1e3,
    "k": 1e3,
    "lakh": 1e5,
    "lakhs": 1e5,
    "lac": 1e5,
    "lacs": 1e5,
    "million": 1e6,
    "millions": 1e6,
    "mn": 1e6,
    "mln": 1e6,
    "m": 1e6,
    "crore": 1e7,
    "crores": 1e7,
    "cr": 1e7,
    "billion": 1e9,
    "billions": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
    "tr": 1e12,
}

_SCALE_ALTERNATION = "|".join(
    sorted((re.escape(w) for w in SCALE_WORDS), key=len, reverse=True)
)

# A number body: 1,234.56 / 1,23,456 / 1234 / .5
_NUM_BODY = r"\d{1,3}(?:[, \s]\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+"

NUMBER_RE = re.compile(
    rf"""
    (?P<open_paren>\()?                       # parenthesised negative
    \s*
    (?P<sign>[-−+])?                     # explicit sign (incl. unicode minus)
    \s*
    (?P<body>{_NUM_BODY})
    \s*
    (?P<close_paren>\))?
    """,
    re.VERBOSE,
)

APPROX_MARKERS = (
    "about",
    "approximately",
    "approx",
    "around",
    "nearly",
    "roughly",
    "circa",
    "~",
    "over",
    "more than",
    "almost",
    "close to",
    "in excess of",
)


class ParsedNumber:
    __slots__ = ("value", "raw", "start", "end", "negative_by_paren", "approximate")

    def __init__(
        self,
        value: float,
        raw: str,
        start: int,
        end: int,
        negative_by_paren: bool = False,
        approximate: bool = False,
    ) -> None:
        self.value = value
        self.raw = raw
        self.start = start
        self.end = end
        self.negative_by_paren = negative_by_paren
        self.approximate = approximate

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ParsedNumber({self.value!r}, raw={self.raw!r})"


def _strip_grouping(body: str) -> str:
    return re.sub(r"[, \s]", "", body)


def plausible_grouping(body: str) -> bool:
    """Reject digit groupings that match neither Western nor Indian conventions.

    Western: 1,234,567. Indian: 12,34,567 (trailing group of 3, then groups of 2).
    A token like "1,23,45,6" is more likely a mangled table cell than a number,
    and turning it into 123456 would invent a fact.
    """
    if "," not in body:
        return True
    integer_part = body.split(".")[0]
    groups = integer_part.split(",")
    if len(groups) == 1:
        return True
    if not (1 <= len(groups[0]) <= 3):
        return False
    tail = groups[1:]
    if all(len(g) == 3 for g in tail):
        return True  # Western
    # Indian: last group is 3 digits, the rest are 2
    if len(tail[-1]) == 3 and all(len(g) == 2 for g in tail[:-1]):
        return True
    return False


def parse_number(text: str) -> ParsedNumber | None:
    """Parse the first number in ``text``. Returns None when nothing parses."""
    for m in NUMBER_RE.finditer(text):
        body = m.group("body")
        if not any(ch.isdigit() for ch in body):
            continue
        if not plausible_grouping(body):
            continue
        cleaned = _strip_grouping(body)
        try:
            value = float(cleaned)
        except ValueError:
            continue
        negative_by_paren = bool(m.group("open_paren") and m.group("close_paren"))
        if m.group("sign") in {"-", "−"}:
            value = -value
        if negative_by_paren:
            value = -abs(value)
        prefix = text[max(0, m.start() - 24) : m.start()].lower()
        approximate = any(marker in prefix for marker in APPROX_MARKERS)
        return ParsedNumber(
            value=value,
            raw=m.group(0).strip(),
            start=m.start(),
            end=m.end(),
            negative_by_paren=negative_by_paren,
            approximate=approximate,
        )
    return None


def scale_for(token: str) -> float | None:
    """Multiplier for a scale word, or None if the token is not one."""
    return SCALE_WORDS.get(token.strip().lower().rstrip("."))


def find_scale_suffix(text: str, from_index: int = 0) -> tuple[float, str] | None:
    """Find a scale word immediately following ``from_index``.

    Handles the glued form ("₹127Cr") as well as the spaced form ("8,142 Cr").
    """
    tail = text[from_index:]
    m = re.match(rf"\s*({_SCALE_ALTERNATION})\b\.?", tail, re.IGNORECASE)
    if not m:
        return None
    token = m.group(1)
    scale = scale_for(token)
    if scale is None:
        return None
    # Guard against a bare "m"/"b"/"k" that is really the start of a word.
    if len(token) == 1 and not re.match(rf"\s*{token}\b", tail, re.IGNORECASE):
        return None
    return scale, token


def format_compact(value: float) -> str:
    """Human-friendly rendering used in explanations."""
    a = abs(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= threshold:
            return f"{value / threshold:,.4g}{suffix}"
    if a == 0:
        return "0"
    if a < 0.01:
        return f"{value:.4g}"
    return f"{value:,.4g}"
