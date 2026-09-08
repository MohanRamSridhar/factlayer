"""Resolving period expressions to concrete date windows.

Documents label time in mutually incompatible ways. In the starter data alone:
"FY24", "2023-24", "FY2024/25", "Q4 FY24", "H2:2024-25", "2025Q2", "as on
March 31, 2024". Two of those look almost identical and mean windows a year
apart.

The rule this module enforces is that comparison happens on resolved date
windows, never on labels. That is what stops the system reporting a contradiction
between a current account deficit of "1.2 per cent of GDP in Q2 of FY25"
(Jul-Sep 2024) and "0.2 percent of GDP in 2025Q2" (Apr-Jun 2025). Both say "Q2";
they are nine months apart.

The fiscal calendar is a *convention*, not a fact about the world, so it is
configurable and -- better -- inferred per document from statements like
"financial year ended March 31, 2024". No jurisdiction is hard-coded.
"""

from __future__ import annotations

import re
from datetime import date

from ..models import Period

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

# Default fiscal year end month when a document gives no evidence either way.
# April-March is the most common convention in the corpora this targets, but it
# is only a fallback: infer_fiscal_year_end() overrides it whenever the document
# states its own year end.
DEFAULT_FY_END_MONTH = 3


def _last_day(year: int, month: int) -> int:
    if month == 12:
        return 31
    nxt = date(year + (month // 12), (month % 12) + 1, 1)
    return (nxt.replace(day=1) - __import__("datetime").timedelta(days=1)).day


def fiscal_year_window(end_year: int, fy_end_month: int = DEFAULT_FY_END_MONTH) -> tuple[date, date]:
    """Window for the fiscal year *labelled* by ``end_year``.

    With a March year end, FY2024 runs 1 Apr 2023 to 31 Mar 2024.
    With a December year end the fiscal year coincides with the calendar year.
    """
    end = date(end_year, fy_end_month, _last_day(end_year, fy_end_month))
    if fy_end_month == 12:
        start = date(end_year, 1, 1)
    else:
        start = date(end_year - 1, fy_end_month + 1, 1)
    return start, end


def fiscal_quarter_window(
    end_year: int, quarter: int, fy_end_month: int = DEFAULT_FY_END_MONTH
) -> tuple[date, date]:
    """Window for quarter ``quarter`` of the fiscal year labelled ``end_year``."""
    fy_start, _ = fiscal_year_window(end_year, fy_end_month)
    start_month_index = fy_start.month - 1 + (quarter - 1) * 3
    start_year = fy_start.year + start_month_index // 12
    start_month = start_month_index % 12 + 1
    start = date(start_year, start_month, 1)
    end_month_index = start_month_index + 2
    end_year_ = fy_start.year + end_month_index // 12
    end_month = end_month_index % 12 + 1
    end = date(end_year_, end_month, _last_day(end_year_, end_month))
    return start, end


def calendar_quarter_window(year: int, quarter: int) -> tuple[date, date]:
    start_month = (quarter - 1) * 3 + 1
    end_month = start_month + 2
    return date(year, start_month, 1), date(year, end_month, _last_day(year, end_month))


def _expand_two_digit(y: str) -> int:
    n = int(y)
    if len(y) == 4:
        return n
    # Two-digit fiscal labels: 24 -> 2024. Anything above 70 is treated as 19xx.
    return 1900 + n if n >= 70 else 2000 + n


def _fy_end_from_span(first: str, second: str) -> int:
    """Resolve '2023-24' / 'FY2024/25' to the ending year."""
    start = _expand_two_digit(first)
    if len(second) <= 2:
        end = start - (start % 100) + int(second)
        if end < start:
            end += 100
        return end
    return _expand_two_digit(second)


# Ordered most-specific first: a quarter pattern must win over the bare year
# pattern embedded inside it.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Q4 FY24 / Q4FY2024 / 4QFY24 / Q4 FY2023-24
    ("fiscal_quarter", re.compile(
        r"\bQ(?P<q>[1-4])\s*[:\-]?\s*FY\s*(?P<y1>\d{2,4})(?:\s*[-/]\s*(?P<y2>\d{2,4}))?\b", re.I)),
    ("fiscal_quarter", re.compile(
        r"\b(?P<q>[1-4])\s*Q\s*FY\s*(?P<y1>\d{2,4})(?:\s*[-/]\s*(?P<y2>\d{2,4}))?\b", re.I)),
    # Q2 of FY25 / Q2 of fiscal 2025
    ("fiscal_quarter", re.compile(
        r"\bQ(?P<q>[1-4])\s+of\s+(?:FY|fiscal(?:\s+year)?)\s*(?P<y1>\d{2,4})(?:\s*[-/]\s*(?P<y2>\d{2,4}))?\b", re.I)),
    # Q2:2024-25 (RBI style) and Q2 2024-25
    ("fiscal_quarter", re.compile(
        r"\bQ(?P<q>[1-4])\s*[:\s]\s*(?P<y1>\d{4})\s*[-/]\s*(?P<y2>\d{2,4})\b", re.I)),
    # H1 FY25 / H2:2024-25
    ("fiscal_half", re.compile(
        r"\bH(?P<h>[12])\s*[:\-]?\s*(?:FY\s*)?(?P<y1>\d{2,4})(?:\s*[-/]\s*(?P<y2>\d{2,4}))?\b", re.I)),
    # 2025Q2 / 2025 Q2 / CY2025Q2  -- calendar quarter
    ("calendar_quarter", re.compile(r"\b(?:CY)?(?P<y>\d{4})\s*[:\-]?\s*Q(?P<q>[1-4])\b", re.I)),
    # Q2 2025 / Q2 CY2025  -- calendar quarter (4-digit year, no span)
    ("calendar_quarter", re.compile(r"\bQ(?P<q>[1-4])\s*[,\s]\s*(?:CY\s*)?(?P<y>\d{4})\b(?!\s*[-/]\s*\d)", re.I)),
    # FY2024-25 / FY 2024/25 / FY24-25
    ("fiscal_year", re.compile(
        r"\bFY\s*(?P<y1>\d{2,4})\s*[-/]\s*(?P<y2>\d{2,4})\b", re.I)),
    # FY24 / FY2024 / fiscal 2024 / fiscal year 2024
    ("fiscal_year", re.compile(
        r"\b(?:FY|fiscal(?:\s+year)?)\s*(?P<y1>\d{2,4})\b(?!\s*[-/]\s*\d)", re.I)),
    # 2023-24 / 2024-25 bare span (very common in Indian official documents)
    ("fiscal_year", re.compile(r"\b(?P<y1>(?:19|20)\d{2})\s*[-/]\s*(?P<y2>\d{2})\b")),
    # CY2024 / calendar year 2024
    ("calendar_year", re.compile(r"\b(?:CY|calendar\s+year)\s*(?P<y>\d{4})\b", re.I)),
    # as on/at/of 31 March 2024 | March 31, 2024
    ("instant", re.compile(
        r"\bas\s+(?:on|at|of)\s+(?P<d>\d{1,2})?\s*(?P<mon>[A-Za-z]{3,9})\.?,?\s*(?P<d2>\d{1,2})?,?\s*(?P<y>\d{4})\b", re.I)),
    # year ended March 31, 2024
    ("fiscal_year", re.compile(
        r"\b(?:year|period)\s+ended\s+(?P<d>\d{1,2})?\s*(?P<mon>[A-Za-z]{3,9})\.?,?\s*(?P<d2>\d{1,2})?,?\s*(?P<y>\d{4})\b", re.I)),
    # March 2025 (month granularity)
    ("month", re.compile(r"\b(?P<mon>[A-Za-z]{3,9})\.?\s+(?P<y>(?:19|20)\d{2})\b")),
    # bare 2024
    ("calendar_year", re.compile(r"\b(?P<y>(?:19|20)\d{2})\b")),
]


def parse_period(
    text: str, fy_end_month: int = DEFAULT_FY_END_MONTH
) -> Period | None:
    """Resolve the first period expression found in ``text``."""
    if not text:
        return None
    for kind, pattern in _PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        g = m.groupdict()
        raw = m.group(0).strip()
        try:
            if kind == "fiscal_quarter":
                end_year = (
                    _fy_end_from_span(g["y1"], g["y2"]) if g.get("y2") else _expand_two_digit(g["y1"])
                )
                q = int(g["q"])
                start, end = fiscal_quarter_window(end_year, q, fy_end_month)
                return Period(
                    raw=raw, kind="fiscal_quarter", start=start, end=end,
                    label=f"Q{q} FY{end_year}", fiscal_year_end_month=fy_end_month,
                )

            if kind == "fiscal_half":
                end_year = (
                    _fy_end_from_span(g["y1"], g["y2"]) if g.get("y2") else _expand_two_digit(g["y1"])
                )
                h = int(g["h"])
                q_start = 1 if h == 1 else 3
                s1, _ = fiscal_quarter_window(end_year, q_start, fy_end_month)
                _, e2 = fiscal_quarter_window(end_year, q_start + 1, fy_end_month)
                return Period(
                    raw=raw, kind="fiscal_half", start=s1, end=e2,
                    label=f"H{h} FY{end_year}", fiscal_year_end_month=fy_end_month,
                )

            if kind == "calendar_quarter":
                year = int(g["y"])
                q = int(g["q"])
                start, end = calendar_quarter_window(year, q)
                return Period(
                    raw=raw, kind="calendar_quarter", start=start, end=end,
                    label=f"{year}Q{q}",
                )

            if kind == "fiscal_year":
                if g.get("mon"):  # "year ended March 31, 2024"
                    mon = MONTHS.get(g["mon"].lower())
                    if mon is None:
                        continue
                    year = int(g["y"])
                    start, end = fiscal_year_window(year, mon)
                    return Period(
                        raw=raw, kind="fiscal_year", start=start, end=end,
                        label=f"FY{year}", fiscal_year_end_month=mon,
                    )
                end_year = (
                    _fy_end_from_span(g["y1"], g["y2"]) if g.get("y2") else _expand_two_digit(g["y1"])
                )
                start, end = fiscal_year_window(end_year, fy_end_month)
                return Period(
                    raw=raw, kind="fiscal_year", start=start, end=end,
                    label=f"FY{end_year}", fiscal_year_end_month=fy_end_month,
                )

            if kind == "calendar_year":
                year = int(g["y"])
                return Period(
                    raw=raw, kind="calendar_year",
                    start=date(year, 1, 1), end=date(year, 12, 31), label=f"CY{year}",
                )

            if kind == "instant":
                mon = MONTHS.get(g["mon"].lower())
                if mon is None:
                    continue
                day = int(g.get("d") or g.get("d2") or 1)
                year = int(g["y"])
                day = min(day, _last_day(year, mon))
                d = date(year, mon, day)
                return Period(raw=raw, kind="instant", start=d, end=d, label=d.isoformat())

            if kind == "month":
                mon = MONTHS.get(g["mon"].lower())
                if mon is None:
                    continue
                year = int(g["y"])
                return Period(
                    raw=raw, kind="month",
                    start=date(year, mon, 1),
                    end=date(year, mon, _last_day(year, mon)),
                    label=f"{year}-{mon:02d}",
                )
        except (ValueError, KeyError, TypeError):
            continue
    return None


_FY_END_HINTS = [
    re.compile(r"(?:financial|fiscal)\s+year\s+ende[dr]\s+(?:\d{1,2}\s+)?([A-Za-z]{3,9})", re.I),
    re.compile(r"(?:year|period)\s+ended\s+([A-Za-z]{3,9})\s+\d{1,2},?\s*\d{4}", re.I),
    re.compile(r"for\s+the\s+year\s+ended\s+(?:\d{1,2}\s+)?([A-Za-z]{3,9})", re.I),
]


def infer_fiscal_year_end(text: str) -> int | None:
    """Infer a document's fiscal year end month from its own wording.

    Preferring the document's own statement over a global default is what keeps
    this generic: a US 10-K saying "fiscal year ended September 30" resolves its
    FY labels correctly with no configuration.
    """
    counts: dict[int, int] = {}
    for pattern in _FY_END_HINTS:
        for m in pattern.finditer(text):
            mon = MONTHS.get(m.group(1).lower())
            if mon:
                counts[mon] = counts.get(mon, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])
