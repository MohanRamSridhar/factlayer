"""Verifying that an extracted quote really appears in the source.

This is the project's main defence against invented facts. A model is asked to
copy a span verbatim; if that span cannot be located in the page it came from,
the fact is rejected and the failure is recorded. Nothing reaches the knowledge
layer without a resolved character offset into a real page.

Matching is tolerant of whitespace only. PDF text extraction reflows lines
unpredictably, so requiring byte equality would reject good facts, but allowing
loose paraphrase would defeat the purpose.
"""

from __future__ import annotations

import difflib
import re


def _normalise(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace, keeping a map from normalised index -> original index."""
    out_chars: list[str] = []
    index_map: list[int] = []
    previous_space = True  # strip leading whitespace
    for i, ch in enumerate(text):
        if ch.isspace():
            if previous_space:
                continue
            out_chars.append(" ")
            index_map.append(i)
            previous_space = True
        else:
            out_chars.append(ch)
            index_map.append(i)
            previous_space = False
    return "".join(out_chars), index_map


class QuoteLocation:
    __slots__ = ("start", "end", "method", "score", "text")

    def __init__(self, start: int, end: int, method: str, score: float, text: str) -> None:
        self.start = start
        self.end = end
        self.method = method
        self.score = score
        self.text = text

    def __repr__(self) -> str:  # pragma: no cover
        return f"QuoteLocation({self.start}, {self.end}, {self.method}, {self.score:.2f})"


def locate_quote(
    haystack: str, quote: str, min_score: float = 0.82
) -> QuoteLocation | None:
    """Find ``quote`` inside ``haystack``.

    Returns offsets into the *original* ``haystack`` so evidence can be
    highlighted, along with how the match was made, which is recorded on the
    fact for auditability.
    """
    if not quote or not haystack:
        return None
    quote = quote.strip()
    if len(quote) < 8:
        return None

    # 1. Exact.
    index = haystack.find(quote)
    if index != -1:
        return QuoteLocation(index, index + len(quote), "exact", 1.0, haystack[index : index + len(quote)])

    # 2. Whitespace-insensitive.
    norm_hay, hay_map = _normalise(haystack)
    norm_quote, _ = _normalise(quote)
    index = norm_hay.find(norm_quote)
    if index != -1:
        start = hay_map[index]
        end_idx = min(index + len(norm_quote) - 1, len(hay_map) - 1)
        end = hay_map[end_idx] + 1
        return QuoteLocation(start, end, "whitespace", 0.99, haystack[start:end])

    # 3. Anchored fuzzy match. Anchor on the rarest long token in the quote so we
    #    only diff a small window rather than the whole page.
    anchor = _pick_anchor(norm_quote, norm_hay)
    if anchor is None:
        return None
    window_radius = len(norm_quote) + 60
    best: tuple[float, int, int] | None = None
    for anchor_index in _find_all(norm_hay, anchor):
        lo = max(0, anchor_index - window_radius)
        hi = min(len(norm_hay), anchor_index + window_radius)
        window = norm_hay[lo:hi]
        matcher = difflib.SequenceMatcher(None, norm_quote, window, autojunk=False)
        block = matcher.find_longest_match(0, len(norm_quote), 0, len(window))
        if block.size < len(norm_quote) * 0.5:
            continue
        cand_start = lo + block.b - block.a
        cand_end = cand_start + len(norm_quote)
        cand_start = max(0, cand_start)
        cand_end = min(len(norm_hay), cand_end)
        score = difflib.SequenceMatcher(
            None, norm_quote, norm_hay[cand_start:cand_end], autojunk=False
        ).ratio()
        if best is None or score > best[0]:
            best = (score, cand_start, cand_end)

    if best is None or best[0] < min_score:
        return None

    score, ns, ne = best
    start = hay_map[min(ns, len(hay_map) - 1)]
    end = hay_map[min(max(ne - 1, 0), len(hay_map) - 1)] + 1
    return QuoteLocation(start, end, "fuzzy", score, haystack[start:end])


def _find_all(haystack: str, needle: str, limit: int = 24) -> list[int]:
    positions: list[int] = []
    start = 0
    while len(positions) < limit:
        index = haystack.find(needle, start)
        if index == -1:
            break
        positions.append(index)
        start = index + 1
    return positions


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,%()\-/]{5,}")


def _pick_anchor(quote: str, haystack: str) -> str | None:
    """Longest token from the quote that also occurs in the haystack.

    Numbers make excellent anchors: they are long, distinctive, and survive the
    reflowing that breaks sentence-level matching.
    """
    tokens = sorted(set(_TOKEN_RE.findall(quote)), key=len, reverse=True)
    for token in tokens[:12]:
        if token in haystack:
            return token
    return None


def contains_value(quote: str, value_raw: str | None) -> bool:
    """Check the quoted span actually contains the value claimed for it.

    Catches the failure where a model finds a real sentence but attaches the
    wrong number to it -- the quote grounds, but grounds something else.
    """
    if not value_raw:
        return True
    digits_in_value = re.sub(r"[^0-9]", "", value_raw)
    if not digits_in_value:
        return True
    quote_digits = re.sub(r"[^0-9]", "", quote)
    return digits_in_value in quote_digits
