"""PDF -> pages -> blocks, with provenance preserved throughout.

Every downstream fact must point back to a place in a file, so extraction never
works on a bare string. A :class:`Block` knows its page, its character span
within that page's text, and its bounding box, which is what lets the UI show a
reviewer exactly where a claim came from.

Two details matter more than they look:

* **Page-level scale declarations.** Financial tables put the unit in a header
  ("(₹ in Million)") and leave the cells bare. Carrying that hint down to the
  numbers is the difference between reading 81,415.38 as eighty-one thousand
  rupees and as ₹81.4 billion.
* **Section headings.** A number in a table cell is meaningless without the row
  and column labels around it, so blocks keep their local heading context.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pymupdf

from ..models import Block, Document
from ..normalize.periods import infer_fiscal_year_end
from ..normalize.units import magnitude_from_header

# Rendering-only artefacts that add noise without adding meaning.
_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "­": "", "​": ""}


def clean_text(text: str) -> str:
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


@dataclass
class Page:
    number: int
    text: str
    blocks: list[Block] = field(default_factory=list)
    magnitude_hint: float | None = None
    unit_hint: str | None = None


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return False
    if stripped.endswith((".", ";", ",")):
        return False
    words = stripped.split()
    if len(words) > 14:
        return False
    letters = [c for c in stripped if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        return True
    return bool(re.match(r"^(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Za-z ,'\-&/()]+$", stripped))


def _looks_like_table(text: str) -> bool:
    """Rough table detector: many short numeric-heavy lines."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return False
    numericish = sum(1 for ln in lines if len(re.findall(r"\d", ln)) >= 3)
    wide_gaps = sum(1 for ln in lines if re.search(r"\S {2,}\S", ln))
    return numericish >= max(2, len(lines) * 0.5) or wide_gaps >= max(2, len(lines) * 0.6)


_NUM_RE = re.compile(r"\d")
_FACTUAL_MARKERS = re.compile(
    r"\b(per cent|percent|%|crore|lakh|million|billion|trillion|revenue|growth|"
    r"increase[d]?|decrease[d]?|total|net|margin|ratio|rate|share|as on|as at|"
    r"appointed|resigned|ceased|effective from|compared|versus|vs\.?|estimated|"
    r"projected|stood at|reached|declined|rose|fell)\b",
    re.IGNORECASE,
)


def density_score(text: str) -> float:
    """How likely a block is to contain checkable facts, in [0, 1].

    Used to spend LLM budget where it will pay off. A 100-page annual report has
    a lot of boilerplate; sending the safe-harbour disclaimer to a model costs
    money and yields nothing.
    """
    if not text.strip():
        return 0.0
    n_chars = len(text)
    digits = len(_NUM_RE.findall(text))
    markers = len(_FACTUAL_MARKERS.findall(text))
    digit_ratio = digits / max(n_chars, 1)
    score = min(1.0, digit_ratio * 6.0) * 0.55 + min(1.0, markers / 6.0) * 0.45
    if n_chars < 60:
        score *= 0.4
    return round(min(score, 1.0), 4)


def _pdf_date(value: str | None) -> date | None:
    if not value:
        return None
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})", value)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


_DATE_IN_TEXT = re.compile(
    r"\b(?P<d>\d{1,2})?\s*(?P<mon>January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(?P<d2>\d{1,2})?,?\s*(?P<y>(?:19|20)\d{2})\b",
    re.IGNORECASE,
)


def guess_published_date(first_pages_text: str, meta_date: date | None) -> date | None:
    """Prefer an explicit date printed on the cover over PDF metadata.

    Metadata dates are frequently the date the file was last re-saved, which can
    be years off. Document vintage drives the revision reasoning, so it is worth
    getting right.
    """
    from ..normalize.periods import MONTHS

    candidates: list[date] = []
    for m in _DATE_IN_TEXT.finditer(first_pages_text[:4000]):
        mon = MONTHS.get(m.group("mon").lower())
        day = int(m.group("d") or m.group("d2") or 1)
        year = int(m.group("y"))
        try:
            candidates.append(date(year, mon, min(day, 28)))
        except (ValueError, TypeError):
            continue
    if candidates:
        return max(candidates)
    return meta_date


# Cover-page furniture that is prominent but says nothing about the document.
_GENERIC_TITLES = {
    "contents", "table of contents", "what's inside", "whats inside", "index",
    "annual report", "introduction", "overview", "disclaimer", "notice",
    "about this report", "highlights", "appendix", "glossary", "abbreviations",
}


def _is_generic_title(text: str) -> bool:
    normalised = re.sub(r"[^a-z' ]", " ", text.lower()).strip()
    normalised = re.sub(r"\s+", " ", normalised)
    return normalised in _GENERIC_TITLES or len(normalised) < 6


def _largest_text_on_page(page: pymupdf.Page) -> str | None:
    """The visually most prominent line on a page.

    Titles are set large; addressee blocks and running heads are not. Reading
    font size beats reading position, which otherwise picks up whatever happens
    to sit at the top of the page.
    """
    best_size, best_text = 0.0, None
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = clean_text("".join(s.get("text", "") for s in spans)).strip()
            if not (6 <= len(text) <= 140):
                continue
            if sum(c.isalpha() for c in text) < len(text) * 0.5:
                continue
            size = max(float(s.get("size", 0)) for s in spans)
            if size > best_size:
                best_size, best_text = size, text
    return best_text


def guess_title(doc: pymupdf.Document, first_page_text: str, filename: str) -> str:
    meta_title = ((doc.metadata or {}).get("title") or "").strip()
    if len(meta_title) > 4 and not meta_title.lower().endswith(".pdf"):
        return meta_title[:200]
    # Scan the first few pages: cover pages are sometimes blank or a logo plate.
    fallback: str | None = None
    for page_index in range(min(4, doc.page_count)):
        candidate = _largest_text_on_page(doc[page_index])
        if not candidate:
            continue
        if not _is_generic_title(candidate):
            return candidate[:200]
        fallback = fallback or candidate
    if fallback:
        return fallback[:200]
    for line in first_page_text.split("\n"):
        s = line.strip()
        if 8 <= len(s) <= 140 and sum(c.isalpha() for c in s) > len(s) * 0.5:
            return s[:200]
    return Path(filename).stem.replace("-", " ").replace("_", " ")[:200]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class IngestedDocument:
    def __init__(self, document: Document, pages: list[Page], full_text: str) -> None:
        self.document = document
        self.pages = pages
        self.full_text = full_text

    @property
    def blocks(self) -> list[Block]:
        return [b for p in self.pages for b in p.blocks]


def detect_two_columns(blocks: list, page_width: float) -> float | None:
    """Find the x of a column gutter, or None if the page is single-column.

    Institutional reports are typeset in two columns far more often than not,
    and sorting their blocks top-to-bottom interleaves the columns line by line.
    The resulting text is still *readable* by a language model -- which is
    precisely the danger. The model silently reassembles the real sentence and
    quotes that, the quote is then not found verbatim on the page, and the fact
    is discarded. An entire document can fail grounding this way while looking
    like a model quality problem.

    Detection is deliberately conservative: a gutter is only accepted when both
    sides carry real content and few blocks straddle it, so a single-column page
    with a stray sidebar is left alone.
    """
    if page_width <= 0 or len(blocks) < 6:
        return None

    gutter = page_width / 2.0
    margin = page_width * 0.04

    left = right = straddling = 0
    for b in blocks:
        x0, x1 = b[0], b[2]
        if x0 < gutter - margin and x1 > gutter + margin:
            straddling += 1
        elif x1 <= gutter + margin:
            left += 1
        else:
            right += 1

    sided = left + right
    if sided == 0:
        return None
    # Both columns must be populated, and full-width blocks (headings, wide
    # tables) must be the exception rather than the rule.
    if min(left, right) < 0.25 * sided:
        return None
    if straddling > 0.35 * len(blocks):
        return None
    return gutter


def sort_reading_order(blocks: list, page_width: float) -> list:
    """Order blocks the way a human reads them.

    Single column: top-to-bottom, then left-to-right. Two columns: the left
    column in full, then the right, with full-width blocks kept in the left
    flow at their vertical position so headings still precede their section.
    """
    blocks = [b for b in blocks if len(b) >= 5]
    gutter = detect_two_columns(blocks, page_width)
    if gutter is None:
        return sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))

    margin = page_width * 0.04

    def column_of(b) -> int:
        x0, x1 = b[0], b[2]
        if x0 < gutter - margin and x1 > gutter + margin:
            return 0  # spans both columns; keep it in the left-hand flow
        return 0 if x1 <= gutter + margin else 1

    return sorted(blocks, key=lambda b: (column_of(b), round(b[1], 1), round(b[0], 1)))


def ingest_pdf(path: str | Path, doc_id: str | None = None) -> IngestedDocument:
    """Read a PDF into pages and provenance-carrying blocks."""
    path = Path(path)
    sha = file_sha256(path)
    doc_id = doc_id or ("d_" + sha[:12])

    pdf = pymupdf.open(path)
    pages: list[Page] = []
    full_chunks: list[str] = []

    try:
        for page_index in range(pdf.page_count):
            page = pdf[page_index]
            raw_blocks = page.get_text("blocks") or []
            raw_blocks = sort_reading_order(raw_blocks, page.rect.width)

            page_text_parts: list[str] = []
            offsets: list[tuple[int, int]] = []
            cleaned_blocks: list[tuple[tuple[float, float, float, float], str]] = []
            cursor = 0
            for b in raw_blocks:
                if len(b) < 6 or b[6] != 0 if len(b) > 6 else False:
                    continue  # image block
                text = clean_text(str(b[4]))
                if not text.strip():
                    continue
                start = cursor
                page_text_parts.append(text)
                cursor += len(text) + 1  # +1 for the joining newline
                offsets.append((start, start + len(text)))
                cleaned_blocks.append(((b[0], b[1], b[2], b[3]), text))

            page_text = "\n".join(page_text_parts)
            magnitude_hint = magnitude_from_header(page_text)
            unit_hint_match = re.search(
                r"\(?\s*(?:₹|Rs\.?|INR|USD|\$|€|£)\s*(?:in\s+)?"
                r"(?:hundred|thousand|lakhs?|millions?|crores?|billions?)\s*\)?",
                page_text,
                re.IGNORECASE,
            )
            unit_hint = unit_hint_match.group(0) if unit_hint_match else None

            page_obj = Page(
                number=page_index + 1,
                text=page_text,
                magnitude_hint=magnitude_hint,
                unit_hint=unit_hint,
            )

            heading = ""
            for i, (bbox, text) in enumerate(cleaned_blocks):
                start, end = offsets[i]
                if _looks_like_heading(text):
                    heading = text.strip()
                    kind = "heading"
                elif _looks_like_table(text):
                    kind = "table"
                else:
                    kind = "prose"
                block_id = f"{doc_id}_p{page_index + 1}_b{i}"
                page_obj.blocks.append(
                    Block(
                        id=block_id,
                        doc_id=doc_id,
                        page=page_index + 1,
                        text=text,
                        char_start=start,
                        char_end=end,
                        bbox=bbox,
                        kind=kind,  # type: ignore[arg-type]
                        density=density_score(text),
                    )
                )
            pages.append(page_obj)
            full_chunks.append(page_text)

        full_text = "\n\n".join(full_chunks)
        head = "\n".join(full_chunks[:3])
        meta = pdf.metadata or {}
        document = Document(
            id=doc_id,
            filename=path.name,
            title=guess_title(pdf, full_chunks[0] if full_chunks else "", path.name),
            publisher=(meta.get("author") or "").strip() or None,
            published=guess_published_date(head, _pdf_date(meta.get("creationDate"))),
            n_pages=pdf.page_count,
            sha256=sha,
            ingested_at=datetime.utcnow().isoformat(timespec="seconds"),
            status="ingested",
            stats={
                "n_blocks": sum(len(p.blocks) for p in pages),
                "n_chars": len(full_text),
                "fiscal_year_end_month": infer_fiscal_year_end(full_text),
            },
        )
    finally:
        pdf.close()

    return IngestedDocument(document=document, pages=pages, full_text=full_text)
