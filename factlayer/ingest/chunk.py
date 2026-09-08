"""Grouping blocks into extraction units.

An extraction unit is what one LLM call sees. Three constraints shape it:

* It must stay inside one page, so evidence keeps a single unambiguous page
  number.
* It must carry enough surrounding context to be interpretable -- a table cell
  reading "81,415.38" needs its column header and the "(₹ in Million)" note, or
  the model will guess.
* It should not be wasted on boilerplate. A 100-page annual report is mostly
  disclaimers and signature blocks; a density filter keeps LLM spend on the
  pages that actually assert things.

The density filter is a budget control, not a correctness mechanism, and it is
tunable: ``min_density=0`` processes everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Block
from .pdf import IngestedDocument, Page


@dataclass
class ExtractionUnit:
    """One page-local group of blocks, plus the context needed to read it."""

    id: str
    doc_id: str
    page: int
    text: str
    blocks: list[Block] = field(default_factory=list)
    heading: str | None = None
    unit_hint: str | None = None
    magnitude_hint: float | None = None
    density: float = 0.0
    # Populated only for batched units, which cover several pages in one call.
    # ``page`` stays the first page so single-page behaviour is unchanged.
    pages: list[int] = field(default_factory=list)

    @property
    def char_start(self) -> int:
        return min((b.char_start for b in self.blocks), default=0)

    @property
    def char_end(self) -> int:
        return max((b.char_end for b in self.blocks), default=0)

    def prompt_context(self) -> str:
        """Header lines prepended to the text when the unit is shown to a model."""
        lines = []
        if self.heading:
            lines.append(f"[section] {self.heading}")
        if self.unit_hint:
            lines.append(
                f"[scale note on this page] {self.unit_hint} "
                "-- bare numbers in tables on this page are expressed in this unit"
            )
        return "\n".join(lines)


def _running_heading(page: Page) -> dict[str, str]:
    """Map each block to the nearest heading above it on the same page."""
    mapping: dict[str, str] = {}
    current = ""
    for block in page.blocks:
        if block.kind == "heading":
            current = block.text.strip()
        mapping[block.id] = current
    return mapping


def build_units(
    doc: IngestedDocument,
    target_chars: int = 2600,
    min_density: float = 0.12,
    overlap_blocks: int = 1,
) -> list[ExtractionUnit]:
    """Group each page's blocks into units of roughly ``target_chars``.

    ``overlap_blocks`` repeats a block at unit boundaries so a sentence split
    across the boundary is still seen whole by one of the two units.
    """
    units: list[ExtractionUnit] = []

    for page in doc.pages:
        headings = _running_heading(page)
        candidates = [b for b in page.blocks if b.kind != "heading" and b.text.strip()]
        if not candidates:
            continue

        current: list[Block] = []
        current_len = 0

        def flush() -> None:
            nonlocal current, current_len
            if not current:
                return
            best_density = max(b.density for b in current)
            # A unit earns its LLM call if any block in it looks factual.
            if best_density >= min_density:
                unit_id = f"{doc.document.id}_p{page.number}_u{len(units)}"
                heading = headings.get(current[0].id) or None
                body = "\n".join(b.text for b in current)
                units.append(
                    ExtractionUnit(
                        id=unit_id,
                        doc_id=doc.document.id,
                        page=page.number,
                        text=body,
                        blocks=list(current),
                        heading=heading,
                        unit_hint=page.unit_hint,
                        magnitude_hint=page.magnitude_hint,
                        density=best_density,
                    )
                )
            current, current_len = [], 0

        for block in candidates:
            if current and current_len + len(block.text) > target_chars:
                tail = current[-overlap_blocks:] if overlap_blocks else []
                flush()
                current = list(tail)
                current_len = sum(len(b.text) for b in current)
            current.append(block)
            current_len += len(block.text)
        flush()

    return units


def build_batches(
    doc: IngestedDocument,
    target_chars: int = 16000,
    min_density: float = 0.12,
    max_pages: int = 10,
) -> list[ExtractionUnit]:
    """Group *pages* into units, many pages to one LLM call.

    ``build_units`` sizes a unit for a small context window: one page, a few
    thousand characters. That was the wrong shape for the model actually being
    used. A 1M-token context has no difficulty with twenty pages at once, and
    the free API tier meters *requests per day*, not tokens -- so page-sized
    chunks were spending the entire daily budget on a fraction of one document.
    Batching cuts calls by roughly ten times at no cost in quality.

    Page provenance survives because each page is introduced by an explicit
    ``[[PAGE n]]`` marker and the model reports which page each claim came
    from. Grounding then verifies the quote against *that page's* text, so a
    misreported page fails the check and is discarded rather than mislabelled.
    Evidence remains as trustworthy as it was with page-local units; only the
    packaging changed.
    """
    batches: list[ExtractionUnit] = []
    current: list[Page] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        pages = [p.number for p in current]
        body = "\n\n".join(f"[[PAGE {p.number}]]\n{p.text}" for p in current)
        blocks = [b for p in current for b in p.blocks]
        densities = [b.density for b in blocks] or [0.0]
        # A scale note is only safe to apply when every page in the batch agrees
        # on it; otherwise a bare table number could be scaled by another page's
        # unit. Ambiguity here silently corrupts magnitudes, so it is dropped.
        hints = {p.unit_hint for p in current}
        magnitudes = {p.magnitude_hint for p in current}
        batches.append(
            ExtractionUnit(
                id=f"{doc.document.id}_b{len(batches)}_p{pages[0]}-{pages[-1]}",
                doc_id=doc.document.id,
                page=pages[0],
                pages=pages,
                text=body,
                blocks=blocks,
                heading=None,
                unit_hint=hints.pop() if len(hints) == 1 else None,
                magnitude_hint=magnitudes.pop() if len(magnitudes) == 1 else None,
                density=max(densities),
            )
        )
        current, current_len = [], 0

    for page in doc.pages:
        blocks = [b for b in page.blocks if b.text.strip()]
        if not blocks:
            continue
        # The density filter still earns its keep: it is now deciding whether a
        # page is worth including in a batch rather than worth its own call.
        if max(b.density for b in blocks) < min_density:
            continue
        if current and (current_len + len(page.text) > target_chars or len(current) >= max_pages):
            flush()
        current.append(page)
        current_len += len(page.text)
    flush()

    return batches


def unit_stats(units: list[ExtractionUnit]) -> dict[str, float]:
    if not units:
        return {"n_units": 0, "n_chars": 0, "mean_density": 0.0}
    total = sum(len(u.text) for u in units)
    return {
        "n_units": len(units),
        "n_chars": total,
        "mean_chars": round(total / len(units), 1),
        "mean_density": round(sum(u.density for u in units) / len(units), 3),
    }
