"""End-to-end ingestion: PDF in, linked knowledge layer out.

The sequence is deliberately linear and each stage is observable, because the
brief asks for a system whose behaviour is clear rather than one that is large.

    ingest -> chunk -> extract -> ground -> normalise -> register -> link

Two properties are worth calling out, since they are design decisions rather
than consequences:

**Adding a document does not rebuild the layer.** Facts are content-addressed,
the measure registry is persistent, and linking is restricted to pairs touching
the new document. The tenth PDF costs what the second cost.

**Budget is a first-class parameter, measured in requests.** The free API tier
meters requests per day, not tokens, and the model has a million-token context.
So pages are batched ten at a time into one call and the density filter drops
boilerplate before batching. A 100-page filing costs a handful of requests
instead of a hundred. The cap is reported, never hidden -- a run that skipped
material says so in its stats.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .extract.llm_extractor import ExtractionResult, LLMExtractor, extract_document_metadata
from .ingest.chunk import build_batches, unit_stats
from .ingest.pdf import ingest_pdf
from .link.candidates import Linker, LinkReport
from .llm import LLMProvider, ResponseCache, build_provider
from .models import Document, ExtractionFailure
from .normalize.periods import DEFAULT_FY_END_MONTH, infer_fiscal_year_end
from .store.db import Store

log = logging.getLogger("factlayer.pipeline")


@dataclass
class Settings:
    """Everything tunable, resolved from the environment in one place."""

    db_path: str = "data/factlayer.db"
    cache_dir: str = "data/cache"
    upload_dir: str = "data/uploads"
    backend: str = "gemini"
    extract_model: str = "gemini-3.8-flash"
    adjudicate_model: str = "gemini-3.8-flash"
    concurrency: int = 4
    max_units_per_doc: int = 14
    batch_target_chars: int = 16000
    batch_max_pages: int = 10
    max_output_tokens: int = 32000
    min_density: float = 0.12
    fy_end_month: int = DEFAULT_FY_END_MONTH

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=os.environ.get("FACTLAYER_DB", "data/factlayer.db"),
            cache_dir=os.environ.get("FACTLAYER_CACHE", "data/cache"),
            upload_dir=os.environ.get("FACTLAYER_UPLOADS", "data/uploads"),
            backend=os.environ.get("FACTLAYER_LLM", "auto"),
            extract_model=os.environ.get("FACTLAYER_EXTRACT_MODEL", "gemini-3.8-flash"),
            adjudicate_model=os.environ.get("FACTLAYER_ADJUDICATE_MODEL", "gemini-3.8-flash"),
            concurrency=int(os.environ.get("FACTLAYER_MAX_CONCURRENCY", "4")),
            max_units_per_doc=int(os.environ.get("FACTLAYER_MAX_UNITS_PER_DOC", "14")),
            batch_target_chars=int(os.environ.get("FACTLAYER_BATCH_CHARS", "16000")),
            batch_max_pages=int(os.environ.get("FACTLAYER_BATCH_PAGES", "10")),
            max_output_tokens=int(os.environ.get("FACTLAYER_MAX_OUTPUT_TOKENS", "32000")),
            min_density=float(os.environ.get("FACTLAYER_MIN_DENSITY", "0.12")),
        )


@dataclass
class IngestReport:
    doc_id: str
    filename: str
    title: str | None = None
    pages: int = 0
    units_total: int = 0
    units_processed: int = 0
    facts: int = 0
    failures: int = 0
    llm_calls: int = 0
    cached_calls: int = 0
    seconds: float = 0.0
    skipped_existing: bool = False
    link: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "title": self.title,
            "pages": self.pages,
            "units_total": self.units_total,
            "units_processed": self.units_processed,
            "facts": self.facts,
            "failures": self.failures,
            "llm_calls": self.llm_calls,
            "cached_calls": self.cached_calls,
            "seconds": round(self.seconds, 1),
            "skipped_existing": self.skipped_existing,
            "link": self.link,
            "notes": self.notes,
        }


class Pipeline:
    """Owns the store, the provider and the measure registry for a run."""

    def __init__(self, settings: Settings | None = None, provider: LLMProvider | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.store = Store(self.settings.db_path)
        self.cache = ResponseCache(self.settings.cache_dir)
        self.provider = provider or build_provider(self.settings.backend, cache=self.cache)
        Path(self.settings.upload_dir).mkdir(parents=True, exist_ok=True)

    # -- ingestion ----------------------------------------------------------

    def ingest(
        self,
        path: str | Path,
        force: bool = False,
        max_units: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> IngestReport:
        started = datetime.now(timezone.utc)
        path = Path(path)
        ingested = ingest_pdf(path)
        document = ingested.document

        report = IngestReport(
            doc_id=document.id, filename=document.filename, pages=document.n_pages
        )

        existing = self.store.document_by_sha(document.sha256)
        if existing is not None and not force:
            # Content-addressed: the same PDF under a different filename is the
            # same document, and re-uploading it is a no-op rather than a
            # duplicate set of facts.
            report.doc_id = existing.id
            report.title = existing.title
            report.skipped_existing = True
            report.facts = len(self.store.facts(existing.id))
            report.notes.append(
                "Document already ingested (identical SHA-256); returning the stored result."
            )
            return report
        if existing is not None and force:
            self.store.delete_document(existing.id)
            report.notes.append("Re-ingesting: previous facts for this document were removed.")

        # -- document-level context -----------------------------------------
        principal_entity: str | None = None
        fy_end_month = infer_fiscal_year_end(ingested.full_text[:60000]) or self.settings.fy_end_month

        if self.provider.available():
            metadata = extract_document_metadata(
                ingested, self.provider, self.settings.extract_model
            )
            document.title = metadata.get("title") or document.title
            document.publisher = metadata.get("publisher") or document.publisher
            principal_entity = metadata.get("principal_entity") or None
            published = _parse_iso_date(metadata.get("published"))
            if published:
                document.published = published
        else:
            report.notes.append(
                f"No LLM backend available ({self.provider.name}); "
                "extraction quality is substantially reduced."
            )

        report.title = document.title

        # -- chunking --------------------------------------------------------
        units = build_batches(
            ingested,
            target_chars=self.settings.batch_target_chars,
            min_density=self.settings.min_density,
            max_pages=self.settings.batch_max_pages,
        )
        report.units_total = len(units)

        cap = max_units if max_units is not None else self.settings.max_units_per_doc
        if cap and len(units) > cap:
            # Highest-density units first: the pages that assert the most.
            units = sorted(units, key=lambda u: -u.density)[:cap]
            report.notes.append(
                f"Budget cap applied: {report.units_total} page batches found, "
                f"{cap} processed (highest density first). Raise FACTLAYER_MAX_UNITS_PER_DOC "
                "for full coverage."
            )
        report.units_processed = len(units)

        # -- extraction ------------------------------------------------------
        page_texts = {page.number: page.text for page in ingested.pages}
        result: ExtractionResult
        if self.provider.available():
            extractor = LLMExtractor(
                self.provider,
                self.settings.extract_model,
                concurrency=self.settings.concurrency,
                max_tokens=self.settings.max_output_tokens,
            )
            result = extractor.extract_units(
                units, page_texts, document.id, principal_entity,
                document.published, fy_end_month, progress=progress,
            )
        else:
            result = ExtractionResult()
            result.failures.append(
                ExtractionFailure(
                    doc_id=document.id, stage="extract", kind="no_llm_backend",
                    detail="no LLM backend configured, so no facts were extracted",
                )
            )

        # -- measure registry ------------------------------------------------
        registry = self.store.load_registry()
        for fact in result.facts:
            kind = fact.quantity.kind if fact.quantity else "other"
            fact.measure_key = registry.assign(fact.measure_raw, kind=kind)
        self.store.save_registry(registry)

        # -- persist ---------------------------------------------------------
        document.status = "ingested"
        document.ingested_at = started.isoformat()
        document.stats = {
            "units_total": report.units_total,
            "units_processed": report.units_processed,
            "facts": len(result.facts),
            "failures": len(result.failures),
            "fy_end_month": fy_end_month,
            "principal_entity": principal_entity,
            **unit_stats(units),
        }
        self.store.upsert_document(document)
        self.store.add_facts(result.facts)
        self.store.add_failures(result.failures)

        report.facts = len(result.facts)
        report.failures = len(result.failures)
        report.llm_calls = self.provider.calls
        report.cached_calls = self.provider.cached_calls

        # -- link into the existing layer -------------------------------------
        link_report = self.link(new_doc_id=document.id)
        report.link = link_report.as_dict()

        report.seconds = (datetime.now(timezone.utc) - started).total_seconds()
        log.info(
            "ingested %s: %d facts, %d failures, %d relations in %.1fs",
            document.filename, report.facts, report.failures,
            link_report.relations_written, report.seconds,
        )
        return report

    # -- linking ------------------------------------------------------------

    def link(self, new_doc_id: str | None = None) -> LinkReport:
        return Linker(self.store).link(new_doc_id=new_doc_id)

    def relink_all(self) -> LinkReport:
        """Re-adjudicate everything. Used after changing the ladder itself."""
        with self.store.connect() as conn:
            conn.execute("DELETE FROM relations")
        return Linker(self.store).link()


def _parse_iso_date(value: Any):
    from datetime import date

    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None
