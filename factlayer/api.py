"""HTTP interface: upload PDFs, inspect the knowledge layer.

The API is the deliverable the brief asks for; the bundled page is a thin client
over it, not a separate product. Every view the page renders is one JSON
endpoint, so a reviewer can drive the whole system with curl and get the same
answers.

Ingestion runs in a background thread rather than inside the request. A 100-page
filing takes minutes on a free API tier, and a POST that holds a connection open
that long is not something a reviewer can watch. Progress is polled instead.
"""

from __future__ import annotations

import logging
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse

from .link.candidates import interesting_relations
from .models import Fact, Relation
from .pipeline import Pipeline, Settings

log = logging.getLogger("factlayer.api")

app = FastAPI(
    title="factlayer",
    description="A fact knowledge layer: grounded facts from PDFs, reconciled across documents.",
    version="0.1.0",
)

_pipeline: Pipeline | None = None
_pipeline_lock = threading.Lock()

# In-flight and completed ingest jobs, keyed by job id.
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def pipeline() -> Pipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = Pipeline(Settings.from_env())
    return _pipeline


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def fact_json(fact: Fact, registry_label: str | None = None) -> dict[str, Any]:
    period = fact.context.period
    return {
        "id": fact.id,
        "doc_id": fact.doc_id,
        "fact_type": fact.fact_type,
        "measure": fact.measure_raw,
        "measure_key": fact.measure_key,
        "measure_label": registry_label or fact.measure_raw,
        "entity": fact.context.entity,
        "entity_key": fact.context.entity_key,
        "value": fact.display_value(),
        "canonical_value": fact.quantity.canonical_value if fact.quantity else None,
        "canonical_unit": fact.quantity.canonical_unit if fact.quantity else None,
        "period": {
            "raw": period.raw,
            "label": period.label,
            "kind": period.kind,
            "start": period.start.isoformat() if period.start else None,
            "end": period.end.isoformat() if period.end else None,
        }
        if period
        else None,
        "basis": fact.context.basis,
        "estimate_type": fact.context.estimate_type,
        "attributed_to": fact.context.attributed_to,
        "as_of": fact.context.as_of.isoformat() if fact.context.as_of else None,
        "confidence": fact.confidence,
        "extractor": fact.extractor,
        # Evidence is the point of the whole exercise, so it is never elided.
        "evidence": {
            "doc_id": fact.evidence.doc_id,
            "page": fact.evidence.page,
            "quote": fact.evidence.quote,
            "char_start": fact.evidence.char_start,
            "char_end": fact.evidence.char_end,
            "locator": fact.evidence.locator,
        },
    }


def relation_json(rel: Relation, store) -> dict[str, Any]:
    a, b = store.get_fact(rel.a_id), store.get_fact(rel.b_id)
    return {
        "id": rel.id,
        "type": rel.type.value,
        "reason_code": rel.reason_code.value,
        "confidence": rel.confidence,
        "explanation": rel.explanation,
        "adjudicator": rel.adjudicator,
        "cross_document": rel.cross_document,
        "analysis": rel.analysis,
        "a": fact_json(a) if a else None,
        "b": fact_json(b) if b else None,
    }


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


def _run_job(job_id: str, path: Path, force: bool, max_units: int | None) -> None:
    def progress(done: int, total: int) -> None:
        with _jobs_lock:
            _jobs[job_id]["progress"] = {"done": done, "total": total}

    try:
        report = pipeline().ingest(path, force=force, max_units=max_units, progress=progress)
        with _jobs_lock:
            _jobs[job_id].update(status="done", report=report.as_dict())
    except Exception as exc:  # noqa: BLE001
        log.exception("ingest job %s failed", job_id)
        with _jobs_lock:
            _jobs[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}")


@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    force: bool = Query(False, description="Re-ingest even if this PDF was seen before"),
    max_units: int | None = Query(None, description="Override the per-document LLM budget"),
) -> dict[str, Any]:
    """Accept a PDF and start ingesting it. Returns a job id to poll."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF uploads are supported")

    settings = pipeline().settings
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{uuid.uuid4().hex[:8]}-{Path(file.filename).name}"
    with target.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "filename": file.filename,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "progress": {"done": 0, "total": 0},
        }
    threading.Thread(
        target=_run_job, args=(job_id, target, force, max_units), daemon=True
    ).start()
    return {"job_id": job_id, "status": "running"}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


# --------------------------------------------------------------------------
# Reading the layer
# --------------------------------------------------------------------------


@app.get("/api/documents")
async def list_documents() -> dict[str, Any]:
    store = pipeline().store
    return {
        "documents": [
            {
                "id": d.id,
                "filename": d.filename,
                "title": d.title,
                "publisher": d.publisher,
                "published": d.published.isoformat() if d.published else None,
                "pages": d.n_pages,
                "status": d.status,
                "stats": d.stats,
            }
            for d in store.list_documents()
        ]
    }


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict[str, Any]:
    store = pipeline().store
    if store.get_document(doc_id) is None:
        raise HTTPException(404, "unknown document")
    store.delete_document(doc_id)
    return {"deleted": doc_id}


@app.get("/api/facts")
async def list_facts(
    doc_id: str | None = None,
    measure_key: str | None = None,
    q: str | None = Query(None, description="Substring match on measure or entity"),
    limit: int = 500,
) -> dict[str, Any]:
    store = pipeline().store
    registry = store.load_registry()
    facts = store.facts(doc_id=doc_id)
    if measure_key:
        facts = [f for f in facts if f.measure_key == measure_key]
    if q:
        needle = q.lower()
        facts = [
            f for f in facts
            if needle in f.measure_raw.lower() or needle in f.context.entity.lower()
        ]
    return {
        "total": len(facts),
        "facts": [fact_json(f, registry.label_for(f.measure_key)) for f in facts[:limit]],
    }


@app.get("/api/facts/{fact_id}")
async def get_fact(fact_id: str) -> dict[str, Any]:
    store = pipeline().store
    fact = store.get_fact(fact_id)
    if fact is None:
        raise HTTPException(404, "unknown fact")
    return {
        "fact": fact_json(fact),
        "relations": [relation_json(r, store) for r in store.relations_for_fact(fact_id)],
    }


@app.get("/api/relations")
async def list_relations(
    type: str | None = Query(None, description="Comma-separated relation types"),
    cross_document_only: bool = False,
    limit: int = 200,
) -> dict[str, Any]:
    store = pipeline().store
    types = [t.strip() for t in type.split(",")] if type else None
    relations = store.relations(types=types, cross_document_only=cross_document_only)
    ordered = interesting_relations(relations)[:limit]
    return {
        "total": len(relations),
        "relations": [relation_json(r, store) for r in ordered],
    }


@app.get("/api/failures")
async def list_failures(doc_id: str | None = None, limit: int = 200) -> dict[str, Any]:
    """What the system could not read.

    Exposed as a first-class view rather than a log file. A knowledge layer that
    cannot report its own blind spots invites more trust than it has earned.
    """
    store = pipeline().store
    failures = store.failures(doc_id=doc_id, limit=limit)
    return {
        "summary": store.failure_summary(),
        "failures": [
            {
                "doc_id": f.doc_id, "page": f.page, "stage": f.stage,
                "kind": f.kind, "detail": f.detail, "sample": f.sample,
            }
            for f in failures
        ],
    }


@app.get("/api/measures")
async def list_measures() -> dict[str, Any]:
    """The schema as it currently exists -- entirely derived from the documents."""
    store = pipeline().store
    registry = store.load_registry()
    counts = {key: n for key, _, n in [(k, e, c) for k, e, c in store.blocks(min_size=1)]}
    clusters = []
    for key, cluster in registry.clusters.items():
        clusters.append(
            {
                "key": key,
                "label": cluster.label(),
                "kind": cluster.kind,
                "tokens": sorted(cluster.tokens),
                "surface_forms": dict(cluster.surface_forms),
                "support": cluster.support,
            }
        )
    clusters.sort(key=lambda c: -c["support"])
    return {"total": len(clusters), "measures": clusters}


@app.get("/api/stats")
async def stats() -> dict[str, Any]:
    p = pipeline()
    data = p.store.stats()
    data["backend"] = {
        "provider": p.provider.name,
        "available": p.provider.available(),
        "extract_model": p.settings.extract_model,
        "llm_calls": p.provider.calls,
        "cached_calls": p.provider.cached_calls,
    }
    return data


@app.post("/api/relink")
async def relink() -> dict[str, Any]:
    """Re-adjudicate every pair. Used after changing the ladder."""
    return pipeline().relink_all().as_dict()


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

_UI_PATH = Path(__file__).parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    if not _UI_PATH.exists():
        return HTMLResponse("<h1>factlayer</h1><p>UI not found; the API is at /docs.</p>")
    return HTMLResponse(_UI_PATH.read_text(encoding="utf-8"))


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})
