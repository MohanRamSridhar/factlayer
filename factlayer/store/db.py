"""SQLite persistence for the knowledge layer.

Chosen over a graph database on purpose. The brief notes that a graph database
alone is not the solution, and that is not just a warning about presentation:
the interesting work here is deciding *whether* two facts relate and *why*,
which is computation, not storage. Once those verdicts exist they are a flat
table of pairs. A single file with no server also keeps the "clone and run"
promise intact.

The schema is hybrid on purpose. Every fact keeps its full Pydantic
serialisation in ``payload`` so nothing is lost and the model can evolve without
a migration, while the handful of fields the linker actually filters on are
lifted into real columns and indexed. Candidate generation is the one operation
that must not degrade as documents accumulate, and it is a two-column index
lookup rather than a scan.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..models import Document, ExtractionFailure, Fact, Relation
from ..normalize.measures import MeasureRegistry

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    title       TEXT,
    publisher   TEXT,
    published   TEXT,
    n_pages     INTEGER DEFAULT 0,
    sha256      TEXT,
    ingested_at TEXT,
    status      TEXT,
    stats       TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256);

CREATE TABLE IF NOT EXISTS facts (
    id              TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    fact_type       TEXT,
    measure_key     TEXT,
    measure_raw     TEXT,
    entity_key      TEXT,
    canonical_value REAL,
    canonical_unit  TEXT,
    period_start    TEXT,
    period_end      TEXT,
    page            INTEGER,
    confidence      REAL,
    payload         TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);
-- The blocking key. Candidate generation depends entirely on this index.
CREATE INDEX IF NOT EXISTS idx_facts_block ON facts(measure_key, entity_key);
CREATE INDEX IF NOT EXISTS idx_facts_doc ON facts(doc_id);

CREATE TABLE IF NOT EXISTS relations (
    id             TEXT PRIMARY KEY,
    a_id           TEXT NOT NULL,
    b_id           TEXT NOT NULL,
    type           TEXT NOT NULL,
    reason_code    TEXT NOT NULL,
    confidence     REAL,
    explanation    TEXT,
    analysis       TEXT,
    adjudicator    TEXT,
    cross_document INTEGER
);
CREATE INDEX IF NOT EXISTS idx_relations_a ON relations(a_id);
CREATE INDEX IF NOT EXISTS idx_relations_b ON relations(b_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(type);

CREATE TABLE IF NOT EXISTS failures (
    id      TEXT PRIMARY KEY,
    doc_id  TEXT,
    page    INTEGER,
    stage   TEXT,
    kind    TEXT,
    detail  TEXT,
    sample  TEXT
);
CREATE INDEX IF NOT EXISTS idx_failures_doc ON failures(doc_id);

-- Single-row table holding the serialised measure registry. It is global state
-- shared by every document, which is exactly why it lives beside them.
CREATE TABLE IF NOT EXISTS registry (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    payload TEXT NOT NULL
);
"""


class Store:
    """Thread-safe SQLite wrapper. One instance per database file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    # -- connection ---------------------------------------------------------

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            # WAL keeps the API readable while a long ingest is writing.
            conn.execute("PRAGMA journal_mode = WAL")
            self._local.conn = conn
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = self._conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # -- documents ----------------------------------------------------------

    def upsert_document(self, doc: Document) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO documents
                     (id, filename, title, publisher, published, n_pages, sha256,
                      ingested_at, status, stats)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     title=excluded.title, publisher=excluded.publisher,
                     published=excluded.published, n_pages=excluded.n_pages,
                     status=excluded.status, stats=excluded.stats""",
                (
                    doc.id,
                    doc.filename,
                    doc.title,
                    doc.publisher,
                    doc.published.isoformat() if doc.published else None,
                    doc.n_pages,
                    doc.sha256,
                    doc.ingested_at,
                    doc.status,
                    json.dumps(doc.stats),
                ),
            )

    def get_document(self, doc_id: str) -> Document | None:
        row = self._conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return _row_to_document(row) if row else None

    def document_by_sha(self, sha256: str) -> Document | None:
        row = self._conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
        return _row_to_document(row) if row else None

    def list_documents(self) -> list[Document]:
        rows = self._conn.execute("SELECT * FROM documents ORDER BY ingested_at").fetchall()
        return [_row_to_document(r) for r in rows]

    def delete_document(self, doc_id: str) -> None:
        """Remove a document and everything derived from it.

        Relations are cleaned up explicitly rather than by cascade because they
        reference two facts and SQLite would only cascade one side.
        """
        with self.connect() as conn:
            fact_ids = [r["id"] for r in conn.execute("SELECT id FROM facts WHERE doc_id = ?", (doc_id,))]
            if fact_ids:
                marks = ",".join("?" * len(fact_ids))
                conn.execute(f"DELETE FROM relations WHERE a_id IN ({marks})", fact_ids)
                conn.execute(f"DELETE FROM relations WHERE b_id IN ({marks})", fact_ids)
            conn.execute("DELETE FROM failures WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM facts WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    # -- facts --------------------------------------------------------------

    def add_facts(self, facts: Iterable[Fact]) -> int:
        rows = []
        for fact in facts:
            quantity = fact.quantity
            period = fact.context.period
            rows.append(
                (
                    fact.id,
                    fact.doc_id,
                    fact.fact_type,
                    fact.measure_key,
                    fact.measure_raw,
                    fact.context.entity_key,
                    quantity.canonical_value if quantity else None,
                    quantity.canonical_unit if quantity else None,
                    period.start.isoformat() if period and period.start else None,
                    period.end.isoformat() if period and period.end else None,
                    fact.evidence.page,
                    fact.confidence,
                    fact.model_dump_json(),
                )
            )
        if not rows:
            return 0
        with self.connect() as conn:
            # Facts are content-addressed, so re-ingesting the same document is
            # idempotent rather than duplicative.
            conn.executemany(
                """INSERT INTO facts
                     (id, doc_id, fact_type, measure_key, measure_raw, entity_key,
                      canonical_value, canonical_unit, period_start, period_end,
                      page, confidence, payload)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO NOTHING""",
                rows,
            )
        return len(rows)

    def get_fact(self, fact_id: str) -> Fact | None:
        row = self._conn.execute("SELECT payload FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return Fact.model_validate_json(row["payload"]) if row else None

    def facts(self, doc_id: str | None = None, limit: int | None = None) -> list[Fact]:
        sql = "SELECT payload FROM facts"
        params: list[Any] = []
        if doc_id:
            sql += " WHERE doc_id = ?"
            params.append(doc_id)
        sql += " ORDER BY doc_id, page"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [Fact.model_validate_json(r["payload"]) for r in self._conn.execute(sql, params)]

    def facts_in_block(self, measure_key: str, entity_key: str) -> list[Fact]:
        """Every fact sharing a blocking key. The linker's only fan-out query."""
        rows = self._conn.execute(
            "SELECT payload FROM facts WHERE measure_key = ? AND entity_key = ?",
            (measure_key, entity_key),
        )
        return [Fact.model_validate_json(r["payload"]) for r in rows]

    def blocks(self, min_size: int = 2) -> list[tuple[str, str, int]]:
        """Blocking keys holding at least ``min_size`` facts, largest first."""
        rows = self._conn.execute(
            """SELECT measure_key, entity_key, COUNT(*) AS n
                 FROM facts
                GROUP BY measure_key, entity_key
               HAVING n >= ?
                ORDER BY n DESC""",
            (min_size,),
        )
        return [(r["measure_key"], r["entity_key"], r["n"]) for r in rows]

    # -- relations ----------------------------------------------------------

    def add_relations(self, relations: Iterable[Relation]) -> int:
        rows = [
            (
                rel.id,
                rel.a_id,
                rel.b_id,
                rel.type.value,
                rel.reason_code.value,
                rel.confidence,
                rel.explanation,
                json.dumps(rel.analysis, default=str),
                rel.adjudicator,
                int(rel.cross_document),
            )
            for rel in relations
        ]
        if not rows:
            return 0
        with self.connect() as conn:
            conn.executemany(
                """INSERT INTO relations
                     (id, a_id, b_id, type, reason_code, confidence, explanation,
                      analysis, adjudicator, cross_document)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     type=excluded.type, reason_code=excluded.reason_code,
                     confidence=excluded.confidence, explanation=excluded.explanation,
                     analysis=excluded.analysis, adjudicator=excluded.adjudicator""",
                rows,
            )
        return len(rows)

    def relations(
        self,
        types: Iterable[str] | None = None,
        cross_document_only: bool = False,
        limit: int | None = None,
    ) -> list[Relation]:
        sql = "SELECT * FROM relations"
        clauses: list[str] = []
        params: list[Any] = []
        if types:
            types = list(types)
            clauses.append(f"type IN ({','.join('?' * len(types))})")
            params.extend(types)
        if cross_document_only:
            clauses.append("cross_document = 1")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY confidence DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [_row_to_relation(r) for r in self._conn.execute(sql, params)]

    def relations_for_fact(self, fact_id: str) -> list[Relation]:
        rows = self._conn.execute(
            "SELECT * FROM relations WHERE a_id = ? OR b_id = ? ORDER BY confidence DESC",
            (fact_id, fact_id),
        )
        return [_row_to_relation(r) for r in rows]

    def existing_relation_pairs(self) -> set[tuple[str, str]]:
        return {
            tuple(sorted((r["a_id"], r["b_id"])))  # type: ignore[misc]
            for r in self._conn.execute("SELECT a_id, b_id FROM relations")
        }

    # -- failures -----------------------------------------------------------

    def add_failures(self, failures: Iterable[ExtractionFailure]) -> int:
        rows = []
        for i, failure in enumerate(failures):
            fid = failure.id or f"x_{failure.doc_id}_{failure.page}_{failure.kind}_{i}"
            rows.append(
                (fid, failure.doc_id, failure.page, failure.stage, failure.kind,
                 failure.detail, failure.sample)
            )
        if not rows:
            return 0
        with self.connect() as conn:
            conn.executemany(
                """INSERT INTO failures (id, doc_id, page, stage, kind, detail, sample)
                   VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO NOTHING""",
                rows,
            )
        return len(rows)

    def failures(self, doc_id: str | None = None, limit: int | None = None) -> list[ExtractionFailure]:
        sql = "SELECT * FROM failures"
        params: list[Any] = []
        if doc_id:
            sql += " WHERE doc_id = ?"
            params.append(doc_id)
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [
            ExtractionFailure(
                id=r["id"], doc_id=r["doc_id"], page=r["page"], stage=r["stage"],
                kind=r["kind"], detail=r["detail"], sample=r["sample"],
            )
            for r in self._conn.execute(sql, params)
        ]

    def failure_summary(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT kind, COUNT(*) n FROM failures GROUP BY kind ORDER BY n DESC"
        )
        return {r["kind"]: r["n"] for r in rows}

    # -- measure registry ---------------------------------------------------

    def load_registry(self) -> MeasureRegistry:
        row = self._conn.execute("SELECT payload FROM registry WHERE id = 1").fetchone()
        if row is None:
            return MeasureRegistry()
        return MeasureRegistry.from_dict(json.loads(row["payload"]))

    def save_registry(self, registry: MeasureRegistry) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO registry (id, payload) VALUES (1, ?)
                   ON CONFLICT(id) DO UPDATE SET payload = excluded.payload""",
                (json.dumps(registry.to_dict()),),
            )

    # -- summary ------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        def count(table: str) -> int:
            return self._conn.execute(f"SELECT COUNT(*) c FROM {table}").fetchone()["c"]

        by_type = {
            r["type"]: r["n"]
            for r in self._conn.execute("SELECT type, COUNT(*) n FROM relations GROUP BY type")
        }
        return {
            "documents": count("documents"),
            "facts": count("facts"),
            "relations": count("relations"),
            "failures": count("failures"),
            "relations_by_type": by_type,
            "failures_by_kind": self.failure_summary(),
        }


def _row_to_document(row: sqlite3.Row) -> Document:
    from datetime import date

    published = row["published"]
    return Document(
        id=row["id"],
        filename=row["filename"],
        title=row["title"],
        publisher=row["publisher"],
        published=date.fromisoformat(published) if published else None,
        n_pages=row["n_pages"] or 0,
        sha256=row["sha256"] or "",
        ingested_at=row["ingested_at"] or "",
        status=row["status"] or "pending",
        stats=json.loads(row["stats"] or "{}"),
    )


def _row_to_relation(row: sqlite3.Row) -> Relation:
    from ..models import ReasonCode, RelationType

    return Relation(
        id=row["id"],
        a_id=row["a_id"],
        b_id=row["b_id"],
        type=RelationType(row["type"]),
        reason_code=ReasonCode(row["reason_code"]),
        confidence=row["confidence"] or 0.0,
        explanation=row["explanation"] or "",
        analysis=json.loads(row["analysis"] or "{}"),
        adjudicator=row["adjudicator"] or "deterministic",
        cross_document=bool(row["cross_document"]),
    )
