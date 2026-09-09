"""Command line entry point.

Exists so the system can be driven without the HTTP layer: batch ingestion, a
re-link after changing the adjudication ladder, and a summary of what the layer
currently believes. The API and the CLI share one pipeline, so neither can drift
from the other.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .link.candidates import interesting_relations
from .pipeline import Pipeline, Settings, load_project_env


def cmd_ingest(args: argparse.Namespace) -> int:
    pipeline = Pipeline(Settings.from_env())
    if not pipeline.provider.available():
        print(
            f"warning: LLM backend {pipeline.provider.name!r} is unavailable; "
            "extraction will produce nothing. Check GEMINI_API_KEY.",
            file=sys.stderr,
        )

    paths: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.pdf")))
        else:
            paths.append(path)

    if not paths:
        print("no PDFs found", file=sys.stderr)
        return 1

    for index, path in enumerate(paths, 1):
        print(f"\n[{index}/{len(paths)}] {path.name}", flush=True)

        def progress(done: int, total: int) -> None:
            print(f"\r    extracting {done}/{total} units", end="", flush=True)

        report = pipeline.ingest(
            path, force=args.force, max_units=args.max_units, progress=progress
        )
        print()
        for note in report.notes:
            print(f"    note: {note}")
        print(
            f"    {report.facts} facts, {report.failures} failures, "
            f"{report.link.get('relations_written', 0)} relations, {report.seconds:.0f}s"
        )

    print("\n" + json.dumps(pipeline.store.stats(), indent=2))
    return 0


def cmd_link(args: argparse.Namespace) -> int:
    pipeline = Pipeline(Settings.from_env())
    report = pipeline.relink_all() if args.rebuild else pipeline.link()
    print(json.dumps(report.as_dict(), indent=2))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    pipeline = Pipeline(Settings.from_env())
    print(json.dumps(pipeline.store.stats(), indent=2))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Print the relationships a reviewer most wants to see."""
    pipeline = Pipeline(Settings.from_env())
    store = pipeline.store
    types = [t.strip() for t in args.type.split(",")] if args.type else None
    relations = interesting_relations(
        store.relations(types=types, cross_document_only=args.cross_document_only)
    )[: args.limit]

    if not relations:
        print("no relations matched")
        return 0

    for rel in relations:
        a, b = store.get_fact(rel.a_id), store.get_fact(rel.b_id)
        print("\n" + "=" * 78)
        print(f"{rel.type.value.upper()}  [{rel.reason_code.value}]  confidence={rel.confidence}")
        print(f"  {rel.explanation}")
        for fact in (a, b):
            if fact is None:
                continue
            period = fact.context.period
            label = (period.label or period.raw) if period else "no period"
            print(f"\n  -- {fact.doc_id} p{fact.evidence.page}")
            print(f"     {fact.measure_raw} = {fact.display_value()}  [{label}]")
            if fact.context.basis:
                print(f"     basis: {', '.join(fact.context.basis)}")
            quote = " ".join(fact.evidence.quote.split())
            print(f"     \"{quote[:200]}\"")
    print()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("factlayer.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_project_env()
    parser = argparse.ArgumentParser(
        prog="factlayer",
        description="Extract grounded facts from PDFs and reconcile them across documents.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest one or more PDFs (or a directory)")
    p_ingest.add_argument("paths", nargs="+")
    p_ingest.add_argument("--force", action="store_true", help="re-ingest even if already stored")
    p_ingest.add_argument("--max-units", type=int, default=None, help="per-document LLM budget")
    p_ingest.set_defaults(func=cmd_ingest)

    p_link = sub.add_parser("link", help="adjudicate unjudged pairs")
    p_link.add_argument("--rebuild", action="store_true", help="discard and re-derive all relations")
    p_link.set_defaults(func=cmd_link)

    p_show = sub.add_parser("show", help="print relationships with their evidence")
    p_show.add_argument("--type", default=None, help="comma-separated relation types")
    p_show.add_argument("--cross-document-only", action="store_true")
    p_show.add_argument("--limit", type=int, default=20)
    p_show.set_defaults(func=cmd_show)

    p_stats = sub.add_parser("stats", help="summarise the knowledge layer")
    p_stats.set_defaults(func=cmd_stats)

    p_serve = sub.add_parser("serve", help="run the API and UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
