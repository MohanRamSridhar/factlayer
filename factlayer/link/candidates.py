"""Choosing which fact pairs are worth judging.

Comparing every fact to every other fact is quadratic, and on six 100-page
filings that is millions of comparisons to find a few dozen interesting ones.
It is also unnecessary: two facts can only corroborate or contradict each other
if they are about the same thing, and "the same thing" is exactly the
``(measure_key, entity_key)`` pair the registry already computes.

So candidate generation is an index lookup, not a search. Facts sharing a
blocking key are compared; facts that do not share one never meet. That keeps
the cost proportional to the number of *collisions* rather than the number of
facts, which is what allows a new document to be linked into an existing
knowledge layer without re-examining everything already stored.

The trade-off is honest and worth stating: recall now depends entirely on the
measure registry. Two documents naming the same quantity in words that share no
rare tokens land in different blocks and are never compared. A vector recall
pass over fact embeddings would widen the funnel here; it is the first thing to
add next, and it would only ever *propose* pairs -- the adjudicator would still
decide.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from ..models import Fact, Relation, RelationType
from ..store.db import Store
from .adjudicate import adjudicate

log = logging.getLogger("factlayer.link")

# A single blocking key holding more facts than this is almost always a sign
# that the registry over-merged (or that a document repeats one figure on every
# page). Comparing all of it would dominate the run for little gain.
MAX_FACTS_PER_BLOCK = 120


@dataclass
class LinkReport:
    blocks_examined: int = 0
    pairs_considered: int = 0
    pairs_skipped_existing: int = 0
    relations_written: int = 0
    oversized_blocks: list[tuple[str, str, int]] = field(default_factory=list)
    by_type: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "blocks_examined": self.blocks_examined,
            "pairs_considered": self.pairs_considered,
            "pairs_skipped_existing": self.pairs_skipped_existing,
            "relations_written": self.relations_written,
            "oversized_blocks": self.oversized_blocks,
            "by_type": self.by_type,
        }


def _pair_priority(a: Fact, b: Fact) -> tuple[int, float]:
    """Cross-document pairs first, then by joint confidence.

    Cross-document relationships are what the brief asks for; two facts from the
    same PDF agreeing with each other is mostly a duplicate-detection result.
    """
    return (0 if a.doc_id != b.doc_id else 1, -(a.confidence + b.confidence))


def candidate_pairs(
    facts: list[Fact],
    restrict_to_doc: str | None = None,
    cross_document_only: bool = False,
) -> Iterator[tuple[Fact, Fact]]:
    """Ordered pairs from one block.

    ``restrict_to_doc`` keeps incremental ingestion cheap: when a new document
    arrives, only pairs touching it are new, so the rest are skipped without
    ever being constructed.
    """
    pairs = itertools.combinations(facts, 2)
    selected = []
    for a, b in pairs:
        if cross_document_only and a.doc_id == b.doc_id:
            continue
        if restrict_to_doc and restrict_to_doc not in (a.doc_id, b.doc_id):
            continue
        selected.append((a, b))
    selected.sort(key=lambda pair: _pair_priority(*pair))
    yield from selected


class Linker:
    """Drives candidate generation and adjudication over a store."""

    def __init__(
        self,
        store: Store,
        cross_document_only: bool = False,
        max_facts_per_block: int = MAX_FACTS_PER_BLOCK,
    ) -> None:
        self.store = store
        self.cross_document_only = cross_document_only
        self.max_facts_per_block = max_facts_per_block

    def link(self, new_doc_id: str | None = None) -> LinkReport:
        """Adjudicate every unjudged candidate pair.

        Passing ``new_doc_id`` restricts work to pairs involving that document,
        which is what makes adding the seventh PDF cost the same as adding the
        second rather than re-deriving the whole layer.
        """
        report = LinkReport()
        existing = self.store.existing_relation_pairs()
        relations: list[Relation] = []

        for measure_key, entity_key, size in self.store.blocks(min_size=2):
            facts = self.store.facts_in_block(measure_key, entity_key)
            if len(facts) > self.max_facts_per_block:
                # Keep the most confident facts and record the truncation rather
                # than silently dropping half a block.
                report.oversized_blocks.append((measure_key, entity_key, len(facts)))
                facts = sorted(facts, key=lambda f: -f.confidence)[: self.max_facts_per_block]

            if new_doc_id and not any(f.doc_id == new_doc_id for f in facts):
                continue

            report.blocks_examined += 1
            for a, b in candidate_pairs(
                facts,
                restrict_to_doc=new_doc_id,
                cross_document_only=self.cross_document_only,
            ):
                key = tuple(sorted((a.id, b.id)))
                if key in existing:
                    report.pairs_skipped_existing += 1
                    continue
                existing.add(key)  # type: ignore[arg-type]
                report.pairs_considered += 1

                relation = adjudicate(a, b)
                if relation is None:
                    continue
                relations.append(relation)
                report.by_type[relation.type.value] = report.by_type.get(relation.type.value, 0) + 1

        report.relations_written = self.store.add_relations(relations)
        log.info(
            "linked %d blocks, %d pairs -> %d relations",
            report.blocks_examined, report.pairs_considered, report.relations_written,
        )
        return report


def interesting_relations(relations: Iterable[Relation]) -> list[Relation]:
    """Order relations the way a reviewer wants to read them.

    Contradictions first because they are the ones that need a human, then
    reconciliations because they are the ones that show the system reasoning,
    then everything else.
    """
    rank = {
        RelationType.CONTRADICTS: 0,
        RelationType.RECONCILED: 1,
        RelationType.SUPERSEDES: 2,
        RelationType.CORROBORATES: 3,
        RelationType.RELATED: 4,
        RelationType.UNDETERMINED: 5,
    }
    return sorted(relations, key=lambda r: (rank.get(r.type, 9), -r.confidence))
