# factlayer

A fact knowledge layer for PDFs. It extracts checkable claims, grounds every one
in verbatim source text, and works out whether claims from different documents
corroborate each other, genuinely conflict, or only appear to conflict because
they describe different things.

The interesting part is not the extraction. It is the last step: deciding that
two numbers which disagree are *not* in conflict, and being able to say exactly
why.

---

## Setup and Run Instructions

Requires Python 3.10+ and a Gemini API key (the free tier is enough — get one at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey)).

```bash
git clone <this-repo> && cd factlayer

python3 -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env
# open .env and paste your key into GEMINI_API_KEY=
```

**Run the UI:**

```bash
factlayer serve
# open http://127.0.0.1:8000
```

Upload a PDF through the page and watch it process. Upload a second document
about the same subject and the relationships appear.

**Or use the CLI:**

```bash
factlayer ingest path/to/docs/            # a directory or individual PDFs
factlayer show --type contradicts         # relationships with their evidence
factlayer show --type reconciled --limit 5
factlayer stats
```

**Or the API directly:**

```bash
curl -F file=@report.pdf localhost:8000/api/documents   # -> {"job_id": ...}
curl localhost:8000/api/jobs/<job_id>                   # poll until done
curl localhost:8000/api/relations?type=contradicts
curl localhost:8000/api/facts?q=inflation
curl localhost:8000/api/failures
```

Interactive API docs are at `/docs`.

### Running without an API key

`FACTLAYER_LLM=none` runs the pipeline with no model at all. Ingestion,
normalisation, linking and adjudication all still work; only claim-finding is
degraded, and the system says so rather than pretending. This exists so the
project can be inspected without credentials.

### Budget, and a warning about the free tier

**Gemini's free tier meters requests per day, per model, and the newest models
are capped hard** — `gemini-3.8-flash` allows 20 requests *per day*. This is easy
to mistake for a per-minute limit and lose a day's budget to. `.env.example`
therefore defaults to `gemini-3.5-flash-lite`, which is far more generous. Check
your own limits at [ai.dev/rate-limit](https://ai.dev/rate-limit).

The pipeline is built around that constraint. Pages are batched into one call
(`FACTLAYER_BATCH_PAGES`, default 25), so the whole three-document dataset costs
**24 requests**, not 200. Low-density pages are dropped before batching, and
`FACTLAYER_MAX_UNITS_PER_DOC` caps batches per document. **Any run that hit the
cap says so** — in the CLI output, the document's stats, and the UI's coverage
column.

Responses are cached by content hash, so re-running after a code change costs
nothing. The full dataset re-ingests from cache in about 3 seconds.

---

## Video Demo

<!-- TODO: paste the link before submitting -->
**[Demo video (3 min)](ADD_LINK_HERE)**

---

## Approach

### The reframe

The brief looks like an extraction task. It is really a reconciliation task.
Three of the four required cases are about *whether a disagreement is real*, and
none of them are answered by better extraction. So the architecture puts its
weight at the far end of the pipeline.

```
ingest → chunk → extract → ground → normalise → register → link → adjudicate
```

### The one rule: the model proposes, the code decides

The LLM finds claims, frames them, and copies the supporting span. It is **never
asked to normalise a number, compare two figures, or judge a relationship.**
Deterministic code parses values, converts units, resolves periods to real
dates, and issues every verdict.

This is the decision everything else follows from:

- A model that is bad at arithmetic cannot corrupt a comparison.
- The same figure extracted twice always normalises identically.
- Every verdict has a **reason code** and the arithmetic behind it, so the
  interface explains a judgement rather than paraphrasing one. Expand *"the
  comparison behind this verdict"* on any relationship card to see the actual
  numbers, tolerances and context fields the decision used.

### The context envelope

A bare number is not a fact. `6.4 per cent` only becomes checkable once you know
what it measures, who it is about, when it applies, on what basis it was
computed, and how firm it is. Every fact carries that bundle:

| Field | Why it exists |
|---|---|
| `entity` / `entity_key` | Documents say "the Company"; comparison needs a name |
| `period` | Resolved to **actual dates**, not kept as a label |
| `basis` | consolidated / standalone / constant prices / seasonally adjusted |
| `estimate_type` | advance estimate → provisional → revised → actual |
| `attributed_to` | The figure's real provenance when a document quotes another source |
| `as_of` | Publication date, which drives vintage reasoning |

Resolving periods to dates rather than strings is load-bearing. "Q2 FY25" (an
Indian fiscal quarter, Jul–Sep 2024) and "2025Q2" (Apr–Jun 2025) both read as
"Q2" and are a year apart. String matching walks straight into that.

**Two figures that disagree are only a contradiction if their envelopes agree.**
When the envelopes differ, the difference usually *explains* the disagreement —
which is exactly case 3.

### Grounding

The extractor must copy a verbatim span. If that span cannot be located in the
page it came from, **the fact is rejected and the rejection is recorded.**
Matching tolerates whitespace only — PDF text reflows unpredictably, but
allowing paraphrase would defeat the purpose. Nothing enters the layer without a
resolved character offset into a real page.

A second check: the claimed value must appear inside its own quote. That catches
the failure where a model reads the right sentence and the wrong column.

### The adjudication ladder

Checks run in order and stop at the first that fires. **Every reconciliation
check runs before the contradiction verdict** — that ordering is the single most
important thing in the codebase.

| # | Check | Verdict |
|---|---|---|
| 1 | Values agree, envelopes agree | `CORROBORATES` (exact / rounded / unit-converted) |
| 2 | Periods resolve to different windows | `RECONCILED` — `PERIOD_MISMATCH` |
| 3 | Periods *written* alike but resolve apart | `RECONCILED` — `PERIOD_LABEL_COLLISION` |
| 4 | `basis` differs | `RECONCILED` — `BASIS_MISMATCH` |
| 5 | Estimate firmness differs | `RECONCILED` — `ESTIMATE_VS_ACTUAL` |
| 6 | Published far apart, same window | `RECONCILED` — `VINTAGE_REVISION` |
| 7 | Later document restates a status | `SUPERSEDES` — `STATE_SUPERSEDED` |
| 8 | Nothing explains the gap | `CONTRADICTS` — `VALUE_DIVERGENCE` |
| 9 | Context too thin to judge | `UNDETERMINED` — `INSUFFICIENT_CONTEXT` |

Tolerance is **rounding-aware**, not a flat percentage. "6.4 per cent" claims to
be nearer 6.4 than 6.3, not to be 6.400; comparing it against 6.37 as though
both were exact manufactures a contradiction out of rounding. The tolerance is
half the last written digit, widened for hedged figures ("about 740 Mn").

State facts get their own branch. "X is a director" and "X ceased to be a
director" do not contradict each other — the later one *replaces* the earlier.
Treating that as a conflict is a mistake worth designing against.

### The schema builds itself

Nothing in the code enumerates domain measures. `measure_key` is assigned at
runtime by a registry that clusters measure names as documents arrive, minting
new keys for unfamiliar ones. A document about rainfall creates rainfall
measures with no code change. The **Schema** tab shows the current measure set
and which surface forms merged into each — it is entirely derived from whatever
has been uploaded.

Clustering uses three cheap signals rather than embeddings:

1. **Rarity weighting** — shared rare tokens are evidence, shared filler is not.
   Computed from the corpus at hand, so it adapts to the domain.
2. **Derivative markers** — a closed class of generic English words (growth,
   share, margin, per) that change *what* is measured. "Revenue" and "revenue
   growth" can never merge.
3. **Kind agreement** — a percentage and a currency amount are different
   measures even when identically worded.

Embeddings were considered and rejected *for now*: they add an API dependency
and a second failure mode, and measure names are short noun phrases drawn from
the documents themselves. The cost of that choice is real and stated under
Limitations.

### Scale and incremental ingestion

Comparing every fact against every other is quadratic. Instead, facts are
blocked on `(measure_key, entity_key)` and only compared within a block, so cost
tracks *collisions* rather than fact count.

Adding a document does not rebuild the layer. Facts are content-addressed, the
registry persists in the database, and linking is restricted to pairs touching
the new document. The tenth PDF costs what the second cost. Re-uploading an
identical PDF is a no-op rather than a duplicate set of facts. LLM responses are
cached by content hash, so re-running after a downstream change costs nothing.

### Storage

SQLite. The brief's warning that a graph database alone is not the solution is
not only about presentation — deciding whether two facts relate is computation,
and once decided the answer is a flat table of pairs. One file, no server, and
the clone-and-run promise survives.

### AI tools used

Claude Code (Opus 5) for implementation throughout, as a pair-programmer:
architecture argued out in conversation, code written and reviewed
collaboratively. Gemini (`gemini-3.8-flash`) is the runtime extraction model.
The prompt-tightening commit came from reading the first real extraction output
and finding that measures were being named too thinly to match across documents.

---

## The Four Required Cases

All four are real output from the three-document India macroeconomy dataset
(Economic Survey 2024-25, RBI Annual Report 2024-25, IMF Article IV 2025).
Reproduce with `factlayer ingest starter-datasets/india-macroeconomy` then
`factlayer show --type <type>`.

Run summary: **1,060 facts, 572 relationships** — 38 corroborations,
345 reconciled, 2 contradictions, 107 undetermined, 138 recorded failures.

### 1. A fact corroborated across documents, expressed differently

RBI and the IMF state India's FY2024-25 real GDP growth in different words, in
documents published six months apart. The system resolves both periods to
1 Apr 2024 – 31 Mar 2025 and matches the values exactly.

> **CORROBORATES** · `exact_match` · confidence 0.95 · cross-document
>
> **RBI Annual Report, p23** — real GDP growth = 6.5 per cent [FY2025]
> *"estimated to have grown by 6.5 per cent in 2024-25, as compared with 9.2 per cent a year ago"*
>
> **IMF Article IV, p10** — real GDP growth = 6.5 percent [FY2025]
> *"India's real GDP grew by 6.5 percent in FY2024/25."*

Note that neither the measure wording ("real GDP growth" vs "India's real GDP
grew"), the period wording ("2024-25" vs "FY2024/25") nor the unit spelling
("per cent" vs "percent") matches. The match happens on resolved dates and
canonical values, not on strings.

A second one shows rounding handled properly — the Survey's *USD 704.9 billion*
of forex reserves and the IMF's *$706 billion* for September 2024 are returned
as `rounded_match`, agreeing within a tolerance derived from how precisely each
was written rather than a flat percentage.

### 2. A genuine or likely contradiction

> **CONTRADICTS** · `value_divergence` · confidence 0.92
>
> USD 8.5 billion vs USD 10.1 billion (difference 15.8%). Same measure, same
> entity, same period, same basis and comparable estimate types — nothing in the
> surrounding context accounts for the difference.
>
> **Economic Survey, p66** — net FDI = USD 8.5 billion [FY2024]
> *"net FDI to India during the first eight months of FY25 stood at USD 0.48 billion compared to USD 8.5 billion in the corresponding period of FY24"*
>
> **Economic Survey, p66** — net FDI = USD 10.1 billion [FY2024]
> *"For FY24 as a whole, the net FDI was USD 10.1 billion."*

Two figures for net FDI in FY24, on the same page, differing by 19%. The system
is right that nothing it extracted explains the gap, and a reader looking at the
evidence sees the answer immediately: the first covers *eight months* of FY24,
the second the *whole* year. The phrase "the corresponding period" carries that
scope, and the extractor recorded the parent fiscal year instead.

**Being straight about this: the system found no genuine contradiction between
the three documents.** That is a finding, not a gap in the demo. These sources
report the same official statistics, and where they differ the difference is
explained — which is why 345 pairs came back reconciled and only 2 as conflicts.
Both surviving conflicts are within a single document and both trace to the same
root cause: a sub-annual period labelled with its parent fiscal year. That is
the most valuable thing this run revealed about the system, and it points
directly at the next fix (see Limitations).

### 3. An apparent contradiction explained by context

The case the whole design exists for. Two documents report "Q2" real GDP growth
2.2 points apart. It looks like a flat contradiction and is not one.

> **RECONCILED** · `period_label_collision` · confidence 0.85 · cross-document
>
> 5.6 per cent vs 7.8 percent (difference 28.2%). The periods are written alike
> (q2) but resolve to different windows: **2024-07-01 to 2024-09-30** against
> **2025-04-01 to 2025-06-30**. Different periods, not conflicting figures.
>
> **RBI Annual Report, p24** — real GDP growth = 5.6 per cent [Q2 FY2025, *fiscal* quarter]
> *"growth softened to 5.6 per cent in Q2"*
>
> **IMF Article IV, p10** — real GDP growth = 7.8 percent [2025Q2, *calendar* quarter]
> *"underpinning real GDP growth of 7.8 percent in 2025Q2."*

RBI's "Q2" is the Indian fiscal quarter Jul–Sep 2024; the IMF's is the calendar
quarter Apr–Jun 2025. They are three quarters apart. Any system comparing period
*labels* reports a contradiction here. Resolving them to dates is what turns a
false alarm into an explanation.

A second flavour, reconciled on basis rather than time:

> **RECONCILED** · `basis_mismatch` · cross-document
>
> **Economic Survey, p34** — unemployment rate = 6 per cent, basis `["aged 15 years and above", "usual status"]`
> **IMF Article IV, p10** — unemployment rate = 5.2 percent, basis `[]`

Same measure, same country, incompatible definitions. The qualifiers were
captured at extraction, so the engine can say *why* the figures differ instead
of ranking one over the other.

### 4. An extraction or reasoning failure

Three, in descending order of how much they cost.

**a) The one that nearly sank the run — two-column reading order.** The RBI
Annual Report is typeset in two columns. PDF blocks were being sorted
top-to-bottom across the full page width, which interleaves the columns line by
line. The resulting text is scrambled but *still readable by a language model* —
so the model silently reassembled each real sentence and quoted that. The quote
then could not be found verbatim on the page, and grounding rejected it. **403
of 419 facts from that document were discarded**, and it looked like a model
quality problem rather than a reading-order bug.

Found by checking whether rejected quotes appeared *anywhere* in the document:
they did not, which ruled out page misattribution and pointed at the text
itself. Fixed with conservative column detection (`detect_two_columns`), applied
only when both columns carry real content and few blocks straddle the gutter.

Result: RBI facts went **174 → 563**, and total ungrounded quotes **429 → 131**.
This is the strongest argument for keeping the failure log as a product surface:
the bug was invisible in the facts that succeeded and obvious in the ones that
did not. Regression tests are in `tests/test_normalize.py`.

**b) A reasoning failure I fixed by declining to judge.** The ladder was
reporting contradictions between two figures pulled from *the same sentence* —
"revised its inflation projection from 4.5 per cent to 4.8 per cent" became a
4.5-vs-4.8 conflict, and "apparel constituting 42 per cent … non-apparel 30 per
cent" became a 42-vs-30 conflict. In both the fault is ours: one statement split
badly, or two neighbouring measures merged by the registry. Blaming the document
for our own ambiguity is the wrong answer, so a shared-evidence guard now
returns `UNDETERMINED` / `shared_sentence_ambiguity` and flags the pair for
review. Contradictions fell from 8 to 2, and the five it removed were all false.

**c) What is still wrong.** 131 quotes remain ungrounded — mostly tables, where
the model reconstructs a row that never existed as contiguous text. 4 values
would not parse, 3 did not appear in their own quote. And the registry still
over-merges occasionally: it briefly treated "depletion in foreign exchange
reserves" as the same measure as "foreign exchange reserves", producing a
nonsense reconciliation between USD 13.8 billion and USD 695 billion. Fixed by
extending the derivative-marker class, but the general problem — a lexical
matcher cannot know which words are operations — is unsolved.

Every one of these is visible in the UI's **What failed** tab and at
`GET /api/failures`, with the discarded input attached.

## Limitations and Next Steps

Honest list of what does not work yet.

**Sub-annual periods are labelled with their parent year.** The most damaging
limitation, and the cause of both surviving false contradictions. "the first
eight months of FY24" and "Q2 of FY24" are both recorded as FY24, so a partial
period is compared against a full year as though they were the same window.
*Next, and first:* teach period parsing to recognise partial-period language
("first N months of", "during April–December of") and narrow the window
accordingly. This is a contained change to `normalize/periods.py` and would
likely remove every false contradiction in this run.

**Measure clustering is lexical.** Genuine synonyms sharing no tokens do not
merge — "forex reserves" and "foreign exchange reserves" are still different
measures, so facts about them never meet. The inverse also happens: the matcher
cannot know which words are operations rather than descriptions, so "depletion
in reserves" merged with "reserves" until the derivative-marker class was
extended by hand. Extending a word list is not a real fix. *Next:* a vector
recall pass over fact embeddings feeding the same deterministic adjudicator —
retrieval would only ever *propose* pairs, never judge them.

**Table extraction is the weakest link that remains.** 131 quotes are still
rejected as ungrounded, most of them table rows the model reconstructs into text
that never existed contiguously. The page-level scale note (`₹ in Million`) is
applied as a hint and deliberately dropped when pages in a batch disagree, but a
table with mixed units on one page will still produce wrong magnitudes.

**Facts do not cross pages.** Grounding is page-local so evidence keeps one
unambiguous page number. A figure whose meaning depends on a header two pages
back is extracted without it, or not at all.

**Context enrichment is not built.** Sometimes the fact explaining an apparent
contradiction is in *neither* fact's chunk — a footnote reading "all figures at
constant 2011-12 prices". The adjudicator should retrieve nearby blocks before
returning `CONTRADICTS`. This was designed and deliberately cut for time; it is
the first thing I would add, and it would directly improve case 3.

**No LLM tiebreaker on measure equivalence.** The `Relation` model carries an
`adjudicator` field with `"llm"` and `"hybrid"` values that nothing currently
produces. The intent was a narrow escalation for borderline measure identity,
recorded as a visibly different class of judgement. Not wired up.

**Document titles are unreliable.** Title detection picked "Climate Transition,
China and Geopolitics" — a chapter heading — for the Economic Survey. Cosmetic,
but visible in the UI and in every explanation that names a document.

**Only one dataset was run.** The Delhivery corpus was not ingested, so the
state-supersession path (a director appearing active in one document and
resigned in a later one) is exercised only by unit tests, not by real documents.
The code path is there and tested; it has not been proven on a real filing.

**Period resolution has gaps.** Ranges like "April to November 2024" resolve to
a month rather than a span, and relative expressions ("the previous year") stay
unresolved. Unresolved periods are handled honestly — they downgrade a verdict
to `UNDETERMINED` rather than being guessed at — but they cost recall.

**The measure registry is order-dependent.** Keys are assigned against whatever
the registry has already seen, so re-ingesting one document into a populated
layer can cluster its measures slightly differently than the original run did —
re-uploading the IMF report changed the relation count from 572 to 570. Nothing
is lost or corrupted, but the layer is not perfectly reproducible under
reordering, and a strict system would either version the registry or re-key
affected facts when a cluster changes.

**Confidence scores are heuristic.** They order results usefully and should not
be read as calibrated probabilities.

**No authentication, no multi-user isolation, one global knowledge layer.** Out
of scope for a prototype, but worth stating.

---

## Additional Notes

**On what I chose not to build.** A graph visualisation would have demoed well
and taught nothing. The brief says so directly, and building it would have come
out of the adjudication ladder's budget. Same reasoning for a chat interface
over the documents: retrieval-augmented answering is the median approach to this
brief, and it cannot do exhaustive extraction, cannot guarantee grounding, and
cannot produce a structured contradiction verdict. Retrieval belongs *inside*
this system — at measure clustering and at context enrichment — not as its
architecture.

**On the failure view.** Rejected claims are surfaced as a first-class tab
rather than a log file. This is a design position: a knowledge layer that cannot
report its own blind spots invites more trust than it has earned. The failure
kinds are specific enough to act on — `ungrounded_quote`, `value_not_in_quote`,
`unparseable_value`, `llm_call_failed` — and each keeps the discarded input.

**On honesty in the output.** Several verdicts deliberately decline to judge.
`UNDETERMINED` with `INSUFFICIENT_CONTEXT` is a real answer, not a failure to
produce one, and it is what the system returns when a period will not resolve or
units are not comparable. Guessing there would produce a more impressive-looking
demo and a less trustworthy system.

**Reproducibility.** LLM responses are cached under `data/cache/` by content
hash, and extraction runs at temperature 0. A second run over the same documents
reproduces the same layer without new API calls.
