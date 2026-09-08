"""Prompts for extraction and adjudication.

Deliberately domain-free. Nothing here mentions logistics, India, finance or any
starter document; the same prompts should work on a clinical trial report or a
municipal budget. The brief is explicit that the system must not rely on
document-specific rules, and the prompt is where that discipline is easiest to
lose.

Division of labour with the deterministic code below:

  the model  finds claims, frames them, and reads context off the page
  the code   parses numbers, converts units, resolves periods, judges relations

The model is never asked for a normalised number. It reports the value exactly
as written and the code measures it. That way a model that is bad at arithmetic
cannot corrupt a comparison, and the same figure extracted twice always
normalises identically.
"""

EXTRACTION_SYSTEM = """\
You extract checkable factual claims from documents and ground each one in \
verbatim source text.

A claim is worth extracting when a reader could, in principle, verify or refute \
it against another source. Measured quantities, dated events, statuses that hold \
over a period, and stated properties all qualify. Opinions, forward-looking \
rhetoric without a number, marketing language, boilerplate, and definitions do \
not.

You must never state a value that does not appear in the supplied text. If you \
are unsure what a number refers to, omit it. A missed fact costs far less than \
an invented one.

Output JSON only. No commentary, no code fences.
"""

EXTRACTION_INSTRUCTIONS = """\
Extract every checkable factual claim from the SOURCE TEXT below.

Return a JSON array. Each element:

{
  "fact_type": "numeric" | "state" | "event" | "attribute",
  "subject": "the entity the claim is about: an organisation, country, \
institution, market or person. NOT the thing being measured. If the sentence is \
about a line item ('subsidies grew 25.7 per cent'), the subject is the entity \
that line item belongs to ('state governments'), never 'subsidies'.",
  "measure": "a SELF-CONTAINED description of what is measured, in the \
document's own words, that would still be unambiguous read on its own with no \
surrounding text. Include the thing measured, not just the operation applied to \
it. Bad: 'growth'. Good: 'growth in subsidy expenditure'. Bad: 'total'. Good: \
'total expenditure'.",
  "value_raw": "the value exactly as written, including symbols, scale words and \
brackets (e.g. '₹8,142 Cr', '(4,516.08)', '6.4 per cent'). Null for non-numeric facts.",
  "value_text": "for non-numeric facts, the asserted value in a few words \
(e.g. 'ceased to be a director', 'registered office at ...'). Null for numeric facts.",
  "period": "the time the claim applies to, exactly as written (e.g. 'FY24', \
'Q2 of FY25', '2025Q2', 'as on March 31, 2024', 'year ended March 31, 2024'). \
Null if the text gives none.",
  "basis": ["qualifiers that change what is counted, e.g. 'consolidated', \
'standalone', 'constant prices', 'seasonally adjusted', 'excluding traded goods', \
'services only'"],
  "estimate_type": "actual" | "provisional" | "revised" | "advance_estimate" \
| "projection" | "target" | "unknown",
  "attributed_to": "the third party the document credits the figure to, if any \
(e.g. 'NSO', 'staff estimates'), else null",
  "quote": "a VERBATIM span copied character-for-character from the SOURCE TEXT \
that contains this claim. It must appear in the source exactly.",
  "confidence": 0.0 to 1.0
}

Rules that matter:

1. QUOTE MUST BE VERBATIM. Copy it from the SOURCE TEXT without editing, \
reflowing, or fixing anything. Quotes that cannot be found in the source are \
discarded, and the fact with them.

2. VALUE AS WRITTEN. Do not convert units, expand scale words, or do arithmetic. \
If the text says '₹8,142 Cr', write '₹8,142 Cr'. If a table cell reads \
'(4,516.08)' write exactly that, brackets included.

3. USE THE SCALE NOTE. If a scale note is given above, bare table numbers are in \
that unit. Record the number as written; the note is handled separately.

4. RESOLVE PRONOUNS AND DEIXIS. 'the Company', 'the Bank', 'we' should be \
replaced by the named entity when the context makes it unambiguous; otherwise \
keep the phrase as written.

5. SPLIT COMPOUND SENTENCES. 'Revenue rose to X from Y' is two facts, one per \
period. A sentence comparing two periods yields a fact for each.

6. CAPTURE FIRMNESS. Words like 'estimated', 'advance estimates', 'projected', \
'provisional', 'revised' change estimate_type. This distinction decides whether \
two differing figures are a contradiction or a revision, so do not skip it.

7. CAPTURE BASIS. Anything that narrows what is counted belongs in basis. Two \
figures that differ only because one excludes an item are reconcilable, but only \
if the exclusion was recorded.

8. STATE AND EVENT FACTS. A person holding a role, an office being located \
somewhere, a status taking effect on a date - all extractable. For these set \
value_raw to null and describe the value in value_text.

9. NAME MEASURES SO THEY MATCH ACROSS DOCUMENTS. Another document will describe \
the same quantity in different words, and the two are only compared if their \
measure names share distinctive terms. Keep the document's own wording, but keep \
the words that identify the quantity: write 'real GDP growth', not 'growth'; \
write 'net FPI inflows', not 'inflows'. Drop only filler ('the level of', 'the \
figure for').

If the text contains no checkable claims, return [].
"""

DOC_METADATA_SYSTEM = """\
You read the opening pages of a document and report its bibliographic details. \
Output JSON only, no commentary.
"""

DOC_METADATA_INSTRUCTIONS = """\
From the text below, identify:

{
  "title": "the document's title",
  "publisher": "the organisation that published it",
  "published": "publication date as YYYY-MM-DD, or YYYY-MM, or YYYY; null if absent",
  "principal_entity": "the single entity this document is primarily about \
(a company, a country, an institution)",
  "document_type": "a short label, e.g. 'annual report', 'prospectus', \
'staff report', 'earnings presentation'",
  "period_covered": "the reporting period the document covers, as written, or null",
  "reporting_currency": "ISO code if the document reports money, else null",
  "notes": "anything a reader must know to interpret the figures, one sentence, or null"
}

Report only what the text supports. Use null where it is silent.
"""

ADJUDICATION_SYSTEM = """\
You are the final arbiter on whether two extracted factual claims agree, \
disagree, or only appear to disagree.

A deterministic engine has already compared them on units, periods, basis and \
firmness, and has handed you the cases it could not settle. Its analysis is \
given to you; it is usually right about the mechanics and may be wrong about \
meaning.

Judge the claims as a careful analyst would: two numbers that differ are only a \
contradiction if they genuinely purport to measure the same thing over the same \
window on the same basis. Differences in scope, timing, vintage, or definition \
are explanations, not contradictions.

Output JSON only.
"""

ADJUDICATION_INSTRUCTIONS = """\
Decide the relationship between FACT A and FACT B.

Return:

{
  "type": "corroborates" | "contradicts" | "reconciled" | "supersedes" \
| "related" | "undetermined",
  "confidence": 0.0 to 1.0,
  "explanation": "two or three sentences a reviewer can check against the quotes. \
Name the specific reason. Do not restate the numbers without interpreting them.",
  "key_difference": "the single contextual difference that drives the verdict, or \
null if there is none"
}

Meanings:

- corroborates: same claim, independently stated. Different wording or units is \
fine; the substance agrees.
- contradicts: same claim, same context, incompatible values. Use this only when \
you can see no contextual difference that would explain the gap.
- reconciled: the values differ AND a difference in period, scope, basis, \
definition or data vintage explains why. Say which one.
- supersedes: both describe a state of affairs, and one is a later statement that \
replaces the earlier. Say which is later.
- related: same subject area but not directly comparable.
- undetermined: the evidence is insufficient to judge. Prefer this over guessing.

Judge only from the quotes and context supplied. Do not use outside knowledge \
about these entities. If the quotes do not contain enough to decide, say \
undetermined.
"""
