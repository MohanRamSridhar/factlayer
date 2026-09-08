"""Entity name normalisation.

Two documents rarely spell an entity the same way: "Delhivery Limited" vs
"Delhivery Ltd." vs "the Company"; "Mr. Suvir Suren Sujan" vs "Suvir Sujan".
Facts about the same entity must collide on one key or they will never be
compared, so this reduces names to a stable form.

The rules here are linguistic, not domain-specific: strip honorifics, strip
corporate suffixes, fold punctuation and case. Nothing knows what a logistics
company is.
"""

from __future__ import annotations

import re
import unicodedata

HONORIFICS = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "shri", "smt", "sri", "sh",
    "hon", "justice", "capt", "col", "gen", "rev", "sir",
}

CORPORATE_SUFFIXES = {
    "limited", "ltd", "llp", "llc", "inc", "incorporated", "corp", "corporation",
    "plc", "pvt", "private", "co", "company", "gmbh", "sa", "nv", "bv", "ag",
    "holdings", "group", "and", "&",
}

# Deictic references that only mean something inside their own document. They are
# resolved to the document's principal entity during extraction, never merged
# blindly across documents.
DEICTIC = {
    "the company", "our company", "the group", "the bank", "the issuer",
    "the corporation", "we", "us", "the organisation", "the organization",
}


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def entity_key(name: str) -> str:
    """Reduce an entity name to a comparison key.

    >>> entity_key("Delhivery Limited")
    'delhivery'
    >>> entity_key("Mr. Suvir Suren Sujan")
    'suvir suren sujan'
    """
    if not name:
        return ""
    text = strip_accents(name).lower()
    text = re.sub(r"[‘’“”]", "'", text)
    text = re.sub(r"[^\w\s&']", " ", text)
    tokens = [t for t in text.split() if t]

    while tokens and tokens[0].rstrip(".") in HONORIFICS:
        tokens.pop(0)
    while tokens and tokens[-1].rstrip(".") in CORPORATE_SUFFIXES:
        tokens.pop()

    tokens = [t for t in tokens if t not in {"the", "of", "a", "an"}] or tokens
    return " ".join(tokens).strip()


def is_deictic(name: str) -> bool:
    return name.strip().lower() in DEICTIC


def initials_form(name: str) -> str:
    """'Suvir Suren Sujan' -> 's s sujan'. Used for loose person matching."""
    tokens = entity_key(name).split()
    if len(tokens) < 2:
        return " ".join(tokens)
    return " ".join([t[0] for t in tokens[:-1]] + [tokens[-1]])


def same_entity(a: str, b: str) -> bool:
    """Conservative equality: exact key match, containment for multi-word names,
    or matching initials form for people."""
    ka, kb = entity_key(a), entity_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    ta, tb = ka.split(), kb.split()
    # "delhivery" vs "delhivery logistics" -- accept containment only when the
    # shorter name is a real prefix, not a single common word.
    if len(ta) >= 1 and len(tb) >= 1:
        short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
        if len(short) >= 1 and long_[: len(short)] == short and len(short[0]) > 3:
            return True
    if len(ta) >= 2 and len(tb) >= 2 and initials_form(a) == initials_form(b):
        return True
    return False
