"""
How we decide whether a chunk contains the answer (used in Steps 6-8).

Each test question stores an "evidence quote": a short, exact sentence copied
from the source document. A chunk counts as RELEVANT if it contains that quote.

Why quotes instead of chunk IDs? Chunk IDs are different for every chunking
strategy, but the quote is the same text everywhere. So one test set can fairly
score all three strategies.
"""

import re


def normalize(text):
    """Lowercase and keep only letters and digits, so formatting differences don't matter."""
    text = text.lower().replace("’", "'")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def contains_quote(text, quote):
    """True if the (normalized) quote appears in the (normalized) text as whole words."""
    return f" {normalize(quote)} " in f" {normalize(text)} "


def relevant_chunk_ids(quotes, chunks):
    """All chunks containing at least one of the evidence quotes."""
    return {c["chunk_id"] for c in chunks if any(contains_quote(c["text"], q) for q in quotes)}


def quotes_found(quotes, chunks):
    """Which evidence quotes appear somewhere in these chunks (e.g. the retrieved top-k)?"""
    return [q for q in quotes if any(contains_quote(c["text"], q) for c in chunks)]
