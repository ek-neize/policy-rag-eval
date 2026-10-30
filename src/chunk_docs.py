"""
Step 2: Split documents into chunks, three different ways.

Why: search tools and language models work best on short passages.
How we split changes what the assistant can find, so we build three
strategies now and measure which works best later (Step 8).

  fixed     - every chunk is ~200 words, with a 40-word overlap.
              Ignores the document's structure. This is the baseline.
  heading   - one chunk per "##" section, so each chunk is a complete
              topic. Long sections are split, tiny ones are merged.
  sentence  - small windows of 3 sentences. Very precise, little context.

Input:  data/raw/manifest.csv and data/raw/docs/*.md   (from Step 1)
Output: data/processed/chunks_<strategy>.jsonl         (one chunk per line)
"""

import csv
import json
import re
import statistics
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# Sizes are kept under ~350 words (~460 tokens) so every chunk fits inside the
# 512-token limit of the embedding models we'll use in Step 3.
FIXED_WORDS = 200          # words per fixed-size chunk
FIXED_OVERLAP = 40         # words shared between neighbouring fixed chunks
SECTION_MAX_WORDS = 350    # split heading-based sections longer than this
SECTION_MIN_WORDS = 30     # merge heading-based sections shorter than this
SENTENCES_PER_CHUNK = 3    # sentence-window size
SENTENCE_STRIDE = 2        # move forward 2 sentences each time (1 overlaps)

# Sections that are navigation, not content. They're removed everywhere.
SKIP_HEADINGS = {"on this page", "related links"}

HEADING = re.compile(r"^(#{1,6}) (.*)$")
# A sentence ends at . ! or ? followed by a space and a capital letter, digit or quote.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")


# ---------------------------------------------------------------------------
# Loading and parsing
# ---------------------------------------------------------------------------
def load_documents():
    """Read the manifest and attach each document's text."""
    with open(RAW_DIR / "manifest.csv", encoding="utf-8") as f:
        docs = list(csv.DictReader(f))
    for doc in docs:
        doc["text"] = (RAW_DIR / "docs" / f"{doc['doc_id']}.md").read_text(encoding="utf-8")
    return docs


def split_sections(text):
    """
    Break a document into (section_title, lines) pairs at each "##" heading.
    Text before the first "##" becomes the "Introduction" section.
    "###" sub-headings stay inside their section. Navigation sections are dropped.
    """
    sections = []
    title, lines = "Introduction", []
    skip_level = None  # while set, we're inside a section we want to drop

    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            level, heading = len(match.group(1)), match.group(2).strip()
            if skip_level is not None and level <= skip_level:
                skip_level = None                 # the skipped section has ended
            if heading.lower() in SKIP_HEADINGS:
                skip_level = level
                continue
            if skip_level is not None or level == 1:
                continue                          # level 1 is the document title
            if level == 2:
                if lines:
                    sections.append((title, lines))
                title, lines = heading, []
                continue
            lines.append(line)                    # keep ### sub-headings as text
            continue
        if skip_level is None and line.strip():
            lines.append(line)

    if lines:
        sections.append((title, lines))
    return sections


def count_words(lines):
    return sum(len(line.split()) for line in lines)


def make_chunk(doc, strategy, number, section, text):
    """One chunk plus everything we need later to cite and evaluate it."""
    return {
        "chunk_id": f"{doc['doc_id']}::{strategy}::{number:03d}",
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "url": doc["url"],
        "topic": doc["topic"],
        "section": section,
        "strategy": strategy,
        "word_count": len(text.split()),
        "text": text,
    }


# ---------------------------------------------------------------------------
# Strategy 1: fixed-size windows
# ---------------------------------------------------------------------------
def chunk_fixed(doc, sections):
    words, owner = [], []          # owner[i] = which section word i came from
    for title, lines in sections:
        section_words = " ".join(lines).split()
        words += section_words
        owner += [title] * len(section_words)

    chunks = []
    step = FIXED_WORDS - FIXED_OVERLAP
    for start in range(0, len(words), step):
        piece = words[start:start + FIXED_WORDS]
        chunks.append(make_chunk(doc, "fixed", len(chunks), owner[start], " ".join(piece)))
        if start + FIXED_WORDS >= len(words):
            break
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: heading-based sections
# ---------------------------------------------------------------------------
def pack_lines(lines, max_words):
    """Group lines into parts of at most max_words, without cutting a line in half."""
    parts, current, size = [], [], 0
    for line in lines:
        n = len(line.split())
        if n > max_words:                         # a single very long line
            if current:
                parts.append(current)
                current, size = [], 0
            words = line.split()
            for i in range(0, len(words), max_words):
                parts.append([" ".join(words[i:i + max_words])])
            continue
        if size + n > max_words and current:
            parts.append(current)
            current, size = [], 0
        current.append(line)
        size += n
    if current:
        parts.append(current)
    return parts


def chunk_by_heading(doc, sections):
    # Merge a tiny section into the one that follows it.
    merged = []
    for title, lines in sections:
        if merged and count_words(merged[-1][1]) < SECTION_MIN_WORDS:
            prev_title, prev_lines = merged[-1]
            merged[-1] = (prev_title, prev_lines + [f"### {title}"] + lines)
        else:
            merged.append((title, list(lines)))

    chunks = []
    for title, lines in merged:
        for part in pack_lines(lines, SECTION_MAX_WORDS):
            # A short header line tells the model (and the search) where this text came from.
            text = f"{doc['title']} — {title}\n" + "\n".join(part)
            chunks.append(make_chunk(doc, "heading", len(chunks), title, text))
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: sentence windows
# ---------------------------------------------------------------------------
def chunk_sentences(doc, sections):
    chunks = []
    for title, lines in sections:
        sentences = []
        for line in lines:
            if HEADING.match(line):
                continue
            sentences += [s.strip() for s in SENTENCE_END.split(line) if s.strip()]
        for start in range(0, len(sentences), SENTENCE_STRIDE):
            window = sentences[start:start + SENTENCES_PER_CHUNK]
            chunks.append(make_chunk(doc, "sentence", len(chunks), title, "\n".join(window)))
            if start + SENTENCES_PER_CHUNK >= len(sentences):
                break
    return chunks


# ---------------------------------------------------------------------------
# Run all three and report
# ---------------------------------------------------------------------------
STRATEGIES = {
    "fixed": chunk_fixed,
    "heading": chunk_by_heading,
    "sentence": chunk_sentences,
}
EXAMPLE_DOC = "credit-reports-score__order-credit-report"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = load_documents()
    print(f"Loaded {len(docs)} documents\n")
    print(f"{'strategy':<10} {'chunks':>7} {'avg words':>10} {'min':>5} {'max':>5}")

    examples = {}
    for name, chunker in STRATEGIES.items():
        all_chunks = []
        for doc in docs:
            all_chunks += chunker(doc, split_sections(doc["text"]))

        with open(OUT_DIR / f"chunks_{name}.jsonl", "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        sizes = [c["word_count"] for c in all_chunks]
        print(f"{name:<10} {len(all_chunks):>7} {statistics.mean(sizes):>10.0f} "
              f"{min(sizes):>5} {max(sizes):>5}")
        examples[name] = next((c for c in all_chunks if c["doc_id"] == EXAMPLE_DOC), None)

    # Show the first chunk of the same document under each strategy, for comparison.
    for name, chunk in examples.items():
        if chunk:
            print(f"\n===== {name}: {chunk['chunk_id']}  (section: {chunk['section']}) =====")
            print(chunk["text"])


if __name__ == "__main__":
    main()
