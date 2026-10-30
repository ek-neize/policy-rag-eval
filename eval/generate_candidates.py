"""
Step 6a: Draft candidate test questions for a human to review.

  single      - Claude writes one question from one passage (one per document).
  paraphrase  - the same question reworded in everyday language that avoids
                the document's key terms (tests meaning-based search).
  multi       - Claude writes a question that needs TWO passages from
                different documents.
  unanswerable- written by hand below; the corpus should NOT answer them.
                The top search results are shown so you can confirm that.

Every answerable question comes with an exact "evidence quote" from its
source passage. The script checks the quote really exists and retries once.

Output: eval/candidates.csv  -> you review it, then run eval/build_testset.py
Cost:   about $0.02-0.04 with Claude Haiku.
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anthropic
import numpy as np
from dotenv import load_dotenv

from assistant import LLM_MODEL, PRICE_PER_MTOK
from embeddings import index_path, load_chunks
from evidence import contains_quote
from retrieval import get_retriever

load_dotenv()

OUT_FILE = Path("eval/candidates.csv")
SEED = 42
N_PARAPHRASE = 10
N_MULTI = 16
MULTI_SIMILARITY = (0.70, 0.90)   # related enough to combine, not near-duplicates

# ---------------------------------------------------------------------------
# Unanswerable questions, written by hand in four types.
# The near-miss and false-premise ones are the hardest: they sound like the
# documents should answer them. VERIFY each one during review.
# ---------------------------------------------------------------------------
UNANSWERABLE = [
    # Bank-specific or real-time data (FCAC never publishes these)
    ("bank-specific", "What is BMO's current prime rate?"),
    ("bank-specific", "What is the annual fee for the TD Aeroplan Visa Infinite card?"),
    ("bank-specific", "What is today's 5-year fixed mortgage rate at RBC?"),
    ("bank-specific", "How much does Scotiabank charge non-customers to use its ATMs?"),
    ("bank-specific", "What is CIBC's daily e-Transfer limit?"),
    # Outside the topic of the documents
    ("out-of-scope", "How do I file my income tax return online?"),
    ("out-of-scope", "How much will I receive from the Canada Pension Plan when I retire?"),
    ("out-of-scope", "What is the minimum wage in Ontario?"),
    ("out-of-scope", "How do I buy shares of an exchange-traded fund?"),
    ("out-of-scope", "How do I apply for Employment Insurance benefits?"),
    # Near-miss: on-topic, but the specific detail should be absent
    ("near-miss", "How many business days does a bank have to process a mortgage discharge in Alberta?"),
    ("near-miss", "What credit score do I need to qualify for a mortgage at a big bank?"),
    ("near-miss", "What is the exact interest rate charged on a Buy Now Pay Later plan from Affirm?"),
    ("near-miss", "How many credit inquiries in one month will lower my credit score by 50 points?"),
    ("near-miss", "What is the maximum amount a joint credit card holder can be sued for in Quebec?"),
    # False premise: the question assumes something untrue
    ("false-premise", "Since FCAC banned overdraft protection, what replaced it?"),
    ("false-premise", "Why are all credit card interest rates in Canada capped at 10%?"),
    ("false-premise", "Which FCAC form do I use to have a late payment removed from my credit report?"),
]

SINGLE_PROMPT = """You are helping build an evaluation test set for a question-answering \
assistant used by bank employees. Read the passage and write ONE test question.

Requirements:
- The question must be answerable from this passage alone, and be specific (not "what does \
this passage say about...").
- Write it the way an employee or customer would ask. Do not mention "the passage", \
"the document" or "this page".
- reference_answer: 1 to 3 sentences, using only facts from the passage.
- evidence_quote: copy EXACTLY, character for character, ONE sentence or ONE list item from \
the passage that contains the key fact. At most 30 words. No "..." and no changes.
- casual_question: the same question as a confused customer might say it in everyday words, \
avoiding the passage's key terms (for example "money I owe" instead of "debt").
- Set usable to false if the passage has no specific, testable fact (only links or headings).

<passage source="{title}" section="{section}">
{text}
</passage>"""

MULTI_PROMPT = """You are helping build an evaluation test set. Below are two passages from \
DIFFERENT documents. Write ONE question that genuinely needs information from BOTH passages \
to answer fully (for example, combining two related rules or comparing two options).

Requirements:
- Neither passage alone should fully answer it. Do not mention "the passages".
- reference_answer: 2 to 4 sentences, using only facts from the passages.
- evidence_quote_a / evidence_quote_b: copy EXACTLY ONE sentence or list item (at most 30 \
words, no "...") from passage A and from passage B respectively, each containing a key fact.
- Set usable to false if there is no natural question that needs both.

<passage id="A" source="{title_a}" section="{section_a}">
{text_a}
</passage>

<passage id="B" source="{title_b}" section="{section_b}">
{text_b}
</passage>"""

SINGLE_TOOL = {
    "name": "write_test_question",
    "description": "Submit the test question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "usable": {"type": "boolean"},
            "question": {"type": "string"},
            "reference_answer": {"type": "string"},
            "evidence_quote": {"type": "string"},
            "casual_question": {"type": "string"},
        },
        "required": ["usable", "question", "reference_answer", "evidence_quote", "casual_question"],
    },
}

MULTI_TOOL = {
    "name": "write_test_question",
    "description": "Submit the test question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "usable": {"type": "boolean"},
            "question": {"type": "string"},
            "reference_answer": {"type": "string"},
            "evidence_quote_a": {"type": "string"},
            "evidence_quote_b": {"type": "string"},
        },
        "required": ["usable", "question", "reference_answer", "evidence_quote_a", "evidence_quote_b"],
    },
}

client = anthropic.Anthropic()
usage_totals = {"input": 0, "output": 0}


def ask_claude(prompt, tool):
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=600,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": prompt}],
    )
    usage_totals["input"] += response.usage.input_tokens
    usage_totals["output"] += response.usage.output_tokens
    reply = next((b.input for b in response.content if b.type == "tool_use"), {})
    # Occasionally the model leaves a field out. Fill gaps with empty text so the
    # script never crashes; the quote check below will flag the row for review.
    for field in tool["input_schema"]["required"]:
        reply.setdefault(field, False if field == "usable" else "")
    return reply


def ask_with_quote_check(prompt, tool, checks):
    """
    Ask Claude; if a quote isn't found word-for-word in its passage, retry once
    with feedback. `checks` maps a reply field to the passage text it must come from.
    """
    reply = ask_claude(prompt, tool)
    bad = [f for f, text in checks.items()
           if reply["usable"] and not (reply[f].strip() and contains_quote(text, reply[f]))]
    if bad:
        feedback = ("\n\nYour previous evidence quote was not copied exactly from the passage. "
                    "Copy a sentence character for character this time.")
        reply = ask_claude(prompt + feedback, tool)
    reply["_quote_ok"] = all(reply[f].strip() and contains_quote(text, reply[f]) for f, text in checks.items())
    if reply["usable"] and not reply["question"].strip():
        reply["usable"] = False           # no question at all: nothing to review
    return reply


def short(text, words=120):
    parts = text.split()
    return " ".join(parts[:words]) + (" ..." if len(parts) > words else "")


def row(id_, group, question, answer="", ev=(), quote_ok="", context="", notes=""):
    r = {"id": id_, "group": group, "question": question, "reference_answer": answer,
         "doc_1": "", "section_1": "", "quote_1": "", "doc_2": "", "section_2": "", "quote_2": "",
         "quote_check": quote_ok, "context": context, "review": "", "notes": notes}
    for n, (doc, section, quote) in enumerate(ev, 1):
        r[f"doc_{n}"], r[f"section_{n}"], r[f"quote_{n}"] = doc, section, quote
    return r


# ---------------------------------------------------------------------------
def make_single_and_paraphrase(chunks, rng):
    by_doc = defaultdict(list)
    for c in chunks:
        if c["word_count"] >= 60:
            by_doc[c["doc_id"]].append(c)

    singles, paraphrase_pool = [], []
    for n, doc_id in enumerate(sorted(by_doc), 1):
        options = [c for c in by_doc[doc_id] if c["section"] != "Introduction"] or by_doc[doc_id]
        chunk = rng.choice(options)
        print(f"  single {n}/{len(by_doc)}: {chunk['title']} > {chunk['section']}")
        reply = ask_with_quote_check(SINGLE_PROMPT.format(**chunk), SINGLE_TOOL,
                                     {"evidence_quote": chunk["text"]})
        if not reply.get("usable"):
            print("      (Claude marked this passage unusable - skipped)")
            continue
        ev = [(chunk["doc_id"], chunk["section"], reply["evidence_quote"])]
        quote_ok = "ok" if reply["_quote_ok"] else "FAIL - fix quote"
        singles.append(row(f"S{len(singles) + 1:02d}", "single", reply["question"],
                           reply["reference_answer"], ev, quote_ok, short(chunk["text"])))
        paraphrase_pool.append((reply, ev, quote_ok, chunk))

    paraphrases = []
    for reply, ev, quote_ok, chunk in rng.sample(paraphrase_pool, min(N_PARAPHRASE, len(paraphrase_pool))):
        paraphrases.append(row(f"P{len(paraphrases) + 1:02d}", "paraphrase", reply["casual_question"],
                               reply["reference_answer"], ev, quote_ok, short(chunk["text"]),
                               notes=f"original: {reply['question']}"))
    return singles, paraphrases


def make_multi(chunks):
    vectors = np.load(index_path("heading", "bge-small"))
    similarity = vectors @ vectors.T
    pairs = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            a, b = chunks[i], chunks[j]
            if a["doc_id"] != b["doc_id"] and min(a["word_count"], b["word_count"]) >= 60:
                if MULTI_SIMILARITY[0] <= similarity[i, j] <= MULTI_SIMILARITY[1]:
                    pairs.append((similarity[i, j], i, j))
    pairs.sort(reverse=True)

    chosen, used_docs, used_pairs = [], defaultdict(int), set()
    for _, i, j in pairs:
        a, b = chunks[i], chunks[j]
        doc_pair = frozenset((a["doc_id"], b["doc_id"]))
        if doc_pair in used_pairs:
            continue                      # each pair of documents is used once
        if used_docs[a["doc_id"]] < 2 and used_docs[b["doc_id"]] < 2:
            chosen.append((a, b))
            used_pairs.add(doc_pair)
            used_docs[a["doc_id"]] += 1
            used_docs[b["doc_id"]] += 1
        if len(chosen) == N_MULTI:
            break

    rows = []
    for n, (a, b) in enumerate(chosen, 1):
        print(f"  multi {n}/{len(chosen)}: {a['title']}  +  {b['title']}")
        prompt = MULTI_PROMPT.format(title_a=a["title"], section_a=a["section"], text_a=a["text"],
                                     title_b=b["title"], section_b=b["section"], text_b=b["text"])
        reply = ask_with_quote_check(prompt, MULTI_TOOL,
                                     {"evidence_quote_a": a["text"], "evidence_quote_b": b["text"]})
        if not reply.get("usable"):
            print("      (no natural question for this pair - skipped)")
            continue
        ev = [(a["doc_id"], a["section"], reply["evidence_quote_a"]),
              (b["doc_id"], b["section"], reply["evidence_quote_b"])]
        context = f"A: {short(a['text'], 80)}\n\nB: {short(b['text'], 80)}"
        rows.append(row(f"M{len(rows) + 1:02d}", "multi", reply["question"], reply["reference_answer"],
                        ev, "ok" if reply["_quote_ok"] else "FAIL - fix quote", context))
    return rows


def make_unanswerable():
    retriever = get_retriever("hybrid", "heading", "bge-small")
    rows = []
    for n, (kind, question) in enumerate(UNANSWERABLE, 1):
        top = retriever.search(question, k=3)
        context = "\n\n".join(f"[{c['title']} > {c['section']}] {short(c['text'], 50)}" for c in top)
        rows.append(row(f"U{n:02d}", "unanswerable", question, context=context,
                        notes=f"type: {kind}. Check the top search results below do NOT answer it."))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing candidates file (this erases your review!)")
    args = parser.parse_args()
    if OUT_FILE.exists() and not args.overwrite:
        sys.exit(f"{OUT_FILE} already exists. It may contain your review work, so I won't "
                 "overwrite it. Use --overwrite if you really want to start again.")

    rng = random.Random(SEED)          # fixed seed: same passages chosen every run
    chunks = load_chunks("heading")

    print("Writing single-passage and paraphrased questions...")
    singles, paraphrases = make_single_and_paraphrase(chunks, rng)
    print("\nWriting multi-document questions...")
    multi = make_multi(chunks)
    print("\nAdding unanswerable questions with their top search results...")
    unanswerable = make_unanswerable()

    rows = singles + paraphrases + multi + unanswerable
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cost = (usage_totals["input"] * PRICE_PER_MTOK["input"]
            + usage_totals["output"] * PRICE_PER_MTOK["output"]) / 1e6
    failed = sum(r["quote_check"].startswith("FAIL") for r in rows)
    print(f"\nWrote {len(rows)} candidates to {OUT_FILE}:")
    print(f"  single {len(singles)}, paraphrase {len(paraphrases)}, multi {len(multi)}, "
          f"unanswerable {len(unanswerable)}")
    print(f"  quotes needing a fix: {failed}")
    print(f"  Claude usage: {usage_totals['input']:,} in / {usage_totals['output']:,} out tokens = ${cost:.4f}")


if __name__ == "__main__":
    main()
