"""
Step 4: Three retrieval methods behind one simple interface.

  bm25    - keyword search: scores chunks by shared words, giving more weight
            to rare words ("prepayment") than common ones ("credit").
  dense   - semantic search with embeddings (built in Step 3).
  hybrid  - runs both and merges their rankings with Reciprocal Rank Fusion.

Every retriever has the same .search(question, k) method, so the evaluation
code in later steps can test any combination without special cases.
"""

import math
import re
from collections import Counter

from embeddings import MODELS, STRATEGIES, VectorIndex, load_chunks

METHODS = ["bm25", "dense", "hybrid"]

# Very common words that carry no meaning for search.
STOPWORDS = set("""
a an and are as at be been but by can could do does for from had has have how i if in
into is it its may me might my no not of on or our should so than that the their them
then there these they this to was we were what when where which who why will with would
you your yours
""".split())


def tokenize(text):
    """Lowercase, split into words, drop stopwords, and strip a plural 's'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens = []
    for w in words:
        if w in STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]                      # "fees" -> "fee", "loans" -> "loan"
        tokens.append(w)
    return tokens


# ---------------------------------------------------------------------------
# Method 1: BM25 keyword search
# ---------------------------------------------------------------------------
class BM25Index:
    """
    BM25 is the classic search-engine scoring formula. For each question word:
      - more occurrences in the chunk  -> higher score (with diminishing returns, k1)
      - rarer across all chunks        -> higher weight (IDF)
      - longer chunk than average      -> score scaled down a little (b)
    """

    def __init__(self, strategy, k1=1.5, b=0.75):
        self.strategy = strategy
        self.k1, self.b = k1, b
        self.chunks = load_chunks(strategy)
        self.doc_tokens = [Counter(tokenize(c["text"])) for c in self.chunks]
        self.doc_lengths = [sum(t.values()) for t in self.doc_tokens]
        self.avg_length = sum(self.doc_lengths) / len(self.doc_lengths)

        # Document frequency: in how many chunks does each word appear?
        df = Counter()
        for tokens in self.doc_tokens:
            df.update(tokens.keys())
        n = len(self.chunks)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, question):
        query = set(tokenize(question))
        results = []
        for tokens, length in zip(self.doc_tokens, self.doc_lengths):
            score = 0.0
            for word in query:
                tf = tokens.get(word, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
                    score += self.idf[word] * tf * (self.k1 + 1) / (tf + norm)
            results.append(score)
        return results

    def search(self, question, k=5):
        scores = self.scores(question)
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{**self.chunks[i], "score": scores[i]} for i in best if scores[i] > 0]


# ---------------------------------------------------------------------------
# Method 2: dense (semantic) search is VectorIndex from embeddings.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Method 3: hybrid with Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------
class HybridIndex:
    """
    BM25 scores and cosine similarities are on different scales, so we can't
    just add them. RRF ignores the scores and uses only the RANK positions:

        score(chunk) = 1/(60 + rank in BM25) + 1/(60 + rank in dense)

    A chunk ranked highly by both methods wins. The constant 60 is the value
    from the original RRF paper; it stops the #1 spot from dominating.
    """

    def __init__(self, strategy, model_key, rrf_k=60, candidates=50):
        self.bm25 = BM25Index(strategy)
        self.dense = VectorIndex(strategy, model_key)
        self.rrf_k, self.candidates = rrf_k, candidates

    def search(self, question, k=5):
        fused, by_id = Counter(), {}
        for retriever in (self.bm25, self.dense):
            for rank, chunk in enumerate(retriever.search(question, k=self.candidates), 1):
                fused[chunk["chunk_id"]] += 1 / (self.rrf_k + rank)
                by_id.setdefault(chunk["chunk_id"], chunk)
        return [{**by_id[cid], "score": score} for cid, score in fused.most_common(k)]


# ---------------------------------------------------------------------------
# One function to build any combination
# ---------------------------------------------------------------------------
def get_retriever(method, strategy, model_key=None):
    if method == "bm25":
        return BM25Index(strategy)
    if method == "dense":
        return VectorIndex(strategy, model_key)
    if method == "hybrid":
        return HybridIndex(strategy, model_key)
    raise ValueError(f"Unknown method: {method}")


def all_configs():
    """Every combination we'll evaluate. BM25 doesn't use an embedding model."""
    configs = []
    for strategy in STRATEGIES:
        configs.append(("bm25", strategy, None))
        for model_key in MODELS:
            configs.append(("dense", strategy, model_key))
            configs.append(("hybrid", strategy, model_key))
    return configs


def config_name(method, strategy, model_key):
    return f"{method}|{strategy}|{model_key or '-'}"
