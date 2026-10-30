"""
Shared helpers for embeddings and vector search (used by Steps 3-9).

An embedding model turns a piece of text into a list of numbers (a "vector")
that represents its meaning. Texts with similar meanings get similar vectors,
even when they use different words.

To search, we embed the question the same way and find the chunks whose
vectors point in the most similar direction ("cosine similarity").
"""

import json
from pathlib import Path

import numpy as np

PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/index")

# ---------------------------------------------------------------------------
# The two embedding models we compare. Both are free and run on your computer.
# ---------------------------------------------------------------------------
MODELS = {
    # A popular, fast, general-purpose model. Our baseline.
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
    },
    # A model trained specifically for search. It expects questions to start
    # with an instruction, but the passages themselves get no prefix.
    "bge-small": {
        "name": "BAAI/bge-small-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
}
STRATEGIES = ["fixed", "heading", "sentence"]

_loaded = {}  # keeps models in memory so each is loaded only once


def load_model(model_key):
    """Load (and on first use, download) an embedding model."""
    if model_key not in _loaded:
        from sentence_transformers import SentenceTransformer  # slow import, so done lazily
        _loaded[model_key] = SentenceTransformer(MODELS[model_key]["name"], device="cpu")
    return _loaded[model_key]


def embed_passages(model_key, texts, show_progress=False):
    """Turn chunk texts into vectors of length 1 (so a dot product = cosine similarity)."""
    model = load_model(model_key)
    vectors = model.encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=show_progress
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(model_key, question):
    """Turn one question into a vector, adding the model's query prefix if it has one."""
    model = load_model(model_key)
    text = MODELS[model_key]["query_prefix"] + question
    return np.asarray(model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)


def count_truncated(model_key, texts):
    """How many texts are longer than the model can read? (The extra is silently cut off.)"""
    model = load_model(model_key)
    limit = model.max_seq_length
    lengths = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
    return sum(n > limit for n in lengths), limit


# ---------------------------------------------------------------------------
# Loading chunks and indexes
# ---------------------------------------------------------------------------
def load_chunks(strategy):
    with open(PROCESSED_DIR / f"chunks_{strategy}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def index_path(strategy, model_key):
    return INDEX_DIR / f"{strategy}__{model_key}.npy"


class VectorIndex:
    """
    A tiny vector "database": a matrix with one row per chunk.

    With a few thousand chunks, comparing the question to every chunk takes
    about a millisecond, so we don't need a separate database program.
    Large systems (millions of chunks) use tools like FAISS or Chroma that
    trade a little accuracy for speed.
    """

    def __init__(self, strategy, model_key):
        self.strategy = strategy
        self.model_key = model_key
        self.chunks = load_chunks(strategy)
        self.vectors = np.load(index_path(strategy, model_key))
        if len(self.vectors) != len(self.chunks):
            raise ValueError(
                f"Index {index_path(strategy, model_key)} is out of date: "
                "re-run src/build_index.py after re-chunking."
            )

    def search(self, question, k=5):
        """Return the k most similar chunks, each with its similarity score."""
        query_vector = embed_query(self.model_key, question)
        scores = self.vectors @ query_vector          # one similarity score per chunk
        best = np.argsort(-scores)[:k]                # positions of the k highest scores
        return [{**self.chunks[i], "score": float(scores[i])} for i in best]
