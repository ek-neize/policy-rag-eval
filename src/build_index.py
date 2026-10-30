"""
Step 3: Embed every chunk with every model and save the results to disk.

3 chunking strategies x 2 embedding models = 6 indexes.
Embedding is the slow part, so we do it once here and reuse the saved files.

Output: data/index/<strategy>__<model>.npy  (one row of numbers per chunk)
        data/index/build_report.csv         (time taken, sizes, truncation)
"""

import csv
import time

from embeddings import INDEX_DIR, MODELS, STRATEGIES, embed_passages, count_truncated, index_path, load_chunks, load_model

import numpy as np


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for model_key in MODELS:
        print(f"\nLoading {MODELS[model_key]['name']} (downloads on first run)...")
        load_model(model_key)

        for strategy in STRATEGIES:
            chunks = load_chunks(strategy)
            texts = [c["text"] for c in chunks]
            truncated, limit = count_truncated(model_key, texts)

            print(f"  Embedding {len(texts)} '{strategy}' chunks...")
            start = time.perf_counter()
            vectors = embed_passages(model_key, texts, show_progress=True)
            seconds = time.perf_counter() - start

            np.save(index_path(strategy, model_key), vectors)
            rows.append({
                "strategy": strategy,
                "model": model_key,
                "chunks": len(texts),
                "dimensions": vectors.shape[1],
                "token_limit": limit,
                "chunks_truncated": truncated,
                "seconds": round(seconds, 1),
                "ms_per_chunk": round(1000 * seconds / len(texts), 1),
                "size_mb": round(vectors.nbytes / 1e6, 2),
            })

    with open(INDEX_DIR / "build_report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'strategy':<9} {'model':<10} {'chunks':>6} {'dims':>5} {'limit':>6} "
          f"{'truncated':>9} {'seconds':>8} {'ms/chunk':>8}")
    for r in rows:
        print(f"{r['strategy']:<9} {r['model']:<10} {r['chunks']:>6} {r['dimensions']:>5} "
              f"{r['token_limit']:>6} {r['chunks_truncated']:>9} {r['seconds']:>8} {r['ms_per_chunk']:>8}")


if __name__ == "__main__":
    main()
