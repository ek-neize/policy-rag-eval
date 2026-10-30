"""
Try a search from the terminal (Steps 3-4).

Examples:
  python src/search.py "How can I get my credit report by mail?"
  python src/search.py "NSF fee" --method bm25
  python src/search.py "What is a prepayment penalty?" --method hybrid --strategy fixed --model minilm
  python src/search.py "Can my bank refuse to open an account?" --compare
"""

import argparse
import textwrap
import time

from embeddings import MODELS, STRATEGIES
from retrieval import METHODS, all_configs, get_retriever


def show(results, preview_words=45):
    for rank, r in enumerate(results, 1):
        preview = " ".join(r["text"].split()[:preview_words])
        print(f"\n  {rank}. score {r['score']:.3f}  |  {r['title']}  >  {r['section']}")
        print(f"     {r['chunk_id']}")
        print(textwrap.indent(textwrap.fill(preview + " ...", 90), "     "))


def main():
    parser = argparse.ArgumentParser(description="Search the FCAC chunks.")
    parser.add_argument("question")
    parser.add_argument("--method", choices=METHODS, default="hybrid")
    parser.add_argument("--strategy", choices=STRATEGIES, default="heading")
    parser.add_argument("--model", choices=list(MODELS), default="bge-small")
    parser.add_argument("-k", type=int, default=3, help="how many results to show")
    parser.add_argument("--compare", action="store_true",
                        help="show the top result from every method/strategy/model combination")
    args = parser.parse_args()

    if args.compare:
        print(f"Question: {args.question}\nTop result for each combination:\n")
        for method, strategy, model_key in all_configs():
            retriever = get_retriever(method, strategy, model_key)
            start = time.perf_counter()
            results = retriever.search(args.question, k=1)
            ms = 1000 * (time.perf_counter() - start)
            label = f"{method:<6} {strategy:<8} {model_key or '-':<9}"
            if results:
                top = results[0]
                print(f"  {label} {ms:6.1f} ms  |  {top['title']}  >  {top['section']}")
            else:
                print(f"  {label} {ms:6.1f} ms  |  (no matching words)")
        return

    retriever = get_retriever(args.method, args.strategy, args.model)
    start = time.perf_counter()
    results = retriever.search(args.question, k=args.k)
    ms = 1000 * (time.perf_counter() - start)
    model_label = args.model if args.method != "bm25" else "-"
    print(f"Question: {args.question}")
    print(f"Method: {args.method}   Strategy: {args.strategy}   Model: {model_label}   ({ms:.1f} ms)")
    if not results:
        print("\n  No results: none of the question's words appear in any chunk.")
    show(results)


if __name__ == "__main__":
    main()
