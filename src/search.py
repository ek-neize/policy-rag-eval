"""
Step 3: Try a search from the terminal.

Examples:
  python src/search.py "How can I get my credit report by mail?"
  python src/search.py "What is the fee for an NSF cheque?" --strategy fixed --model minilm
  python src/search.py "Can my bank refuse to open an account?" --compare
"""

import argparse
import textwrap

from embeddings import MODELS, STRATEGIES, VectorIndex


def show(results, preview_words=45):
    for rank, r in enumerate(results, 1):
        preview = " ".join(r["text"].split()[:preview_words])
        print(f"\n  {rank}. score {r['score']:.3f}  |  {r['title']}  >  {r['section']}")
        print(f"     {r['chunk_id']}")
        print(textwrap.indent(textwrap.fill(preview + " ...", 90), "     "))


def main():
    parser = argparse.ArgumentParser(description="Search the FCAC chunks by meaning.")
    parser.add_argument("question")
    parser.add_argument("--strategy", choices=STRATEGIES, default="heading")
    parser.add_argument("--model", choices=list(MODELS), default="bge-small")
    parser.add_argument("-k", type=int, default=3, help="how many results to show")
    parser.add_argument("--compare", action="store_true",
                        help="show the top result from all 6 strategy/model combinations")
    args = parser.parse_args()

    if args.compare:
        print(f"Question: {args.question}\nTop result for each combination:")
        for model_key in MODELS:
            for strategy in STRATEGIES:
                top = VectorIndex(strategy, model_key).search(args.question, k=1)[0]
                print(f"\n  [{strategy:<8} + {model_key:<9}] score {top['score']:.3f}"
                      f"  |  {top['title']}  >  {top['section']}")
        return

    index = VectorIndex(args.strategy, args.model)
    print(f"Question: {args.question}\nStrategy: {args.strategy}   Model: {args.model}")
    show(index.search(args.question, k=args.k))


if __name__ == "__main__":
    main()
