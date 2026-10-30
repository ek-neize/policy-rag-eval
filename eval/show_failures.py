"""
Error analysis: print the cases that went wrong in an evaluation run, with the
question, the answer, the reference, and the judge's explanation.

Usage:
  python eval/show_failures.py                         # default config
  python eval/show_failures.py --run dense-fixed-minilm
  python eval/show_failures.py --ids I08 I09 P03        # specific cases
"""

import argparse
import json
import textwrap
from pathlib import Path


def is_failure(r):
    if r.get("group") == "injection":
        return r.get("attack_success") or (r["scores"].get("correctness") or 0) < 1
    if r["expected_answerable"] != r["answered"]:
        return True
    if r["expected_answerable"]:
        s = r["scores"]
        return (s.get("correctness") or 0) < 1 or (s.get("groundedness") is not None and s["groundedness"] < 1)
    return False


def wrap(label, text):
    return textwrap.fill(f"{label}{text}", 100, subsequent_indent=" " * len(label))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="hybrid-heading-bge-small")
    parser.add_argument("--ids", nargs="*")
    args = parser.parse_args()

    path = Path("results/runs") / args.run / "cases.jsonl"
    results = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    chosen = [r for r in results if r["id"] in args.ids] if args.ids else [r for r in results if is_failure(r)]

    for r in chosen:
        s, ret = r["scores"], r.get("retrieval", {})
        print("=" * 100)
        print(f"{r['id']}  [{r['group']}]  answered={r['answered']}  "
              f"evidence rank={ret.get('first_relevant_rank', '-')}  "
              f"correctness={s.get('correctness', '-')}  groundedness={s.get('groundedness', '-')}"
              + (f"  attack={r['attack_name']} {'WORKED' if r['attack_success'] else 'resisted'}"
                 if r.get("attack_name") else ""))
        print(wrap("QUESTION:  ", r["question"]))
        print(wrap("ANSWER:    ", r["answer"]))
        if r["reference_answer"]:
            print(wrap("REFERENCE: ", r["reference_answer"]))
        j = r.get("judgement")
        if j:
            print(wrap("JUDGE:     ", f"{j.get('correctness')} - {j.get('explanation', '')}"))
            for c in j.get("claims", []):
                if not c.get("supported_by_any_passage"):
                    print(wrap("  UNSUPPORTED CLAIM: ", c.get("claim", "")))
    print("=" * 100)
    print(f"{len(chosen)} case(s) shown out of {len(results)}")


if __name__ == "__main__":
    main()
