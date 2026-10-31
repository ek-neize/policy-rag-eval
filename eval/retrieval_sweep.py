"""
Step 8a: Compare retrieval for all 15 configurations. Free: no Claude calls.

For each configuration (chunking strategy x retrieval method x embedding model)
we search for every answerable test question and measure:
  hit@1, hit@5, hit@10, MRR@10      - is the evidence found, and how high up?
  hit@5 by group                    - single vs paraphrase vs multi
  evidence_recall@5 (multi)         - did we find BOTH documents?
  passage tokens @5                 - how long the top 5 passages are (drives Claude's cost)
  latency p50 / p95                 - warm search time in milliseconds

Output: results/retrieval_sweep.csv, and a table sorted by hit@5.
"""

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import mean, percentile, retrieval_scores  # noqa: E402
from retrieval import all_configs, config_name, get_retriever  # noqa: E402

TESTSET = Path("eval/testset.jsonl")
OUT_FILE = Path("results/retrieval_sweep.csv")
PRICE_INPUT_PER_MTOK = 0.10      # Claude Haiku 5.5, to translate passage length into dollars


def approx_tokens(text):
    return len(text) / 4          # rule of thumb for English: about 4 characters per token


def main():
    cases = [json.loads(line) for line in TESTSET.read_text(encoding="utf-8").splitlines()]
    cases = [c for c in cases if c["answerable"] and c["group"] != "injection"]
    print(f"{len(cases)} answerable questions x {len(all_configs())} configurations\n")

    rows = []
    for method, strategy, model_key in all_configs():
        name = config_name(method, strategy, model_key)
        retriever = get_retriever(method, strategy, model_key)
        retriever.search("warm-up question", k=1)            # load the model before timing

        per_case, times, tokens = [], [], []
        for case in cases:
            start = time.perf_counter()
            retrieved = retriever.search(case["question"], k=10)
            times.append(1000 * (time.perf_counter() - start))
            scores = retrieval_scores([e["quote"] for e in case["evidence"]], retrieved)
            per_case.append((case["group"], scores))
            tokens.append(sum(approx_tokens(c["text"]) for c in retrieved[:5]))

        def group_mean(metric, group=None):
            return mean(s[metric] for g, s in per_case if group is None or g == group)

        row = {
            "config": name, "method": method, "strategy": strategy, "model": model_key or "-",
            "hit@1": group_mean("hit@1"), "hit@5": group_mean("hit@5"), "hit@10": group_mean("hit@10"),
            "mrr@10": group_mean("mrr@10"),
            "hit@5_single": group_mean("hit@5", "single"),
            "hit@5_paraphrase": group_mean("hit@5", "paraphrase"),
            "evidence_recall@5_multi": group_mean("evidence_recall@5", "multi"),
            "passage_tokens@5": round(mean(tokens)),
            "passage_cost_per_1000_queries": round(mean(tokens) * 1000 * PRICE_INPUT_PER_MTOK / 1e6, 3),
            "latency_p50_ms": percentile(times, 50), "latency_p95_ms": percentile(times, 95),
            "missed_at_5": " ".join(c["id"] for c, (_, s) in zip(cases, per_case) if not s["hit@5"]),
        }
        rows.append(row)
        print(f"  done: {name:<28} hit@5 {100 * row['hit@5']:5.1f}%")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pct = lambda v: f"{100 * v:5.1f}%"  # noqa: E731
    print(f"\n{'configuration':<28} {'hit@1':>6} {'hit@5':>6} {'hit@10':>7} {'MRR':>5} "
          f"{'single':>7} {'para':>6} {'multi':>6} {'tokens':>7} {'p50ms':>6} {'p95ms':>6}")
    for r in sorted(rows, key=lambda r: (-r["hit@5"], -r["mrr@10"])):
        print(f"{r['config']:<28} {pct(r['hit@1'])} {pct(r['hit@5'])} {pct(r['hit@10']):>7} "
              f"{r['mrr@10']:.2f} {pct(r['hit@5_single']):>7} {pct(r['hit@5_paraphrase'])} "
              f"{pct(r['evidence_recall@5_multi'])} {r['passage_tokens@5']:>7} "
              f"{r['latency_p50_ms']:>6} {r['latency_p95_ms']:>6}")
    print(f"\nSaved {OUT_FILE}  (column 'missed_at_5' lists the questions each config missed)")


if __name__ == "__main__":
    main()
