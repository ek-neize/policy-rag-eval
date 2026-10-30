"""
Compare evaluation runs side by side (uses each run's summary.json).

Usage:
  python eval/compare_runs.py hybrid-heading-bge-small-prompt-v1 hybrid-heading-bge-small
  python eval/compare_runs.py                       # every run in results/runs
"""

import json
import sys
from pathlib import Path

RUNS_DIR = Path("results/runs")

ROWS = [  # (label, section, key, kind, higher_is_better)
    ("hit@5", "retrieval", "hit@5", "pct", True),
    ("MRR@10", "retrieval", "mrr@10", "num", True),
    ("correctness", "answer_quality", "correctness", "pct", True),
    ("groundedness", "answer_quality", "groundedness", "pct", True),
    ("fully grounded answers", "answer_quality", "fully_grounded_rate", "pct", True),
    ("citation support", "answer_quality", "citation_support", "pct", True),
    ("citation hit", "citations", "citation_hit", "pct", True),
    ("correct refusals", "abstention", "correct_refusal_rate", "pct", True),
    ("false refusals", "abstention", "false_refusal_rate", "pct", False),
    ("attack success", "injection", "attack_success_rate", "pct", False),
    ("injection detected", "injection", "detection_rate", "pct", True),
    ("p50 total ms", "latency_ms", "total_p50", "ms", False),
    ("p95 total ms", "latency_ms", "total_p95", "ms", False),
    ("cost per query $", "cost_usd", "per_query", "usd", False),
    ("avg input tokens", "cost_usd", "avg_input_tokens", "int", False),
]


def fmt(value, kind):
    if value is None:
        return "n/a"
    return {"pct": f"{100 * value:.1f}%", "num": f"{value:.3f}", "ms": f"{value:,.0f}",
            "usd": f"{value:.5f}", "int": f"{value:,.0f}"}[kind]


def main():
    names = sys.argv[1:] or sorted(p.name for p in RUNS_DIR.iterdir() if (p / "summary.json").exists())
    summaries = {n: json.loads((RUNS_DIR / n / "summary.json").read_text()) for n in names}

    width = max(14, *(len(n) for n in names)) + 2
    print(f"{'metric':<24}" + "".join(f"{n:>{width}}" for n in names))
    print(f"{'prompt version':<24}" + "".join(f"{s.get('prompt_version', 'v1'):>{width}}" for s in summaries.values()))
    for label, section, key, kind, higher in ROWS:
        values = [s[section].get(key) for s in summaries.values()]
        known = [v for v in values if v is not None]
        best = (max(known) if higher else min(known)) if len(known) > 1 else None
        cells = [fmt(v, kind) + ("*" if v is not None and v == best and len(set(known)) > 1 else " ")
                 for v in values]
        print(f"{label:<24}" + "".join(f"{c:>{width}}" for c in cells))
    print("\n* = best value in that row")


if __name__ == "__main__":
    main()
