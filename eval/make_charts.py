"""
Step 8c: Turn the experiment results into charts for the README.

Reads  results/retrieval_sweep.csv and results/runs/*/summary.json
Writes results/charts/*.png

  1_retrieval_by_config.png   hit@5 for all 15 retrieval configurations
  2_paraphrase_gap.png        normal vs everyday-wording questions, per retrieval setup
  3_quality_vs_cost.png       correctness vs cost per query, with the A/A noise band
  4_metric_noise.png          how much each metric moved between two identical runs
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                 # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

RESULTS = Path("results")
CHARTS = RESULTS / "charts"

# Colours: a validated colour-blind-safe palette, used in a fixed order.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

SETUPS = [  # (method, model, label) - one colour each, same order in every chart
    ("bm25", "-", "BM25 (keywords)"),
    ("dense", "minilm", "Dense · MiniLM"),
    ("dense", "bge-small", "Dense · BGE"),
    ("hybrid", "minilm", "Hybrid · MiniLM"),
    ("hybrid", "bge-small", "Hybrid · BGE"),
]
STRATEGIES = [("fixed", "Fixed-size"), ("heading", "Heading-based"), ("sentence", "Sentence-window")]

RUNS = [  # (run folder, label) for the end-to-end comparison
    ("hybrid-heading-bge-small-run1", "Default (run 1)"),
    ("hybrid-heading-bge-small-run2", "Default (run 2)"),
    ("hybrid-heading-minilm", "Hybrid · MiniLM"),
    ("bm25-heading-none", "BM25"),
    ("dense-sentence-bge-small", "Sentence chunks"),
    ("hybrid-heading-bge-small-k3", "Default, k=3"),
    ("hybrid-heading-bge-small-k8", "Default, k=8"),
]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
})


def style(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def title(fig, main, sub):
    fig.text(0.01, 0.97, main, fontsize=13, fontweight="bold", va="top")
    fig.text(0.01, 0.905, sub, fontsize=9.5, color=INK_2, va="top")


def save(fig, name):
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / name, dpi=200)
    plt.close(fig)
    print(f"  saved {CHARTS / name}")


def load_sweep():
    with open(RESULTS / "retrieval_sweep.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["method"], r["strategy"], r["model"]): r for r in rows}


# ---------------------------------------------------------------------------
def chart_retrieval(sweep):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.17)
    width, gap = 0.16, 0.012
    for s_i, (strategy, _) in enumerate(STRATEGIES):
        for m_i, (method, model, label) in enumerate(SETUPS):
            value = 100 * float(sweep[(method, strategy, model)]["hit@5"])
            x = s_i + (m_i - 2) * width
            ax.bar(x, value, width - gap, color=SERIES[m_i], edgecolor=SURFACE, linewidth=1.5,
                   label=label if s_i == 0 else None)
            ax.text(x, value + 1, f"{value:.0f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(STRATEGIES)), [s[1] for s in STRATEGIES], fontsize=10.5, color=INK)
    ax.set_ylim(0, 105)
    ax.set_ylabel("hit@5 (% of questions with evidence in the top 5)")
    style(ax)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=9)
    title(fig, "Chunking matters more than the retrieval method",
          "hit@5 on 66 answerable questions. Sentence-window chunks trail by 10–20 points; "
          "within the other two, setups differ by 1–6 questions.")
    save(fig, "1_retrieval_by_config.png")


def chart_paraphrase(sweep):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.72, bottom=0.11)
    labels = [label for _, _, label in SETUPS]
    height = 0.36
    for i, (method, model, _) in enumerate(SETUPS):
        row = sweep[(method, "heading", model)]
        for j, (key, colour, name) in enumerate([("hit@5_single", SERIES[0], "Normal wording (47 questions)"),
                                                 ("hit@5_paraphrase", SERIES[1], "Everyday wording (10 questions)")]):
            value = 100 * float(row[key])
            y = i + (j - 0.5) * height
            ax.barh(y, value, height - 0.03, color=colour, edgecolor=SURFACE, linewidth=1.5,
                    label=name if i == 0 else None)
            ax.text(value + 1, y, f"{value:.0f}%", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(labels)), labels, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("hit@5 (%), heading-based chunks")
    style(ax, grid_axis="x")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=9)
    title(fig, "Everyday wording is where semantic search earns its keep",
          "Same facts, reworded to avoid the documents' key terms. With only 10 paraphrases, "
          "each one is worth 10 points.")
    save(fig, "2_paraphrase_gap.png")


def load_runs():
    runs = []
    for folder, label in RUNS:
        path = RESULTS / "runs" / folder / "summary.json"
        if path.exists():
            runs.append((label, json.loads(path.read_text())))
        else:
            print(f"  (skipping {folder}: no summary.json)")
    return runs


def aa_noise(runs):
    by_label = dict(runs)
    a, b = by_label.get("Default (run 1)"), by_label.get("Default (run 2)")
    return (a, b) if a and b else (None, None)


def cost_per_1000(s):
    """Precise cost: run total / number of cases (the per-query field is rounded)."""
    c = s["cost_usd"]
    if c.get("total_assistant") and s.get("cases"):
        return 1000 * c["total_assistant"] / (s["cases"] - s.get("errors", 0))
    return 1000 * c["per_query"]


def chart_quality_cost(runs):
    a, b = aa_noise(runs)
    noise = abs(a["answer_quality"]["correctness"] - b["answer_quality"]["correctness"]) * 100 if a else 0
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.12)
    points = [(label, s) for label, s in runs if label != "Default (run 2)"]
    for label, s in points:
        cost = cost_per_1000(s)                           # dollars per 1,000 questions
        corr = 100 * s["answer_quality"]["correctness"]
        ax.errorbar(cost, corr, yerr=noise, fmt="o", markersize=8, color=SERIES[0],
                    ecolor=GRID, elinewidth=6, capsize=0, markeredgecolor=SURFACE, markeredgewidth=2)
        name = "Default" if label == "Default (run 1)" else label
        ax.annotate(name, (cost, corr), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xlabel("Cost per 1,000 questions (US$, Claude Haiku 5.5)")
    ax.set_ylabel("Correctness (%)")
    ax.set_ylim(min(100 * s["answer_quality"]["correctness"] for _, s in points) - 8, 100)
    xs = [cost_per_1000(s) for _, s in points]
    ax.set_xlim(min(xs) - 0.05, max(xs) + 0.15)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("$%.2f"))
    style(ax, grid_axis="both")
    title(fig, "Better retrieval improved answers at no extra cost",
          f"Grey bars show ±{noise:.1f} points: the change seen between two identical runs (A/A test). "
          "Gaps smaller than this may be noise.")
    save(fig, "3_quality_vs_cost.png")


def chart_noise(runs):
    a, b = aa_noise(runs)
    if not a:
        print("  (skipping noise chart: needs both default runs)")
        return
    metrics = [
        ("Retrieval hit@5", a["retrieval"]["hit@5"], b["retrieval"]["hit@5"]),
        ("Correct refusals", a["abstention"]["correct_refusal_rate"], b["abstention"]["correct_refusal_rate"]),
        ("False refusals", a["abstention"]["false_refusal_rate"], b["abstention"]["false_refusal_rate"]),
        ("Attack success", a["injection"]["attack_success_rate"], b["injection"]["attack_success_rate"]),
        ("Citation support", a["answer_quality"]["citation_support"], b["answer_quality"]["citation_support"]),
        ("Groundedness (per claim)", a["answer_quality"]["groundedness"], b["answer_quality"]["groundedness"]),
        ("Correctness", a["answer_quality"]["correctness"], b["answer_quality"]["correctness"]),
        ("Fully grounded answers", a["answer_quality"]["fully_grounded_rate"],
         b["answer_quality"]["fully_grounded_rate"]),
    ]
    diffs = [(name, abs(x - y) * 100) for name, x, y in metrics]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.78, bottom=0.12)
    for i, (name, d) in enumerate(diffs):
        if d > 0:
            ax.barh(i, d, 0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
        ax.text(d + 0.15, i, "0 (identical)" if d == 0 else f"{d:.1f} pts",
                va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(diffs)), [d[0] for d in diffs], fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(d for _, d in diffs) * 1.25 + 0.5)
    ax.set_xlabel("Difference between two identical runs (percentage points)")
    style(ax, grid_axis="x")
    title(fig, "Which metrics can be compared across single runs?",
          "Same configuration run twice. Only differences larger than these bars should be "
          "treated as real.")
    save(fig, "4_metric_noise.png")


def main():
    sweep = load_sweep()
    chart_retrieval(sweep)
    chart_paraphrase(sweep)
    runs = load_runs()
    chart_quality_cost(runs)
    chart_noise(runs)


if __name__ == "__main__":
    main()
