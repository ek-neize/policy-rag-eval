"""
Step 7b: How much can we trust the LLM judge?

  python eval/judge_agreement.py sample   -> writes eval/human_grades.csv (20 answers, judge hidden)
  ... you grade them in a spreadsheet ...
  python eval/judge_agreement.py score    -> compares your grades with the judge's

Agreement rate = how often you and the judge gave the same grade.
Cohen's kappa  = agreement corrected for luck. If both of you say "correct" 90% of
                 the time, you'll agree often just by chance; kappa removes that.
                 Rough guide: <0.4 weak, 0.4-0.6 moderate, 0.6-0.8 substantial, >0.8 near-perfect.
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from embeddings import load_chunks  # noqa: E402

RUN = "hybrid-heading-bge-small"
GRADES_FILE = Path("eval/human_grades.csv")
REPORT_FILE = Path("results/judge_agreement.json")
SAMPLE_SIZE = 20
SEED = 7
GROUPS = ("single", "paraphrase", "multi")   # injection passages contain planted text, so left out

JUDGE_TO_HUMAN = {"correct": "correct", "partially_correct": "partial", "incorrect": "incorrect"}


def load_results(run):
    path = Path("results/runs") / run / "cases.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sample(run):
    if GRADES_FILE.exists():
        sys.exit(f"{GRADES_FILE} already exists and may contain your grades. Delete it first to resample.")
    strategy = run.split("-")[1]
    chunks = {c["chunk_id"]: c for c in load_chunks(strategy)}

    eligible = [r for r in load_results(run)
                if r["group"] in GROUPS and r["answered"] and r.get("judgement")]
    chosen = sorted(random.Random(SEED).sample(eligible, min(SAMPLE_SIZE, len(eligible))), key=lambda r: r["id"])

    rows = []
    for r in chosen:
        passages = "\n\n".join(f"[S{i}] {chunks[cid]['text']}" for i, cid in enumerate(r["passages_shown"], 1))
        rows.append({"id": r["id"], "group": r["group"], "question": r["question"],
                     "reference_answer": r["reference_answer"], "answer": r["answer"], "passages": passages,
                     "human_correctness": "", "human_fully_supported": "", "human_notes": ""})
    with open(GRADES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} answers to {GRADES_FILE}: {dict(Counter(r['group'] for r in rows))}")
    print("Fill in human_correctness (correct / partial / incorrect) and "
          "human_fully_supported (yes / no) for every row.")


def cohens_kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0   # both graders always gave the same single label
    return (observed - expected) / (1 - expected)


def compare(name, human, judge, labels):
    agree = sum(h == j for h, j in zip(human, judge)) / len(human)
    kappa = cohens_kappa(human, judge)
    print(f"\n{name}")
    print(f"  agreement {100 * agree:.0f}%   Cohen's kappa {kappa:.2f}")
    print(f"  {'':<18}" + "".join(f"judge:{lab:<11}" for lab in labels))
    for h_lab in labels:
        counts = [sum(1 for h, j in zip(human, judge) if h == h_lab and j == j_lab) for j_lab in labels]
        print(f"  you:{h_lab:<14}" + "".join(f"{c:<17}" for c in counts))
    lenient = sum(labels.index(j) < labels.index(h) for h, j in zip(human, judge))
    strict = sum(labels.index(j) > labels.index(h) for h, j in zip(human, judge))
    print(f"  judge more lenient than you: {lenient}   judge stricter than you: {strict}")
    return {"agreement": round(agree, 3), "kappa": round(kappa, 3),
            "judge_more_lenient": lenient, "judge_stricter": strict}


def read_grades(path):
    if not path.exists():
        sys.exit(f"{path} not found.")
    with open(path, encoding="utf-8") as f:
        grades = list(csv.DictReader(f))
    problems = []
    for g in grades:
        g["human_correctness"] = g["human_correctness"].strip().lower()
        g["human_fully_supported"] = g["human_fully_supported"].strip().lower()
        if g["human_correctness"] not in ("correct", "partial", "incorrect"):
            problems.append(f"{g['id']}: human_correctness must be correct / partial / incorrect")
        if g["human_fully_supported"] not in ("yes", "no"):
            problems.append(f"{g['id']}: human_fully_supported must be yes / no")
    if problems:
        sys.exit(f"Fix these rows in {path} first:\n  " + "\n  ".join(problems))
    return grades


def raters(file_a, file_b):
    """How often do two graders (e.g. you and Claude) agree with EACH OTHER?"""
    a = {g["id"]: g for g in read_grades(file_a)}
    b = {g["id"]: g for g in read_grades(file_b)}
    ids = sorted(set(a) & set(b))
    print(f"Comparing {file_a.name} (A) with {file_b.name} (B) on {len(ids)} answers")
    compare("CORRECTNESS", [a[i]["human_correctness"] for i in ids],
            [b[i]["human_correctness"] for i in ids], ["correct", "partial", "incorrect"])
    compare("FULLY SUPPORTED", [a[i]["human_fully_supported"] for i in ids],
            [b[i]["human_fully_supported"] for i in ids], ["yes", "no"])
    print("\n(In the tables above, 'you' = file A and 'judge' = file B.)")
    diffs = [i for i in ids if (a[i]["human_correctness"], a[i]["human_fully_supported"])
             != (b[i]["human_correctness"], b[i]["human_fully_supported"])]
    print(f"\nANSWERS WHERE A AND B DIFFER ({len(diffs)})")
    for i in diffs:
        print(f"  {i}: A {a[i]['human_correctness']}/{a[i]['human_fully_supported']}   "
              f"B {b[i]['human_correctness']}/{b[i]['human_fully_supported']}")
        for label, g in (("A", a[i]), ("B", b[i])):
            if g["human_notes"]:
                print(f"      {label} note: {g['human_notes'][:150]}")


def score(run, grades_file=GRADES_FILE):
    grades = read_grades(grades_file)
    results = {r["id"]: r for r in load_results(run)}
    h_corr, j_corr, h_sup, j_sup, rows = [], [], [], [], []
    for g in grades:
        j = results[g["id"]]["judgement"]
        jc = JUDGE_TO_HUMAN.get(j.get("correctness"), "incorrect")
        js = "yes" if j.get("claims") and all(c.get("supported_by_any_passage") for c in j["claims"]) else "no"
        h_corr.append(g["human_correctness"]); j_corr.append(jc)
        h_sup.append(g["human_fully_supported"]); j_sup.append(js)
        rows.append((g, jc, js, j.get("explanation", "")))

    print(f"Compared {len(grades)} answers from {grades_file.name} with the judge (run '{run}')")
    report = {
        "run": run, "n": len(grades), "grades_file": str(grades_file),
        "correctness": compare("CORRECTNESS", h_corr, j_corr, ["correct", "partial", "incorrect"]),
        "fully_supported": compare("FULLY SUPPORTED (no unsupported claims)", h_sup, j_sup, ["yes", "no"]),
    }
    binary_h = ["correct" if x == "correct" else "not" for x in h_corr]
    binary_j = ["correct" if x == "correct" else "not" for x in j_corr]
    report["correct_vs_not"] = {"agreement": round(sum(a == b for a, b in zip(binary_h, binary_j)) / len(binary_h), 3),
                                "kappa": round(cohens_kappa(binary_h, binary_j), 3)}
    print(f"\nCORRECT vs NOT CORRECT (partial and incorrect merged)\n"
          f"  agreement {100 * report['correct_vs_not']['agreement']:.0f}%   "
          f"Cohen's kappa {report['correct_vs_not']['kappa']:.2f}")

    disagreements = [(g, jc, js, ex) for g, jc, js, ex in rows
                     if jc != g["human_correctness"] or js != g["human_fully_supported"]]
    print(f"\nDISAGREEMENTS ({len(disagreements)})")
    for g, jc, js, ex in disagreements:
        print(f"  {g['id']}: you {g['human_correctness']}/{g['human_fully_supported']}  "
              f"judge {jc}/{js}")
        print(f"      judge said: {ex[:160]}")
        if g["human_notes"]:
            print(f"      your note:  {g['human_notes'][:160]}")
    report["disagreements"] = [{"id": g["id"], "human": [g["human_correctness"], g["human_fully_supported"]],
                                "judge": [jc, js], "judge_explanation": ex, "human_notes": g["human_notes"]}
                               for g, jc, js, ex in disagreements]

    out = REPORT_FILE.with_name(f"judge_agreement_{grades_file.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["sample", "score", "raters"])
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--grades", default=str(GRADES_FILE),
                        help="grades file to score against the judge")
    parser.add_argument("--other", default="eval/claude_grades.csv",
                        help="second grades file for 'raters'")
    args = parser.parse_args()
    if args.action == "sample":
        sample(args.run)
    elif args.action == "score":
        score(args.run, Path(args.grades))
    else:
        raters(Path(args.grades), Path(args.other))


if __name__ == "__main__":
    main()
