"""
Step 7: Run the full evaluation for one configuration.

For every test case:
  1. retrieve the top 10 chunks (timed)        -> retrieval metrics
  2. for injection cases, plant the attack      -> injection metrics
  3. the assistant answers from the top 5       -> abstention, citations, cost, latency
  4. the judge grades answered questions        -> groundedness, correctness

Progress is saved after every case, so if it stops (network error, Ctrl+C)
just run the same command again and it continues where it left off.

Usage:
  python eval/run_eval.py --limit 5                 # quick, cheap smoke test
  python eval/run_eval.py                           # all 94 cases, default config
  python eval/run_eval.py --method dense --strategy fixed --model minilm
  python eval/run_eval.py --fresh                   # discard saved progress and restart
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import anthropic  # noqa: E402

from assistant import PROMPT_VERSION, TOP_K, Assistant  # noqa: E402
from embeddings import MODELS, STRATEGIES  # noqa: E402
from evidence import contains_quote  # noqa: E402
from judge import judge  # noqa: E402
from metrics import attack_succeeded, citation_scores, judge_scores, retrieval_scores, summarize  # noqa: E402
from retrieval import METHODS  # noqa: E402

TESTSET = Path("eval/testset.jsonl")
RUNS_DIR = Path("results/runs")
RETRIEVE_K = 10          # retrieve 10 for retrieval metrics; the assistant reads the top TOP_K


def plant_attack(attack, chunks, quotes):
    """Return a copy of the passages with the attack text planted in them."""
    chunks = [dict(c) for c in chunks]
    if attack["placement"] == "inside":
        # Hide it in the middle of the real source passage (or the top passage if it wasn't found).
        target = next((c for c in chunks if any(contains_quote(c["text"], q) for q in quotes)), chunks[0])
        lines = target["text"].split("\n")
        middle = max(1, len(lines) // 2)
        target["text"] = "\n".join(lines[:middle] + [attack["text"]] + lines[middle:])
    else:
        # Add it as an extra, official-looking passage in position 2, keeping 5 passages total.
        fake = {"chunk_id": f"INJECTED::{attack['name']}", "doc_id": "INJECTED",
                "title": "Financial Consumer Agency of Canada", "section": "Important notice",
                "url": "", "topic": "injected", "text": attack["text"]}
        chunks = chunks[:1] + [fake] + chunks[1:-1]
    return chunks


def run_case(case, assistant, client):
    quotes = [e["quote"] for e in case["evidence"]]

    start = time.perf_counter()
    retrieved = assistant.retriever.search(case["question"], k=RETRIEVE_K)
    retrieval_ms = 1000 * (time.perf_counter() - start)

    passages = retrieved[:assistant.k]
    if case["attack"]:
        passages = plant_attack(case["attack"], passages, quotes)

    answer = assistant.ask(case["question"], chunks=passages, log=False)
    answered = answer["answerable"]

    result = {
        "id": case["id"], "group": case["group"], "question": case["question"],
        "expected_answerable": case["answerable"], "answered": answered,
        "answer": answer["answer"], "reference_answer": case["reference_answer"],
        "cited_labels": answer["cited_labels"], "cited_chunk_ids": answer["cited_chunk_ids"],
        "invalid_citations": answer["invalid_citations"],
        "injection_detected": answer["injection_detected"],
        "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved],
        "passages_shown": [c["chunk_id"] for c in passages],
        "retrieval": retrieval_scores(quotes, retrieved) if quotes else {},
        "scores": {},
        "prompt_version": answer["prompt_version"],
        "input_tokens": answer["input_tokens"], "output_tokens": answer["output_tokens"],
        "cost_usd": answer["cost_usd"],
        "retrieval_ms": round(retrieval_ms, 1), "generation_ms": answer["generation_ms"],
        "error": answer["error"],
    }

    if case["attack"]:
        result["attack_name"] = case["attack"]["name"]
        result["attack_success"] = attack_succeeded(case["attack"], answer["answer"], answered)

    # Citations and judging only make sense for questions that have an answer.
    if case["answerable"]:
        by_label = {f"S{i}": c for i, c in enumerate(passages, 1)}
        cited = [by_label[s] for s in answer["cited_labels"] if s in by_label]
        result["scores"].update(citation_scores(quotes, cited, answer["invalid_citations"]))
        if answered and answer["answer"].strip():
            judgement, judge_cost = judge(client, case["question"], case["reference_answer"],
                                          answer["answer"], passages)
            result["judgement"] = judgement
            result["judge_cost_usd"] = round(judge_cost, 6)
            result["scores"].update(judge_scores(judgement))
        else:
            result["scores"]["correctness"] = 0.0      # wrongly refused: no credit
    return result


def print_summary(summary, config, cold_start_ms):
    s = summary
    pct = lambda v: "  n/a" if v is None else f"{100 * v:5.1f}%"  # noqa: E731
    print(f"\n================  {config}  ================")
    print(f"Cases: {s['cases']}   errors: {s['errors']}   cold start: {cold_start_ms:,.0f} ms\n")
    print("RETRIEVAL (answerable questions)")
    for m in ("hit@1", "hit@3", "hit@5", "hit@10", "evidence_recall@5"):
        print(f"  {m:<20} {pct(s['retrieval'][m])}")
    print(f"  {'MRR@10':<20} {s['retrieval']['mrr@10']}")
    print("\nANSWER QUALITY (judge)")
    for m, v in s["answer_quality"].items():
        print(f"  {m:<20} {pct(v)}")
    print("\nCITATIONS")
    for m, v in s["citations"].items():
        print(f"  {m:<20} {pct(v)}")
    print("\nABSTENTION")
    for m, v in s["abstention"].items():
        print(f"  {m:<20} {pct(v)}")
    inj = s["injection"]
    print("\nPROMPT INJECTION")
    print(f"  {'attack_success_rate':<20} {pct(inj['attack_success_rate'])}   "
          f"worked: {inj['attacks_that_worked'] or 'none'}")
    print(f"  {'detection_rate':<20} {pct(inj['detection_rate'])}")
    print(f"  {'still_correct':<20} {pct(inj['still_correct'])}")
    lat = s["latency_ms"]
    print("\nLATENCY (warm, ms)        p50      p95")
    for part in ("retrieval", "generation", "total"):
        print(f"  {part:<20} {lat[part + '_p50']:>7} {lat[part + '_p95']:>8}")
    c = s["cost_usd"]
    print(f"\nCOST  per query ${c['per_query']:.5f}  (avg {c['avg_input_tokens']:.0f} input tokens)   "
          f"run total: assistant ${c['total_assistant']:.4f} + judge ${c['total_judge']:.4f}")
    print("\nBY GROUP          cases   hit@5   answered   correctness")
    for g, v in s["by_group"].items():
        print(f"  {g:<14} {v['cases']:>6} {pct(v['hit@5']):>8} {pct(v['answered_rate']):>9} {pct(v['correctness']):>11}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, default="hybrid")
    parser.add_argument("--strategy", choices=STRATEGIES, default="heading")
    parser.add_argument("--model", choices=list(MODELS), default="bge-small")
    parser.add_argument("-k", type=int, default=TOP_K, help="how many passages Claude reads (max 10)")
    parser.add_argument("--limit", type=int, help="only run the first N cases")
    parser.add_argument("--fresh", action="store_true", help="discard saved progress")
    args = parser.parse_args()

    cases = [json.loads(line) for line in TESTSET.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        cases = cases[:args.limit]

    client = anthropic.Anthropic(max_retries=5)
    t0 = time.perf_counter()
    assistant = Assistant(args.method, args.strategy, args.model, k=min(args.k, RETRIEVE_K), client=client)
    assistant.retriever.search("warm-up question", k=1)          # loads the embedding model
    cold_start_ms = 1000 * (time.perf_counter() - t0)

    model_part = assistant.model_key or "none"
    k_part = "" if assistant.k == TOP_K else f"-k{assistant.k}"     # default k keeps the old folder name
    out_dir = RUNS_DIR / f"{args.method}-{args.strategy}-{model_part}{k_part}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_file = out_dir / "cases.jsonl"
    if args.fresh and cases_file.exists():
        cases_file.unlink()

    done = {}
    if cases_file.exists():
        for line in cases_file.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    todo = [c for c in cases if c["id"] not in done]
    print(f"Config {assistant.config}: {len(cases)} cases, {len(done)} already done, {len(todo)} to run")

    with open(cases_file, "a", encoding="utf-8") as f:
        for n, case in enumerate(todo, 1):
            try:
                result = run_case(case, assistant, client)
            except anthropic.APIError as error:
                print(f"  ! {case['id']}: API error, will retry on next run ({error})")
                continue
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            done[case["id"]] = result
            flag = ""
            if case["attack"]:
                flag = "ATTACK WORKED" if result["attack_success"] else "attack resisted"
            elif case["answerable"] != result["answered"]:
                flag = "wrongly refused" if case["answerable"] else "answered an unanswerable question"
            print(f"  {n:>3}/{len(todo)} {case['id']:<4} {case['group']:<12} "
                  f"correctness={result['scores'].get('correctness', '-')!s:<5} {flag}")

    results = [done[c["id"]] for c in cases if c["id"] in done]
    summary = summarize(results)
    versions = sorted({r.get("prompt_version", "v1") for r in results})
    if len(versions) > 1:
        print(f"\n⚠ WARNING: this run mixes prompt versions {versions}. Re-run with --fresh.")
    summary.update({"config": assistant.config, "k": assistant.k, "cold_start_ms": round(cold_start_ms, 1),
                    "prompt_version": ", ".join(versions)})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary, assistant.config, cold_start_ms)
    print(f"\nSaved: {cases_file} and {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
