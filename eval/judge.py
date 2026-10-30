"""
Step 7: LLM-as-a-judge.

A second Claude call grades each answer:
  1. Split the answer into individual factual claims.
  2. For each claim: is it supported by the passages it cites? By any passage?
  3. Compare the whole answer with the reference answer: correct / partial / incorrect.

Judges make mistakes too, so in Step 7b you hand-grade a sample and measure
how often you and the judge agree.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from assistant import LLM_MODEL, PRICE_PER_MTOK, format_passages  # noqa: E402

JUDGE_MODEL = LLM_MODEL   # same model; a known limitation, checked by human agreement

JUDGE_PROMPT = """You are a strict, fair evaluator of a question-answering assistant. \
Grade the ANSWER using only the PASSAGES and the REFERENCE ANSWER below.

Step 1 - Claims. Split the answer into its individual factual claims (one fact each). \
For each claim record the passage labels it cites (e.g. ["S1"], or [] if none), then decide:
- supported_by_cited: true only if the cited passage(s) state this fact. A claim with no \
citation is false here.
- supported_by_any_passage: true if ANY passage states this fact.
A claim is supported only if the passage says it; plausible outside knowledge does not count. \
Paraphrasing is fine. Generic advice such as "contact your bank" is supported only if a \
passage says so.

Step 2 - Correctness. Compare the answer to the reference answer:
- correct: contains the reference's key facts and contradicts nothing
- partially_correct: some key facts missing or slightly wrong, nothing seriously wrong
- incorrect: misses the main point, contradicts the reference, or includes a harmful error
Extra details that are correct and supported should not lower the grade.

<question>{question}</question>

<reference_answer>{reference}</reference_answer>

<passages>
{passages}
</passages>

<answer>{answer}</answer>"""

JUDGE_TOOL = {
    "name": "grade_answer",
    "description": "Submit the grade.",
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "cited_labels": {"type": "array", "items": {"type": "string"}},
                        "supported_by_cited": {"type": "boolean"},
                        "supported_by_any_passage": {"type": "boolean"},
                    },
                    "required": ["claim", "cited_labels", "supported_by_cited", "supported_by_any_passage"],
                },
            },
            "correctness": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]},
            "explanation": {"type": "string", "description": "One or two sentences."},
        },
        "required": ["claims", "correctness", "explanation"],
    },
}


def judge(client, question, reference, answer, passages):
    """Returns (judgement dict, cost in USD)."""
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1500,
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "grade_answer"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, reference=reference, answer=answer,
            passages=format_passages(passages))}],
    )
    cost = (response.usage.input_tokens * PRICE_PER_MTOK["input"]
            + response.usage.output_tokens * PRICE_PER_MTOK["output"]) / 1e6
    result = next((b.input for b in response.content if b.type == "tool_use"), {})
    result.setdefault("claims", [])
    result.setdefault("correctness", None)
    result.setdefault("explanation", "")
    return result, cost
