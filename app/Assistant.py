"""
Step 9: The web app - ask a question, get an evidence-backed answer.

Run locally:   streamlit run app/app.py
"""

import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)                                   # all data paths are relative to the project folder

# On Streamlit Community Cloud the API key comes from the app's Secrets settings.
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass                                         # running locally: assistant.py reads .env

from assistant import LOG_FILE, PROMPT_VERSION, Assistant, write_log  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: the best configuration from Step 8
# ---------------------------------------------------------------------------
METHOD, STRATEGY, MODEL, TOP_K = "hybrid", "heading", "minilm", 5
MAX_QUESTION_CHARS = 400
MAX_QUESTIONS_PER_SESSION = 25        # protects your API credit if the app is public
DAILY_BUDGET_USD = 0.50               # ~1,000 questions; the app pauses after this each day
FEEDBACK_LOG = Path("results/feedback_log.jsonl")

EXAMPLES = [
    "Can my bank refuse to open an account for me, and what can I do if it does?",
    "How long does a bankruptcy stay on my credit report?",
    "What is BMO's current prime rate?",
]

st.set_page_config(page_title="FCAC Policy Assistant", page_icon="🏦", layout="centered")


# ---------------------------------------------------------------------------
# One-time setup (cached: runs once per server, not on every click)
# ---------------------------------------------------------------------------
def ensure_index():
    """Build the chunks and the vector index if they don't exist (e.g. on a fresh cloud deploy)."""
    from embeddings import embed_passages, index_path, load_chunks
    import numpy as np

    if not Path(f"data/processed/chunks_{STRATEGY}.jsonl").exists():
        import chunk_docs
        chunk_docs.main()
    if not index_path(STRATEGY, MODEL).exists():
        index_path(STRATEGY, MODEL).parent.mkdir(parents=True, exist_ok=True)
        texts = [c["text"] for c in load_chunks(STRATEGY)]
        np.save(index_path(STRATEGY, MODEL), embed_passages(MODEL, texts))


@st.cache_resource(show_spinner="Loading the search index (first start takes about a minute)...")
def get_assistant():
    ensure_index()
    assistant = Assistant(METHOD, STRATEGY, MODEL, k=TOP_K)
    assistant.retriever.search("warm-up", k=1)          # load the embedding model now
    return assistant


def spent_today():
    """Total assistant cost logged by the app today (UTC)."""
    if not LOG_FILE.exists():
        return 0.0
    today = datetime.now(timezone.utc).date().isoformat()
    total = 0.0
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        if entry.get("source") == "app" and entry.get("timestamp", "").startswith(today):
            total += entry.get("cost_usd", 0)
    return total


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def render_answer_text(result):
    """Escape $ (Streamlit treats $...$ as maths) and turn [S1] into links to the source page."""
    text = result["answer"].replace("$", r"\$")
    urls = {f"S{i}": c["url"] for i, c in enumerate(result["retrieved"], 1)}
    return re.sub(r"\[(S\d+)\]", lambda m: f"[[{m.group(1)}]]({urls[m.group(1)]})" if m.group(1) in urls
                  else m.group(0), text)


def save_feedback(query_id, rating):
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "query_id": query_id, "rating": rating}
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def show_result(result):
    if result["answerable"]:
        st.markdown(render_answer_text(result))
    else:
        st.warning("**Evidence missing.** " + result["answer"].replace("$", r"\$"))
    if result["injection_detected"]:
        st.info("🛡️ A retrieved passage contained instructions aimed at the assistant. They were ignored.")

    by_label = {f"S{i}": c for i, c in enumerate(result["retrieved"], 1)}
    cited = [s for s in result["cited_labels"] if s in by_label]
    if cited:
        st.markdown("**Sources**")
        for label in cited:
            c = by_label[label]
            with st.expander(f"[{label}] {c['title']} — {c['section']}"):
                st.markdown(f"[Open the FCAC page ↗]({c['url']})")
                st.text(c["text"])

    st.caption(f"Answered in {result['total_ms'] / 1000:.1f} s · cost ${result['cost_usd']:.5f} · "
               f"{len(result['retrieved'])} passages searched · prompt {PROMPT_VERSION}")

    rating = st.feedback("thumbs", key=f"fb_{result['query_id']}")
    if rating is not None and not st.session_state.get(f"saved_{result['query_id']}"):
        save_feedback(result["query_id"], "up" if rating == 1 else "down")
        st.session_state[f"saved_{result['query_id']}"] = True
    if st.session_state.get(f"saved_{result['query_id']}"):
        st.caption("Thanks, your feedback was saved.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🏦 FCAC Policy Assistant")
st.write("Answers questions about Canadian financial consumer guidance using **only** "
         "[FCAC](https://www.canada.ca/en/financial-consumer-agency.html) pages, with a citation for "
         "every claim. If the pages don't contain the answer, it says so.")

with st.sidebar:
    st.header("About")
    st.markdown(
        f"- **Retrieval:** {METHOD} (BM25 + embeddings), {STRATEGY}-based chunks, {MODEL}\n"
        f"- **Answers:** Claude Haiku 5.5, top {TOP_K} passages\n"
        "- **Evaluated on 94 test questions:** 96% correct, 100% correct refusals, "
        "0/10 prompt-injection attacks succeeded\n"
        "- **Sources:** 50 FCAC pages. Not financial advice."
    )
    st.caption("Source code and evaluation: see the project README on GitHub.")

assistant = get_assistant()          # loads once per server; cached for every visitor after that

st.session_state.setdefault("count", 0)
st.session_state.setdefault("history", [])

st.write("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, width="stretch"):
        st.session_state["question"] = example

question = st.text_input("Your question", key="question", max_chars=MAX_QUESTION_CHARS,
                         placeholder="e.g. How do I dispute an error on my credit report?")
ask = st.button("Ask", type="primary", disabled=not question.strip())

if ask:
    if st.session_state["count"] >= MAX_QUESTIONS_PER_SESSION:
        st.error("You've reached the question limit for this session. Refresh the page to start again.")
    elif spent_today() >= DAILY_BUDGET_USD:
        st.error("The demo's daily budget has been reached. Please try again tomorrow.")
    else:
        try:
            with st.spinner("Searching FCAC guidance and writing an answer..."):
                result = assistant.ask(question.strip(), log=False)
        except anthropic.AuthenticationError:
            st.error("The assistant isn't configured correctly (API key problem). "
                     "The owner has been notified in the logs.")
            st.stop()
        except anthropic.APIError:
            st.error("The assistant is temporarily unavailable. Please try again in a minute.")
            st.stop()
        result["query_id"] = uuid.uuid4().hex[:12]
        result["source"] = "app"
        write_log(result)
        st.session_state["count"] += 1
        st.session_state["history"].insert(0, result)

for i, result in enumerate(st.session_state["history"]):
    if i == 1:
        st.divider()
        st.caption("Earlier questions")
    st.subheader(result["question"])
    show_result(result)
