"""
Step 9: Monitoring page - how is the assistant doing in use?

Reads results/query_log.jsonl (every question) and results/feedback_log.jsonl (thumbs up/down).
Alert thresholds below are where a team would start investigating.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)

QUERY_LOG = Path("results/query_log.jsonl")
FEEDBACK_LOG = Path("results/feedback_log.jsonl")

# Alert thresholds, based on the offline evaluation (Step 8): in testing the assistant
# wrongly refused ~0-5% of answerable questions and p95 latency was ~2-3 s.
ALERTS = {
    "refusal_rate": 0.30,        # many refusals = users ask things the documents don't cover, or retrieval broke
    "p95_seconds": 5.0,
    "thumbs_down_rate": 0.25,
    "invalid_citation_rate": 0.01,
}

st.set_page_config(page_title="Monitoring", page_icon="📊", layout="wide")
st.title("📊 Monitoring")


def read_jsonl(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line])


logs = read_jsonl(QUERY_LOG)
if logs.empty or "source" not in logs:
    st.info("No questions from the app yet. Ask something on the main page, then come back.")
    st.stop()

only_app = st.toggle("Only questions asked in the app", value=True,
                     help="Off = also include questions asked from the command line (src/assistant.py).")
if only_app:
    logs = logs[logs["source"] == "app"]
if logs.empty:
    st.info("No questions from the app yet.")
    st.stop()

logs["time"] = pd.to_datetime(logs["timestamp"])
logs["refused"] = ~logs["answerable"].astype(bool)
logs["invalid_citation"] = logs["invalid_citations"].apply(lambda x: bool(x))
feedback = read_jsonl(FEEDBACK_LOG)
if not feedback.empty:
    feedback = feedback.drop_duplicates("query_id", keep="last")
    logs = logs.merge(feedback[["query_id", "rating"]], on="query_id", how="left")
else:
    logs["rating"] = None

n = len(logs)
refusal_rate = logs["refused"].mean()
p50, p95 = np.percentile(logs["total_ms"], [50, 95]) / 1000
rated = logs["rating"].dropna()
down_rate = (rated == "down").mean() if len(rated) else 0.0
invalid_rate = logs["invalid_citation"].mean()

# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Questions", f"{n:,}")
c2.metric("Refused (evidence missing)", f"{refusal_rate:.0%}")
c3.metric("Injection attempts flagged", int(logs["injection_detected"].sum()))
c4.metric("Latency p50 / p95", f"{p50:.1f} s / {p95:.1f} s")
c5.metric("Total cost", f"${logs['cost_usd'].sum():.4f}", f"${logs['cost_usd'].mean():.5f} per question",
          delta_color="off")

c6, c7, c8 = st.columns(3)
c6.metric("👍 / 👎", f"{(rated == 'up').sum()} / {(rated == 'down').sum()}")
c7.metric("Invalid citations", f"{invalid_rate:.1%}")
c8.metric("Prompt versions in use", ", ".join(sorted(logs["prompt_version"].dropna().unique())) or "-")

# Alerts: the point of monitoring is to notice problems without reading every log line.
alerts = []
if n >= 10 and refusal_rate > ALERTS["refusal_rate"]:
    alerts.append(f"Refusal rate {refusal_rate:.0%} is above {ALERTS['refusal_rate']:.0%}. Check what users "
                  "are asking: new topics the documents don't cover, or a retrieval problem?")
if p95 > ALERTS["p95_seconds"]:
    alerts.append(f"p95 latency {p95:.1f} s is above {ALERTS['p95_seconds']:.0f} s.")
if len(rated) >= 5 and down_rate > ALERTS["thumbs_down_rate"]:
    alerts.append(f"{down_rate:.0%} of rated answers got 👎 (threshold {ALERTS['thumbs_down_rate']:.0%}).")
if invalid_rate > ALERTS["invalid_citation_rate"]:
    alerts.append(f"Invalid citations in {invalid_rate:.1%} of answers: the model cited passages it wasn't given.")
for alert in alerts:
    st.warning("⚠️ " + alert)
if not alerts:
    st.success("No alerts. All monitored metrics are within their thresholds.")

st.subheader("Questions per day")
st.bar_chart(logs.set_index("time").resample("D").size().rename("questions"), color="#2a78d6")

st.subheader("👎 answers: candidates for the test set")
st.caption("Real failures are the most valuable new test cases. Review these and add good ones "
           "to eval/candidates.csv.")
bad = logs[logs["rating"] == "down"]
if bad.empty:
    st.write("None yet.")
else:
    st.dataframe(bad[["timestamp", "question", "answer", "answerable"]], width="stretch", hide_index=True)

st.subheader("Recent questions")
recent = logs.sort_values("time", ascending=False).head(50)
st.dataframe(
    recent[["timestamp", "question", "answerable", "injection_detected", "rating", "total_ms", "cost_usd"]],
    width="stretch", hide_index=True,
    column_config={"total_ms": st.column_config.NumberColumn("ms", format="%d"),
                   "cost_usd": st.column_config.NumberColumn("cost $", format="%.5f")},
)
