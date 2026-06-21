"""
Sandbox demo (Streamlit) — runs the ranking system end-to-end on a small
candidate sample and shows the ranked shortlist with grounded reasoning and the
transparent sub-scores behind each pick.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud / HuggingFace Spaces (CPU, free tier).

It accepts an uploaded .jsonl / .json (a list, or one JSON object per line) of up
to ~200 candidates, or falls back to the bundled sample_candidates.json.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import sys

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from redrob_ranker import (  # noqa: E402
    behavioral, config, embed, honeypot, reasoning, scoring, structured, textbuild,
)

REF = dt.date.fromisoformat(config.REFERENCE_DATE)


def _load_candidates(raw: bytes):
    text = raw.decode("utf-8")
    text_stripped = text.lstrip()
    if text_stripped.startswith("["):
        return json.loads(text)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def rank_small(cands, embed_backend="auto"):
    """Single-pass ranking for a small sample (full facts for everyone)."""
    facts_list, beh_list, hp_list, texts, cids = [], [], [], [], []
    for c in cands:
        blob = textbuild.full_profile_blob(c)
        f = structured.score_structured(c, blob)
        f["candidate_id"] = c["candidate_id"]
        facts_list.append(f)
        beh_list.append(behavioral.availability(c, REF))
        hp_list.append(honeypot.detect(c, REF)["is_honeypot"])
        texts.append(textbuild.build_embedding_text(c))
        cids.append(c["candidate_id"])

    semantic = embed.compute_semantic_scores(
        texts, config.JD_POSITIVE_ANCHOR, config.JD_NEGATIVE_ANCHOR,
        backend=embed_backend, verbose=False,
    )
    final = scoring.combine(
        [f["structured_fit"] for f in facts_list], semantic,
        [b["modifier"] for b in beh_list], hp_list,
    )
    ranked = scoring.select_and_order(cids, final, top=len(cids))
    fact_by_id = {f["candidate_id"]: (f, b) for f, b in zip(facts_list, beh_list)}
    rows = []
    for rank_pos, (i, cid, disp) in enumerate(ranked, start=1):
        f, b = fact_by_id[cid]
        rows.append({
            "rank": rank_pos,
            "candidate_id": cid,
            "title": f.get("current_title", ""),
            "score": round(disp, 4),
            "reasoning": reasoning.build_reasoning(f, b, rank_pos),
            "title_fit": round(f["sub"]["role_title"], 2),
            "evidence": round(f["sub"]["career_evidence"], 2),
            "skill_trust": round(f["sub"]["skill_trust"], 2),
            "availability": round(b["avail_index"], 2),
            "honeypot": "⚠" if hp_list[i] else "",
        })
    return rows


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="Redrob Candidate Ranker", page_icon="🎯", layout="wide")
st.title("🎯 Redrob — Intelligent Candidate Ranking")
st.caption("Ranks candidates against the *Senior AI Engineer (Founding Team)* JD — "
           "by reading the profile, not counting keywords. CPU-only, no LLM at inference.")

with st.expander("How the score works", expanded=False):
    st.markdown(
        "**score = (0.70 · structured_fit + 0.30 · semantic_fit) · behavioral_modifier · honeypot_gate**\n\n"
        "- **structured_fit** — title coherence, real career evidence (retrieval/ranking in "
        "production with NDCG/MAP eval), trust-weighted skills (stuffed lists ignored), "
        "domain, product-vs-services, location.\n"
        "- **semantic_fit** — contrastive static embeddings: `cos(cand, JD-ideal) − cos(cand, JD-anti-pattern)`.\n"
        "- **behavioral_modifier** — down-weights the dormant / unresponsive.\n"
        "- **honeypot_gate** — removes internally-impossible profiles.")

uploaded = st.file_uploader("Upload candidates (.jsonl or .json, ≤ ~200)", type=["jsonl", "json"])
default_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sample_candidates.json")

cands = None
if uploaded is not None:
    cands = _load_candidates(uploaded.read())
elif os.path.exists(default_path):
    st.info("No file uploaded — using the bundled `sample_candidates.json`.")
    with open(default_path, "rb") as f:
        cands = _load_candidates(f.read())

if cands:
    cands = cands[:200]
    st.write(f"Ranking **{len(cands)}** candidates…")
    with st.spinner("Scoring…"):
        rows = rank_small(cands)
    st.success("Done.")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    csv_lines = ["candidate_id,rank,score,reasoning"]
    import csv as _csv
    buf = io.StringIO()
    w = _csv.writer(buf)
    for r in rows:
        w.writerow([r["candidate_id"], r["rank"], f'{r["score"]:.6f}', r["reasoning"]])
    st.download_button("⬇ Download ranked CSV", buf.getvalue(),
                       file_name="submission.csv", mime="text/csv")
else:
    st.warning("Upload a candidate file to begin.")
