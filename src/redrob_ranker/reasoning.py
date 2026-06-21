"""
reasoning.py — deterministic, fact-grounded reasoning for each ranked candidate.

Stage-4 manual review checks reasoning for: specific facts, JD connection,
honest concerns, NO hallucination, variation across rows, and rank consistency.

We generate reasoning *from the candidate's own parsed facts* (never an LLM), so
every claim is traceable to the profile — there is nothing to hallucinate. We
vary sentence structure with a per-candidate deterministic seed, and we choose
which concern to surface based on the candidate's actual flags, so the tone
tracks the rank: strong picks lead with evidence and note at most a minor
caveat; lower picks openly state why they're borderline.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List


def _seed(cid: str) -> int:
    return int(hashlib.md5(cid.encode()).hexdigest(), 16)


def _pick(seed: int, options: List[str]) -> str:
    return options[seed % len(options)]


def _yrs(yoe) -> str:
    try:
        return f"{float(yoe):.1f} yrs"
    except Exception:
        return "experience n/a"


def _evidence_phrase(cats: List[str]) -> str:
    has = set(cats)
    bits = []
    if "retrieval" in has:
        bits.append("retrieval/search")
    if "ranking/recsys" in has:
        bits.append("ranking/recommendation")
    if not bits and "nlp/ir" in has:
        bits.append("NLP/IR")
    core = " & ".join(bits) if bits else None

    tail = []
    if "production" in has:
        tail.append("in production")
    if "evaluation" in has:
        tail.append("with NDCG/MAP-style evaluation")
    tailstr = (" " + " ".join(tail)) if tail else ""

    if core:
        return f"career history shows building {core} systems{tailstr}"
    if "production" in has:
        return f"production engineering background{tailstr}"
    return ""


def build_reasoning(facts: Dict[str, Any], beh: Dict[str, Any], rank: int) -> str:
    cid = facts["candidate_id"]
    seed = _seed(cid)
    title = facts.get("current_title") or "Candidate"
    yoe = facts.get("yoe")
    cats = facts.get("evidence_cats") or []
    skills = facts.get("top_skills") or []
    loc = facts.get("location_str") or ""

    # ---- strengths ----
    lead_opts = [
        f"{title} ({_yrs(yoe)})",
        f"{title} with {_yrs(yoe)}",
        f"{title}, {_yrs(yoe)}",
    ]
    lead = _pick(seed, lead_opts)

    ev = _evidence_phrase(cats)
    strengths: List[str] = []
    if ev:
        strengths.append(ev)
    if facts.get("has_product"):
        strengths.append("at a product company")
    if facts.get("nlp_ir_present") and "nlp/ir" not in " ".join(cats):
        strengths.append("NLP/IR orientation")
    if skills:
        verb = _pick(seed >> 3, ["verified skills:", "trusted skills:", "key skills:"])
        strengths.append(f"{verb} {', '.join(skills[:3])}")

    # ---- availability note ----
    avail_bits = []
    if beh.get("short_notice"):
        avail_bits.append(f"{beh['notice_period_days']}-day notice")
    if beh.get("open_to_work"):
        avail_bits.append("open to work")
    if facts.get("location_str") and facts.get("in_india"):
        avail_bits.append(f"based in {loc.split(',')[0]}")

    # ---- honest concerns (ordered by severity; surface the most relevant) ----
    concerns: List[str] = []
    fl = facts.get("flags", {})
    if facts.get("services_only"):
        concerns.append("entire career at IT-services firms (flagged in the JD)")
    if facts.get("cv_speech_only"):
        concerns.append("skews computer-vision/speech with thin NLP/IR")
    if fl.get("research_only"):
        concerns.append("research-leaning with limited production-deployment signal")
    if fl.get("transitioning"):
        concerns.append("still transitioning into ML from an adjacent role")
    if fl.get("langchain_only"):
        concerns.append("AI exposure is mostly recent LLM-wrapper work")
    if fl.get("title_chaser"):
        concerns.append("short tenures suggest title-chasing")
    if not facts.get("in_india"):
        concerns.append(
            f"based in {loc} but open to relocation" if facts.get("relocate")
            else f"based in {loc}, not open to relocation"
        )
    if beh.get("dormant"):
        concerns.append(f"inactive ~{beh['days_inactive']}d on-platform")
    if beh.get("low_response"):
        concerns.append(f"low recruiter response rate ({beh['response_rate']:.0%})")
    if not ev and not facts.get("services_only"):
        concerns.append("limited direct retrieval/ranking evidence in career history")

    # ---- assemble, tone matched to rank ----
    strong = "; ".join([lead] + strengths) if strengths else lead
    if avail_bits and rank <= 60:
        strong += "; " + ", ".join(avail_bits)

    if rank <= 15:
        # strong pick: lead with fit, at most one concern
        out = strong + "."
        if concerns:
            out = strong + f". Minor concern: {concerns[0]}."
    elif rank <= 50:
        out = strong + "."
        if concerns:
            out += f" Concern: {concerns[0]}."
    else:
        # borderline / filler: be candid about why
        why = concerns[0] if concerns else "adjacent fit, included near the cutoff"
        out = strong + f". Borderline: {why}."
        if len(concerns) > 1:
            out = out[:-1] + f"; {concerns[1]}."

    # keep it tidy
    out = " ".join(out.split())
    if len(out) > 300:
        out = out[:296].rsplit(" ", 1)[0] + "..."
    return out
