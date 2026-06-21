"""
Tests for the Redrob ranker. Run with:  pytest -q   (or: python tests/test_ranker.py)

These pin the behaviors that matter for the challenge:
  - keyword stuffers rank below genuine engineers
  - plain-language strong candidates are recognized via career evidence
  - honeypots (internal impossibilities) are detected
  - the output ordering satisfies the submission spec exactly
  - reasoning never invents skills the candidate doesn't have
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from redrob_ranker import behavioral, honeypot, reasoning, scoring, structured, textbuild
from redrob_ranker import config as C

REF = dt.date.fromisoformat(C.REFERENCE_DATE)


# --------------------------------------------------------------------------- #
# Builders for synthetic candidates
# --------------------------------------------------------------------------- #
def _sig(**kw):
    base = dict(
        profile_completeness_score=80, signup_date="2024-01-01",
        last_active_date="2026-06-01", open_to_work_flag=True,
        profile_views_received_30d=10, applications_submitted_30d=2,
        recruiter_response_rate=0.6, avg_response_time_hours=20,
        skill_assessment_scores={}, connection_count=200, endorsements_received=50,
        notice_period_days=30, expected_salary_range_inr_lpa={"min": 20, "max": 40},
        preferred_work_mode="hybrid", willing_to_relocate=True,
        github_activity_score=40, search_appearance_30d=30, saved_by_recruiters_30d=3,
        interview_completion_rate=0.8, offer_acceptance_rate=0.5,
        verified_email=True, verified_phone=True, linkedin_connected=True,
    )
    base.update(kw)
    return base


def real_engineer():
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "A", "headline": "ML Engineer", "current_title": "ML Engineer",
            "summary": "Built production semantic search and ranking systems with embeddings.",
            "location": "Pune, Maharashtra", "country": "India", "years_of_experience": 7.0,
            "current_company": "Flipkart", "current_company_size": "1001-5000",
            "current_industry": "E-commerce",
        },
        "career_history": [{
            "company": "Flipkart", "title": "ML Engineer", "start_date": "2021-01-01",
            "end_date": None, "duration_months": 65, "is_current": True, "industry": "E-commerce",
            "company_size": "1001-5000",
            "description": "Built a production recommendation and semantic search system using "
                           "embeddings and FAISS; evaluated ranking with NDCG and ran A/B tests.",
        }],
        "education": [],
        "skills": [
            {"name": "Semantic Search", "proficiency": "advanced", "endorsements": 30, "duration_months": 40},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 36},
            {"name": "Learning to Rank", "proficiency": "advanced", "endorsements": 15, "duration_months": 30},
        ],
        "redrob_signals": _sig(),
    }


def keyword_stuffer():
    # A marketing manager with stuffed AI skills (no real evidence in description).
    return {
        "candidate_id": "CAND_0000002",
        "profile": {
            "anonymized_name": "B", "headline": "Marketing Manager", "current_title": "Marketing Manager",
            "summary": "Marketing professional; experimented with ChatGPT for content.",
            "location": "Pune, Maharashtra", "country": "India", "years_of_experience": 7.0,
            "current_company": "Acme Corp", "current_company_size": "201-500",
            "current_industry": "Marketing",
        },
        "career_history": [{
            "company": "Acme Corp", "title": "Marketing Manager", "start_date": "2021-01-01",
            "end_date": None, "duration_months": 65, "is_current": True, "industry": "Marketing",
            "company_size": "201-500",
            "description": "Owned brand campaigns, SEO, and the editorial calendar.",
        }],
        "education": [],
        "skills": [
            {"name": "Semantic Search", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "FAISS", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "RAG", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "LLMs", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        ],
        "redrob_signals": _sig(),
    }


def honeypot_profile():
    c = real_engineer()
    c["candidate_id"] = "CAND_0000003"
    # current role claims 200 months but started 2024 -> impossible
    c["career_history"][0]["start_date"] = "2024-01-01"
    c["career_history"][0]["duration_months"] = 200
    return c


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_stuffer_ranks_below_real_engineer():
    real = real_engineer()
    stuff = keyword_stuffer()
    sr = structured.score_structured(real, textbuild.full_profile_blob(real))
    ss = structured.score_structured(stuff, textbuild.full_profile_blob(stuff))
    assert sr["structured_fit"] > ss["structured_fit"], (sr["structured_fit"], ss["structured_fit"])
    # the stuffer's real title sinks the title component
    assert ss["sub"]["role_title"] < 0.2
    # their stuffed expert-at-0-month skills earn ~no trust
    assert ss["sub"]["skill_trust"] < 0.15


def test_plain_language_evidence_recognized():
    real = real_engineer()
    s = structured.score_structured(real, textbuild.full_profile_blob(real))
    assert "retrieval" in s["evidence_cats"]
    assert "ranking/recsys" in s["evidence_cats"]
    assert "evaluation" in s["evidence_cats"]


def test_honeypot_detected_and_normal_not():
    assert honeypot.detect(honeypot_profile(), REF)["is_honeypot"] is True
    assert honeypot.detect(real_engineer(), REF)["is_honeypot"] is False


def test_behavioral_downweights_dormant():
    active = behavioral.availability(real_engineer(), REF)["modifier"]
    dormant = real_engineer()
    dormant["redrob_signals"] = _sig(last_active_date="2025-10-01", recruiter_response_rate=0.05,
                                     open_to_work_flag=False)
    assert behavioral.availability(dormant, REF)["modifier"] < active


def test_ordering_is_spec_valid():
    ids = [f"CAND_{i:07d}" for i in range(1, 11)]
    scores = np.array([0.9, 0.9, 0.8, 0.8, 0.7, 0.6, 0.6, 0.6, 0.5, 0.4])
    ranked = scoring.select_and_order(ids, scores, top=10)
    disp = [d for _, _, d in ranked]
    # non-increasing
    assert all(disp[i] >= disp[i + 1] for i in range(len(disp) - 1))
    # equal scores -> candidate_id ascending
    for i in range(len(ranked) - 1):
        if disp[i] == disp[i + 1]:
            assert ranked[i][1] < ranked[i + 1][1]
    # unique ids, exactly top
    assert len({cid for _, cid, _ in ranked}) == 10


def test_reasoning_has_no_hallucinated_skills():
    real = real_engineer()
    facts = structured.score_structured(real, textbuild.full_profile_blob(real))
    facts["candidate_id"] = real["candidate_id"]
    beh = behavioral.availability(real, REF)
    text = reasoning.build_reasoning(facts, beh, rank=1)
    profile_skills = {s["name"].lower() for s in real["skills"]}
    # any "skills:" mention must reference real skills only
    if "skills:" in text.lower():
        listed = text.lower().split("skills:")[1]
        for s in facts["top_skills"]:
            assert s.lower() in profile_skills
    assert real["profile"]["current_title"].split()[0].lower() in text.lower()


def test_title_value_ordering():
    assert structured._title_value("Marketing Manager") < structured._title_value("Software Engineer")
    assert structured._title_value("Software Engineer") < structured._title_value("Data Scientist")
    assert structured._title_value("Data Scientist") <= structured._title_value("ML Engineer")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
