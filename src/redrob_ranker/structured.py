"""
structured.py — the JD-logic layer.

Turns each candidate into interpretable sub-scores in [0, 1] plus a bag of
derived facts (used by reasoning.py). This is where the dataset's traps are
actually defeated:

  * keyword stuffers  -> low role_title (their real title is non-tech) and low
                         skill_trust (their AI "skills" are untrusted claims).
  * plain-language T5  -> high career_evidence + semantics even with an adjacent
                         title, because evidence is read from descriptions.
  * CV/speech-only     -> domain penalty.
  * research-only      -> research_only flag penalty.
  * services-only      -> product_vs_services penalty.
  * langchain-wrappers -> langchain_only flag penalty.

Performance notes:
  * Free-text evidence is scanned with Python's C-level substring search (fast,
    stemming-friendly: token "rank" matches "ranking").
  * Skill tags are matched by EXACT set membership (skills are clean canonical
    tags), which is O(1) and avoids substring traps like "ml" in "html".
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from . import config as C
from .textbuild import norm

_PROF_W = {"beginner": 0.30, "intermediate": 0.55, "advanced": 0.85, "expert": 1.0}

# Exact-membership sets for skill tags.
_AI_RELEVANT_SKILLS = frozenset(
    C.RETRIEVAL_RANKING_SKILLS | C.VECTOR_DB_SKILLS | C.EMBEDDING_SKILLS
    | C.NLP_SKILLS | C.LLM_FINETUNE_SKILLS | C.CORE_ML_SKILLS
    | {"hugging face transformers", "mlops", "recommendation systems"}
)
_NLP_SKILLS = frozenset(C.NLP_SKILLS | {"hugging face transformers"})
_CV_SKILLS = frozenset(C.CV_SPEECH_ROBOTICS_SKILLS)

_TRANSITION_PHRASES = (
    "transition", "transitioning", "pivot", "building competence",
    "looking to move into", "interested in transitioning", "break into",
    "learning modern ml", "upskilling", "self-directed ml", "curious about how ai",
    "experimented with chatgpt", "exploring ai",
)

_CV_BLOB_TOKENS = ("computer vision", "image classification", "speech recognition",
                   "tts", " asr", "robotics", "object detection", "segmentation",
                   "image generation")


def _any(blob: str, toks) -> bool:
    return any(t in blob for t in toks)


def _count(blob: str, toks) -> int:
    return sum(1 for t in toks if t in blob)


# --------------------------------------------------------------------------- #
# Title coherence
# --------------------------------------------------------------------------- #
_TITLE_CACHE: Dict[str, float] = {}


def _title_value(title: str) -> float:
    cached = _TITLE_CACHE.get(title)
    if cached is not None:
        return cached
    t = norm(title)
    if not t:
        _TITLE_CACHE[title] = 0.35
        return 0.35
    if any(k in t for k in C.TITLE_BULLSEYE):
        v = 1.0
    elif any(k in t for k in C.TITLE_CORE_AI):
        v = 0.85
    elif any(k in t for k in C.TITLE_CAUTION_AI):
        v = 0.62
    elif any(k in t for k in C.TITLE_NON_TECH):
        v = 0.06
    elif any(k in t for k in C.TITLE_ADJACENT):
        v = 0.50
    else:
        v = 0.35
    if any(k in t for k in ("junior", "intern", "trainee", "associate")):
        v *= 0.72
    _TITLE_CACHE[title] = v
    return v


def _role_title_score(cand: Dict[str, Any]) -> Tuple[float, str, List[str]]:
    profile = cand.get("profile") or {}
    cur = profile.get("current_title") or ""
    cur_v = _title_value(cur)

    hist = cand.get("career_history") or []
    hist_vals = [_title_value(r.get("title") or "") for r in hist[:3]]
    best_recent = max(hist_vals) if hist_vals else cur_v

    all_vals = [_title_value(r.get("title") or "") for r in hist] or [cur_v]
    frac_ai = sum(1 for v in all_vals if v >= 0.62) / len(all_vals)

    score = 0.60 * cur_v + 0.25 * best_recent + 0.15 * frac_ai
    titles = [cur] + [r.get("title", "") for r in hist[:3]]
    return min(1.0, score), cur, titles


# --------------------------------------------------------------------------- #
# Career evidence (read from descriptions + summary, never from skill tags)
# --------------------------------------------------------------------------- #
# Each free-text category is scanned exactly once per candidate, then reused by
# career-evidence, domain, and the disqualifier flags.
def _scan_presence(blob: str) -> Dict[str, bool]:
    return {
        "retrieval": _any(blob, C.EVIDENCE_RETRIEVAL),
        "ranking/recsys": _any(blob, C.EVIDENCE_RANKING_RECSYS),
        "production": _any(blob, C.EVIDENCE_PRODUCTION),
        "evaluation": _any(blob, C.EVIDENCE_EVAL),
        "nlp/ir": _any(blob, C.EVIDENCE_NLP_IR),
        "research": _any(blob, C.EVIDENCE_RESEARCH_ONLY),
        "langchain": _any(blob, C.EVIDENCE_LANGCHAIN_WRAPPER),
        "core_ml": _any(blob, C.CORE_ML_SKILLS),
    }


_EVIDENCE_WEIGHTS = [
    ("retrieval", 0.30), ("ranking/recsys", 0.25), ("production", 0.20),
    ("evaluation", 0.15), ("nlp/ir", 0.10),
]


def _career_evidence(present: Dict[str, bool]) -> Tuple[float, List[str]]:
    score = 0.0
    cats: List[str] = []
    for label, w in _EVIDENCE_WEIGHTS:
        if present[label]:
            score += w
            cats.append(label)
    return min(1.0, score), cats


# --------------------------------------------------------------------------- #
# Experience band: ideal 6-8, soft 5-9
# --------------------------------------------------------------------------- #
def _experience_score(yoe: float) -> float:
    if yoe is None:
        return 0.5
    base = math.exp(-((yoe - 7.0) / 4.0) ** 2)
    if C.EXP_SOFT_LOW <= yoe <= C.EXP_SOFT_HIGH:
        base = max(base, 0.88)
    if C.EXP_IDEAL_LOW <= yoe <= C.EXP_IDEAL_HIGH:
        base = max(base, 0.97)
    return float(min(1.0, base))


# --------------------------------------------------------------------------- #
# Skill trust (anti-stuffing) + AI skill inventory for reasoning
# --------------------------------------------------------------------------- #
def _skill_trust(cand: Dict[str, Any]) -> Tuple[float, int, List[str], int]:
    sig = cand.get("redrob_signals") or {}
    assess = {norm(k): v for k, v in (sig.get("skill_assessment_scores") or {}).items()}

    trusted_mass = 0.0
    n_ai_claimed = 0
    cv_count = 0
    top_trusted: List[Tuple[float, str]] = []

    for s in cand.get("skills") or []:
        name = norm(s.get("name"))
        if not name:
            continue
        if name in _CV_SKILLS:
            cv_count += 1
        if name not in _AI_RELEVANT_SKILLS:
            continue
        n_ai_claimed += 1
        prof = _PROF_W.get(s.get("proficiency", ""), 0.4)
        endo = min(1.0, (s.get("endorsements") or 0) / 20.0)
        dur = min(1.0, (s.get("duration_months") or 0) / 36.0)
        trust = prof * (0.5 * endo + 0.5 * dur)
        if name in assess:
            trust = max(trust, assess[name] / 100.0)
        trusted_mass += trust
        top_trusted.append((trust, s.get("name")))

    top_trusted.sort(reverse=True)
    names = [n for _, n in top_trusted[:4]]
    score = min(1.0, trusted_mass / 2.5)
    return score, n_ai_claimed, names, cv_count


# --------------------------------------------------------------------------- #
# Domain (NLP/IR positive; CV/speech-only negative)
# --------------------------------------------------------------------------- #
def _domain_score(blob, present, cv_count: int, has_nlp_skill: bool) -> Tuple[float, bool, bool]:
    nlp_ir_present = (
        has_nlp_skill or present["nlp/ir"] or present["retrieval"] or present["ranking/recsys"]
    )
    cv_heavy = _count(blob, _CV_BLOB_TOKENS) >= 2 or cv_count >= 3
    cv_speech_only = cv_heavy and not nlp_ir_present
    if nlp_ir_present:
        score = 1.0
    elif cv_speech_only:
        score = 0.15
    else:
        score = 0.5
    return score, nlp_ir_present, cv_speech_only


# --------------------------------------------------------------------------- #
# Product vs services
# --------------------------------------------------------------------------- #
_COMPANY_CACHE: Dict[str, Tuple[bool, bool]] = {}


def _company_flags(c: str) -> Tuple[bool, bool]:
    """(is_services_firm, is_product_company) for a normalized company name, memoized."""
    v = _COMPANY_CACHE.get(c)
    if v is None:
        v = (any(f in c for f in C.CONSULTING_FIRMS),
             any(p in c for p in C.PRODUCT_COMPANY_HINTS))
        _COMPANY_CACHE[c] = v
    return v


def _product_services(cand: Dict[str, Any]) -> Tuple[float, bool, bool]:
    comps = [norm(r.get("company")) for r in (cand.get("career_history") or []) if r.get("company")]
    comps.append(norm((cand.get("profile") or {}).get("current_company")))
    comps = [c for c in comps if c]
    if not comps:
        return 0.6, False, False
    flags = [_company_flags(c) for c in comps]
    is_serv = [f[0] for f in flags]
    has_prod = any(f[1] for f in flags)
    services_only = all(is_serv) and any(is_serv)
    if has_prod:
        return 1.0, services_only, has_prod
    if services_only:
        return 0.20, True, False
    return 0.62, False, False


# --------------------------------------------------------------------------- #
# Location
# --------------------------------------------------------------------------- #
def _location_score(cand: Dict[str, Any]) -> Tuple[float, str, bool, bool]:
    profile = cand.get("profile") or {}
    loc = norm(profile.get("location"))
    country = norm(profile.get("country"))
    city = loc.split(",")[0].strip()
    sig = cand.get("redrob_signals") or {}
    relocate = bool(sig.get("willing_to_relocate"))
    in_india = ("india" in country) or any(m in loc for m in C.INDIA_MARKERS)

    if city in C.PREFERRED_CITIES:
        return 1.0, profile.get("location", ""), in_india, relocate
    if any(c in city for c in C.INDIA_TIER1_CITIES) or (in_india and city):
        return 0.90, profile.get("location", ""), in_india, relocate
    if in_india:
        return 0.78, profile.get("location", ""), in_india, relocate
    return (0.55 if relocate else 0.28), profile.get("location", ""), in_india, relocate


# --------------------------------------------------------------------------- #
# Top-level structured scorer
# --------------------------------------------------------------------------- #
def score_structured(cand: Dict[str, Any], blob: str) -> Dict[str, Any]:
    profile = cand.get("profile") or {}
    yoe = profile.get("years_of_experience")

    present = _scan_presence(blob)
    role_title, current_title, titles = _role_title_score(cand)
    career_evidence, evidence_cats = _career_evidence(present)
    experience = _experience_score(yoe)
    skill_trust, n_ai_claimed, top_skills, cv_count = _skill_trust(cand)
    has_nlp_skill = any(
        norm(s.get("name")) in _NLP_SKILLS for s in (cand.get("skills") or [])
    )
    domain, nlp_ir_present, cv_speech_only = _domain_score(blob, present, cv_count, has_nlp_skill)
    product_services, services_only, has_product = _product_services(cand)
    location, loc_str, in_india, relocate = _location_score(cand)

    w = C.STRUCT_WEIGHTS
    base = (
        w["role_title"] * role_title
        + w["career_evidence"] * career_evidence
        + w["experience"] * experience
        + w["skill_trust"] * skill_trust
        + w["domain_nlp_ir"] * domain
        + w["product_vs_services"] * product_services
        + w["location"] * location
    ) / sum(w.values())

    # disqualifier flags (JD's explicit "do NOT want" list)
    summary = norm(profile.get("summary"))
    research_only = present["research"] and not present["production"]
    langchain_only = present["langchain"] and not (
        present["retrieval"] or present["ranking/recsys"] or present["core_ml"]
    )
    roles = cand.get("career_history") or []
    avg_tenure = (sum(r.get("duration_months") or 0 for r in roles) / len(roles)) if roles else 99
    title_chaser = len(roles) >= 4 and avg_tenure < 18
    transitioning = _any(summary, _TRANSITION_PHRASES) and role_title < 0.62

    penalty = 1.0
    if research_only:
        penalty *= 0.55
    if langchain_only:
        penalty *= 0.82
    if title_chaser:
        penalty *= 0.85
    if transitioning:
        penalty *= 0.80

    structured_fit = base * penalty

    return {
        "structured_fit": structured_fit,
        "sub": {
            "role_title": role_title,
            "career_evidence": career_evidence,
            "experience": experience,
            "skill_trust": skill_trust,
            "domain": domain,
            "product_services": product_services,
            "location": location,
        },
        "current_title": current_title,
        "titles": titles,
        "yoe": yoe,
        "evidence_cats": evidence_cats,
        "top_skills": top_skills,
        "n_ai_claimed": n_ai_claimed,
        "cv_count": cv_count,
        "nlp_ir_present": nlp_ir_present,
        "cv_speech_only": cv_speech_only,
        "services_only": services_only,
        "has_product": has_product,
        "in_india": in_india,
        "relocate": relocate,
        "location_str": loc_str,
        "flags": {
            "research_only": research_only,
            "langchain_only": langchain_only,
            "title_chaser": title_chaser,
            "transitioning": transitioning,
        },
    }
