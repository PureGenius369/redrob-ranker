"""
honeypot.py — impossibility detection.

The spec seeds ~80 honeypots with "subtly impossible" profiles (e.g. 8 years at
a company founded 3 years ago; 'expert' in 10 skills with 0 months used) and
forces them to relevance tier 0. Ranking them in the top 100 risks
disqualification (honeypot rate > 10% in top 100).

We don't special-case specific IDs — we detect *internal contradictions* a real
profile cannot have. EDA confirmed the documented patterns are present and rare:
  - >=3 advanced/expert skills with 0 months of use
  - a current role claiming more tenure than time elapsed since it started
  - total role months far exceeding total years of experience
A profile failing any hard check is gated out of the rankable region.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional


def _pdate(s: Optional[str]) -> Optional[dt.date]:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def _months(a: dt.date, b: dt.date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def detect(cand: Dict[str, Any], ref: dt.date) -> Dict[str, Any]:
    reasons: List[str] = []
    profile = cand.get("profile") or {}
    yoe = profile.get("years_of_experience") or 0
    yoe_months = yoe * 12

    # 1. expert/advanced skills with zero months of use (the "'expert' in 10
    #    skills with 0 years used" pattern). Real profiles essentially never
    #    show >=3 of these; EDA found only ~21 candidates pool-wide.
    ez = 0
    for s in cand.get("skills") or []:
        dm = s.get("duration_months") or 0
        if s.get("proficiency") in ("advanced", "expert") and dm == 0:
            ez += 1
    if ez >= 3:
        reasons.append(f"{ez} advanced/expert skills claimed with 0 months of use")

    # 2. tenure consistency (the "8 years at a company founded 3 years ago"
    #    pattern shows up as a current role claiming more months than have
    #    elapsed since it began, or roles summing past the stated career).
    sum_dur = 0
    for r in cand.get("career_history") or []:
        d = r.get("duration_months") or 0
        sum_dur += d
        sdt, edt = _pdate(r.get("start_date")), _pdate(r.get("end_date"))
        if sdt and sdt > ref:
            reasons.append("a role starts in the future")
        if sdt and edt and edt < sdt:
            reasons.append("a role ends before it starts")
        if r.get("is_current") and sdt:
            elapsed = _months(sdt, ref)
            if d > elapsed + 4:
                reasons.append(
                    f"current role claims {d} months but only {elapsed} have elapsed since it began"
                )

    # Aggregate role months can't far exceed total stated experience (allow a
    # generous 3-year slack for overlaps / rounding before calling it impossible).
    if yoe_months and sum_dur > yoe_months + 36:
        reasons.append(
            f"career roles sum to {sum_dur} months but stated experience is only {int(yoe_months)} months"
        )

    # NOTE: we deliberately do NOT flag on signup-vs-last-active ordering or on
    # expected-salary min>max. EDA shows ~19% of profiles have salary min>max and
    # the signup/active dates are independently sampled — both are dataset noise,
    # not the "subtly impossible" career/skill contradictions the honeypots use.
    is_honeypot = len(reasons) > 0
    return {"is_honeypot": is_honeypot, "reasons": reasons, "n_reasons": len(reasons)}
