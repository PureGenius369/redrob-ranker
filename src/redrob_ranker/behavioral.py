"""
behavioral.py — the availability modifier.

The JD is explicit: "a perfect-on-paper candidate who hasn't logged in for 6
months and has a 5% recruiter response rate is, for hiring purposes, not
actually available. Down-weight them appropriately."

So behavioral signals don't add fit — they *modulate* it. We compute an
availability index in [0, 1] from the 23 redrob signals and map it to a bounded
multiplier (config.BEHAVIOR_MIN_MULT .. BEHAVIOR_MAX_MULT). Bounded on purpose:
a strong-fit candidate who's a bit slow to reply shouldn't be buried, but a
dormant/unresponsive one is meaningfully discounted.

Thresholds are calibrated to the observed pool: response_rate p10/p50/p90 =
0.14/0.44/0.73; everyone is >=45 days inactive (p50=130, p90=231 days).
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, Optional

from . import config as C


def _pdate(s: Optional[str]) -> Optional[dt.date]:
    if not s:
        return None
    try:
        return dt.date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def availability(cand: Dict[str, Any], ref: dt.date) -> Dict[str, Any]:
    s = cand.get("redrob_signals") or {}

    rr = s.get("recruiter_response_rate")
    rr = 0.3 if rr is None else rr
    f_response = _clip01(rr / 0.70)  # 0.73 -> ~1.0

    la = _pdate(s.get("last_active_date"))
    days_inactive = (ref - la).days if la else 240
    f_recency = _clip01((240 - days_inactive) / (240 - 45))  # 45d fresh, 240d cold

    f_otw = 1.0 if s.get("open_to_work_flag") else 0.40

    demand_raw = (
        (s.get("saved_by_recruiters_30d") or 0)
        + 0.2 * (s.get("search_appearance_30d") or 0)
        + 0.3 * (s.get("profile_views_received_30d") or 0)
    )
    f_demand = _clip01(math.log1p(demand_raw) / math.log1p(60))

    icr = s.get("interview_completion_rate")
    f_interview = 0.6 if icr is None else _clip01(icr)

    notice = s.get("notice_period_days")
    notice = 60 if notice is None else notice
    f_notice = _clip01((90 - notice) / (90 - 30)) if notice > 30 else 1.0
    f_notice = max(0.15, f_notice)

    f_complete = _clip01((s.get("profile_completeness_score") or 50) / 100.0)

    verified = sum(
        bool(s.get(k)) for k in ("verified_email", "verified_phone", "linkedin_connected")
    ) / 3.0

    gh = s.get("github_activity_score")
    f_github = 0.40 if (gh is None or gh < 0) else _clip01(gh / 60.0)

    avail = (
        0.22 * f_response
        + 0.22 * f_recency
        + 0.12 * f_otw
        + 0.10 * f_demand
        + 0.08 * f_interview
        + 0.08 * f_notice
        + 0.06 * f_complete
        + 0.06 * verified
        + 0.06 * f_github
    )

    mult = C.BEHAVIOR_MIN_MULT + (C.BEHAVIOR_MAX_MULT - C.BEHAVIOR_MIN_MULT) * avail

    return {
        "modifier": mult,
        "avail_index": avail,
        "days_inactive": days_inactive,
        "response_rate": rr,
        "open_to_work": bool(s.get("open_to_work_flag")),
        "notice_period_days": notice,
        "github_activity_score": gh,
        "dormant": days_inactive >= 200,
        "low_response": rr < 0.18,
        "short_notice": notice <= 30,
    }
