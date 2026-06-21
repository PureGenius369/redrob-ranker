"""
Exploratory data analysis over the full candidate pool.
Streams candidates.jsonl, prints compact summaries used to calibrate the ranker:
title/skill/location vocabularies, behavioral distributions, and the
date/skill impossibility patterns that identify honeypots.

Usage:
    python scripts/eda.py <path-to-candidates.jsonl[.gz]>
"""
from __future__ import annotations
import sys, os, datetime as dt
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from redrob_ranker.loader import iter_candidates  # noqa: E402

REF = dt.date(2026, 6, 21)


def pdate(s):
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s[:10])
    except Exception:
        return None


def months_between(a, b):
    if not a or not b:
        return None
    return (b.year - a.year) * 12 + (b.month - a.month)


def main(path):
    n = 0
    id_min, id_max = None, None
    titles = Counter()
    countries = Counter()
    cities = Counter()
    skills = Counter()
    prof = Counter()
    yoe_buckets = Counter()
    # honeypot signals
    expert_zero_dur = Counter()         # # skills with adv/expert prof but 0 months, per cand
    n_overclaim_tenure = 0              # role claims more months than elapsed since start
    n_sum_dur_gt_yoe = 0               # sum(role months) >> yoe*12
    n_end_before_start = 0
    n_dur_mismatch = 0                 # duration_months vs computed (end-start)
    n_future_start = 0
    n_skill_expert0_ge5 = 0           # >=5 adv/expert skills with 0 duration
    # behavioral
    resp = []
    last_active_days = []
    open_to_work = 0
    github_linked = 0
    n_signals = 0
    skills_per = []
    examples_honeypot = []

    for c in iter_candidates(path):
        n += 1
        cid = c.get("candidate_id", "")
        if id_min is None or cid < id_min:
            id_min = cid
        if id_max is None or cid > id_max:
            id_max = cid
        p = c.get("profile", {}) or {}
        titles[(p.get("current_title") or "").strip()] += 1
        countries[(p.get("country") or "").strip()] += 1
        cities[(p.get("location") or "").strip().split(",")[0].lower()] += 1
        yoe = p.get("years_of_experience") or 0
        yoe_buckets[int(yoe // 2) * 2] += 1

        sk = c.get("skills") or []
        skills_per.append(len(sk))
        ez = 0
        for s in sk:
            nm = (s.get("name") or "").strip()
            skills[nm.lower()] += 1
            pr = s.get("proficiency", "")
            prof[pr] += 1
            if pr in ("advanced", "expert") and (s.get("duration_months") or 0) == 0:
                ez += 1
        expert_zero_dur[ez] += 1
        if ez >= 5:
            n_skill_expert0_ge5 += 1

        # date / tenure consistency
        sum_dur = 0
        bad = False
        for r in c.get("career_history") or []:
            d = r.get("duration_months") or 0
            sum_dur += d
            sdt, edt = pdate(r.get("start_date")), pdate(r.get("end_date"))
            if sdt and sdt > REF:
                n_future_start += 1
                bad = True
            if sdt and edt:
                if edt < sdt:
                    n_end_before_start += 1
                    bad = True
                comp = months_between(sdt, edt)
                if comp is not None and abs(comp - d) > 6:
                    n_dur_mismatch += 1
            if r.get("is_current") and sdt:
                elapsed = months_between(sdt, REF)
                if elapsed is not None and d > elapsed + 4:
                    n_overclaim_tenure += 1
                    bad = True
        if yoe and sum_dur > yoe * 12 + 18:
            n_sum_dur_gt_yoe += 1
            bad = True
        if bad and len(examples_honeypot) < 6:
            examples_honeypot.append(cid)

        sig = c.get("redrob_signals") or {}
        if sig:
            n_signals += 1
            rr = sig.get("recruiter_response_rate")
            if rr is not None:
                resp.append(rr)
            la = pdate(sig.get("last_active_date"))
            if la:
                last_active_days.append((REF - la).days)
            if sig.get("open_to_work_flag"):
                open_to_work += 1
            if (sig.get("github_activity_score") or -1) >= 0:
                github_linked += 1

    import numpy as np
    print(f"TOTAL CANDIDATES: {n}")
    print(f"ID range: {id_min} .. {id_max}")
    print(f"\n--- current_title top 45 ---")
    for t, c in titles.most_common(45):
        print(f"  {c:6d}  {t}")
    print(f"\n--- country top 15 ---")
    for t, c in countries.most_common(15):
        print(f"  {c:6d}  {t}")
    print(f"\n--- city top 25 ---")
    for t, c in cities.most_common(25):
        print(f"  {c:6d}  {t}")
    print(f"\n--- years_of_experience buckets ---")
    for k in sorted(yoe_buckets):
        print(f"  {k:>3}-{k+1}: {yoe_buckets[k]}")
    print(f"\n--- skills top 70 ---")
    for t, c in skills.most_common(70):
        print(f"  {c:6d}  {t}")
    print(f"\n--- proficiency dist ---  {dict(prof)}")
    print(f"skills per candidate: mean={np.mean(skills_per):.1f} p50={np.percentile(skills_per,50):.0f} p95={np.percentile(skills_per,95):.0f} max={max(skills_per)}")

    print(f"\n=== HONEYPOT / IMPOSSIBILITY SIGNALS ===")
    print(f"  candidates w/ >=5 adv/expert skills at 0 months: {n_skill_expert0_ge5}")
    print(f"  expert_zero_dur per-cand dist (count->#cands), top: {expert_zero_dur.most_common(8)}")
    print(f"  current-role overclaims tenure (> elapsed since start): {n_overclaim_tenure}")
    print(f"  sum(role months) > yoe*12+18: {n_sum_dur_gt_yoe}")
    print(f"  end_date < start_date: {n_end_before_start}")
    print(f"  duration_months vs (end-start) mismatch >6mo: {n_dur_mismatch}")
    print(f"  future start_date: {n_future_start}")
    print(f"  example suspicious ids: {examples_honeypot}")

    print(f"\n=== BEHAVIORAL ===")
    print(f"  has redrob_signals: {n_signals}/{n}")
    print(f"  recruiter_response_rate: mean={np.mean(resp):.2f} p10={np.percentile(resp,10):.2f} p50={np.percentile(resp,50):.2f} p90={np.percentile(resp,90):.2f}")
    print(f"  days since last_active: p10={np.percentile(last_active_days,10):.0f} p50={np.percentile(last_active_days,50):.0f} p90={np.percentile(last_active_days,90):.0f} max={max(last_active_days)}")
    print(f"  open_to_work: {open_to_work} ({100*open_to_work/n:.0f}%)   github_linked: {github_linked} ({100*github_linked/n:.0f}%)")


if __name__ == "__main__":
    main(sys.argv[1])
