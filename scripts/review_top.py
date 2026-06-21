"""Quality review of a submission: join the ranked CSV back to candidate data and
print a compact table + summary, to sanity-check the shortlist by eye.

    python scripts/review_top.py --candidates ./candidates.jsonl --submission ./submission.csv
"""
from __future__ import annotations
import argparse, csv, os, sys, datetime as dt
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from redrob_ranker import loader, textbuild, structured, behavioral, honeypot, config

REF = dt.date.fromisoformat(config.REFERENCE_DATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission", required=True)
    a = ap.parse_args()

    order = {}
    with open(a.submission, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            order[row["candidate_id"]] = (int(row["rank"]), row["score"])
    recs = {}
    for c in loader.iter_candidates(a.candidates):
        if c["candidate_id"] in order:
            recs[c["candidate_id"]] = c
            if len(recs) == len(order):
                break

    rows = []
    titles = Counter()
    n_india = n_otw = n_serv = n_cv = n_research = n_hp = 0
    for cid, (rank, score) in order.items():
        c = recs[cid]
        f = structured.score_structured(c, textbuild.full_profile_blob(c))
        b = behavioral.availability(c, REF)
        hp = honeypot.detect(c, REF)["is_honeypot"]
        titles[f["current_title"]] += 1
        n_india += f["in_india"]; n_otw += b["open_to_work"]
        n_serv += f["services_only"]; n_cv += f["cv_speech_only"]
        n_research += f["flags"]["research_only"]; n_hp += hp
        flags = []
        if f["services_only"]: flags.append("SERVICES")
        if f["cv_speech_only"]: flags.append("CV/SPEECH")
        if f["flags"]["research_only"]: flags.append("RESEARCH")
        if not f["in_india"]: flags.append("non-IN" + ("+reloc" if f["relocate"] else ""))
        if b["dormant"]: flags.append("DORMANT")
        if b["low_response"]: flags.append("LOWRESP")
        if hp: flags.append("HONEYPOT")
        rows.append((rank, score, f["current_title"], (f["location_str"] or "?").split(",")[0],
                     len(f["evidence_cats"]), ",".join(flags)))

    rows.sort()
    print(f"{'#':>3} {'score':>7}  {'title':28} {'city':14} ev  flags")
    for rank, score, title, city, ev, flags in rows:
        print(f"{rank:>3} {score:>7}  {title[:28]:28} {city[:14]:14} {ev}/5  {flags}")

    n = len(rows)
    print(f"\n=== top-{n} summary ===")
    print(f"in_india={n_india}  open_to_work={n_otw}  services_only={n_serv}  "
          f"cv_speech_only={n_cv}  research_only={n_research}  HONEYPOTS={n_hp}")
    print("title distribution:", dict(titles.most_common()))


if __name__ == "__main__":
    main()
