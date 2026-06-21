"""Tally honeypot reason categories to calibrate the detector."""
import sys, os, datetime as dt
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from redrob_ranker.loader import iter_candidates
from redrob_ranker import honeypot

REF = dt.date(2026, 6, 21)
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
cats = Counter()
n = flagged = 0
for c in iter_candidates(sys.argv[1]):
    n += 1
    r = honeypot.detect(c, REF)
    if r["is_honeypot"]:
        flagged += 1
        for reason in r["reasons"]:
            key = reason.split(" but ")[0].split(" claims ")[0][:45]
            cats[key] += 1
    if n >= limit:
        break
print(f"flagged {flagged}/{n} ({100*flagged/n:.1f}%)")
for k, v in cats.most_common():
    print(f"  {v:6d}  {k}")
