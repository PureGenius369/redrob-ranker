"""cProfile the pass-1 hot path to find the real bottleneck."""
import sys, os, cProfile, pstats, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from redrob_ranker import loader, textbuild, structured, behavioral, honeypot, config
import datetime as dt

REF = dt.date.fromisoformat(config.REFERENCE_DATE)
PATH = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8000


def work():
    k = 0
    for cand in loader.iter_candidates(PATH):
        blob = textbuild.full_profile_blob(cand)
        structured.score_structured(cand, blob)
        behavioral.availability(cand, REF)
        honeypot.detect(cand, REF)
        textbuild.build_embedding_text(cand)
        k += 1
        if k >= N:
            break


pr = cProfile.Profile()
pr.enable()
work()
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
ps.print_stats(18)
print(s.getvalue())
print(f"(profiled {N} candidates)")
