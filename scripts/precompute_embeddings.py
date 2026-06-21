"""
Pre-computation step (allowed to exceed the 5-minute ranking budget): embed the
full candidate pool once with the static model and save vectors + both JD anchors
to artifacts/cand_embeddings.npz. The ranking step then loads this and stays
fully offline and well within budget — mirroring a real system where embeddings
are indexed offline and ranking is a fast scoring pass.

    python scripts/precompute_embeddings.py --candidates ./candidates.jsonl
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from redrob_ranker import loader, textbuild, config, embed  # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "cand_embeddings.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()

    t0 = time.time()
    ids, texts = [], []
    for c in loader.iter_candidates(args.candidates):
        ids.append(c["candidate_id"])
        texts.append(textbuild.build_embedding_text(c))
    print(f"[precompute] built {len(texts)} texts ({time.time()-t0:.0f}s)")

    mp = embed._resolve_model_path(args.model_path)
    if mp is None:
        sys.exit("No model2vec model found. Run: python scripts/fetch_model.py")
    be = embed._Model2VecBackend(mp)
    vecs = be.encode(texts)                      # L2-normalized float32
    anchors = be.encode([config.JD_POSITIVE_ANCHOR, config.JD_NEGATIVE_ANCHOR])
    print(f"[precompute] encoded ({time.time()-t0:.0f}s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        ids=np.array(ids),
        vecs=vecs.astype(np.float16),            # ~100 MB for 100K x 512
        pos=anchors[0].astype(np.float16),
        neg=anchors[1].astype(np.float16),
    )
    print(f"[precompute] saved {args.out} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
