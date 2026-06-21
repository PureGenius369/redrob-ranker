#!/usr/bin/env python3
"""
rank.py — produce the top-100 submission CSV from candidates.jsonl.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

CPU-only, no network, well under the 5-minute / 16 GB budget. Two passes:
  Pass 1 streams the full pool and computes scalar scores (structured fit,
         behavioral modifier, honeypot gate) + the text for the semantic layer.
  Pass 2 re-reads only the selected top-N to compute full facts and write
         grounded reasoning. This keeps peak memory low.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from redrob_ranker import (  # noqa: E402
    behavioral,
    config,
    embed,
    honeypot,
    loader,
    reasoning,
    scoring,
    structured,
    textbuild,
)


def parse_args():
    p = argparse.ArgumentParser(description="Redrob top-100 candidate ranker")
    p.add_argument("--candidates", required=True, help="path to candidates.jsonl[.gz]")
    p.add_argument("--out", default="submission.csv", help="output CSV path")
    p.add_argument("--top", type=int, default=100, help="how many to rank")
    p.add_argument("--embed", default="auto", choices=["auto", "lsa", "model2vec", "precomputed"],
                   help="semantic backend. 'auto' uses precomputed embeddings if present "
                        "(fastest, fully offline), else vendored model2vec, else LSA.")
    p.add_argument("--emb-file", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "artifacts", "cand_embeddings.npz"),
        help="precomputed embeddings artifact (see scripts/precompute_embeddings.py)")
    p.add_argument("--ref-date", default=config.REFERENCE_DATE,
                   help="reference 'today' for recency/date math (YYYY-MM-DD)")
    p.add_argument("--limit", type=int, default=0, help="only read first N candidates (debug)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ref = dt.date.fromisoformat(args.ref_date)
    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
    t0 = time.time()

    # Use precomputed embeddings when available (fastest, fully offline ranking).
    precomp = None
    if args.embed in ("auto", "precomputed") and os.path.exists(args.emb_file):
        precomp = embed.load_precomputed(args.emb_file)
        log(f"[embed] using precomputed embeddings from {args.emb_file}")
    elif args.embed == "precomputed":
        log(f"[embed] WARNING: {args.emb_file} not found; encoding inline instead")

    # ---------------- Pass 1: scalar scoring over the full pool ----------------
    cids, embed_texts = [], []
    struct_fit, beh_mult, hp_flag = [], [], []
    n_honeypot = 0
    for cand in loader.iter_candidates(args.candidates):
        blob = textbuild.full_profile_blob(cand)
        st = structured.score_structured(cand, blob)
        bh = behavioral.availability(cand, ref)
        hp = honeypot.detect(cand, ref)

        cids.append(cand["candidate_id"])
        if precomp is None:
            embed_texts.append(textbuild.build_embedding_text(cand))
        struct_fit.append(st["structured_fit"])
        beh_mult.append(bh["modifier"])
        hp_flag.append(hp["is_honeypot"])
        n_honeypot += int(hp["is_honeypot"])

        if args.limit and len(cids) >= args.limit:
            break
        if not args.quiet and len(cids) % 20000 == 0:
            log(f"  scored {len(cids)} ... ({time.time()-t0:.0f}s)")

    log(f"[pass1] {len(cids)} candidates scored, {n_honeypot} honeypots gated "
        f"({time.time()-t0:.0f}s)")

    # ---------------- Semantic layer ----------------
    if precomp is not None:
        semantic = embed.semantic_from_precomputed(cids, precomp)
    else:
        semantic = embed.compute_semantic_scores(
            embed_texts, config.JD_POSITIVE_ANCHOR, config.JD_NEGATIVE_ANCHOR,
            backend=("auto" if args.embed in ("auto", "precomputed") else args.embed),
            verbose=not args.quiet,
        )
    log(f"[semantic] computed ({time.time()-t0:.0f}s)")
    del embed_texts  # free memory before pass 2

    # ---------------- Combine + select ----------------
    final = scoring.combine(struct_fit, semantic, beh_mult, hp_flag)
    ranked = scoring.select_and_order(cids, final, top=args.top)
    top_ids = {cid for _, cid, _ in ranked}
    log(f"[select] top {len(ranked)} chosen ({time.time()-t0:.0f}s)")

    # ---------------- Pass 2: full facts + reasoning for the top-N ----------------
    facts_by_id = {}
    for cand in loader.iter_candidates(args.candidates):
        cid = cand["candidate_id"]
        if cid not in top_ids:
            continue
        blob = textbuild.full_profile_blob(cand)
        st = structured.score_structured(cand, blob)
        st["candidate_id"] = cid
        bh = behavioral.availability(cand, ref)
        facts_by_id[cid] = (st, bh)
        if len(facts_by_id) == len(top_ids):
            break

    # ---------------- Write CSV ----------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_pos, (_, cid, disp) in enumerate(ranked, start=1):
            st, bh = facts_by_id[cid]
            reason = reasoning.build_reasoning(st, bh, rank_pos)
            w.writerow([cid, rank_pos, f"{disp:.6f}", reason])

    log(f"[done] wrote {args.out} ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
