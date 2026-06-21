"""
scoring.py — blend the layers into one score and produce a spec-valid ordering.

    fit   = BLEND_STRUCTURED * structured_fit + BLEND_SEMANTIC * semantic
    score = fit * behavioral_modifier * (honeypot_gate if impossible else 1)

The ordering guarantees the validator's requirements: score is non-increasing
with rank, and ties are broken by candidate_id ascending.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from . import config as C


def combine(structured_fit, semantic, behavioral_mult, is_honeypot) -> np.ndarray:
    structured_fit = np.asarray(structured_fit, dtype=np.float64)
    semantic = np.asarray(semantic, dtype=np.float64)
    behavioral_mult = np.asarray(behavioral_mult, dtype=np.float64)
    is_honeypot = np.asarray(is_honeypot, dtype=bool)

    fit = C.BLEND_STRUCTURED * structured_fit + C.BLEND_SEMANTIC * semantic
    score = fit * behavioral_mult
    score = np.where(is_honeypot, score * C.HONEYPOT_GATE_MULT, score)
    return score


def select_and_order(
    candidate_ids: Sequence[str], scores: np.ndarray, top: int = 100
) -> List[Tuple[int, str, float]]:
    """
    Return [(orig_index, candidate_id, display_score)] of length `top`, ordered
    by rank. Selection is by precise score; output order and display score are
    rounded and re-sorted so equal scores are tie-broken by candidate_id asc.
    """
    n = len(candidate_ids)
    top = min(top, n)

    # select best `top` by precise score, tie-break cid asc
    order = sorted(range(n), key=lambda i: (-scores[i], candidate_ids[i]))
    chosen = order[:top]

    # normalize to a tidy [.., 0.99] display range (monotonic -> order preserved)
    smax = max(scores[i] for i in chosen) or 1.0
    disp = {i: round(float(scores[i] / smax) * 0.99, 6) for i in chosen}

    # re-sort by (rounded desc, cid asc) so the CSV satisfies the tie rule exactly
    chosen.sort(key=lambda i: (-disp[i], candidate_ids[i]))
    return [(i, candidate_ids[i], disp[i]) for i in chosen]
