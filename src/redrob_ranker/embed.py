"""
embed.py — the semantic layer.

We score each candidate by a *contrastive* semantic signal:

        sem_raw = cos(candidate, JD_POSITIVE_ANCHOR)
                - cos(candidate, JD_NEGATIVE_ANCHOR)

Subtracting similarity to the explicitly-unwanted profile is what lets the
semantic layer help (not hurt) on the trap candidates: a stuffed
"Marketing Manager + AI skills" profile is close to BOTH anchors, so its
contrastive score is muted, while a plain-language engineer who *describes*
building search/ranking systems scores high on positive, low on negative.

Two interchangeable backends, both CPU-only and offline at ranking time:

  * model2vec  — distilled static embeddings (real transformer semantics at
                 hashing speed). Used if installed AND a local model is present
                 (vendored in artifacts/model). No network needed at run time.
  * lsa        — TF-IDF (1-2 grams) + TruncatedSVD. Pure scikit-learn, zero
                 downloads, deterministic. The safe default that always runs
                 inside the Stage-3 sandbox.

`--embed auto` (default) prefers model2vec when available, else LSA.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class _Model2VecBackend:
    name = "model2vec"

    def __init__(self, model_path: str):
        from model2vec import StaticModel  # imported lazily

        self.model = StaticModel.from_pretrained(model_path)

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, show_progress_bar=False)
        return _l2_normalize(np.asarray(vecs, dtype=np.float32))


class _LSABackend:
    """TF-IDF + TruncatedSVD fit on the candidate corpus (anchors included)."""

    name = "lsa"

    def __init__(self, corpus: List[str], n_components: int = 256, seed: int = 13):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.6,
            max_features=200_000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        X = self.vectorizer.fit_transform(corpus)
        n_components = min(n_components, max(2, min(X.shape) - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=seed)
        self._doc_vectors = _l2_normalize(self.svd.fit_transform(X).astype(np.float32))

    def doc_vectors(self) -> np.ndarray:
        return self._doc_vectors

    def encode(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return _l2_normalize(self.svd.transform(X).astype(np.float32))


def scale_contrastive(sem_raw: np.ndarray) -> np.ndarray:
    """Robustly scale a contrastive score vector to [0, 1] (2nd-98th pct clip)."""
    lo, hi = np.percentile(sem_raw, [2, 98])
    if hi <= lo:
        return np.full_like(sem_raw, 0.5)
    return np.clip((sem_raw - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def load_precomputed(path: str) -> dict:
    """Load a precomputed embedding artifact: candidate vectors + both JD anchors."""
    d = np.load(path, allow_pickle=False)
    id2idx = {str(cid): i for i, cid in enumerate(d["ids"])}
    return {"id2idx": id2idx, "vecs": d["vecs"], "pos": d["pos"], "neg": d["neg"]}


def semantic_from_precomputed(ordered_ids, precomp: dict) -> np.ndarray:
    """Contrastive semantic scores for candidates (in stream order) from a
    precomputed artifact — no model needed, so the ranking step stays offline."""
    id2idx = precomp["id2idx"]
    vecs = precomp["vecs"]
    idx = np.fromiter((id2idx[str(c)] for c in ordered_ids), dtype=np.int64,
                      count=len(ordered_ids))
    V = vecs[idx].astype(np.float32)
    pos = precomp["pos"].astype(np.float32)
    neg = precomp["neg"].astype(np.float32)
    return scale_contrastive(V @ pos - V @ neg)


def _resolve_model_path(model_path: Optional[str]) -> Optional[str]:
    """Find a vendored model2vec directory if one exists."""
    candidates = []
    if model_path:
        candidates.append(model_path)
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates.append(os.path.join(here, "artifacts", "model"))
    for c in candidates:
        if c and os.path.isdir(c) and os.listdir(c):
            return c
    # Allow a bare HF id only if offline mode is not forced (build-time use).
    if model_path and "/" in model_path and not os.environ.get("HF_HUB_OFFLINE"):
        return model_path
    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def compute_semantic_scores(
    cand_texts: List[str],
    pos_anchor: str,
    neg_anchor: str,
    backend: str = "auto",
    model_path: Optional[str] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Return a contrastive semantic score per candidate, robustly scaled to [0, 1]
    across the pool (2nd-98th percentile clipping to resist outliers).
    """
    chosen = None
    if backend in ("auto", "model2vec"):
        mp = _resolve_model_path(model_path)
        if mp is not None:
            try:
                be = _Model2VecBackend(mp)
                anchors = be.encode([pos_anchor, neg_anchor])
                docs = be.encode(cand_texts)
                chosen = (be, docs, anchors)
                if verbose:
                    print(f"[embed] using model2vec from {mp}")
            except Exception as e:  # pragma: no cover
                if backend == "model2vec":
                    raise
                if verbose:
                    print(f"[embed] model2vec unavailable ({e}); falling back to LSA")

    if chosen is None:
        n = len(cand_texts)
        be = _LSABackend(cand_texts + [pos_anchor, neg_anchor])
        all_docs = be.doc_vectors()           # candidates + 2 anchors, same SVD space
        docs = all_docs[:n]
        anchors = all_docs[n:n + 2]
        chosen = (be, docs, anchors)
        if verbose:
            print(f"[embed] using LSA (TF-IDF+SVD), dim={docs.shape[1]}")

    _, docs, anchors = chosen
    return scale_contrastive(docs @ anchors[0] - docs @ anchors[1])
