"""
textbuild.py — turn a candidate record into the text we embed, plus small
normalization helpers shared across the scoring modules.

For the semantic layer we deliberately weight the *truth-bearing* fields
(summary + career descriptions + titles) over the *claim* field (skills), by
repeating the former. Skills are appended once, plainly, so genuine semantic
content still contributes but stuffed skill lists can't dominate the vector.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_WS = re.compile(r"\s+")


def norm(s: Any) -> str:
    """Lowercase + strip. Safe on None. (No regex — called ~25x/candidate.)"""
    if not s:
        return ""
    return str(s).strip().lower()


def candidate_skill_names(cand: Dict[str, Any]) -> List[str]:
    return [norm(s.get("name")) for s in (cand.get("skills") or []) if s.get("name")]


def career_text(cand: Dict[str, Any]) -> str:
    parts: List[str] = []
    for role in cand.get("career_history") or []:
        title = role.get("title") or ""
        desc = role.get("description") or ""
        comp = role.get("company") or ""
        parts.append(f"{title} at {comp}. {desc}")
    return " ".join(parts)


def build_embedding_text(cand: Dict[str, Any], max_chars: int = 600) -> str:
    """
    Compose a compact document to embed. Static embeddings are a mean of token
    vectors, so document *length* mostly adds tokenization cost, not signal — we
    therefore keep only the truth-bearing fields (title, headline, summary, and
    the two most recent role descriptions) and cap the length. The random skill
    tag-cloud is deliberately excluded so stuffed skills can't move the vector.
    """
    profile = cand.get("profile") or {}
    title = profile.get("current_title") or ""
    headline = profile.get("headline") or ""
    summary = profile.get("summary") or ""

    descs = " ".join(
        (r.get("title") or "") + ": " + (r.get("description") or "")
        for r in (cand.get("career_history") or [])[:2]
    )
    doc = f"{title}. {headline}. {summary} {descs}"
    return _WS.sub(" ", doc).strip()[:max_chars]


def full_profile_blob(cand: Dict[str, Any]) -> str:
    """Lowercased concatenation of all free text — used for evidence scanning."""
    profile = cand.get("profile") or {}
    chunks = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
        career_text(cand),
    ]
    text = " ".join(c for c in chunks if c)
    return " ".join(text.split()).lower()  # collapse whitespace fast, no regex
