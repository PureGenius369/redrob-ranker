"""
loader.py — stream candidates from JSONL (optionally gzipped) with low memory.

We never hold the raw 465 MB of JSON in memory as Python dicts longer than we
must. The pipeline streams candidates, extracts a compact feature record per
candidate, and keeps only those compact records + embeddings. This is what
keeps us comfortably inside the 16 GB budget on the full 100K pool.
"""

from __future__ import annotations

import gzip
import io
import json
from typing import Iterator, Dict, Any

try:  # orjson is ~3x faster to parse; optional.
    import orjson  # type: ignore

    def _loads(b: bytes) -> Any:
        return orjson.loads(b)

    _ORJSON = True
except Exception:  # pragma: no cover - fallback
    def _loads(b: bytes) -> Any:
        return json.loads(b)

    _ORJSON = False


def _open_maybe_gzip(path: str):
    """Open a path as text, transparently handling .gz."""
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path: str) -> Iterator[Dict[str, Any]]:
    """Yield one parsed candidate dict per non-blank line of JSONL/JSONL.gz."""
    with _open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if _ORJSON:
                yield _loads(line.encode("utf-8") if isinstance(line, str) else line)
            else:
                yield json.loads(line)


def count_candidates(path: str) -> int:
    n = 0
    with _open_maybe_gzip(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n
