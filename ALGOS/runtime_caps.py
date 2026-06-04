"""Small hot-loop runtime caps shared by LUCIDOTA algo organs.

These helpers are deliberately boring: clamp scalar knobs, cap text/JSON
surfaces, and reject oversized array-like payloads before prototype algos turn
into laptop-murder buttons.
"""
from __future__ import annotations

import json
from typing import Any

MAX_TEXT_CHARS = 20_000
MAX_JSON_CHARS = 8_000
MAX_LABELS = 64
MAX_SPANS = 256
MAX_ARRAY_ELEMS = 2_000_000
MAX_DB_ROWS = 1_000
MAX_HIDDEN = 512
MAX_DIM = 1024
MAX_GA_DIM = 32
MAX_FLUID_SLOTS = 256
MAX_SUBSTEPS = 512
MAX_EVIDENCE_REFS = 32
MAX_REF_CHARS = 256
MAX_RATIONALE = 512


def clamp_int(x: Any, lo: int, hi: int, name: str) -> int:
    try:
        value = int(x)
    except Exception as exc:  # noqa: BLE001 - preserve small dependency-free helper
        raise ValueError(f"{name} must be int") from exc
    if value < lo or value > hi:
        raise ValueError(f"{name}={value} outside [{lo},{hi}]")
    return value


def cap_text(s: Any, limit: int = MAX_TEXT_CHARS) -> tuple[str, bool]:
    text = "" if s is None else str(s)
    limit = clamp_int(limit, 0, max(MAX_TEXT_CHARS, int(limit)), "limit")
    return text[:limit], len(text) > limit


def bounded_payload(payload: Any, max_chars: int = MAX_JSON_CHARS) -> tuple[str, bool]:
    max_chars = clamp_int(max_chars, 0, max(MAX_JSON_CHARS, int(max_chars)), "max_chars")
    raw = json.dumps(payload or {}, sort_keys=True, default=str)
    return raw[:max_chars], len(raw) > max_chars


def assert_array_budget(*arrays: Any, max_elems: int = MAX_ARRAY_ELEMS) -> None:
    total = 0
    for arr in arrays:
        if arr is not None:
            total += int(getattr(arr, "size", 0) or 0)
    if total > max_elems:
        raise MemoryError(f"array budget exceeded: {total}>{max_elems}")
