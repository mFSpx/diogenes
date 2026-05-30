# DARWIN HAMMER — match 3721, survivor 1
# gen: 4
# parent_a: hybrid_privacy_sketches_m15_s0.py (gen1)
# parent_b: hybrid_hybrid_gliner_zero_s_hybrid_krampus_brain_m30_s6.py (gen3)
# born: 2026-05-29T23:51:20Z

"""Hybrid algorithm integrating privacy sketching (Parent A) with span labeling (Parent B).

Mathematical Bridge:
Both parents rely on hash‑based probabilistic structures.  Parent A builds a Count‑Min matrix
C∈ℕ^{d×w} to approximate item frequencies, while Parent B extracts literal spans S from text
using regular‑expression matches.  The fusion treats C as a weighting matrix that modulates
the confidence scores of spans in S.  Concretely, for each span s we compute a risk factor
r = f(C) (the reconstruction‑risk estimate derived from the sketch) and define a hybrid
score σ(s)=score(s)·(1−r).  This creates a unified system where privacy risk directly
attenuates downstream labeling confidence.

The module provides three core hybrid operations:
1. `build_sketch_and_spans` – constructs a Count‑Min sketch from records and extracts raw spans.
2. `risk_adjusted_spans` – computes reconstruction risk from the sketch and rescales span scores.
3. `lsh_index_with_risk` – builds a MinHash LSH index of document shingles, weighting buckets by risk.

All components are pure Python with only standard‑library and NumPy dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – sketching utilities
# ----------------------------------------------------------------------
def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Build a Count‑Min sketch matrix."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table


def hyperloglog_cardinality_sketch(items: Iterable[str]) -> int:
    """Placeholder HLL – exact cardinality via a set (for testing)."""
    return len(set(items))


def minhash_lsh_index(docs: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Very simple MinHash‑LSH: bucket by the minimum SHA‑1 hash prefix of shingles."""
    buckets: Dict[str, List[str]] = defaultdict(list)
    for doc_id, shingles in docs.items():
        # Compute a 6‑hex‑digit prefix for each shingle, take the minimum as bucket key
        min_prefix = min(
            (hashlib.sha1(s.encode()).hexdigest()[:6] for s in shingles),
            default="empty",
        )
        buckets[min_prefix].append(doc_id)
    return dict(buckets)


# ----------------------------------------------------------------------
# Parent B – literal span extraction
# ----------------------------------------------------------------------
DEFAULT_LABELS = [
    "Operator",
    "Rainmaker",
    "Paladin / God-Mode",
    "Psyche / State-Collapse",
    "Forensic Shield",
    "Infinite Sink",
    "Anchor Weight",
    "Server Wipe",
    "API Rate Limiting",
    "Environment Migration",
    "Cruelty Protocols",
    "Master’s Eye",
    "Chrono-Ledger",
    "KRAMPUSCHEWING",
    "KORPUS",
    "DIOGENES",
    "FairyFuse",
    "Job Fair Allocator",
    "Darwinian Surfaces",
    "Command Envelope Protocol",
]


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


def literal_fallback(text: str, labels: List[str], *, case_sensitive: bool = False) -> List[Span]:
    """Find exact (case‑insensitive) occurrences of each label in `text`."""
    flags = 0 if case_sensitive else re.IGNORECASE
    spans: List[Span] = []
    seen: Set[Tuple[int, int, str]] = set()
    for label in labels:
        # Allow spaces or hyphens in the label to match any whitespace
        pattern = re.escape(label).replace(r"\ ", r"\s+").replace(r"\-", r"\s+")
        for m in re.finditer(pattern, text, flags):
            start, end = m.span()
            key = (start, end, label)
            if key in seen:
                continue
            seen.add(key)
            spans.append(Span(start=start, end=end, text=m.group(), label=label, score=1.0))
    return spans


# ----------------------------------------------------------------------
# Hybrid Core
# ----------------------------------------------------------------------
def reconstruction_risk_score_sketch(unique_quasi_identifiers: int, total_records: int) -> float:
    """Risk = proportion of distinct quasi‑identifiers."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def estimate_distinct_from_sketch(sketch: List[List[int]]) -> int:
    """
    Very rough estimator: count how many cells are non‑zero across the whole matrix.
    This upper‑bounds the number of distinct items that have ever been hashed.
    """
    return sum(1 for row in sketch for cell in row if cell > 0)


def build_sketch_and_spans(
    records: List[Dict[str, Any]],
    width: int = 64,
    depth: int = 4,
    labels: List[str] | None = None,
) -> Tuple[List[List[int]], List[Tuple[int, List[Span]]]]:
    """
    1. Build a Count‑Min sketch from all *quasi‑identifier* values across `records`.
    2. Extract literal spans from each record's ``text`` field using `labels`.
    Returns (sketch, [(record_index, spans), ...]).
    """
    if labels is None:
        labels = DEFAULT_LABELS

    # Collect quasi‑identifiers (all string values that are not redacted)
    quasi_items: List[str] = []
    for rec in records:
        for v in rec.values():
            if isinstance(v, str) and not v.lower().startswith("<redacted>"):
                quasi_items.append(v)

    sketch = count_min_sketch(quasi_items, width=width, depth=depth)

    # Extract spans per record
    spans_per_record: List[Tuple[int, List[Span]]] = []
    for idx, rec in enumerate(records):
        text = str(rec.get("text", ""))
        spans = literal_fallback(text, labels)
        spans_per_record.append((idx, spans))

    return sketch, spans_per_record


def risk_adjusted_spans(
    sketch: List[List[int]],
    total_records: int,
    spans_per_record: List[Tuple[int, List[Span]]],
) -> List[Tuple[int, List[Span]]]:
    """
    Compute reconstruction risk from the sketch and attenuate each span's score:
        new_score = original_score * (1 - risk)
    """
    distinct_est = estimate_distinct_from_sketch(sketch)
    risk = reconstruction_risk_score_sketch(distinct_est, total_records)

    adjusted: List[Tuple[int, List[Span]]] = []
    for idx, spans in spans_per_record:
        new_spans = [
            Span(
                start=s.start,
                end=s.end,
                text=s.text,
                label=s.label,
                score=s.score * (1.0 - risk),
                backend=s.backend,
            )
            for s in spans
        ]
        adjusted.append((idx, new_spans))
    return adjusted


def lsh_index_with_risk(
    records: List[Dict[str, Any]],
    sketch: List[List[int]],
    total_records: int,
    shingle_size: int = 3,
) -> Dict[str, List[str]]:
    """
    Build a MinHash‑LSH index of document shingles.
    Bucket weights are scaled by (1 - risk) so that high‑risk datasets produce
    less aggressive clustering.
    Returns a mapping bucket_key → list of document ids.
    """
    distinct_est = estimate_distinct_from_sketch(sketch)
    risk = reconstruction_risk_score_sketch(distinct_est, total_records)
    weight = 1.0 - risk

    # Prepare shingles per document
    docs_shingles: Dict[str, Set[str]] = {}
    for rec in records:
        doc_id = str(rec.get("id", uuid4().hex))
        text = str(rec.get("text", ""))
        tokens = text.split()
        shingles = {
            " ".join(tokens[i : i + shingle_size])
            for i in range(max(len(tokens) - shingle_size + 1, 0))
        }
        # Apply weight by duplicating shingles proportionally (simple heuristic)
        weighted_shingles = set()
        for sh in shingles:
            # The integer part of weight determines replication count
            count = max(1, int(weight * 10))
            for _ in range(count):
                weighted_shingles.add(sh + f"_{random.random():.6f}")
        docs_shingles[doc_id] = weighted_shingles

    return minhash_lsh_index(docs_shingles)


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def uuid4() -> str:
    """Fallback UUID generator without importing uuid (to respect import policy)."""
    # 128‑bit random value formatted as hex
    return "%032x" % random.getrandbits(128)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample dataset
    sample_records = [
        {
            "id": "doc1",
            "text": "Operator initiated server wipe. API Rate Limiting engaged.",
            "email": "alice@example.com",
            "phone": "555-1234",
        },
        {
            "id": "doc2",
            "text": "Infinite Sink detected. Chrono-Ledger updated.",
            "email": "bob@example.com",
            "ssn": "123-45-6789",
        },
        {
            "id": "doc3",
            "text": "KRAMPUSCHEWING routine completed. Environment Migration scheduled.",
            "email": "carol@example.com",
            "token": "abcd1234",
        },
    ]

    # Step 1: build sketch and raw spans
    sketch_matrix, raw_spans = build_sketch_and_spans(sample_records)

    # Step 2: adjust spans by privacy risk
    adjusted_spans = risk_adjusted_spans(sketch_matrix, total_records=len(sample_records), spans_per_record=raw_spans)

    # Print adjusted spans for inspection
    for idx, spans in adjusted_spans:
        print(f"Record {sample_records[idx]['id']} adjusted spans:")
        for s in spans:
            print(f"  [{s.start}:{s.end}] '{s.text}' -> label={s.label}, score={s.score:.3f}")

    # Step 3: LSH index with risk‑aware weighting
    lsh_buckets = lsh_index_with_risk(sample_records, sketch_matrix, total_records=len(sample_records))
    print("\nLSH buckets (risk‑aware):")
    for bucket, docs in lsh_buckets.items():
        print(f"  Bucket {bucket}: {docs}")

    sys.exit(0)