# DARWIN HAMMER — match 3603, survivor 2
# gen: 5
# parent_a: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s4.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hdc_serpentin_m619_s1.py (gen4)
# born: 2026-05-29T23:51:00Z

"""Hybrid Span‑MinHash‑Morphology Fusion

This module merges the core mathematics of two parent algorithms:

* **Parent A** – a deterministic literal‑fallback label matcher that extracts
  `Span` objects (start, end, text, label, score). Its essential operations are
  regular‑expression scanning, hashing utilities and timestamp generation.

* **Parent B** – a hyperdimensional MinHash / morphology framework that builds a
  MinHash signature of a token set, creates a high‑dimensional morphology vector
  from physical attributes, and binds two vectors via element‑wise multiplication.

**Mathematical Bridge**

A `Span` is interpreted as a *token set* (the words of its `text`).  
The MinHash signature of this token set (`minhash_signature`) becomes a compact
numeric fingerprint. Simultaneously, the geometric attributes of the span
(`length = end‑start`, `score`, etc.) are mapped onto a `Morphology` instance,
which yields a high‑dimensional vector (`morphology_vector`).  

The two representations are fused by the **bind** operation (element‑wise
multiplication) producing a single hyperdimensional vector that encodes both
lexical similarity (via MinHash) and geometric/contextual information (via
morphology). This unified representation can be compared across spans with
standard similarity measures (e.g. cosine similarity)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities from Parent A
# ----------------------------------------------------------------------
DEFAULT_LABELS = [
    "Operator", "Rainmaker", "Paladin / God-Mode", "Psyche / State-Collapse",
    "Forensic Shield", "Infinite Sink", "Anchor Weight", "Server Wipe",
    "API Rate Limiting", "Environment Migration", "Cruelty Protocols",
    "Master’s Eye", "Chrono-Ledger", "KRAMPUSCHEWING", "KORPUS",
    "DIOGENES", "FairyFuse", "Job Fair Allocator", "Darwinian Surfaces",
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


def now_iso() -> str:
    """Current UTC timestamp in ISO‑8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    """SHA‑256 hash of the supplied Unicode text."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def parse_labels(raw: str | None) -> List[str]:
    """Parse a JSON file, comma‑separated string or fallback to defaults."""
    if not raw:
        return list(DEFAULT_LABELS)
    p = Path(raw)
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            labels = data.get("required_exact_labels") or data.get("labels") or []
        else:
            labels = data
        return [str(x).strip() for x in labels if str(x).strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]


def literal_fallback(text: str, labels: List[str], *, case_sensitive: bool = False) -> List[Span]:
    """Pure‑Python label matcher that returns deterministic spans."""
    flags = 0 if case_sensitive else re.IGNORECASE
    spans: List[Span] = []
    seen: set[Tuple[int, int, str]] = set()
    for label in labels:
        # generate simple variants (e.g. replace “ / ” and “-” with a space)
        variants = {label, label.replace("/", " "), label.replace("-", " ")}
        for variant in variants:
            for m in re.finditer(re.escape(variant), text, flags):
                start, end = m.span()
                key = (start, end, variant)
                if key in seen:
                    continue
                seen.add(key)
                # deterministic pseudo‑score based on hash of the match
                score = int.from_bytes(hashlib.sha1(f"{text}{start}{end}{variant}".encode()).digest()[:4], "big")
                score = (score % 1000) / 1000.0  # map to [0,1)
                spans.append(Span(start, end, m.group(), variant, score))
    # sort by start position for reproducibility
    spans.sort(key=lambda s: (s.start, s.end))
    return spans


# ----------------------------------------------------------------------
# Hyperdimensional utilities from Parent B
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1
Vector = List[float]


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float
    tokens: List[str]


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of a token with a seed."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def morphology_vector(m: Morphology, dim: int = 10000) -> Vector:
    """Generate a hyper‑dimensional vector from morphology attributes."""
    # deterministic seed from the four numeric attributes
    seed_bytes = hashlib.sha256(f"{m.length}{m.width}{m.height}{m.mass}".encode()).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    random.seed(seed)  # make the random component reproducible per morphology
    base = np.array([random.random() for _ in range(dim)], dtype=np.float32)

    # tile the four attributes to the vector length
    attr_cycle = np.array([m.length, m.width, m.height, m.mass], dtype=np.float32)
    repeats = dim // 4 + 1
    attrs = np.tile(attr_cycle, repeats)[:dim]

    return (base * attrs).tolist()


def bind(a: Vector, b: Vector) -> Vector:
    """Bind two vectors via element‑wise multiplication (Hadamard product)."""
    if len(a) != len(b):
        raise ValueError("Vectors must be of equal length to bind")
    return (np.multiply(a, b)).tolist()


def cosine_similarity(v1: Vector, v2: Vector) -> float:
    """Cosine similarity between two vectors."""
    a = np.array(v1, dtype=np.float64)
    b = np.array(v2, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def extract_spans(text: str, label_source: str | None = None) -> List[Span]:
    """Extract deterministic spans using Parent‑A's literal fallback."""
    labels = parse_labels(label_source)
    return literal_fallback(text, labels)


def span_to_morphology(span: Span) -> Morphology:
    """
    Convert a Span into a Morphology instance.

    - length  → span width (end‑start)
    - width   → span score scaled to [0, 10]
    - height  → label index (deterministic hash modulo 10)
    - mass    → constant base mass (1.0) plus score
    - tokens  → whitespace‑split tokens of the span text
    """
    length = float(span.end - span.start)
    width = span.score * 10.0
    # deterministic label index using a small hash
    label_hash = int.from_bytes(hashlib.sha1(span.label.encode()).digest()[:4], "big")
    height = float(label_hash % 10)
    mass = 1.0 + span.score
    tokens = span.text.split()
    return Morphology(length, width, height, mass, tokens)


def span_fused_vector(span: Span, mh_k: int = 128, dim: int = 10000) -> Vector:
    """
    Produce a fused hyper‑dimensional vector for a span.

    1. Compute a MinHash signature of the span's token set.
    2. Convert the integer signature to a float vector (normalised).
    3. Build a morphology vector from span geometry.
    4. Bind the two vectors.
    """
    # Step 1 – MinHash
    mh_sig = minhash_signature(span.text.split(), k=mh_k)

    # Step 2 – normalise signature to [0,1] float vector and pad/truncate to `dim`
    sig_float = np.array([s / MAX64 for s in mh_sig], dtype=np.float32)
    if dim > mh_k:
        pad = np.zeros(dim - mh_k, dtype=np.float32)
        sig_vec = np.concatenate([sig_float, pad])
    else:
        sig_vec = sig_float[:dim]

    # Step 3 – morphology vector
    morph_vec = np.array(morphology_vector(span_to_morphology(span), dim=dim), dtype=np.float32)

    # Step 4 – bind
    fused = bind(sig_vec.tolist(), morph_vec.tolist())
    return fused


def compare_spans(span_a: Span, span_b: Span) -> float:
    """
    Compute similarity between two spans using their fused hyper‑dimensional vectors.
    Returns a cosine similarity in [0,1].
    """
    vec_a = span_fused_vector(span_a)
    vec_b = span_fused_vector(span_b)
    return cosine_similarity(vec_a, vec_b)


# ----------------------------------------------------------------------
# Command‑line interface / smoke test
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    sample_text = (
        "The Operator initiated the Server Wipe while the Rainmaker "
        "monitored the API Rate Limiting. An Infinite Sink was deployed."
    )
    spans = extract_spans(sample_text)
    if not spans:
        print("No spans detected – check label list.")
        return

    print(f"Extracted {len(spans)} spans:")
    for i, s in enumerate(spans, 1):
        print(f"{i}. [{s.start}:{s.end}] \"{s.text}\" → label={s.label}, score={s.score:.3f}")

    # Compute pairwise similarities
    print("\nPairwise span similarities (cosine of fused vectors):")
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            sim = compare_spans(spans[i], spans[j])
            print(f"Span {i + 1} ↔ Span {j + 1}: {sim:.4f}")

    # Demonstrate deterministic hash & timestamp utilities
    print("\nUtility demo:")
    print("Current ISO timestamp:", now_iso())
    print("SHA‑256 of first span text:", sha256_text(spans[0].text))


if __name__ == "__main__":
    _smoke_test()