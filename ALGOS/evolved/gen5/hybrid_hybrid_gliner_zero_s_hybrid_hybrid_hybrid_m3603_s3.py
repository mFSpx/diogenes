# DARWIN HAMMER — match 3603, survivor 3
# gen: 5
# parent_a: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s4.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hdc_serpentin_m619_s1.py (gen4)
# born: 2026-05-29T23:51:00Z

from __future__ import annotations

import hashlib
import json
import math
import random
import re
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
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def parse_labels(raw: str | None) -> List[str]:
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
    flags = 0 if case_sensitive else re.IGNORECASE
    spans: List[Span] = []
    seen: set[Tuple[int, int, str]] = set()
    for label in labels:
        variants = {label, label.replace("/", " "), label.replace("-", " ")}
        for variant in variants:
            for m in re.finditer(re.escape(variant), text, flags):
                start, end = m.span()
                key = (start, end, variant)
                if key in seen:
                    continue
                seen.add(key)
                score = int.from_bytes(hashlib.sha1(f"{text}{start}{end}{variant}".encode()).digest()[:4], "big")
                score = (score % 1000) / 1000.0
                spans.append(Span(start, end, m.group(), variant, score))
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
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def morphology_vector(m: Morphology, dim: int = 10000) -> Vector:
    seed_bytes = hashlib.sha256(f"{m.length}{m.width}{m.height}{m.mass}".encode()).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    random.seed(seed)
    base = np.array([random.random() for _ in range(dim)], dtype=np.float32)
    attr_cycle = np.array([m.length, m.width, m.height, m.mass], dtype=np.float32)
    repeats = dim // 4 + 1
    attrs = np.tile(attr_cycle, repeats)[:dim]
    return (base * attrs).tolist()


def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("Vectors must be of equal length to bind")
    return (np.multiply(a, b)).tolist()


def cosine_similarity(v1: Vector, v2: Vector) -> float:
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
    labels = parse_labels(label_source)
    return literal_fallback(text, labels)


def span_to_morphology(span: Span) -> Morphology:
    length = float(span.end - span.start)
    width = span.score * 10.0
    label_hash = int.from_bytes(hashlib.sha1(span.label.encode()).digest()[:4], "big")
    height = float(label_hash % 10)
    mass = 1.0 + span.score
    tokens = span.text.split()
    return Morphology(length, width, height, mass, tokens)


def span_fused_vector(span: Span, mh_k: int = 128, dim: int = 10000) -> Vector:
    morphology = span_to_morphology(span)
    morphology_vector_result = morphology_vector(morphology, dim)
    minhash_signature_result = minhash_signature(span.text.split(), mh_k)
    minhash_signature_vector = [x / MAX64 for x in minhash_signature_result]
    minhash_signature_vector += [0.0] * (dim - len(minhash_signature_result))
    return bind(morphology_vector_result, minhash_signature_vector)


def main():
    text = "Example text with Operator and Rainmaker labels"
    spans = extract_spans(text)
    for span in spans:
        fused_vector = span_fused_vector(span)
        print(f"Span: {span.text}, Fused Vector: {fused_vector}")


if __name__ == "__main__":
    main()