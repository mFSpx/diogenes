# DARWIN HAMMER — match 359, survivor 1
# gen: 4
# parent_a: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s4.py (gen1)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hoeffding_tre_m7_s6.py (gen3)
# born: 2026-05-29T23:28:25Z

"""Hybrid algorithm merging span extraction (Parent A) with similarity & split decision (Parent B).

Mathematical bridge:
- Each extracted `Span` becomes a graph node.
- A deterministic feature vector for a node is built from:
    * Normalized start/end positions,
    * The span's confidence score,
    * Binary bits of the SHA‑256 hash of its text (converted to a 256‑bit vector and truncated to 64 bits).
- The 64‑bit perceptual hash (`phash`) derived from the hash bits feeds the **Hamming‑based similarity** (Parent B).
- The numeric part of the vector feeds an **RBF kernel** (Parent B) using Euclidean distance.
- The two similarity matrices are linearly combined (α·RBF + (1‑α)·Hamming) to obtain a unified similarity measure.
- Finally, a Hoeffding‑bound based test decides whether the current node set should be split into finer clusters.

The code below implements this pipeline with three core functions:
`extract_spans`, `compute_span_features`, `hybrid_similarity`, and `should_split`.
"""

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
from typing import Any, Dict, List, Tuple, Sequence, Hashable, Set

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities from Parent A (span extraction)
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
    """Pure‑Python deterministic label matcher."""
    flags = 0 if case_sensitive else re.IGNORECASE
    spans: List[Span] = []
    seen: Set[Tuple[int, int, str]] = set()
    for label in labels:
        # simple variant: replace slashes and hyphens with spaces
        pattern = re.escape(label.replace("/", " ").replace("-", " "))
        for m in re.finditer(pattern, text, flags):
            start, end = m.start(), m.end()
            key = (start, end, label)
            if key in seen:
                continue
            seen.add(key)
            # deterministic pseudo‑score based on hash of the matched substring
            raw = text[start:end]
            h = int(hashlib.sha1(raw.encode()).hexdigest(), 16)
            score = (h % 1000) / 1000.0  # 0.0‑0.999
            spans.append(Span(start, end, raw, label, score))
    return spans

# ----------------------------------------------------------------------
# Utilities from Parent B (similarity & Hoeffding bound)
# ----------------------------------------------------------------------
Node = Hashable
FeatureVec = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: FeatureVec, b: FeatureVec) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    """Compress a float list to a 64‑bit perceptual hash."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def rbf_kernel_matrix(features: Dict[Node, FeatureVec], epsilon: float = 1.0) -> Tuple[np.ndarray, List[Node]]:
    nodes = list(features.keys())
    n = len(nodes)
    K = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val
    return K, nodes

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))

@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    epsilon: float
    gain_gap: float
    reason: str

def should_split(best_gain: float, second_best_gain: float, r: float,
                 delta: float, n: int, tie_threshold: float = 0.05) -> SplitDecision:
    eps = hoeffding_bound(r, delta, n)
    gap = best_gain - second_best_gain
    if gap > eps and gap > tie_threshold:
        return SplitDecision(True, eps, gap, "Statistically significant gain")
    return SplitDecision(False, eps, gap, "Insufficient evidence to split")

# ----------------------------------------------------------------------
# Hybrid core: combine span extraction with similarity & split logic
# ----------------------------------------------------------------------
def extract_spans(text: str, label_source: str | None = None) -> List[Span]:
    """Extract deterministic spans from `text` using labels from `label_source`."""
    labels = parse_labels(label_source)
    return literal_fallback(text, labels)

def _hash_bits_64(text: str) -> List[int]:
    """Return the first 64 bits of the SHA‑256 hash as a list of 0/1 integers."""
    h = hashlib.sha256(text.encode()).digest()
    bits = []
    for byte in h[:8]:  # 8 bytes = 64 bits
        for i in reversed(range(8)):
            bits.append((byte >> i) & 1)
    return bits

def compute_span_features(spans: List[Span]) -> Dict[Span, FeatureVec]:
    """
    Build a numeric feature vector for each span.

    Vector components:
        0‑1: normalized start position (relative to document length)
        0‑1: normalized end position
        0‑1: confidence score (already 0‑1)
        0‑1 (64 dims): bits from SHA‑256 hash of the span text
    """
    if not spans:
        return {}
    # Approximate document length as max end
    doc_len = max(s.end for s in spans) or 1
    feats: Dict[Span, FeatureVec] = {}
    for s in spans:
        norm_start = s.start / doc_len
        norm_end = s.end / doc_len
        bits = _hash_bits_64(s.text)  # 64 binary values
        vec = [norm_start, norm_end, s.score] + bits
        feats[s] = vec
    return feats

def hybrid_similarity(features: Dict[Span, FeatureVec],
                     alpha: float = 0.5,
                     epsilon: float = 1.0) -> Tuple[np.ndarray, List[Span]]:
    """
    Compute a blended similarity matrix.

    - RBF part uses the full numeric vector (including hash bits).
    - Hamming part uses a perceptual hash derived from the numeric vector.
    - The final similarity is `alpha * RBF + (1-alpha) * Hamming`.
    """
    if not features:
        return np.empty((0, 0)), []
    # RBF kernel
    rbf_mat, nodes = rbf_kernel_matrix(features, epsilon)

    # Hamming similarity via phash
    hashes = [compute_phash(list(features[n])) for n in nodes]
    n = len(nodes)
    ham_mat = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            d = hamming_distance(hashes[i], hashes[j])
            sim = 1.0 - d / 64.0
            ham_mat[i, j] = sim
            ham_mat[j, i] = sim

    combined = alpha * rbf_mat + (1.0 - alpha) * ham_mat
    return combined, nodes

def evaluate_split(similarity: np.ndarray,
                   nodes: List[Span],
                   delta: float = 0.05,
                   tie_threshold: float = 0.05) -> SplitDecision:
    """
    Very coarse split evaluator:
    - Treat each node's average similarity to others as its "gain".
    - Compare best vs second‑best average gain.
    - Use Hoeffding bound to decide split.
    """
    if similarity.size == 0:
        return SplitDecision(False, 0.0, 0.0, "Empty similarity matrix")
    n = similarity.shape[0]
    # Exclude self‑similarity by masking diagonal
    mask = ~np.eye(n, dtype=bool)
    avg_gains = similarity[mask].reshape(n, n - 1).mean(axis=1)
    best_gain = float(np.max(avg_gains))
    second_best_gain = float(np.partition(avg_gains, -2)[-2]) if n > 1 else 0.0
    # r is the range of possible gain values (0‑1)
    return should_split(best_gain, second_best_gain, r=1.0,
                        delta=delta, n=n, tie_threshold=tie_threshold)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "Operator initiated a Server Wipe while the API Rate Limiting "
        "mechanism was active. The Paladin / God-Mode engaged the Infinite Sink."
    )
    # 1. Extract spans
    spans = extract_spans(sample_text)
    print(f"Extracted {len(spans)} spans:")
    for s in spans:
        print(f"  [{s.start}:{s.end}] {s.label!r} (score={s.score:.3f})")

    # 2. Compute features
    feats = compute_span_features(spans)

    # 3. Hybrid similarity
    sim_mat, nodes = hybrid_similarity(feats, alpha=0.6, epsilon=0.8)
    print("\nHybrid similarity matrix (rounded):")
    print(np.round(sim_mat, 3))

    # 4. Decide whether to split
    decision = evaluate_split(sim_mat, nodes)
    print("\nSplit decision:")
    print(decision)