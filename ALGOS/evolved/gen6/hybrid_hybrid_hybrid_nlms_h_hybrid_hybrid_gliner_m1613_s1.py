# DARWIN HAMMER — match 1613, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_nlms_hybrid_h_hybrid_hard_truth_ma_m1061_s2.py (gen5)
# parent_b: hybrid_hybrid_gliner_zero_s_hybrid_krampus_brain_m30_s5.py (gen3)
# born: 2026-05-29T23:37:47Z

"""Hybrid algorithm combining Parent A's perceptual‑hash Hamming similarity and RBF kernel
with Parent B's label‑handling utilities.

Mathematical bridge
------------------
Parent A provides two symmetric similarity matrices for a set of feature vectors:

* **Hamming‑based similarity** `S` derived from perceptual hashes of the vectors.
* **RBF kernel** `K` derived from Euclidean distances via a Gaussian.

Both matrices live in the same ℝⁿˣⁿ space (n = number of feature nodes) and are
positive‑definite.  The fusion consists of a convex combination

    F = α·S + (1‑α)·K ,   0 ≤ α ≤ 1

producing a unified similarity kernel `F`.  Parent B supplies label‑parsing,
hashing and deterministic span extraction utilities.  By embedding each label
into the same feature space (simple SHA‑256‑derived float vectors) we can
project the fused kernel onto the label space and obtain a unified
label‑ranking score.

The module therefore:
1. Builds `S` and `K` from raw numeric features.
2. Fuses them into `F`.
3. Generates deterministic label embeddings.
4. Scores each label by aggregating its similarity to all feature nodes via `F`.

The result is a single, mathematically‑coherent system that leverages both
parents' core topologies.
"""

import math
import random
import re
import sys
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities (from Parent B)
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
        for m in re.finditer(re.escape(label), text, flags):
            key = (m.start(), m.end(), label)
            if key in seen:
                continue
            seen.add(key)
            spans.append(
                Span(
                    start=m.start(),
                    end=m.end(),
                    text=m.group(),
                    label=label,
                    score=1.0,
                )
            )
    return spans

# ----------------------------------------------------------------------
# Parent A core functions (similarity & kernel)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: threshold each value against the mean."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:  # limit to 64 bits for a 64‑bit integer hash
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def similarity_matrix(features: Dict[int, List[float]]) -> Tuple[np.ndarray, List[int]]:
    """Hamming‑based similarity matrix S (values in [0,1])."""
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    hashes = [compute_phash(list(features[node])) for node in nodes]

    for i in range(n):
        for j in range(i, n):
            d = hamming_distance(hashes[i], hashes[j])
            sim = 1.0 - d / 64.0
            S[i, j] = sim
            S[j, i] = sim
    return S, nodes

def rbf_kernel_matrix(features: Dict[int, List[float]], epsilon: float = 1.0) -> Tuple[np.ndarray, List[int]]:
    """Gaussian RBF kernel matrix K."""
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

# ----------------------------------------------------------------------
# Fusion layer (new hybrid)
# ----------------------------------------------------------------------
def fused_similarity(
    features: Dict[int, List[float]],
    epsilon: float = 1.0,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute the convex combination F = α·S + (1‑α)·K.

    Parameters
    ----------
    features : dict[int, list[float]]
        Mapping from node identifier to its numeric feature vector.
    epsilon : float, optional
        Bandwidth for the Gaussian RBF kernel (default 1.0).
    alpha : float, optional
        Weight for the Hamming‑based similarity (default 0.5).

    Returns
    -------
    F : np.ndarray
        Fused similarity matrix (n×n) with entries in [0,1].
    nodes : list[int]
        Order of node identifiers corresponding to rows/columns of F.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1")
    S, nodes_s = similarity_matrix(features)
    K, nodes_k = rbf_kernel_matrix(features, epsilon)

    # Nodes order is identical because both functions iterate over the same dict keys.
    F = alpha * S + (1.0 - alpha) * K
    return F, nodes_s

def label_to_vector(label: str, dim: int = 16) -> List[float]:
    """
    Deterministic embedding of a textual label into a fixed‑size float vector.

    The SHA‑256 digest is split into 8‑byte chunks, each interpreted as an
    unsigned 64‑bit integer and scaled to [0, 1].  If the required dimension
    exceeds the number of chunks, the digest is recycled.
    """
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    needed = dim * 8  # 8 bytes per float
    # Repeat the digest if needed
    while len(digest) < needed:
        digest += hashlib.sha256(digest).digest()
    vec = []
    for i in range(dim):
        chunk = digest[i * 8 : (i + 1) * 8]
        intval = int.from_bytes(chunk, byteorder="big", signed=False)
        vec.append(intval / (2**64 - 1))
    return vec

def rank_labels(
    features: Dict[int, List[float]],
    labels: List[str],
    epsilon: float = 1.0,
    alpha: float = 0.5,
    agg: str = "mean",
) -> List[Tuple[str, float]]:
    """
    Score each label by aggregating its similarity to all feature nodes via
    the fused kernel.

    Parameters
    ----------
    features : dict[int, list[float]]
        Numeric feature vectors for the data nodes.
    labels : list[str]
        Candidate textual labels.
    epsilon, alpha : float
        Passed to :func:`fused_similarity`.
    agg : {"mean", "max"}
        Aggregation method across nodes.

    Returns
    -------
    list of (label, score) sorted descending by score.
    """
    F, nodes = fused_similarity(features, epsilon, alpha)  # n×n matrix
    # Build label embeddings matrix L (m × d) where d = dimension of feature vectors
    # For simplicity we project label vectors onto the same dimension as the
    # feature vectors by truncating or padding with zeros.
    dim = len(next(iter(features.values())))  # assume uniform dimensionality
    L = np.empty((len(labels), dim), dtype=np.float64)
    for idx, lab in enumerate(labels):
        vec = label_to_vector(lab, dim)
        L[idx, :] = np.array(vec, dtype=np.float64)

    # Compute similarity between each label embedding and each node feature.
    # We'll reuse the same Gaussian kernel idea but on the label vectors.
    # First, compute a kernel between labels and nodes using Euclidean distance.
    label_node_kernel = np.empty((len(labels), len(nodes)), dtype=np.float64)
    for i, lab_vec in enumerate(L):
        for j, node_id in enumerate(nodes):
            dist = euclidean(lab_vec.tolist(), features[node_id])
            label_node_kernel[i, j] = gaussian(dist, epsilon)

    # Fuse label‑node kernel with node‑node fused similarity:
    #   score_i = agg_j ( label_node_kernel[i, j] * F[j, j] )
    # Here we weight each node's contribution by its self‑similarity F[j, j] (always 1
    # for properly normalised kernels, but we keep the term for generality).
    weighted = label_node_kernel * np.diag(F)  # broadcast over rows
    if agg == "mean":
        scores = weighted.mean(axis=1)
    elif agg == "max":
        scores = weighted.max(axis=1)
    else:
        raise ValueError("agg must be 'mean' or 'max'")

    ranked = sorted(zip(labels, scores.tolist()), key=lambda x: x[1], reverse=True)
    return ranked

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic feature set (5 nodes, 8‑dimensional vectors)
    random.seed(42)
    np.random.seed(42)
    features = {i: (np.random.rand(8) * 10).tolist() for i in range(5)}

    # Use a subset of default labels
    test_labels = ["Operator", "Infinite Sink", "Chrono-Ledger", "Unknown Entity"]

    # Compute fused matrix and print a summary
    fused_mat, node_order = fused_similarity(features, epsilon=0.8, alpha=0.6)
    print("Fused similarity matrix (rounded):")
    print(np.round(fused_mat, 3))

    # Rank labels
    ranking = rank_labels(features, test_labels, epsilon=0.8, alpha=0.6, agg="mean")
    print("\nLabel ranking:")
    for lbl, sc in ranking:
        print(f"{lbl:20s} -> {sc:.4f}")

    # Demonstrate literal fallback on a simple text
    sample_text = "The Operator triggered the Infinite Sink while the Chrono‑Ledger was active."
    spans = literal_fallback(sample_text, test_labels, case_sensitive=False)
    print("\nDetected spans:")
    for sp in spans:
        print(asdict(sp))