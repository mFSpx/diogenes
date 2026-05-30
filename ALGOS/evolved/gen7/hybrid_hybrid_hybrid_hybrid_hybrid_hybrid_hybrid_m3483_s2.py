# DARWIN HAMMER — match 3483, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_semantic_neig_m1894_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m1009_s2.py (gen4)
# born: 2026-05-29T23:50:21Z

"""Hybrid Algorithm: Fusion of
- hybrid_hybrid_hybrid_hybrid_hybrid_semantic_neig_m1894_s1.py (Parent A)
- hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m1009_s2.py (Parent B)

Mathematical Bridge
-------------------
Parent A treats SSIM values between document vectors as raw pheromone signals that
drive a probability distribution over *semantic neighbors*.  
Parent B creates a stochastic *weekday weight vector* for a fixed set of groups
by applying a sinusoidal rotation based on the current weekday.

The hybrid algorithm interprets the SSIM‑derived pheromone vector as a
*semantic pheromone distribution* and **multiplies** it element‑wise with the
weekday weight vector.  The product is then renormalised, yielding a unified
distribution that simultaneously respects image‑style similarity (via SSIM)
and temporal/week‑day dynamics (via the sinusoidal weighting).  This operation
is linear in the logarithmic domain and preserves the stochastic nature of both
parents, providing a mathematically coherent fusion."""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# In‑memory semantic enclave (Parent A)
# ----------------------------------------------------------------------
_ENCLAVE: Dict[str, np.ndarray] = {}
_PHEROMONE: Dict[Tuple[str, str], float] = {}  # (source, target) -> pheromone strength


def clear_enclave() -> None:
    """Remove all registered documents and pheromone traces."""
    _ENCLAVE.clear()
    _PHEROMONE.clear()


def register_document(doc_id: str, vector: List[float]) -> None:
    """Store a document vector as a NumPy array for fast linear algebra."""
    _ENCLAVE[doc_id] = np.array(vector, dtype=float)


def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index Measure between two 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (Parent A helper)."""
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return 0.0
    return float(np.dot(a, b) / den)


# ----------------------------------------------------------------------
# Weekday weight vector (Parent B)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1


def weekday_weight_vector(groups: Tuple[str, ...], dow: int) -> np.ndarray:
    """
    Normalised weight vector for *groups* based on weekday ``dow``.
    Sinusoidal rotation yields a row‑stochastic vector.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------


def pheromone_vector(source_id: str) -> np.ndarray:
    """
    Build a pheromone vector for ``source_id`` by computing SSIM between the
    source document and every other document in the enclave.  The resulting
    values are interpreted as raw pheromone signals and normalised to sum to 1.
    """
    if source_id not in _ENCLAVE:
        raise KeyError(f"Document {source_id!r} not registered.")
    src_vec = _ENCLAVE[source_id]
    pheromones = []
    keys = []
    for doc_id, vec in _ENCLAVE.items():
        if doc_id == source_id:
            continue
        ssim = compute_ssim(src_vec.tolist(), vec.tolist())
        pheromones.append(ssim)
        keys.append(doc_id)
    if not pheromones:
        # No neighbours – return a uniform distribution over zero length
        return np.array([], dtype=np.float64)
    pher_arr = np.array(pheromones, dtype=np.float64)
    # Shift to non‑negative domain (SSIM ∈ [-1, 1])
    pher_arr = pher_arr - pher_arr.min() + 1e-12
    pher_arr /= pher_arr.sum()
    # Store for optional later inspection
    for k, val in zip(keys, pher_arr):
        _PHEROMONE[(source_id, k)] = float(val)
    return pher_arr


def fused_distribution(source_id: str, dow: int) -> np.ndarray:
    """
    Fuse the SSIM‑derived pheromone vector with the weekday weight vector.
    The two vectors are aligned on length by truncating or padding with the
    last value; the element‑wise product is then renormalised to a proper
    probability distribution.
    """
    pher_vec = pheromone_vector(source_id)
    week_vec = weekday_weight_vector(GROUPS, dow)

    # Align lengths
    L = max(len(pher_vec), len(week_vec))
    if len(pher_vec) < L:
        # Pad pheromone vector with its mean value
        pad_val = pher_vec.mean() if pher_vec.size else 0.0
        pher_vec = np.pad(pher_vec, (0, L - len(pher_vec)), constant_values=pad_val)
    if len(week_vec) < L:
        week_vec = np.pad(week_vec, (0, L - len(week_vec)), constant_values=week_vec.mean())

    fused = pher_vec * week_vec
    if fused.sum() == 0:
        # Avoid division by zero – fall back to uniform distribution
        return np.full(L, 1.0 / L, dtype=np.float64)
    return fused / fused.sum()


def update_pheromone_via_feedback(source_id: str, target_id: str, reward: float) -> None:
    """
    Simulate a reinforcement step: increase the stored pheromone strength for a
    (source, target) pair proportionally to ``reward`` and decay all others
    slightly.  This mirrors the pheromone‑update dynamics of ant‑colony algorithms.
    """
    decay = 0.95
    # Decay existing pheromones for the source
    for (s, t) in list(_PHEROMONE.keys()):
        if s == source_id:
            _PHEROMONE[(s, t)] *= decay

    # Apply reward to the specific edge
    key = (source_id, target_id)
    current = _PHEROMONE.get(key, 0.0)
    _PHEROMONE[key] = current + reward * (1 - decay)


def semantic_neighbor_distribution(source_id: str) -> Dict[str, float]:
    """
    Return a dictionary mapping neighbour document IDs to their normalised
    pheromone (SSIM) values.  This is a user‑friendly view of the raw vector.
    """
    vec = pheromone_vector(source_id)
    neighbours = [doc for doc in _ENCLAVE.keys() if doc != source_id]
    return {doc: float(val) for doc, val in zip(neighbours, vec)}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Register a few toy documents (4 vectors → matches GROUPS length)
    register_document("doc_A", [0.1, 0.3, 0.5, 0.7])
    register_document("doc_B", [0.2, 0.2, 0.6, 0.8])
    register_document("doc_C", [0.15, 0.35, 0.55, 0.75])
    register_document("doc_D", [0.05, 0.25, 0.45, 0.65])

    # Choose a source and a weekday (e.g., Wednesday → dow=3)
    src = "doc_A"
    dow = 3

    print(f"Semantic neighbor pheromones for {src!r}:")
    print(semantic_neighbor_distribution(src))

    fused = fused_distribution(src, dow)
    print("\nFused distribution (pheromone × weekday weight):")
    for idx, val in enumerate(fused):
        label = GROUPS[idx] if idx < len(GROUPS) else f"pad_{idx}"
        print(f"  {label}: {val:.6f}")

    # Simulate feedback that doc_A prefers doc_B
    update_pheromone_via_feedback(src, "doc_B", reward=0.4)

    print("\nPheromone after feedback:")
    print(semantic_neighbor_distribution(src))
    sys.exit(0)