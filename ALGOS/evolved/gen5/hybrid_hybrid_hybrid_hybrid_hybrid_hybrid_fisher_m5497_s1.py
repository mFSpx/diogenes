# DARWIN HAMMER — match 5497, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_sheaf__m1068_s1.py (gen4)
# parent_b: hybrid_hybrid_fisher_locali_jepa_energy_m52_s1.py (gen2)
# born: 2026-05-30T00:02:17Z

"""Hybrid Fisher‑Sheaf Gaussian Algorithm
Integrates:
- Parent A: similarity matrix of sheaf sections using phash & Gaussian‑modeled uncertainty.
- Parent B: Fisher information derived from Gaussian‑beam derivatives.

Mathematical bridge:
Both parents employ a Gaussian beam kernel `g(θ)=exp(-½((θ−c)/w)²)`.  
In the hybrid we (1) build a similarity matrix `S` from sheaf feature vectors,
(2) compute a Fisher‑information weight `F_i = fisher_score(θ_i, c, w)` for each node,
and (3) form a weighted similarity `H = S ∘ (F Fᵀ)` where `∘` denotes element‑wise product.
The system energy is defined as the squared Frobenius norm of the deviation of `H`
from the identity, i.e. `E = ‖H−I‖_F²`, measuring how far the weighted similarity
deviates from perfect self‑similarity.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

# ----------------------------------------------------------------------
# Shared Gaussian beam (present in both parents)
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

# ----------------------------------------------------------------------
# Parent A – similarity matrix based on phash & Hamming distance
Node = int
FeatureVec = tuple[float, ...]
def compute_phash(values: list[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def similarity_matrix(features: dict[Node, FeatureVec]) -> tuple[np.ndarray, list[Node]]:
    """Return S (n×n) where S_ij = 1 - Hamming/64 between phashes of node i and j."""
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = compute_phash(list(features[ni]))
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
            else:
                hj = compute_phash(list(features[nj]))
                d = hamming_distance(hi, hj)
                S[i, j] = 1.0 - d / 64.0
    return S, nodes

# ----------------------------------------------------------------------
# Parent B – Fisher information from Gaussian beam
def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a scalar observation with Gaussian beam likelihood."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----------------------------------------------------------------------
# Hybrid operations -----------------------------------------------------

def compute_fisher_weights(thetas: dict[Node, float],
                           center: float,
                           width: float) -> dict[Node, float]:
    """Map each node to its Fisher weight using its associated angle θ."""
    return {node: fisher_score(theta, center, width) for node, theta in thetas.items()}

def weighted_similarity(features: dict[Node, FeatureVec],
                        thetas: dict[Node, float],
                        center: float,
                        width: float) -> tuple[np.ndarray, list[Node]]:
    """
    Build the hybrid matrix H = S ∘ (F Fᵀ).

    - S : similarity matrix from sheaf feature vectors (Parent A).
    - F : vector of Fisher scores per node (Parent B).
    - ∘ : element‑wise product.
    Returns H and the node ordering.
    """
    S, nodes = similarity_matrix(features)

    # Align Fisher scores with the node ordering
    F_vec = np.array([thetas.get(node, 0.0) for node in nodes], dtype=np.float64)
    # Convert raw angles to Fisher information
    F_vec = np.array([fisher_score(theta, center, width) for theta in F_vec],
                     dtype=np.float64)

    # Outer product gives a symmetric weight matrix
    W = np.outer(F_vec, F_vec)
    H = S * W
    return H, nodes

def hybrid_energy(H: np.ndarray) -> float:
    """
    Energy = squared Frobenius norm of (H - I).
    Measures deviation of the weighted similarity from perfect self‑similarity.
    """
    n = H.shape[0]
    I = np.eye(n, dtype=H.dtype)
    diff = H - I
    return float(np.linalg.norm(diff, 'fro') ** 2)

# ----------------------------------------------------------------------
# Utility – parse optional datetime strings attached to nodes (illustrative)

def parse_node_datetimes(node_strings: dict[Node, str]) -> dict[Node, datetime | None]:
    """
    Convert raw string timestamps attached to nodes into UTC datetimes.
    Uses the same loose parser from Parent B.
    """
    def parse_loose_datetime(raw: str) -> datetime | None:
        text = raw.strip().strip("'\"`[]()")
        if not text:
            return None
        try:
            val = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if val.tzinfo is None:
                val = val.replace(tzinfo=timezone.utc)
            return val.astimezone(timezone.utc)
        except ValueError:
            return None

    return {node: parse_loose_datetime(s) for node, s in node_strings.items()}

# ----------------------------------------------------------------------
# Smoke test -----------------------------------------------------------

if __name__ == "__main__":
    # Construct a tiny synthetic problem
    random.seed(0)
    # Feature vectors (2‑D for simplicity)
    feats = {
        0: (random.random(), random.random()),
        1: (random.random(), random.random()),
        2: (random.random(), random.random()),
        3: (random.random(), random.random()),
    }

    # Associate a random angle θ ∈ [0, 2π) with each node
    thetas = {node: random.random() * 2 * math.pi for node in feats}

    # Parameters for Gaussian beam / Fisher score
    center = math.pi          # aim at π radians
    width = 0.5               # moderate spread

    # Compute hybrid matrix and energy
    H, order = weighted_similarity(feats, thetas, center, width)
    E = hybrid_energy(H)

    print("Node order:", order)
    print("Weighted similarity matrix H:\n", H)
    print(f"Hybrid energy: {E:.6f}")

    # Demonstrate datetime parsing utility
    raw_dates = {
        0: "2023-01-15T12:34:56Z",
        1: "invalid-date",
        2: "2022-07-01",
        3: "",
    }
    parsed = parse_node_datetimes(raw_dates)
    print("Parsed datetimes:", parsed)