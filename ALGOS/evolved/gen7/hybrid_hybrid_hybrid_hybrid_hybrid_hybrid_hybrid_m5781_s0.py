# DARWIN HAMMER — match 5781, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1223_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1235_s0.py (gen6)
# born: 2026-05-30T00:04:45Z

"""Hybrid Algorithm combining:
- Parent A: Koopman operator on morphology vectors with hyperdimensional
  fractional binding.
- Parent B: Hoeffding bound, Least Squares Magnitude (LSM) vector,
  Real Log Canonical Threshold (RLCT) and tropical‑style scaling.

Mathematical bridge:
The LSM vector derived from a graph is interpreted as a probability
distribution over dimensions.  It is used to weight the Koopman
operator matrix (K) via element‑wise multiplication, producing a
‘contextualised’ operator K̂ = K ⊙ w where w is the outer product of the
LSM vector with itself.  The Hoeffding bound (H) provides a confidence
scale that attenuates the transformed morphology, while the RLCT (R)
produces an exponential decay factor exp(‑R).  The final hybrid
representation is obtained by applying K̂ to the morphology vector,
binding the result with a random hyperdimensional vector and finally
raising each component to a fractional power (α) to encode non‑linear
interactions.
"""

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Vector = List[float]
Node = int
Graph = Dict[Node, Set[Node]]

# ----------------------------------------------------------------------
# Parent A – Morphology & Koopman utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def morphology_to_vector(morph: Morphology) -> np.ndarray:
    """Convert a Morphology instance to a normalized 4‑dimensional vector."""
    vec = np.array([morph.length, morph.width, morph.height, morph.mass], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

def random_orthogonal_matrix(dim: int, seed: int | None = None) -> np.ndarray:
    """Generate a random orthogonal matrix via QR decomposition."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(a)
    # Ensure deterministic sign
    d = np.diag(r)
    ph = np.sign(d)
    q *= ph
    return q

def fractional_power_binding(vec: np.ndarray, power: float = 0.5) -> np.ndarray:
    """Apply a fractional power element‑wise (non‑linear binding)."""
    # Preserve sign for real-valued vectors
    sign = np.sign(vec)
    return sign * (np.abs(vec) ** power)

def hyperdimensional_random_vector(dim: int, seed: int | None = None) -> np.ndarray:
    """Generate a random hyperdimensional binary‑valued vector (+1 / -1)."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=dim)

# ----------------------------------------------------------------------
# Parent B – Probabilistic & regret utilities
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    if total_phases < 1 or current_phase < 1:
        raise ValueError('phases and phase must be positive')
    return min(1.0, 1.0 / (2 ** max(0, total_phases - current_phase)))

def compute_hoeffding_bound(observed_gain: float, delta: float, n: int) -> float:
    return math.sqrt((observed_gain * math.log(2 / delta)) / (2 * n))

def compute_lsm_vector(graph: Graph) -> np.ndarray:
    """Least Squares Magnitude vector – normalized degree distribution."""
    num_nodes = len(graph)
    if num_nodes == 0:
        raise ValueError("graph must contain at least one node")
    lsm = np.zeros(num_nodes, dtype=float)
    for i, node in enumerate(sorted(graph)):
        lsm[i] = len(graph[node])
    total = np.sum(lsm)
    return lsm / total if total != 0 else lsm

def compute_rlct(gain: float, lsm_vector: np.ndarray) -> float:
    """Real Log Canonical Threshold based on gain and LSM vector."""
    dot = np.dot(lsm_vector, lsm_vector)
    # Guard against log(0)
    return gain * math.log(dot + 1e-12)

# ----------------------------------------------------------------------
# Hybrid core functions (integrating both parents)
# ----------------------------------------------------------------------
def contextual_koopman_operator(dim: int,
                                lsm_vector: np.ndarray,
                                seed: int | None = None) -> np.ndarray:
    """
    Build a Koopman operator K and weight it with the outer product of the LSM vector.
    K̂ = K ⊙ (w ⊗ w) where w = lsm_vector reshaped to (dim, 1).
    """
    if lsm_vector.shape[0] != dim:
        raise ValueError("LSM vector size must match operator dimension")
    K = random_orthogonal_matrix(dim, seed)
    w = lsm_vector.reshape(dim, 1)
    weighting = w @ w.T                     # outer product
    return K * weighting                    # element‑wise weighting

def hybrid_representation(morph: Morphology,
                          graph: Graph,
                          observed_gain: float,
                          delta: float,
                          n: int,
                          fractional_power: float = 0.5,
                          seed: int | None = None) -> np.ndarray:
    """
    Produce a hybrid high‑dimensional representation.

    Steps:
    1. Convert morphology to a normalized vector (dim = 4).
    2. Compute LSM vector from the graph and expand it to the same dimension.
    3. Build a contextualised Koopman operator K̂ using the LSM vector.
    4. Apply K̂ to the morphology vector.
    5. Scale by confidence factor derived from Hoeffding bound.
    6. Apply exponential decay using RLCT.
    7. Bind with a random hyperdimensional vector.
    8. Apply fractional power binding.
    """
    # 1. Morphology vector
    v = morphology_to_vector(morph)          # shape (4,)

    # 2. LSM vector (may have different size than 4)
    lsm = compute_lsm_vector(graph)
    # If needed, pad or truncate to match dimension
    dim = v.shape[0]
    if lsm.shape[0] < dim:
        lsm_padded = np.pad(lsm, (0, dim - lsm.shape[0]), constant_values=0.0)
    else:
        lsm_padded = lsm[:dim]

    # 3. Contextual Koopman operator
    K_hat = contextual_koopman_operator(dim, lsm_padded, seed)

    # 4. Linear transformation
    transformed = K_hat @ v                # shape (dim,)

    # 5. Confidence scaling (Hoeffding)
    H = compute_hoeffding_bound(observed_gain, delta, n)
    confidence_scale = 1.0 / (1.0 + H)      # diminishes with larger bound

    # 6. RLCT decay
    R = compute_rlct(observed_gain, lsm_padded)
    decay = math.exp(-R)

    scaled = transformed * confidence_scale * decay

    # 7. Hyperdimensional binding
    hd_vec = hyperdimensional_random_vector(dim, seed)
    bound = scaled * hd_vec                # element‑wise binding (multiplication)

    # 8. Fractional power binding
    final = fractional_power_binding(bound, fractional_power)

    return final

def evaluate_hybrid_confidence(morph: Morphology,
                               graph: Graph,
                               observed_gain: float,
                               delta: float,
                               n: int,
                               seed: int | None = None) -> Tuple[float, float]:
    """
    Return the two scalar confidence measures used inside the hybrid:
    (Hoeffding bound, RLCT).
    """
    H = compute_hoeffding_bound(observed_gain, delta, n)
    lsm = compute_lsm_vector(graph)
    # Align LSM size to morphology dimension for RLCT
    dim = 4
    if lsm.shape[0] < dim:
        lsm_adj = np.pad(lsm, (0, dim - lsm.shape[0]), constant_values=0.0)
    else:
        lsm_adj = lsm[:dim]
    R = compute_rlct(observed_gain, lsm_adj)
    return H, R

def hybrid_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Tropical‑style distance (max‑plus) between two hybrid vectors.
    d(a,b) = max_i (a_i - b_i) .
    """
    if vec_a.shape != vec_b.shape:
        raise ValueError("vectors must have the same shape")
    return float(np.max(vec_a - vec_b))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple morphology
    m = Morphology(length=2.0, width=1.0, height=0.5, mass=3.0)

    # Tiny graph: 3 nodes with varying degrees
    g: Graph = {
        0: {1, 2},
        1: {0},
        2: {0}
    }

    observed_gain = 0.8
    delta = 0.05
    n = 100
    seed = 42

    hybrid_vec = hybrid_representation(
        morph=m,
        graph=g,
        observed_gain=observed_gain,
        delta=delta,
        n=n,
        fractional_power=0.6,
        seed=seed
    )
    print("Hybrid vector:", hybrid_vec)

    H, R = evaluate_hybrid_confidence(m, g, observed_gain, delta, n, seed)
    print(f"Hoeffding bound = {H:.6f}, RLCT = {R:.6f}")

    # Distance to a perturbed version
    perturbed = hybrid_vec * 0.9
    dist = hybrid_distance(hybrid_vec, perturbed)
    print(f"Tropical max‑plus distance to perturbed vector: {dist:.6f}")