# DARWIN HAMMER — match 1866, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_sketches_rlct_hybrid_hybrid_distri_m194_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m775_s0.py (gen5)
# born: 2026-05-29T23:39:20Z

"""Hybrid Sketch‑Fisher‑SSIM Algorithm

Parents:
* hybrid_hybrid_sketches_rlct_hybrid_hybrid_distri_m194_s3.py – provides a
  Count‑Min sketch, an RLCT estimator, a tropical (max‑plus) broadcast, a
  Hoeffding‑bound test and a simulated‑annealing acceptance rule.
* hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m775_s0.py – provides a
  Fisher‑information scalar for a Gaussian beam, an SSIM similarity measure and
  a linear ternary router updated by a Fisher‑scaled gradient.

Mathematical bridge:
The sketch yields a high‑dimensional frequency vector **v**.  From the non‑zero
entries we estimate the Real Log Canonical Threshold **Λ** which quantifies the
information loss of the sketch.  **Λ** is used as a *temperature* for the
simulated‑annealing acceptance in leader election and also scales the Fisher
information **γ** that modulates the learning‑rate of the router.  The tropical
broadcast produces a vector **b** of “gain strengths”.  The Structural
Similarity Index Measure **ρ = SSIM(v, b)** supplies an error signal **e = 1‑ρ**
that drives the outer‑product update of the router matrix **W**:
    ΔW = – η·γ·Λ·e·(y·vᵀ) ,   y = softmax(W·v)
Thus the three core topologies – sketch + RLCT, tropical broadcast, and
Fisher‑scaled ternary routing – are mathematically fused into a single unified
system.

The module implements the combined workflow and provides three public functions:
`extract_features`, `router_update`, and `leader_election`.
"""

import hashlib
import math
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Sketch & RLCT utilities (from Parent A)
# ----------------------------------------------------------------------
def count_min_sketch(items: List[Any], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Classic Count‑Min sketch."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(), 16) % width
            table[d][idx] += 1
    return table


def flatten_sketch(sketch: List[List[int]]) -> np.ndarray:
    """Flatten the 2‑D sketch table into a 1‑D frequency vector."""
    return np.asarray([c for row in sketch for c in row], dtype=np.float64)


def estimate_rlct_from_counts(counts: np.ndarray) -> float:
    """
    Very simple proxy for RLCT: ratio of non‑zero entries to total entries,
    passed through a log‑log transform to mimic the theoretical definition.
    """
    nz = np.count_nonzero(counts)
    total = counts.size
    if nz == 0:
        return 0.0
    ratio = nz / total
    # Ensure arguments to log are > 0
    return math.log(max(ratio, 1e-12)) / math.log(max(total, math.e))


# ----------------------------------------------------------------------
# Tropical broadcast (max‑plus) utilities (from Parent A)
# ----------------------------------------------------------------------
def tropical_broadcast(adj: Dict[int, List[int]], init_vals: np.ndarray) -> np.ndarray:
    """
    Perform one iteration of max‑plus (tropical) broadcast.
    `adj` maps node → list of neighbours.
    `init_vals` holds the current broadcast strength for each node.
    Returns the updated strengths.
    """
    n = len(init_vals)
    new_vals = init_vals.copy()
    for i in range(n):
        incoming = [init_vals[j] for j in adj.get(i, [])]
        if incoming:
            # max‑plus aggregation: max of (incoming value + edge weight)
            # Edge weight is assumed 0 for simplicity.
            new_vals[i] = max(incoming)
    return new_vals


# ----------------------------------------------------------------------
# Hoeffding bound and simulated annealing (from Parent A)
# ----------------------------------------------------------------------
def hoeffding_test(successes: int, trials: int, epsilon: float = 0.05) -> bool:
    """
    Returns True if the empirical mean deviates from 0.5 by more than ε
    with probability bounded by Hoeffding's inequality.
    """
    if trials == 0:
        return False
    mean = successes / trials
    bound = math.exp(-2 * trials * (epsilon ** 2))
    # Simple test: accept if bound is smaller than a tiny threshold
    return bound < 1e-6 and abs(mean - 0.5) > epsilon


def simulated_annealing_accept(delta: float, temperature: float) -> bool:
    """
    Accept a move with probability exp(-Δ / T) if Δ > 0,
    otherwise accept unconditionally.
    """
    if delta <= 0:
        return True
    prob = math.exp(-delta / max(temperature, 1e-12))
    return random.random() < prob


# ----------------------------------------------------------------------
# Fisher information, SSIM and ternary router (from Parent B)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian envelope centred at `center` with std‑dev `width`."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Scalar Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("Input shapes must match")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shift = z - np.max(z)
    exp_z = np.exp(shift)
    return exp_z / exp_z.sum()


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def extract_features(items: List[Any],
                     width: int = 64,
                     depth: int = 4) -> Tuple[np.ndarray, float]:
    """
    Build a Count‑Min sketch from `items`, flatten it to a vector `v`,
    and compute the RLCT estimate `Λ`.
    Returns (v, Λ).
    """
    sketch = count_min_sketch(items, width, depth)
    v = flatten_sketch(sketch)
    Λ = estimate_rlct_from_counts(v)
    return v, Λ


def router_update(W: np.ndarray,
                  x: np.ndarray,
                  rlct: float,
                  fisher: float,
                  broadcast_vec: np.ndarray,
                  eta: float = 0.01) -> np.ndarray:
    """
    Perform one Fisher‑scaled ternary router update.
    - `W` : weight matrix (num_outputs × num_inputs)
    - `x` : input feature vector
    - `rlct` : RLCT estimate (acts as a temperature‑like scaling)
    - `fisher` : Fisher information scalar
    - `broadcast_vec` : tropical broadcast strengths (same shape as `x`)
    - `eta` : base learning rate
    Returns the updated weight matrix.
    """
    # Forward pass
    logits = W @ x
    y = softmax(logits)

    # Similarity between sketch features and broadcast strengths
    rho = ssim(x, broadcast_vec)
    e = 1.0 - rho  # error signal

    # Gradient (outer product) scaled by Fisher and RLCT
    grad = np.outer(y, x)  # shape (out, in)
    delta_W = -eta * fisher * rlct * e * grad
    return W + delta_W


def leader_election(broadcast_vec: np.ndarray,
                    sketch_counts: np.ndarray,
                    rlct: float,
                    temperature_factor: float = 0.5) -> List[int]:
    """
    Decide which nodes become candidate leaders.
    - Nodes with high broadcast strength are examined.
    - Hoeffding test on the corresponding sketch count decides eligibility.
    - Simulated annealing acceptance uses temperature proportional to RLCT.
    Returns the list of candidate node indices.
    """
    candidates = []
    temperature = temperature_factor * max(rlct, 1e-12)

    for idx, strength in enumerate(broadcast_vec):
        # Map strength to a pseudo‑trial count (for demonstration)
        trials = int(max(strength, 1))
        successes = int(sketch_counts[idx] if idx < sketch_counts.size else 0)

        if hoeffding_test(successes, trials):
            # Simulated‑annealing acceptance on the strength itself
            delta = -strength  # we prefer lower “energy” (higher strength)
            if simulated_annealing_accept(delta, temperature):
                candidates.append(idx)

    return candidates


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Synthetic data stream
    items = [random.randint(0, 1000) for _ in range(500)]

    # 1. Feature extraction
    v, Λ = extract_features(items, width=64, depth=4)

    # 2. Build a tiny graph for tropical broadcast (10 nodes)
    n_nodes = 10
    adjacency = {i: [(i - 1) % n_nodes, (i + 1) % n_nodes] for i in range(n_nodes)}
    init_strengths = np.random.rand(n_nodes) * 5.0
    broadcast = tropical_broadcast(adjacency, init_strengths)

    # 3. Fisher information (use arbitrary parameters)
    θ = random.uniform(-1, 1)
    γ = fisher_score(theta=θ, center=0.0, width=1.0)

    # 4. Initialize router matrix (output dim = 5)
    out_dim = 5
    W = np.random.randn(out_dim, v.size) * 0.01

    # 5. Router update
    W_new = router_update(W, v, rlct=Λ, fisher=γ, broadcast_vec=broadcast, eta=0.05)

    # 6. Leader election
    leaders = leader_election(broadcast_vec=broadcast,
                              sketch_counts=v,
                              rlct=Λ)

    print("RLCT (Λ):", Λ)
    print("Fisher scalar (γ):", γ)
    print("Broadcast strengths:", broadcast[:5], "...")
    print("Number of leaders elected:", len(leaders))
    print("Leader indices:", leaders)