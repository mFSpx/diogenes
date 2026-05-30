# DARWIN HAMMER — match 4473, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m1388_s4.py (gen6)
# parent_b: hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m2434_s0.py (gen6)
# born: 2026-05-29T23:55:59Z

"""Hybrid Algorithm combining Hyperdimensional Computing (A) and RLCT‑Grokking (B)

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m1388_s4.py (A)
- hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m2434_s0.py (B)

Mathematical Bridge:
Both parents manipulate high‑dimensional binary (+1/‑1) vectors and use
information‑theoretic quantities (Fisher scores, entropy, pheromone
signals) to weight combinatorial operations (binding, bundling,
Shapley weighting).  The bridge is therefore a *Fisher‑driven pheromone
scalar* that modulates a *Shapley‑weighted bundle* of bound hypervectors.
The resulting hypervector simultaneously encodes statistical (Fisher)
and combinatorial (Shapley) information while being guided by an
energy‑based RLCT term derived from a Gaussian likelihood.

The implementation below fuses:
1. Vector generation, bind, bundle, similarity (A).
2. Fisher‑score‑based pheromone calculation, Shapley weighting,
   ternary routing, and an RLCT‑style free‑energy term (B).

Three core functions illustrate the hybrid pipeline:
- `shapley_weighted_hypervector`
- `ternary_route_with_energy`
- `hybrid_predictor`
"""

import math
import random
import hashlib
import datetime
import sys
import pathlib
import numpy as np
from itertools import combinations
from functools import reduce
from typing import List, Tuple, Dict

Vector = List[int]

# ----------------------------------------------------------------------
# Hyperdimensional primitives (from Parent A)
# ----------------------------------------------------------------------
def random_vector(dim: int = 10000, seed: int | str | None = None) -> Vector:
    rng = random.Random(seed)
    return [1 if rng.getrandbits(1) else -1 for _ in range(dim)]

def symbol_vector(symbol: str, dim: int = 10000) -> Vector:
    # Deterministic seed from symbol hash
    seed_bytes = hashlib.sha256(symbol.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    return random_vector(dim, seed)

def bind(a: Vector, b: Vector) -> Vector:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return [x * y for x, y in zip(a, b)]

def bundle(vectors: List[Vector]) -> Vector:
    if not vectors:
        raise ValueError("bundle requires at least one vector")
    dim = len(vectors[0])
    for v in vectors:
        if len(v) != dim:
            raise ValueError("all vectors must have same dimension")
    summed = [sum(comp) for comp in zip(*vectors)]
    return [1 if s >= 0 else -1 for s in summed]

def similarity(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    dot = sum(x * y for x, y in zip(a, b))
    return dot / len(a)

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float = 0.0, width: float = 1.0, eps: float = 1e-12) -> float:
    """Fisher information for a univariate Gaussian N(center, width^2)."""
    # I(θ) = (d/dθ log p)^2 averaged over data → (θ‑center)^2/width^4 + 1/width^2
    # Use absolute value to keep positivity.
    return max(1.0 / (width * width) + ((theta - center) ** 2) / (width ** 4), eps)

# ----------------------------------------------------------------------
# Pheromone system & Shapley weighting (from Parent B)
# ----------------------------------------------------------------------
class PheromoneSystem:
    """Tracks pheromone signals with exponential decay."""
    def __init__(self):
        self.signals: Dict[str, float] = {}

    def update(self, key: str, increment: float) -> None:
        self.signals[key] = self.signals.get(key, 0.0) + increment

    def strength(self, key: str, half_life_seconds: float) -> float:
        """Return decayed strength; for simplicity we ignore real time."""
        base = self.signals.get(key, 0.0)
        # Decay factor per call (placeholder, real implementation would use timestamps)
        decay = 0.5 ** (1.0 / max(half_life_seconds, 1e-6))
        return base * decay

def shapley_kernel_weight(n: int, k: int) -> float:
    """
    Approximate Shapley kernel weight for an element that appears in a
    coalition of size k out of n total elements.
    Exact weight: 1 / (n * C(n‑1, k‑1))
    """
    if n == 0:
        return 0.0
    from math import comb
    return 1.0 / (n * comb(n - 1, k - 1))

def shapley_weighted_hypervector(vectors: List[Vector]) -> Vector:
    """
    Compute a weighted bundle where each vector is multiplied (bound) by its
    Shapley kernel weight before bundling.
    """
    n = len(vectors)
    weighted = []
    for idx, vec in enumerate(vectors):
        k = idx + 1  # treat ordering as coalition size for illustration
        w = shapley_kernel_weight(n, k)
        weighted.append([int(x * w) for x in vec])  # scale components
    # Sum then binarize
    dim = len(vectors[0])
    summed = [sum(comp) for comp in zip(*weighted)]
    return [1 if s >= 0 else -1 for s in summed]

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def ternary_route_with_energy(
    candidates: List[Vector],
    pheromone_sys: PheromoneSystem,
    surface_key: str,
    half_life: float = 30.0,
) -> Tuple[int, float]:
    """
    Route to one of three bins (0,1,2) based on a combination of:
    - similarity to a reference (first candidate)
    - pheromone strength for the surface_key
    - RLCT‑style free energy using a Gaussian likelihood

    Returns (chosen_index, free_energy).
    """
    if len(candidates) < 3:
        raise ValueError("need at least three candidates for ternary routing")
    ref = candidates[0]
    sims = [similarity(ref, c) for c in candidates[:3]]
    pher = pheromone_sys.strength(surface_key, half_life)

    # Gaussian likelihood centred at 0 with width 1 for each similarity value
    likelihoods = [gaussian_beam(s, 0.0, 1.0) for s in sims]
    # RLCT free energy: -log(likelihood) + complexity term (here proportional to 1/pheromone)
    eps = 1e-12
    free_energies = [
        -math.log(l + eps) + (1.0 / (pher + eps))
        for l in likelihoods
    ]
    chosen = int(np.argmin(free_energies))
    return chosen, free_energies[chosen]

def hybrid_fisher_pheromone_vector(
    symbol: str,
    theta: float,
    pheromone_sys: PheromoneSystem,
    surface_key: str,
) -> Vector:
    """
    Build a hypervector for `symbol` whose magnitude is modulated by a
    Fisher‑derived pheromone signal.
    """
    base_vec = symbol_vector(symbol)
    fisher = fisher_score(theta)
    # Update pheromone with Fisher magnitude
    pheromone_sys.update(surface_key, fisher)
    strength = pheromone_sys.strength(surface_key, half_life_seconds=60.0)
    # Scale base vector by the (normalized) pheromone strength before binding
    scale = math.tanh(strength)  # keep scaling in (-1,1)
    scaled_vec = [int(x * scale) for x in base_vec]
    # Bind with a fixed query vector to embed the Fisher information
    query = random_vector(len(base_vec), seed="FISHER_QUERY")
    return bind(scaled_vec, query)

def hybrid_predictor(
    input_symbol: str,
    theta: float,
    target_symbol: str,
    pheromone_sys: PheromoneSystem,
    surface_key: str,
) -> float:
    """
    End‑to‑end hybrid prediction:
    1. Generate a Fisher‑phermone hypervector for the input.
    2. Generate a set of candidate hypervectors (including the target).
    3. Apply Shapley‑weighted bundling to the candidates.
    4. Route using ternary energy and return similarity to the bundled vector.
    """
    # Step 1
    inp_vec = hybrid_fisher_pheromone_vector(input_symbol, theta, pheromone_sys, surface_key)

    # Step 2 – create two decoys plus the true target
    decoy1 = random_vector(len(inp_vec), seed="DECOY1")
    decoy2 = random_vector(len(inp_vec), seed="DECOY2")
    target_vec = symbol_vector(target_symbol)

    candidates = [inp_vec, decoy1, decoy2, target_vec]

    # Step 3 – Shapley weighted bundle of all candidates
    bundled = shapley_weighted_hypervector(candidates)

    # Step 4 – ternary routing (use first three candidates)
    chosen_idx, energy = ternary_route_with_energy(candidates[:3], pheromone_sys, surface_key)

    # Final similarity between the routed candidate and the bundled representation
    routed_vec = candidates[chosen_idx]
    sim = similarity(routed_vec, bundled)
    # Return a score that combines similarity and inverse energy (higher is better)
    return sim - 0.1 * energy

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pher_sys = PheromoneSystem()
    surface = "demo_surface"
    score = hybrid_predictor(
        input_symbol="alpha",
        theta=0.3,
        target_symbol="beta",
        pheromone_sys=pher_sys,
        surface_key=surface,
    )
    print(f"Hybrid prediction score: {score:.4f}")
    # Verify that pheromone values are being tracked
    print(f"Pheromone strength after prediction: {pher_sys.strength(surface, 60.0):.4f}")