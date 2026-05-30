# DARWIN HAMMER — match 917, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_pherom_m96_s0.py (gen4)
# parent_b: hybrid_hybrid_sketches_rlct_hybrid_hybrid_bayes__m98_s0.py (gen4)
# born: 2026-05-29T23:31:37Z

"""Hybrid Algorithm combining:
- Parent A: Geometric Algebra with Koopman operator dynamics (hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_pherom_m96_s0.py)
- Parent B: Count‑Min sketch, Bayesian probability updates and feature extraction (hybrid_hybrid_sketches_rlct_hybrid_hybrid_bayes__m98_s0.py)

Mathematical Bridge:
The frequency table produced by a Count‑Min sketch is interpreted as a multivector
in the Clifford algebra Cl(N,0) where each hash bucket corresponds to a 1‑vector
basis blade.  The Koopman operator is learned from paired state matrices (X, X′)
and applied to the coefficient vector of this multivector, yielding a linear
evolution in the algebraic space.  The resulting coefficients are normalised to
form a probability distribution which is then refined by a Bayesian update
using a Beta prior per bucket.  Finally, Shannon entropy of the posterior
distribution modulates pheromone‑like weights that drive a discrete action
selection.  This pipeline fuses the linear‑operator dynamics of Parent A with the
probabilistic sketch‑based inference of Parent B into a single decision engine.
"""

import math
import random
import sys
import pathlib
import hashlib
import numpy as np
from collections import defaultdict
from typing import Iterable, List, Dict, Tuple, FrozenSet

# ----------------------------------------------------------------------
# Parent A – Geometric Algebra core
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j : j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def coefficient_vector(self) -> np.ndarray:
        """Return an ordered vector of coefficients for 1‑vector blades (size n)."""
        vec = np.zeros(self.n)
        for blade, coef in self.components.items():
            if len(blade) == 1:
                idx = next(iter(blade))
                if idx < self.n:
                    vec[idx] = coef
        return vec

    @staticmethod
    def from_coefficient_vector(vec: np.ndarray) -> "Multivector":
        comps = {frozenset({i}): float(v) for i, v in enumerate(vec) if abs(v) > 1e-15}
        return Multivector(comps, vec.shape[0])


def koopman_operator(multivector: Multivector, X: np.ndarray, X_prime: np.ndarray) -> Multivector:
    """
    Learn a Koopman matrix K such that X_prime ≈ K @ X (least‑squares),
    then apply K to the multivector's coefficient vector.
    """
    if X.shape != X_prime.shape:
        raise ValueError("X and X_prime must have the same shape")
    # Least‑squares solution K = X_prime @ pinv(X)
    K = X_prime @ np.linalg.pinv(X)
    # Apply K to the multivector coefficients
    c = multivector.coefficient_vector()
    if K.shape[1] != c.shape[0]:
        raise ValueError("Koopman matrix dimensions incompatible with multivector size")
    c_new = K @ c
    return Multivector.from_coefficient_vector(c_new)


# ----------------------------------------------------------------------
# Parent B – Count‑Min sketch & Bayesian update
# ----------------------------------------------------------------------
def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Count‑Min sketch of item frequencies."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            hash_bytes = hashlib.md5((str(d) + item).encode()).digest()
            index = int.from_bytes(hash_bytes, "big") % width
            table[d][index] += 1
    return table


def flatten_sketch(sketch: List[List[int]]) -> np.ndarray:
    """Aggregate depth rows into a single frequency vector."""
    return np.mean(np.array(sketch, dtype=float), axis=0)


def bayesian_update_counts(
    counts: np.ndarray,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> np.ndarray:
    """
    Perform a Beta‑Bernoulli Bayesian update for each bucket.
    Returns the posterior mean probability for each bucket.
    """
    alpha_post = prior_alpha + counts
    beta_post = prior_beta + np.max(counts) - counts  # simple complementary pseudo‑counts
    return alpha_post / (alpha_post + beta_post)


def shannon_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution."""
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs))


# ----------------------------------------------------------------------
# Hybrid functions (the fusion)
# ----------------------------------------------------------------------
def multivector_from_sketch(sketch: List[List[int]]) -> Multivector:
    """
    Map a Count‑Min sketch to a multivector:
    - Flatten the sketch to a frequency vector.
    - Normalise to unit L1 norm.
    - Use each bucket as the coefficient of a 1‑vector basis blade.
    """
    freq_vec = flatten_sketch(sketch)
    total = np.sum(freq_vec) + 1e-12
    norm_vec = freq_vec / total
    n = len(norm_vec)
    return Multivector.from_coefficient_vector(norm_vec)


def hybrid_koopman_bayesian_step(
    items: Iterable[str],
    X: np.ndarray,
    X_prime: np.ndarray,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> Tuple[int, float]:
    """
    Execute one hybrid decision step:
    1. Build a Count‑Min sketch from the input items.
    2. Convert the sketch into a multivector.
    3. Evolve the multivector with the Koopman operator learned from (X, X′).
    4. Extract the evolved coefficient vector, treat it as a raw probability
       estimate and refine it with a Bayesian update.
    5. Compute Shannon entropy of the posterior and use it to weight the
       posterior (pheromone‑like modulation).
    6. Return the selected bucket index (action) and the final weight.
    """
    # 1. Sketch
    sketch = count_min_sketch(items)
    # 2. Multivector conversion
    mv = multivector_from_sketch(sketch)
    # 3. Koopman evolution
    mv_evolved = koopman_operator(mv, X, X_prime)
    # 4. Bayesian refinement
    raw_probs = mv_evolved.coefficient_vector()
    raw_probs = np.clip(raw_probs, 0.0, None)
    if raw_probs.sum() == 0:
        raw_probs = np.ones_like(raw_probs) / raw_probs.size
    else:
        raw_probs /= raw_probs.sum()
    posterior = bayesian_update_counts(raw_probs * 1e3, prior_alpha, prior_beta)  # scale to pseudo‑counts
    posterior /= posterior.sum()
    # 5. Entropy‑based pheromone weighting
    entropy = shannon_entropy(posterior)
    pheromone = np.exp(-entropy) * posterior
    pheromone /= pheromone.sum()
    # 6. Action selection (argmax)
    action = int(np.argmax(pheromone))
    weight = float(pheromone[action])
    return action, weight


def generate_random_koopman_matrices(dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Utility to produce a random (X, X′) pair suitable for Koopman learning."""
    X = np.random.rand(dim, dim)
    # Simulate a linear dynamical system with a hidden matrix A
    A = np.random.rand(dim, dim) * 0.5
    X_prime = A @ X
    return X, X_prime


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample data stream
    sample_items = [f"item_{random.randint(0, 100)}" for _ in range(200)]

    # Dimension for the geometric algebra (choose width of sketch)
    WIDTH = 32
    DIM = WIDTH

    # Generate compatible Koopman matrices
    X_mat, Xp_mat = generate_random_koopman_matrices(DIM)

    # Run the hybrid decision step
    chosen_bucket, confidence = hybrid_koopman_bayesian_step(
        items=sample_items,
        X=X_mat,
        X_prime=Xp_mat,
        prior_alpha=2.0,
        prior_beta=2.0,
    )

    print(f"Chosen bucket (action): {chosen_bucket}")
    print(f"Associated pheromone weight: {confidence:.4f}")