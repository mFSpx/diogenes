# DARWIN HAMMER — match 5359, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_fisher_m984_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hoeffd_hybrid_hybrid_hybrid_m1973_s1.py (gen6)
# born: 2026-05-30T00:01:32Z

"""Hybrid module combining:
- Parent A: Fisher‑weighted Shapley values, Count‑Min sketch, RL‑based adaptive width.
- Parent B: Hoeffding bound, Gini coefficient, morphological sphericity/flatness.

Mathematical bridge:
Both parents define a ``Morphology`` dataclass and expose a *sphericity* measure.
We map the sphericity to a characteristic angle ``θ`` (geometric ↔ statistical
interface).  The angle feeds a Gaussian‑beam Fisher information
``I(θ)`` which becomes a multiplicative relevance weight for Shapley
contributions.  The resulting *Fisher‑weighted Shapley* can be computed
exactly for small feature sets or approximated with a Count‑Min sketch for
large sets.  The sketch width is chosen adaptively using the Hoeffding bound
(from Parent B) together with the Gini inequality of the underlying
contributions, thus fusing the information‑theoretic and concentration‑
inequality perspectives of the two lineages.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Set, FrozenSet, Iterable, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Shared building block – Morphology (present in both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of an entity."""
    length: float
    width: float
    height: float
    mass: float


def morphological_indices(morph: Morphology) -> Tuple[float, float]:
    """Return (sphericity, flatness) for a morphology.

    The formulas are taken from Parent B.
    """
    volume = morph.length * morph.width * morph.height
    surface_area = 2 * (
        morph.length * morph.width
        + morph.width * morph.height
        + morph.height * morph.length
    )
    # sphericity ≈ volume / (surface_area / 6)^(1/3)
    sphericity = volume / ((surface_area / 6) ** (1 / 3))
    # flatness – a simple proxy used in Parent B (truncated in the source)
    flatness = (
        morph.length + morph.width + morph.height
    ) / (2 * math.sqrt(
        (morph.length * morph.width)
        + (morph.width * morph.height)
        + (morph.height * morph.length)
    ))
    return sphericity, flatness


# ----------------------------------------------------------------------
# Parent B utilities
# ----------------------------------------------------------------------
def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound ε for range ``r``, confidence ``1‑δ`` and sample size ``n``."""
    if r <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("r > 0, 0 < delta < 1, n > 0 required")
    return math.sqrt((r * r * math.log(1.0 / delta)) / (2.0 * n))


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini inequality coefficient for a non‑negative iterable."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non‑negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


# ----------------------------------------------------------------------
# Parent A core – Shapley & Count‑Min sketch (simplified)
# ----------------------------------------------------------------------
def exact_shapley(values: List[float]) -> List[float]:
    """Exact Shapley values for an additive characteristic function.

    For an additive game v(S)=∑_{i∈S} values[i] the Shapley value of each
    player equals its own value, but we keep the combinatorial formula for
    illustration.
    """
    n = len(values)
    if n == 0:
        return []
    # Pre‑compute factorials to avoid repeated calls
    fact = [1] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = fact[i - 1] * i

    shapley = [0.0] * n
    # Enumerate all subsets via bitmask – feasible only for small n
    for i in range(n):
        phi = 0.0
        for mask in range(1 << n):
            if mask & (1 << i):
                continue  # subset contains i → skip
            s = bin(mask).count("1")
            weight = fact[s] * fact[n - s - 1] / fact[n]
            # marginal contribution of i to subset S
            marginal = values[i]  # additive game
            phi += weight * marginal
        shapley[i] = phi
    return shapley


class CountMinSketch:
    """Very small Count‑Min sketch implementation (Parent A style)."""

    def __init__(self, width: int, depth: int, seed: int = 0):
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive integers")
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=np.float64)
        rng = random.Random(seed)
        self.hash_seeds = [rng.randint(1, 2 ** 31 - 1) for _ in range(depth)]

    def _hash(self, key: int, i: int) -> int:
        # Simple mixed‑multiply hash; deterministic for reproducibility
        return (key * self.hash_seeds[i]) % self.width

    def update(self, key: int, increment: float = 1.0) -> None:
        for i in range(self.depth):
            idx = self._hash(key, i)
            self.tables[i, idx] += increment

    def estimate(self, key: int) -> float:
        """Return the minimum count over the hash rows – unbiased upper bound."""
        estimates = [
            self.tables[i, self._hash(key, i)] for i in range(self.depth)
        ]
        return min(estimates)


# ----------------------------------------------------------------------
# Hybrid mathematics
# ----------------------------------------------------------------------
def characteristic_angle(morph: Morphology) -> float:
    """Map sphericity to an angle θ ∈ [0, π/2].

    The mapping is monotonic: higher sphericity → smaller angle.
    """
    sphericity, _ = morphological_indices(morph)
    # Normalise by a heuristic maximal sphericity (cube of side = max dimension)
    max_dim = max(morph.length, morph.width, morph.height)
    max_vol = max_dim ** 3
    max_sa = 6 * (max_dim ** 2)
    max_sphericity = max_vol / ((max_sa / 6) ** (1 / 3))
    norm = max(0.0, min(1.0, sphericity / max_sphericity))
    # θ = arccos(norm) maps 1 → 0, 0 → π/2
    return math.acos(norm)


def fisher_information(theta: float, sigma: float = 1.0) -> float:
    """Fisher information of a 1‑D Gaussian beam w.r.t. angle θ.

    For a Gaussian with mean μ(θ)=θ and variance σ², I(θ)=1/σ².
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return 1.0 / (sigma ** 2)


def weighted_shapley(morph: Morphology, values: List[float]) -> List[float]:
    """Compute Fisher‑weighted Shapley values.

    1. Compute θ from morphology.
    2. Obtain I(θ) as relevance weight.
    3. Multiply exact Shapley contributions by I(θ).
    """
    theta = characteristic_angle(morph)
    w_f = fisher_information(theta)
    shap = exact_shapley(values)
    return [phi * w_f for phi in shap]


def adaptive_sketch_width(
    morph: Morphology,
    epsilon: float,
    delta: float,
    n_samples: int,
    values: List[float],
) -> int:
    """Determine a sketch width that respects a Hoeffding‑style error bound.

    The Count‑Min sketch error for a positive stream is ≤ total_sum / width
    with probability 1‑δ.  We replace ``total_sum`` by an upper bound derived
    from the Hoeffding inequality applied to the (normalised) value stream.
    """
    if epsilon <= 0 or delta <= 0 or n_samples <= 0:
        raise ValueError("epsilon, delta, n_samples must be positive")
    # Normalise values to [0, 1] (range r = 1)
    max_val = max(values) if values else 1.0
    norm_vals = [v / max_val for v in values]
    # Hoeffding bound on the empirical mean of the normalised stream
    bound = hoeffding_bound(r=1.0, delta=delta, n=n_samples)
    # Upper bound on total sum ≈ n_samples * (mean + bound)
    est_mean = sum(norm_vals) / len(norm_vals) if norm_vals else 0.0
    total_upper = n_samples * (est_mean + bound)
    # Desired error ε translates to width ≥ total_upper / ε
    width = max(1, int(math.ceil(total_upper / epsilon)))
    return width


def sketch_fisher_shapley(
    morph: Morphology,
    stream: Iterable[Tuple[int, float]],
    epsilon: float,
    delta: float,
    n_samples: int,
) -> Dict[int, float]:
    """Approximate Fisher‑weighted Shapley values using a Count‑Min sketch.

    The sketch dimensions are chosen adaptively via ``adaptive_sketch_width``.
    The stream yields (feature_id, contribution) pairs.
    """
    # Collect raw contributions to compute adaptive width
    raw_vals = [inc for _, inc in stream]
    width = adaptive_sketch_width(morph, epsilon, delta, n_samples, raw_vals)
    depth = 5  # fixed depth; could also be tuned via Gini
    sketch = CountMinSketch(width=width, depth=depth, seed=42)

    # Populate sketch
    for fid, inc in stream:
        sketch.update(fid, increment=inc)

    # Estimate each feature's count
    estimates = {fid: sketch.estimate(fid) for fid, _ in stream}
    theta = characteristic_angle(morph)
    w_f = fisher_information(theta)
    # Apply Fisher weight
    weighted = {fid: est * w_f for fid, est in estimates.items()}
    return weighted


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple morphology
    morph = Morphology(length=2.0, width=2.0, height=2.0, mass=5.0)

    # Small feature set for exact computation
    vals = [1.2, 0.8, 3.4]
    print("Exact weighted Shapley:", weighted_shapley(morph, vals))

    # Large synthetic stream for sketching
    random.seed(0)
    large_stream = [(i, random.random()) for i in range(1000)]

    # Parameters for the hybrid sketch
    eps = 0.01
    delta = 0.05
    n_samples = len(large_stream)

    approx = sketch_fisher_shapley(
        morph, large_stream, epsilon=eps, delta=delta, n_samples=n_samples
    )
    # Show a few entries
    sample_keys = list(approx.keys())[:5]
    print("Sketch‑based weighted estimates (sample):")
    for k in sample_keys:
        print(f"  feature {k}: {approx[k]:.4f}")

    # Demonstrate adaptive width computation
    width = adaptive_sketch_width(morph, eps, delta, n_samples, [inc for _, inc in large_stream])
    print("Adaptive sketch width:", width)