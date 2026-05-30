# DARWIN HAMMER — match 5402, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1444_s2.py (gen6)
# parent_b: thanatosis.py (gen0)
# born: 2026-05-30T00:01:48Z

"""Hybrid algorithm merging:
- Parent A: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1444_s2.py (SSIM & geometric algebra utilities)
- Parent B: thanatosis.py (simulated annealing with dormancy)

Mathematical bridge:
The Structural Similarity Index (SSIM) from Parent A quantifies the similarity between two
state encodings (treated as multivectors).  This similarity is used to modulate the
annealing temperature of Parent B: a high SSIM (states already similar) cools the system
more aggressively, while a low SSIM keeps a higher temperature to explore the space.
The GA utilities provide a deterministic way to embed raw vectors into a multivector
space, ensuring the SSIM operates on a common geometric representation.
"""

import sys
import math
import random
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Parent A – SSIM and geometric‑algebra utilities
# ----------------------------------------------------------------------


def compute_ssim(
    x: list[float],
    y: list[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Return the Structural Similarity Index between two equal‑length sequences."""
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


def _blade_sign(indices: list[int]) -> tuple[list[int], int]:
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


def _multiply_blades(blade_a: tuple[int, ...], blade_b: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    # Cancel duplicate indices (e_i ^ e_i = 0) and compute sign from swaps
    result, sign = _blade_sign(combined)
    return tuple(result), sign


def encode_as_multivector(vec: np.ndarray) -> dict[tuple[int, ...], float]:
    """
    Encode a real vector into a simple multivector representation:
    each component e_i gets weight vec[i]; higher‑grade blades are omitted for brevity.
    Returns a dictionary mapping basis blades (as index tuples) to scalar coefficients.
    """
    mv = {}
    for i, coeff in enumerate(vec):
        if coeff != 0.0:
            mv[(i,)] = float(coeff)
    return mv


def multivector_inner(mv1: dict[tuple[int, ...], float], mv2: dict[tuple[int, ...], float]) -> float:
    """
    Compute a scalar inner product between two multivectors limited to grade‑1 blades.
    The result is the ordinary dot product of the underlying vectors.
    """
    common = set(mv1) & set(mv2)
    return sum(mv1[b] * mv2[b] for b in common)


# ----------------------------------------------------------------------
# Parent B – Simulated annealing primitives
# ----------------------------------------------------------------------


def acceptance_probability(delta_e: float, temperature: float) -> float:
    """Metropolis acceptance probability."""
    if delta_e < 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(-delta_e / temperature)


def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Geometric cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)


@dataclass(frozen=True)
class DormancyDecision:
    accept: bool
    probability: float
    dormant: bool


# ----------------------------------------------------------------------
# Hybrid layer – three core functions
# ----------------------------------------------------------------------


def hybrid_temperature(
    k: int,
    state_a: np.ndarray,
    state_b: np.ndarray,
    t0: float = 1.0,
    alpha: float = 0.95,
) -> float:
    """
    Compute an annealing temperature that is scaled by the SSIM between two
    multivector encodings of the current states.

    High similarity → stronger cooling (temperature multiplied by (1‑ssim)).
    """
    base_temp = cooling_temperature(k, t0, alpha)

    # Encode as multivectors, then extract the grade‑1 vector for SSIM.
    mv_a = encode_as_multivector(state_a)
    mv_b = encode_as_multivector(state_b)

    # Convert back to plain vectors for SSIM (order must match).
    vec_a = [mv_a.get((i,), 0.0) for i in range(len(state_a))]
    vec_b = [mv_b.get((i,), 0.0) for i in range(len(state_b))]

    ssim = compute_ssim(vec_a, vec_b)
    # Clamp to [0,1] just in case of numeric noise.
    ssim = max(0.0, min(1.0, ssim))

    scaled_temp = base_temp * (1.0 - ssim)
    return scaled_temp


def hybrid_acceptance(delta_e: float, temperature: float, similarity: float) -> float:
    """
    Adjust the Metropolis probability using SSIM as a confidence factor.
    When similarity is high we trust the temperature more (probability unchanged);
    when similarity is low we soften the acceptance by blending with a uniform 0.5.
    """
    base_p = acceptance_probability(delta_e, temperature)
    # Blend: p' = (1‑sim) * 0.5 + sim * base_p
    return (1.0 - similarity) * 0.5 + similarity * base_p


def hybrid_decide(
    delta_e: float,
    k: int,
    state_a: np.ndarray,
    state_b: np.ndarray,
    t0: float = 1.0,
    alpha: float = 0.95,
    dormancy_floor: float = 0.05,
    seed: int | str | None = None,
) -> DormancyDecision:
    """
    Full hybrid decision:
    1. Compute temperature modulated by SSIM of the two states.
    2. Compute similarity (SSIM) itself.
    3. Derive a blended acceptance probability.
    4. Return a DormancyDecision that also flags dormancy when the temperature
       falls below `dormancy_floor` and the move is non‑improving.
    """
    temp = hybrid_temperature(k, state_a, state_b, t0, alpha)
    similarity = compute_ssim(state_a.tolist(), state_b.tolist())
    prob = hybrid_acceptance(delta_e, temp, similarity)

    rng = random.Random(seed)
    accept = rng.random() <= prob
    dormant = temp <= dormancy_floor and delta_e >= 0
    return DormancyDecision(accept=accept, probability=prob, dormant=dormant)


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------


def _demo():
    # Two arbitrary state vectors
    state1 = np.array([0.2, 0.5, 0.3, 0.7, 0.1])
    state2 = np.array([0.25, 0.45, 0.35, 0.65, 0.15])

    # Simulated energy difference (e.g., from a cost function)
    delta_energy = 0.04

    # Iteration index
    iteration = 7

    decision = hybrid_decide(
        delta_e=delta_energy,
        k=iteration,
        state_a=state1,
        state_b=state2,
        t0=1.0,
        alpha=0.9,
        dormancy_floor=0.02,
        seed=42,
    )

    print("Hybrid decision:")
    print(f"  accept   = {decision.accept}")
    print(f"  prob     = {decision.probability:.4f}")
    print(f"  dormant  = {decision.dormant}")


if __name__ == "__main__":
    _demo()