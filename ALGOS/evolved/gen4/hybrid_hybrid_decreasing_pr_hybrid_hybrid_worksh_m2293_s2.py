# DARWIN HAMMER — match 2293, survivor 2
# gen: 4
# parent_a: hybrid_decreasing_pruning_hybrid_hybrid_bandit_m365_s2.py (gen3)
# parent_b: hybrid_hybrid_workshare_all_hybrid_liquid_time_c_m39_s0.py (gen2)
# born: 2026-05-29T23:41:45Z

import math
import hashlib
import datetime as dt
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# 1. Thermodynamic developmental rate (Schoolfield model)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters for the Schoolfield temperature‑dependent rate model."""
    rho_25: float = 1.0          # Rate at the reference temperature 25 °C (298.15 K)
    delta_h_activation: float = 12_000.0   # Activation enthalpy (J mol⁻¹)
    t_low: float = 283.15        # Lower bound temperature (K)
    t_high: float = 307.15       # Upper bound temperature (K)
    delta_h_low: float = -45_000.0   # Enthalpy below t_low (J mol⁻¹)
    delta_h_high: float = 65_000.0   # Enthalpy above t_high (J mol⁻¹)
    r_gas: float = 8.314         # Universal gas constant (J mol⁻¹ K⁻¹)


def developmental_rate(params: SchoolfieldParams, T: float) -> float:
    """
    Compute the temperature‑dependent developmental rate ρ(T) using a
    piece‑wise Schoolfield formulation.

    The original code clipped the result to [0, 1], which artificially
    limited biologically plausible rates >1.  Here we return the raw
    rate and let downstream components decide on scaling.
    """
    if T < params.t_low:
        delta_h = params.delta_h_low
        T_ref = params.t_low
    elif T > params.t_high:
        delta_h = params.delta_h_high
        T_ref = params.t_high
    else:
        # Within the optimal range we use the activation enthalpy.
        delta_h = params.delta_h_activation
        T_ref = 298.15  # reference temperature (25 °C)

    exponent = (delta_h / params.r_gas) * (1.0 / T_ref - 1.0 / T)
    rho = params.rho_25 * math.exp(exponent)
    return rho


# ----------------------------------------------------------------------
# 2. Calendar‑dependent weight vector
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """
    Produce a weight vector of length ``len(groups)`` that varies
    smoothly with the day‑of‑week (dow ∈ {0,…,6} where 0 = Monday).

    The previous implementation used a fixed amplitude and a sine
    wave that could produce near‑zero totals for pathological inputs.
    The new version guarantees positivity and normalises to sum‑to‑1.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")

    # Base angles equally spaced around the unit circle.
    base_angles = np.arange(n) * (2.0 * math.pi) / n

    # Phase shift proportional to the weekday.
    phase = (2.0 * math.pi) * (dow % 7) / 7.0

    # Amplitude chosen so that the minimum weight stays >0.
    amplitude = 0.4
    raw = 1.0 + amplitude * np.sin(base_angles + phase)

    # Ensure strict positivity before normalisation.
    raw = np.clip(raw, a_min=1e-8, a_max=None)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


# ----------------------------------------------------------------------
# 3. MinHash utilities
# ----------------------------------------------------------------------
_MAX_HASH = 2 ** 64 - 1  # Maximum value of a 64‑bit unsigned integer


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on MD5."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.md5(data).digest()[:8], "big")


def minhash_signature(tokens: Sequence[str], k: int = 128) -> List[int]:
    """
    Compute a k‑length MinHash signature for a set of tokens.
    Empty token sets yield the maximal hash value for every position.
    """
    token_set = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not token_set:
        return [_MAX_HASH] * k
    return [min(_hash(i, t) for t in token_set) for i in range(k)]


def _hash_to_fraction(h: int) -> float:
    """Map a 64‑bit hash to the interval [0, 1]."""
    return h / _MAX_HASH


# ----------------------------------------------------------------------
# 4. Core hybrid step
# ----------------------------------------------------------------------
def hybrid_step(
    params: SchoolfieldParams,
    tokens: Sequence[str],
    T: float,
    dow: int,
    groups: Sequence[str] = ("codex", "groq", "cohere", "local_models"),
    k: int = 128,
) -> Tuple[float, List[int]]:
    """
    Compute a context‑aware similarity score.

    1. ρ(T) modulates the overall magnitude.
    2. A weekday‑dependent weight vector distributes influence across
       semantic groups.
    3. The MinHash signature is split evenly among the groups; each
       group's average hash‑fraction contributes to the weighted sum.
    """
    # 1. Temperature scaling
    rho = developmental_rate(params, T)

    # 2. Calendar weighting
    weight_vec = weekday_weight_vector(groups, dow)

    # 3. MinHash signature
    signature = minhash_signature(tokens, k=k)

    # 4. Allocate signature slots to groups (equal partition)
    slots_per_group = max(1, k // len(groups))
    group_means = []
    for i in range(len(groups)):
        start = i * slots_per_group
        end = start + slots_per_group
        slice_hashes = signature[start:end] if end <= k else signature[start:] + signature[: end - k]
        fractions = [_hash_to_fraction(h) for h in slice_hashes]
        group_means.append(np.mean(fractions))

    # 5. Weighted combination, finally scaled by the developmental rate.
    modulated_similarity = float(np.dot(weight_vec, np.array(group_means))) * rho
    return modulated_similarity, signature


# ----------------------------------------------------------------------
# 5. Edge‑pruning schedule
# ----------------------------------------------------------------------
def prune_edges(lambda_: float, alpha: float, t: float, T: float, params: SchoolfieldParams) -> float:
    """
    Compute a pruning probability that decays exponentially with time
    and is further modulated by the temperature‑dependent rate.
    """
    rho = developmental_rate(params, T)
    p = lambda_ * math.exp(-alpha * t) * rho
    return max(0.0, min(1.0, p))


# ----------------------------------------------------------------------
# 6. Helper for day‑of‑week (Monday = 0, Sunday = 6)
# ----------------------------------------------------------------------
def weekday_index(year: int, month: int, day: int) -> int:
    """Return the ISO weekday index (0 = Monday)."""
    return dt.date(year, month, day).weekday()


# ----------------------------------------------------------------------
# 7. Demonstration when run as a script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example parameters
    params = SchoolfieldParams()
    tokens = ["example", "tokens", "for", "minhash", "integration", "test"]
    T = 298.15                     # 25 °C in Kelvin
    dow = weekday_index(2024, 1, 1)  # Monday → 0

    # Run the hybrid step
    similarity, signature = hybrid_step(params, tokens, T, dow)
    print(f"Modulated similarity: {similarity:.6f}")

    # Edge‑pruning example
    lambda_ = 0.1
    alpha = 0.01
    t = 10.0
    prob = prune_edges(lambda_, alpha, t, T, params)
    print(f"Pruned edge probability: {prob:.6f}")