# DARWIN HAMMER — match 5594, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1876_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_bandit_router_m1212_s5.py (gen3)
# born: 2026-05-30T00:03:18Z

"""Hybrid Algorithm integrating:
- Parent A: Schoolfield temperature‑dependent developmental rate and pheromone‑based probability handling.
- Parent B: MinHash similarity, Shannon entropy, and contextual bandit action selection.

Mathematical bridge:
    1. MinHash similarity s ∈ [0,1] is linearly mapped to a Kelvin temperature T∈[t_low,t_high].
    2. The Schoolfield developmental_rate(T) yields a scalar r.
    3. r modulates pheromone probability vectors and bandit propensities, acting as a
       biochemical “gain” that couples information‑theoretic similarity with
       temperature‑dependent physiology.
The resulting system produces temperature‑aware entropy measures and
rate‑scaled action choices in a single unified workflow.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core physical model (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0      # cal·mol⁻¹
    t_low: float = 283.15                     # K
    t_high: float = 307.15                    # K
    delta_h_low: float = -45_000.0            # cal·mol⁻¹
    delta_h_high: float = 65_000.0            # cal·mol⁻¹
    r_cal: float = 1.987                     # cal·mol⁻¹·K⁻¹


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield–Rollinson poikilotherm developmental rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)


# ----------------------------------------------------------------------
# Pheromone handling (simplified version of Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PheromoneParams:
    surface_key: str
    limit: int = 10                     # maximum number of surface bins
    base_prob: float = 0.1              # baseline probability for each bin


def generate_pheromone_vector(p_params: PheromoneParams, temp_k: float) -> List[float]:
    """
    Produce a probability vector of length `limit`.  The raw vector is a
    uniform baseline perturbed by a temperature‑dependent factor derived from
    the developmental rate.  The vector is then normalized.
    """
    rate = developmental_rate(temp_k)
    raw = np.full(p_params.limit, p_params.base_prob)
    # Temperature‑dependent perturbation: a simple sinusoidal modulation
    # that respects the physical rate magnitude.
    for i in range(p_params.limit):
        raw[i] *= (1.0 + 0.5 * math.sin(rate * (i + 1)))
    probs = raw / raw.sum()
    return probs.tolist()


# ----------------------------------------------------------------------
# MinHash and entropy utilities (Parent B)
# ----------------------------------------------------------------------
def deterministic_hash(token: str, seed: int) -> int:
    """Deterministic 64‑bit integer hash for a token + seed."""
    # Simple xor‑folded built‑in hash to stay within stdlib.
    combined = f"{token}:{seed}"
    h = hash(combined) & ((1 << 64) - 1)
    return h


def minhash_signature(tokens: List[str], num_hash_functions: int) -> List[int]:
    """Deterministic MinHash signature."""
    signature = []
    for i in range(num_hash_functions):
        min_hash = (1 << 64) - 1
        for token in tokens:
            h = deterministic_hash(token, i)
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    return signature


def minhash_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Jaccard‑like similarity based on identical hash positions."""
    if not sig1 or len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def shannon_entropy(probs: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probs)
    if total <= 0:
        raise ValueError("positive probability mass required")
    p = np.array(probs) / total
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))


# ----------------------------------------------------------------------
# Hybrid functions (the new unified system)
# ----------------------------------------------------------------------
def similarity_to_temperature(similarity: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Linearly map a similarity score s∈[0,1] to a temperature T∈[t_low,t_high].
    """
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("similarity must be in [0,1]")
    return params.t_low + similarity * (params.t_high - params.t_low)


def hybrid_entropy_with_rate(
    tokens_a: List[str],
    tokens_b: List[str],
    num_hash_functions: int,
    p_params: PheromoneParams,
) -> Tuple[float, float]:
    """
    Compute a temperature‑aware entropy:
        1. MinHash similarity → temperature T.
        2. Developmental rate r = f(T).
        3. Pheromone probability vector v(T) scaled by r.
        4. Shannon entropy H(v) multiplied by r to obtain a rate‑weighted metric.
    Returns (entropy, rate) for downstream use.
    """
    sig_a = minhash_signature(tokens_a, num_hash_functions)
    sig_b = minhash_signature(tokens_b, num_hash_functions)
    sim = minhash_similarity(sig_a, sig_b)
    temp_k = similarity_to_temperature(sim)
    rate = developmental_rate(temp_k)
    pheromone_vec = generate_pheromone_vector(p_params, temp_k)
    entropy = shannon_entropy(pheromone_vec)
    weighted_entropy = entropy * rate
    return weighted_entropy, rate


def rate_scaled_bandit_action(
    propensities: List[float],
    similarity: float,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> int:
    """
    Choose an action index from a contextual bandit.
    The developmental rate r(T) derived from similarity scales the raw propensities
    before normalization, biasing the selection toward actions favored under the
    current biochemical “temperature”.
    """
    temp_k = similarity_to_temperature(similarity, params)
    rate = developmental_rate(temp_k, params)
    scaled = np.array(propensities) * rate
    if scaled.sum() == 0:
        # fallback to uniform random choice
        return random.randrange(len(propensities))
    probs = scaled / scaled.sum()
    return int(np.random.choice(len(propensities), p=probs))


def hybrid_decision_engine(
    tokens_a: List[str],
    tokens_b: List[str],
    num_hash_functions: int,
    p_params: PheromoneParams,
    bandit_propensities: List[float],
) -> dict:
    """
    Full pipeline:
        • Compute similarity‑driven temperature and rate.
        • Produce a rate‑weighted entropy measure.
        • Select a bandit action using the same rate.
        • Return a dictionary summarizing the intermediate values.
    """
    # Step 1: similarity & temperature
    sig_a = minhash_signature(tokens_a, num_hash_functions)
    sig_b = minhash_signature(tokens_b, num_hash_functions)
    sim = minhash_similarity(sig_a, sig_b)

    # Step 2: temperature & rate
    temp_k = similarity_to_temperature(sim)
    rate = developmental_rate(temp_k)

    # Step 3: pheromone vector & weighted entropy
    pheromone_vec = generate_pheromone_vector(p_params, temp_k)
    entropy = shannon_entropy(pheromone_vec)
    weighted_entropy = entropy * rate

    # Step 4: bandit action
    action = rate_scaled_bandit_action(bandit_propensities, sim, SchoolfieldParams())

    return {
        "similarity": sim,
        "temperature_K": temp_k,
        "developmental_rate": rate,
        "pheromone_vector": pheromone_vec,
        "entropy": entropy,
        "weighted_entropy": weighted_entropy,
        "selected_action": action,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy token sets
    tokens_x = ["alpha", "beta", "gamma", "delta"]
    tokens_y = ["beta", "epsilon", "zeta", "eta"]

    # Pheromone parameters (no external DB)
    pher_params = PheromoneParams(surface_key="test_surface", limit=8, base_prob=0.05)

    # Bandit propensities (example)
    bandit_props = [0.2, 0.5, 0.1, 0.2]

    # Run hybrid decision engine
    result = hybrid_decision_engine(
        tokens_a=tokens_x,
        tokens_b=tokens_y,
        num_hash_functions=64,
        p_params=pher_params,
        bandit_propensities=bandit_props,
    )

    for key, val in result.items():
        print(f"{key}: {val}")