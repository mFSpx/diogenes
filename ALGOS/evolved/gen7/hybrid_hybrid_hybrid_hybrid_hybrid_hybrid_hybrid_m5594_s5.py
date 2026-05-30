# DARWIN HAMMER — match 5594, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1876_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_bandit_router_m1212_s5.py (gen3)
# born: 2026-05-30T00:03:18Z

"""Hybrid algorithm combining:
- Parent A: Schoolfield temperature‑dependent developmental rate and pheromone‑based probability update.
- Parent B: MinHash similarity, Shannon entropy, and contextual bandit propensities.

Mathematical bridge:
A MinHash Jaccard‑like similarity s∈[0,1] is linearly mapped to a Kelvin temperature T∈[t_low,t_high] of the Schoolfield model.
The resulting developmental_rate r(T) modulates:
    * pheromone probabilities (multiplicative scaling before renormalisation),
    * bandit action propensities (multiplicative scaling before soft‑max),
    * entropy weighting (entropy computed on the scaled distribution).

Thus information‑theoretic similarity and biochemical temperature‑rate physics co‑determine the unified decision dynamics."""
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Schoolfield model and simple pheromone handling
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0          # baseline rate at 25 °C
    delta_h_activation: float = 12_000.0   # activation enthalpy (cal mol⁻¹ K⁻¹)
    t_low: float = 283.15        # low temperature bound (K)
    t_high: float = 307.15       # high temperature bound (K)
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987         # gas constant (cal mol⁻¹ K⁻¹)

def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield temperature‑dependent developmental rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

def normalize(probs: List[float]) -> List[float]:
    total = sum(probs)
    if total == 0:
        raise ValueError("Probability vector sums to zero")
    return [p / total for p in probs]

# ----------------------------------------------------------------------
# Parent B – MinHash similarity, entropy, and bandit utilities
# ----------------------------------------------------------------------
def deterministic_hash(token: str, seed: int) -> int:
    """Deterministic 64‑bit integer hash for a token + seed."""
    import hashlib
    h = hashlib.sha256(f"{token}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)

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

def calculate_entropy(probs: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probs)
    if total <= 0:
        raise ValueError("positive probability mass required")
    probs = np.array(probs) / total
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs))

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def similarity_to_temperature(similarity: float,
                              params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Linearly map similarity ∈[0,1] to temperature ∈[t_low, t_high]."""
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("similarity must be in [0,1]")
    return params.t_low + similarity * (params.t_high - params.t_low)

def hybrid_pheromone_distribution(tokens_a: List[str],
                                  tokens_b: List[str],
                                  raw_counts: List[int],
                                  params: SchoolfieldParams = SchoolfieldParams(),
                                  num_hash: int = 128) -> List[float]:
    """
    Compute pheromone probabilities scaled by the developmental rate derived from
    MinHash similarity between two token sets.
    """
    # 1. similarity via MinHash
    sig_a = minhash_signature(tokens_a, num_hash)
    sig_b = minhash_signature(tokens_b, num_hash)
    sim = minhash_similarity(sig_a, sig_b)

    # 2. map similarity → temperature → rate
    temp_k = similarity_to_temperature(sim, params)
    rate = developmental_rate(temp_k, params)

    # 3. base pheromone probabilities
    base_probs = normalize([float(c) for c in raw_counts])

    # 4. scale by rate and renormalise
    scaled = [p * rate for p in base_probs]
    return normalize(scaled)

def hybrid_bandit_action(propensities: List[float],
                         tokens_a: List[str],
                         tokens_b: List[str],
                         params: SchoolfieldParams = SchoolfieldParams(),
                         num_hash: int = 64) -> int:
    """
    Choose an action index using bandit propensities modulated by the
    developmental rate derived from MinHash similarity.
    Returns the selected index.
    """
    # similarity → temperature → rate
    sim = minhash_similarity(
        minhash_signature(tokens_a, num_hash),
        minhash_signature(tokens_b, num_hash)
    )
    temp_k = similarity_to_temperature(sim, params)
    rate = developmental_rate(temp_k, params)

    # scale propensities
    scaled = np.array(propensities) * rate
    # soft‑max to obtain a distribution
    max_val = np.max(scaled)
    exp_vals = np.exp(scaled - max_val)  # stability
    probs = exp_vals / exp_vals.sum()
    # sample according to the distribution
    return int(np.random.choice(len(propensities), p=probs))

def hybrid_entropy_analysis(tokens_a: List[str],
                            tokens_b: List[str],
                            raw_counts: List[int],
                            params: SchoolfieldParams = SchoolfieldParams(),
                            num_hash: int = 128) -> Tuple[float, float]:
    """
    Returns a tuple (entropy_before, entropy_after) where:
        * entropy_before is the Shannon entropy of the raw pheromone distribution,
        * entropy_after is the entropy after scaling by the developmental rate.
    """
    base_probs = normalize([float(c) for c in raw_counts])
    entropy_before = calculate_entropy(base_probs)

    after_probs = hybrid_pheromone_distribution(
        tokens_a, tokens_b, raw_counts, params, num_hash
    )
    entropy_after = calculate_entropy(after_probs)
    return entropy_before, entropy_after

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Example token sets
    tokens1 = ["alpha", "beta", "gamma", "delta"]
    tokens2 = ["beta", "epsilon", "gamma", "zeta"]

    # Example pheromone counts (e.g., visits to 4 surfaces)
    counts = [10, 5, 15, 20]

    # Example bandit propensities for 4 actions
    propensities = [0.2, 0.5, 0.1, 0.2]

    # Hybrid pheromone distribution
    pher_dist = hybrid_pheromone_distribution(tokens1, tokens2, counts)
    print("Hybrid pheromone distribution:", pher_dist)

    # Hybrid bandit action selection
    action = hybrid_bandit_action(propensities, tokens1, tokens2)
    print("Selected bandit action index:", action)

    # Entropy analysis
    ent_before, ent_after = hybrid_entropy_analysis(tokens1, tokens2, counts)
    print(f"Entropy before scaling: {ent_before:.4f}")
    print(f"Entropy after scaling : {ent_after:.4f}")

    # Verify that distributions sum to 1
    assert abs(sum(pher_dist) - 1.0) < 1e-9, "Pheromone distribution not normalised"
    print("Smoke test completed successfully.")