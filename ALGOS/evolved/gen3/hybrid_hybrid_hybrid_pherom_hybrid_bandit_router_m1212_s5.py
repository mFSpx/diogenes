# DARWIN HAMMER — match 1212, survivor 5
# gen: 3
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_liquid_time_c_m288_s1.py (gen2)
# parent_b: hybrid_bandit_router_poikilotherm_schoolf_m20_s0.py (gen1)
# born: 2026-05-29T23:34:35Z

"""Hybrid algorithm merging:
- Parent A: MinHash-based similarity, entropy-driven action selection.
- Parent B: Contextual bandit routing with temperature‑dependent developmental rate (Schoolfield model).

Mathematical bridge:
Similarity ∈ [0,1] from Parent A is linearly mapped to a Kelvin temperature range
[t_low, t_high] of the Schoolfield model (Parent B). That temperature modulates the
developmental_rate, which we use as a multiplicative factor on bandit propensities
and on the expected‑entropy weighting. Thus the two topologies are fused into a
single decision engine where information‑theoretic confidence and biochemical‑rate
physics co‑determine the chosen action.
"""

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (MinHash, entropy)
# ----------------------------------------------------------------------
def deterministic_hash(token: str, seed: int) -> int:
    """Deterministic 64‑bit integer hash for a token + seed."""
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
    return -float(np.sum(probs * np.log(probs)))


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    """Weighted expected entropy for a hit/miss scenario."""
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError("p_hit must be in [0, 1]")
    return p_hit * calculate_entropy(hit_state) + (1.0 - p_hit) * calculate_entropy(miss_state)


# ----------------------------------------------------------------------
# Parent B utilities (Bandit + Schoolfield temperature model)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987


# Global policy store (action_id -> [cumulative_reward, count])
_POLICY: Dict[str, List[float]] = {}


def reset_policy() -> None:
    _POLICY.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


def average_reward(action_id: str) -> float:
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield‑Rollinson temperature dependent rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)


# ----------------------------------------------------------------------
# Hybrid core – mathematical bridge
# ----------------------------------------------------------------------
def similarity_to_temperature(similarity: float,
                              low_k: float = 283.15,
                              high_k: float = 307.15) -> float:
    """
    Map a similarity value ∈[0,1] linearly onto the Kelvin temperature interval.
    """
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("similarity must be in [0,1]")
    return low_k + similarity * (high_k - low_k)


def hybrid_propensity(base_propensity: float, temperature_k: float) -> float:
    """
    Scale a bandit propensity by the developmental rate at the given temperature.
    The rate acts as a biologically‑inspired temperature‑sensitivity factor.
    """
    rate = developmental_rate(temperature_k)
    return base_propensity * rate


def hybrid_expected_entropy(p_hit: float,
                            hit_state: List[float],
                            miss_state: List[float],
                            temperature_k: float) -> float:
    """
    Temperature‑adjusted expected entropy.
    The developmental rate modulates the contribution of hit/miss entropy,
    giving hotter contexts (higher rate) more influence.
    """
    rate = developmental_rate(temperature_k)
    base = expected_entropy(p_hit, hit_state, miss_state)
    return base * rate


def select_hybrid_action(action_dict: Dict[Any, Tuple[float, List[float], List[float]]],
                         temperature_k: float) -> Any:
    """
    Choose the action with the lowest temperature‑adjusted expected entropy.
    action_dict maps an action identifier to a tuple:
        (p_hit, hit_state_distribution, miss_state_distribution)
    """
    best = None
    best_score = float("inf")
    for a_id, (p_hit, hit_dist, miss_dist) in action_dict.items():
        score = hybrid_expected_entropy(p_hit, hit_dist, miss_dist, temperature_k)
        if score < best_score:
            best = a_id
            best_score = score
    return best


# ----------------------------------------------------------------------
# Example high‑level hybrid routine
# ----------------------------------------------------------------------
def hybrid_decision_process(tokens_a: List[str],
                            tokens_b: List[str],
                            base_propensities: Dict[str, float],
                            hit_state: List[float],
                            miss_state: List[float]) -> Tuple[str, float]:
    """
    End‑to‑end hybrid decision:
    1. Compute MinHash similarity between two token sets.
    2. Translate similarity → temperature.
    3. Adjust propensities with developmental_rate.
    4. Build an action dictionary where each action's p_hit is the normalized
       temperature‑scaled propensity.
    5. Return the selected action and the temperature used.
    """
    # 1. Similarity
    sig_a = minhash_signature(tokens_a, num_hash_functions=64)
    sig_b = minhash_signature(tokens_b, num_hash_functions=64)
    sim = minhash_similarity(sig_a, sig_b)

    # 2. Temperature mapping
    temp_k = similarity_to_temperature(sim)

    # 3. Scale propensities
    scaled = {
        aid: hybrid_propensity(prop, temp_k)
        for aid, prop in base_propensities.items()
    }
    total = sum(scaled.values()) or 1.0
    normalized = {aid: val / total for aid, val in scaled.items()}

    # 4. Build action dict for entropy evaluation
    action_dict = {
        aid: (p_hit, hit_state, miss_state) for aid, p_hit in normalized.items()
    }

    # 5. Select action
    chosen = select_hybrid_action(action_dict, temp_k)
    return chosen, temp_k


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Sample token sets
    tokens_x = ["alpha", "beta", "gamma", "delta"]
    tokens_y = ["beta", "epsilon", "zeta", "eta"]

    # Base propensities for three hypothetical actions
    base_props = {"A": 1.0, "B": 0.8, "C": 0.5}

    # Simple hit/miss state distributions (must sum >0)
    hit_dist = [0.7, 0.2, 0.1]
    miss_dist = [0.4, 0.4, 0.2]

    action, temperature = hybrid_decision_process(
        tokens_x,
        tokens_y,
        base_props,
        hit_dist,
        miss_dist,
    )

    print(f"Chosen action: {action}")
    print(f"Derived temperature (K): {temperature:.2f}")
    print(f"Developmental rate at this temperature: {developmental_rate(temperature):.4f}")