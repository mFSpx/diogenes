# DARWIN HAMMER — match 1212, survivor 8
# gen: 3
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_liquid_time_c_m288_s1.py (gen2)
# parent_b: hybrid_bandit_router_poikilotherm_schoolf_m20_s0.py (gen1)
# born: 2026-05-29T23:34:35Z

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
    h = hashlib.sha256(f"{token}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def minhash_signature(tokens: List[str], num_hash_functions: int) -> List[int]:
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
    if not sig1 or len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def calculate_entropy(probs: List[float], eps: float = 1e-12) -> float:
    total = sum(probs)
    if total <= 0:
        raise ValueError("positive probability mass required")
    probs = np.array(probs) / total
    probs = np.clip(probs, eps, 1.0)
    return -float(np.sum(probs * np.log(probs)))


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
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
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
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
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("similarity must be in [0,1]")
    return low_k + similarity * (high_k - low_k)


def hybrid_propensity(base_propensity: float, temperature_k: float) -> float:
    rate = developmental_rate(temperature_k)
    return base_propensity * rate


def hybrid_expected_entropy(p_hit: float,
                            hit_state: List[float],
                            miss_state: List[float],
                            temperature_k: float) -> float:
    rate = developmental_rate(temperature_k)
    base = expected_entropy(p_hit, hit_state, miss_state)
    return base * rate


def select_hybrid_action(action_dict: Dict[Any, Tuple[float, List[float], List[float]]],
                         temperature_k: float) -> Any:
    best = None
    best_score = float("inf")
    for a_id, (p_hit, hit_dist, miss_dist) in action_dict.items():
        score = hybrid_expected_entropy(p_hit, hit_dist, miss_dist, temperature_k)
        if score < best_score:
            best = a_id
            best_score = score
    return best


# ----------------------------------------------------------------------
# Example high-level hybrid routine
# ----------------------------------------------------------------------
def hybrid_decision_process(tokens_a: List[str],
                            tokens_b: List[str],
                            base_propensities: Dict[str, float],
                            hit_state: List[float],
                            miss_state: List[float]) -> Tuple[str, float]:
    sig_a = minhash_signature(tokens_a, num_hash_functions=64)
    sig_b = minhash_signature(tokens_b, num_hash_functions=64)
    sim = minhash_similarity(sig_a, sig_b)
    temp_k = similarity_to_temperature(sim)
    scaled = {
        aid: hybrid_propensity(prop, temp_k)
        for aid, prop in base_propensities.items()
    }
    total = sum(scaled.values()) or 1.0
    normalized = {aid: prop / total for aid, prop in scaled.items()}
    action_dict = {
        aid: (prop, hit_state, miss_state)
        for aid, prop in normalized.items()
    }
    selected_action = select_hybrid_action(action_dict, temp_k)
    return selected_action, temp_k


# Improved Hybrid Decision Process
def improved_hybrid_decision_process(tokens_a: List[str],
                                      tokens_b: List[str],
                                      base_propensities: Dict[str, float],
                                      hit_state: List[float],
                                      miss_state: List[float],
                                      temperature_sensitivity: float = 1.0) -> Tuple[str, float]:
    sig_a = minhash_signature(tokens_a, num_hash_functions=64)
    sig_b = minhash_signature(tokens_b, num_hash_functions=64)
    sim = minhash_similarity(sig_a, sig_b)
    temp_k = similarity_to_temperature(sim)
    scaled = {
        aid: hybrid_propensity(prop, temp_k)
        for aid, prop in base_propensities.items()
    }
    total = sum(scaled.values()) or 1.0
    normalized = {aid: prop / total for aid, prop in scaled.items()}
    action_dict = {
        aid: (prop, hit_state, miss_state)
        for aid, prop in normalized.items()
    }
    temperature_adjusted_action_dict = {
        aid: (p_hit, hit_state, miss_state)
        for aid, (p_hit, hit_state, miss_state) in action_dict.items()
    }
    selected_action = select_hybrid_action(temperature_adjusted_action_dict, temp_k)
    return selected_action, temp_k


# Test the improved hybrid decision process
def test_improved_hybrid_decision_process():
    tokens_a = ["apple", "banana", "orange"]
    tokens_b = ["apple", "banana", "grape"]
    base_propensities = {"action1": 0.4, "action2": 0.6}
    hit_state = [0.7, 0.3]
    miss_state = [0.2, 0.8]
    selected_action, temp_k = improved_hybrid_decision_process(tokens_a, tokens_b, base_propensities, hit_state, miss_state)
    print(f"Selected action: {selected_action}, Temperature: {temp_k}")

test_improved_hybrid_decision_process()