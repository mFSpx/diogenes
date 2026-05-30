# DARWIN HAMMER — match 5594, survivor 7
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1876_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_bandit_router_m1212_s5.py (gen3)
# born: 2026-05-30T00:03:18Z

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0      
    t_low: float = 283.15                     
    t_high: float = 307.15                    
    delta_h_low: float = -45_000.0            
    delta_h_high: float = 65_000.0            
    r_cal: float = 1.987                     


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin-positive and rho_25 non-negative")
    numerator = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)


@dataclass(frozen=True)
class PheromoneParams:
    surface_key: str
    limit: int = 10                     
    base_prob: float = 0.1              


def generate_pheromone_vector(p_params: PheromoneParams, temp_k: float) -> List[float]:
    rate = developmental_rate(temp_k)
    raw = np.full(p_params.limit, p_params.base_prob)
    for i in range(p_params.limit):
        raw[i] *= (1.0 + 0.5 * math.sin(rate * (i + 1)))
    probs = raw / raw.sum()
    return probs.tolist()


def deterministic_hash(token: str, seed: int) -> int:
    combined = f"{token}:{seed}"
    h = hash(combined) & ((1 << 64) - 1)
    return h


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


def shannon_entropy(probs: List[float], eps: float = 1e-12) -> float:
    total = sum(probs)
    if total <= 0:
        raise ValueError("positive probability mass required")
    p = np.array(probs) / total
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))


def similarity_to_temperature(similarity: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("similarity must be in [0,1]")
    return params.t_low + similarity * (params.t_high - params.t_low)


def hybrid_entropy_with_rate(
    tokens_a: List[str],
    tokens_b: List[str],
    num_hash_functions: int,
    p_params: PheromoneParams,
) -> Tuple[float, float]:
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
    temp_k = similarity_to_temperature(similarity, params)
    rate = developmental_rate(temp_k, params)
    scaled = np.array(propensities) * rate
    if scaled.sum() == 0:
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
    sig_a = minhash_signature(tokens_a, num_hash_functions)
    sig_b = minhash_signature(tokens_b, num_hash_functions)
    sim = minhash_similarity(sig_a, sig_b)
    temp_k = similarity_to_temperature(sim)
    rate = developmental_rate(temp_k)
    pheromone_vec = generate_pheromone_vector(p_params, temp_k)
    entropy = shannon_entropy(pheromone_vec)
    weighted_entropy = entropy * rate
    action = rate_scaled_bandit_action(bandit_propensities, sim)
    return {
        "similarity": sim,
        "temperature": temp_k,
        "rate": rate,
        "pheromone_vector": pheromone_vec,
        "entropy": entropy,
        "weighted_entropy": weighted_entropy,
        "action": action,
    }