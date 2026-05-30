# DARWIN HAMMER — match 1365, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_liquid_hybrid_path_signatur_m113_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m813_s1.py (gen5)
# born: 2026-05-29T23:35:32Z

"""
This module implements a novel hybrid algorithm that mathematically fuses the core topologies of 
two parent algorithms: hybrid_hybrid_hybrid_liquid_hybrid_path_signatur_m113_s3 and 
hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m813_s1.

The mathematical bridge between the two parents lies in the use of MinHash signatures and 
Hoeffding bound. The first parent uses MinHash signatures to compute the similarity between 
two token sets, while the second parent uses Hoeffding bound to evaluate the uncertainty of 
the regret-weighted strategy. This hybrid algorithm combines these two concepts by using 
the MinHash signatures to compute the similarity between the token sets and then using the 
Hoeffding bound to evaluate the uncertainty of the regret-weighted strategy.

The hybrid algorithm works as follows: for each token set, it computes the MinHash signature, 
then uses the Hoeffding bound to evaluate the uncertainty of the regret-weighted strategy 
based on the similarity between the token sets.
"""

import numpy as np
import math
import random
import sys
import pathlib
import hashlib

MAX64 = (1 << 64) - 1

def _hash(seed: int, token: str) -> int:
    """Deterministic 64-bit hash of a token with a seed."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def minhash_signature(tokens: list[str], k: int = 128) -> list[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: list[int], sig_b: list[int]) -> float:
    """Jaccard-like similarity of two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def hoeffding_bound(n: int, epsilon: float, delta: float) -> float:
    return math.sqrt((2 * math.log(2 / delta)) / n) + (epsilon * (n - 1) / n)

def compute_regret_weighted_strategy(
    actions: list, counterfactuals: list, token_sets: list
) -> dict[str, float]:
    if not actions:
        return {}
    exp_map = {a['id']: a['expected_value'] for a in actions}
    regrets = {a['id']: 0.0 for a in actions}
    similarities = []
    for i in range(len(token_sets)):
        sig_a = minhash_signature(token_sets[i])
        similarities.append([])
        for j in range(len(token_sets)):
            if i == j:
                similarities[-1].append(1.0)
                continue
            sig_b = minhash_signature(token_sets[j])
            similarities[-1].append(similarity(sig_a, sig_b))

    for cf in counterfactuals:
        if cf['action_id'] not in exp_map:
            continue
        diff = cf['outcome_value'] - exp_map[cf['action_id']]
        regrets[cf['action_id']] += diff * cf['probability']

    max_r = max(regrets.values())
    exp_vals = {aid: math.exp(r - max_r) for aid, r in regrets.items()}
    z = sum(exp_vals.values())
    probs = {aid: v / z for aid, v in exp_vals.items()}
    uncertainties = []
    for i in range(len(token_sets)):
        n = len(token_sets[i])
        epsilon = 0.1
        delta = 0.1
        uncertainty = hoeffding_bound(n, epsilon, delta)
        uncertainties.append(uncertainty)

    return probs, similarities, uncertainties

def shingles(text: str, width: int = 5) -> set[str]:
    """Extract width-wise word shingles from a string."""
    words = text.split()
    if width <= 0:
        raise ValueError("width must be positive")
    if len(words) < width:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + width]) for i in range(len(words) - width + 1)}

def noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """
    Diffusion-forcing schedule ᾱₜ ∈ (0,1] for t=0…T.
    Cosine schedule follows the DDPM formulation; linear is a simple
    decreasing schedule.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    steps = np.arange(T + 1, dtype=np.float64)

    if schedule == "cosine":
        s = 0.008
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        alpha_bars = np.clip(alpha_bars, 1e-9, 1.0)
        return alpha_bars
    elif schedule == "linear":
        beta_start = 1e-4
        beta_end = 0.02
        beta = beta_start + (beta_end - beta_start) * steps / T
        alpha_bars = np.cumprod(1 - beta)
        return alpha_bars

if __name__ == "__main__":
    token_sets = [["apple", "banana", "orange"], ["banana", "orange", "grape"]]
    actions = [{"id": 1, "expected_value": 10.0}, {"id": 2, "expected_value": 20.0}]
    counterfactuals = [
        {"action_id": 1, "outcome_value": 15.0, "probability": 0.5},
        {"action_id": 2, "outcome_value": 25.0, "probability": 0.5},
    ]
    probs, similarities, uncertainties = compute_regret_weighted_strategy(actions, counterfactuals, token_sets)
    print(probs)
    print(similarities)
    print(uncertainties)