# DARWIN HAMMER — match 1212, survivor 6
# gen: 3
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_liquid_time_c_m288_s1.py (gen2)
# parent_b: hybrid_bandit_router_poikilotherm_schoolf_m20_s0.py (gen1)
# born: 2026-05-29T23:34:35Z

"""Hybrid algorithm merging MinHash‑based infotaxis (Parent A) with temperature‑dependent bandit routing (Parent B).

Mathematical bridge:
- Parent A selects actions by minimizing the *expected Shannon entropy* of hit/miss states,
  where the hit probability `p_hit` can be interpreted as a similarity measure.
- Parent B provides a *developmental rate* `ρ(T)` (Schoolfield model) that varies with temperature
  and acts as a temperature‑dependent scaling factor for the certainty of a hit.

In this hybrid we compute `p_hit` from MinHash similarity, evaluate the expected entropy,
and then modulate that entropy by `1/ρ(T)`.  The resulting *temperature‑aware entropy* is used
as the selection criterion for a contextual bandit‑style routing decision."""


import hashlib
import math
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Parent A utilities (MinHash & entropy)
# ----------------------------------------------------------------------
def deterministic_hash(token: str, seed: int) -> int:
    """Deterministic 64‑bit integer hash of ``token`` combined with ``seed``."""
    h = hashlib.sha256(f"{token}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def minhash_signature(tokens: List[str], num_hash_functions: int) -> List[int]:
    """Compute a deterministic MinHash signature for a list of tokens."""
    signature: List[int] = []
    for i in range(num_hash_functions):
        min_hash = (1 << 64) - 1
        for token in tokens:
            h = deterministic_hash(token, i)
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    return signature


def minhash_similarity(sig1: List[int], sig2: List[int]) -> float:
    """Jaccard‑like similarity based on identical positions in two signatures."""
    if len(sig1) != len(sig2) or not sig1:
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def calculate_entropy(probs: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability distribution."""
    total = sum(probs)
    if total <= 0:
        raise ValueError("positive probability mass required")
    p = np.array(probs) / total
    p = np.clip(p, eps, 1.0)
    return -float(np.sum(p * np.log(p)))


def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
    """Weighted expected entropy for a hit/miss scenario."""
    if not 0.0 <= p_hit <= 1.0:
        raise ValueError("p_hit must be in [0, 1]")
    return p_hit * calculate_entropy(hit_state) + (1.0 - p_hit) * calculate_entropy(miss_state)


# ----------------------------------------------------------------------
# Parent B utilities (Schoolfield temperature model & bandit policy)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0               # rate at 25 °C (298.15 K)
    delta_h_activation: float = 12_000.0  # activation enthalpy (J mol⁻¹)
    t_low: float = 283.15            # low‑temperature breakpoint (K)
    t_high: float = 307.15           # high‑temperature breakpoint (K)
    delta_h_low: float = -45_000.0   # low‑temp deactivation enthalpy (J mol⁻¹)
    delta_h_high: float = 65_000.0   # high‑temp deactivation enthalpy (J mol⁻¹)
    r_cal: float = 1.987             # gas constant (cal mol⁻¹ K⁻¹)


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield‑Rollinson temperature‑dependent rate (dimensionless)."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    # Core Arrhenius term
    arr = params.rho_25 * (temp_k / 298.15) * math.exp(
        (params.delta_h_activation / params.r_cal) * ((1.0 / 298.15) - (1.0 / temp_k))
    )
    # Low‑ and high‑temperature deactivation factors
    low = math.exp((params.delta_h_low / params.r_cal) * ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) * ((1.0 / params.t_high) - (1.0 / temp_k)))
    return arr / (1.0 + low + high)


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def hybrid_expected_entropy(
    action_tokens: List[str],
    reference_tokens: List[str],
    temperature_c: float,
    num_minhash: int = 128,
) -> float:
    """
    Compute a temperature‑aware expected entropy for ``action_tokens`` against
    ``reference_tokens``.

    Steps
    -----
    1. Build MinHash signatures and obtain a similarity ``p_hit``.
    2. Derive hit/miss state frequency vectors from token counts.
    3. Compute the raw expected entropy using ``expected_entropy``.
    4. Scale the entropy by ``1 / developmental_rate(T)`` where ``T`` is Kelvin.
       Higher developmental rates (optimal temperatures) therefore *reduce* the
       entropy, biasing the selector toward actions that are both similar and
       thermally favourable.
    """
    # 1. similarity as hit probability
    sig_a = minhash_signature(action_tokens, num_minhash)
    sig_ref = minhash_signature(reference_tokens, num_minhash)
    p_hit = minhash_similarity(sig_a, sig_ref)

    # 2. frequency vectors (simple bag‑of‑words counts)
    vocab = list(set(action_tokens + reference_tokens))
    idx = {w: i for i, w in enumerate(vocab)}
    hit_counts = [0.0] * len(vocab)
    miss_counts = [0.0] * len(vocab)
    for w in action_tokens:
        hit_counts[idx[w]] += 1.0
    for w in reference_tokens:
        miss_counts[idx[w]] += 1.0

    # 3. raw expected entropy
    raw_entropy = expected_entropy(p_hit, hit_counts, miss_counts)

    # 4. temperature scaling
    temp_k = temperature_c + 273.15
    rate = developmental_rate(temp_k)
    if rate <= 0:
        # Guard against division by zero – fall back to raw entropy
        return raw_entropy
    return raw_entropy / rate


def select_best_action(
    actions: Dict[str, List[str]],
    reference: List[str],
    temperature_c: float,
    num_minhash: int = 128,
) -> Tuple[str, float]:
    """
    Choose the action with the lowest temperature‑aware expected entropy.

    Returns
    -------
    (action_id, hybrid_entropy)
    """
    best_id = None
    best_score = math.inf
    for aid, tokens in actions.items():
        score = hybrid_expected_entropy(tokens, reference, temperature_c, num_minhash)
        if score < best_score:
            best_score = score
            best_id = aid
    if best_id is None:
        raise RuntimeError("No actions provided")
    return best_id, best_score


def update_action_statistics(
    stats: Dict[str, Dict[str, int]],
    updates: List[Tuple[str, List[str]]],
) -> None:
    """
    Increment token occurrence counters for actions.

    Parameters
    ----------
    stats
        Mapping ``action_id -> {token -> count}``.
    updates
        Iterable of ``(action_id, token_list)`` tuples representing new observations.
    """
    for aid, tokens in updates:
        token_map = stats.setdefault(aid, {})
        for t in tokens:
            token_map[t] = token_map.get(t, 0) + 1


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny vocabulary and three candidate actions
    actions = {
        "A": ["alpha", "beta", "gamma", "delta"],
        "B": ["epsilon", "zeta", "eta", "theta", "iota"],
        "C": ["alpha", "eta", "kappa", "lambda"],
    }

    # Reference token set (e.g., user query or target document)
    reference = ["alpha", "beta", "eta", "mu", "nu"]

    # Simulated environmental temperature (°C)
    temperature_c = random.uniform(5.0, 35.0)

    # Select the best action under current temperature
    chosen_id, score = select_best_action(actions, reference, temperature_c)
    print(f"Temperature: {temperature_c:.2f} °C")
    print(f"Chosen action: {chosen_id} with hybrid entropy {score:.4f}")

    # Demonstrate statistics update
    stats: Dict[str, Dict[str, int]] = {}
    update_action_statistics(stats, [("A", actions["A"]), ("B", actions["B"])])
    print("Updated token statistics (partial view):")
    for aid, cnts in stats.items():
        print(f"  {aid}: {cnts}")