# DARWIN HAMMER — match 25, survivor 6
# gen: 4
# parent_a: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s4.py (gen2)
# parent_b: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s8.py (gen3)
# born: 2026-05-29T23:26:33Z

from __future__ import annotations

import math
import random
import sys
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, List, Mapping, Tuple, Dict, Hashable

import hashlib

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


# ----------------------------------------------------------------------
# Parent‑A primitives 
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    if total_phases < 1 or current_phase < 1:
        raise ValueError("phases and phase must be positive")
    decay = 2 ** max(0, total_phases - current_phase)
    return min(1.0, 1.0 / decay)


def hoeffding_bound(R: float, delta: float, n: int) -> float:
    if n <= 0:
        raise ValueError("sample size n must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2.0 * n))


def tropical_max_plus_gain(gains: np.ndarray) -> float:
    return float(np.max(gains))


# ----------------------------------------------------------------------
# Parent‑B primitives 
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        np.frombuffer(
            np.frombuffer(
                bytearray(hashlib.blake2b(data, digest_size=8).digest()),
                dtype=np.uint8,
            ),
            dtype=np.uint64,
        ),
        "big",
    )


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def compute_regret_weighted_strategy(
    actions: List[MathAction], counterfactuals: List[MathCounterfactual]
) -> Dict[str, float]:
    if not actions:
        return {}
    exp_map = {a.id: a.expected_value for a in actions}
    regrets = {a.id: 0.0 for a in actions}
    for cf in counterfactuals:
        if cf.action_id not in exp_map:
            continue
        diff = cf.outcome_value - exp_map[cf.action_id]
        regrets[cf.action_id] += diff * cf.probability

    max_r = max(regrets.values())
    exp_vals = {aid: math.exp(r - max_r) for aid, r in regrets.items()}
    z = sum(exp_vals.values())
    return {aid: v / z for aid, v in exp_vals.items()}


# ----------------------------------------------------------------------
# Hybrid core mathematics
# ----------------------------------------------------------------------
def compute_split_metric(
    sample_gains: np.ndarray,
    R: float,
    delta: float,
    n_samples: int,
) -> Tuple[float, float, float]:
    epsilon = hoeffding_bound(R, delta, n_samples)
    G = tropical_max_plus_gain(sample_gains)
    delta_E = epsilon - G
    return epsilon, G, delta_E


def simulated_annealing_accept(delta_E: float, T: float, sigma: float = 0.0, lam: float = 0.5) -> bool:
    if T <= 0:
        raise ValueError("Temperature T must be positive")
    T_eff = T / (1.0 + lam * sigma)
    prob = math.exp(-delta_E / T_eff) if delta_E > 0 else 1.0
    return random.random() < prob


def hybrid_leader_selection(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    leader_tokens: Iterable[str],
    candidate_tokens: Iterable[str],
    temperature: float,
    lam: float = 0.5,
) -> Tuple[str, float]:
    base_probs = compute_regret_weighted_strategy(actions, counterfactuals)

    leader_sig = signature(leader_tokens)
    cand_sig = signature(candidate_tokens)

    sim = similarity(leader_sig, cand_sig)  

    T_eff = temperature / (1.0 + lam * sim)

    ids, probs = zip(*base_probs.items())
    probs = np.array(probs, dtype=float)
    probs /= probs.sum()
    chosen_id = np.random.choice(ids, p=probs)

    accept_prob = math.exp(- (1.0 - sim) / T_eff)
    return chosen_id, accept_prob


def hybrid_operation(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    leader_tokens: Iterable[str],
    candidate_tokens: Iterable[str],
    sample_gains: np.ndarray,
    R: float,
    delta: float,
    n_samples: int,
    temperature: float,
    lam: float = 0.5,
) -> Tuple[float, float, float, str, float]:
    epsilon, G, delta_E = compute_split_metric(sample_gains, R, delta, n_samples)
    chosen_id, accept_prob = hybrid_leader_selection(
        actions, counterfactuals, leader_tokens, candidate_tokens, temperature, lam
    )
    return epsilon, G, delta_E, chosen_id, accept_prob


# Example usage
if __name__ == "__main__":
    actions = [MathAction("action1", 10.0), MathAction("action2", 20.0)]
    counterfactuals = [
        MathCounterfactual("action1", 15.0),
        MathCounterfactual("action2", 25.0),
    ]
    leader_tokens = ["token1", "token2"]
    candidate_tokens = ["token3", "token4"]
    sample_gains = np.array([10.0, 20.0])
    R = 10.0
    delta = 0.1
    n_samples = 100
    temperature = 1000.0

    epsilon, G, delta_E, chosen_id, accept_prob = hybrid_operation(
        actions,
        counterfactuals,
        leader_tokens,
        candidate_tokens,
        sample_gains,
        R,
        delta,
        n_samples,
        temperature,
    )

    print(f"epsilon: {epsilon}, G: {G}, delta_E: {delta_E}, chosen_id: {chosen_id}, accept_prob: {accept_prob}")