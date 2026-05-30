# DARWIN HAMMER — match 25, survivor 5
# gen: 4
# parent_a: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s4.py (gen2)
# parent_b: hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s8.py (gen3)
# born: 2026-05-29T23:26:33Z

"""Hybrid Leader‑Election & Regret‑Weighted Tree with Tropical Max‑Plus and Hoeffding Bounds.

Parents:
- hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s4 (simulated‑annealing
  acceptance of Hoeffding‑tree splits, tropical max‑plus gain as “energy”)
- hybrid_hybrid_regret_engine_hybrid_ternary_lens__m3_s8 (signature based
  similarity, regret‑weighted probability distribution)

Mathematical bridge:
Both parents decide *whether* a structural change (a split, a broadcast, a
leader promotion) should be kept.  The tree parent provides a scalar energy
E = G (tropical gain) and a Hoeffding bound ε that plays the role of a
temperature‑like uncertainty.  The regret engine supplies a probability
distribution π over actions (leaders) and a similarity measure σ between
their signatures.  We fuse them by defining a combined ΔE = ε – G and an
effective temperature

    T_eff = T / (1 + λ·σ)

where λ controls how much similarity of leader signatures cools the system.
The acceptance probability is then

    p_accept = exp( –ΔE / T_eff ).

The hybrid algorithm thus simultaneously:
* evaluates candidate splits with Hoeffding‑bound + tropical max‑plus,
* selects leaders with regret‑weighted probabilities,
* modulates simulated‑annealing acceptance by signature similarity.

The module implements the core mathematics and provides three public
functions demonstrating the hybrid operation.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, List, Mapping, Tuple, Dict, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (derived from Parent B)
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
# Parent‑A primitives (broadcast & Hoeffding‑tree)
# ----------------------------------------------------------------------
def broadcast_probability(total_phases: int, current_phase: int) -> float:
    """Probability that a node broadcasts in the current phase.

    Mirrors the original parent implementation: early phases broadcast more
    aggressively, later phases decay exponentially.
    """
    if total_phases < 1 or current_phase < 1:
        raise ValueError("phases and phase must be positive")
    # Decay factor grows with the distance from the final phase.
    decay = 2 ** max(0, total_phases - current_phase)
    return min(1.0, 1.0 / decay)


def hoeffding_bound(R: float, delta: float, n: int) -> float:
    """Classic Hoeffding bound ε = sqrt( (R^2 * ln(1/δ)) / (2n) )."""
    if n <= 0:
        raise ValueError("sample size n must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2.0 * n))


def tropical_max_plus_gain(gains: np.ndarray) -> float:
    """Tropical max‑plus evaluation: max_i (gain_i + weight_i) with weight=0."""
    # In pure max‑plus algebra the “addition” is max and “multiplication” is +.
    # With unit weights the gain reduces to the ordinary maximum.
    return float(np.max(gains))


# ----------------------------------------------------------------------
# Parent‑B primitives (signatures, regret)
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
    """Min‑hash style signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two min‑hash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def compute_regret_weighted_strategy(
    actions: List[MathAction], counterfactuals: List[MathCounterfactual]
) -> Dict[str, float]:
    """Soft‑max over cumulative regret (Parent‑B core)."""
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
    """Return (ε, G, ΔE) for a candidate split.

    ε   – Hoeffding bound (uncertainty)
    G   – Tropical max‑plus gain (energy, higher is better)
    ΔE  – ε – G  (the lower the better; negative ΔE favours acceptance)
    """
    epsilon = hoeffding_bound(R, delta, n_samples)
    G = tropical_max_plus_gain(sample_gains)
    delta_E = epsilon - G
    return epsilon, G, delta_E


def simulated_annealing_accept(delta_E: float, T: float, sigma: float = 0.0, lam: float = 0.5) -> bool:
    """Accept a candidate with probability exp(‑ΔE / T_eff).

    T_eff = T / (1 + λ·σ) where σ∈[0,1] is a similarity score between the
    candidate’s signature and the current leader’s signature.
    """
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
    """Select a leader among candidates using regret‑weighted probabilities
    modulated by signature similarity.

    Returns (selected_action_id, acceptance_probability).
    """
    # 1. Regret‑weighted base distribution
    base_probs = compute_regret_weighted_strategy(actions, counterfactuals)

    # 2. Signature similarity between current leader and each candidate
    leader_sig = signature(leader_tokens)
    cand_sig = signature(candidate_tokens)

    sim = similarity(leader_sig, cand_sig)  # σ ∈ [0,1]

    # 3. Temperature adjustment
    T_eff = temperature / (1.0 + lam * sim)

    # 4. Sample according to the softened distribution
    ids, probs = zip(*base_probs.items())
    probs = np.array(probs, dtype=float)
    probs /= probs.sum()
    chosen_id = random.choices(ids, weights=probs, k=1)[0]

    # 5. Acceptance probability (higher similarity cools the system)
    accept_prob = math.exp(- (1.0 - sim) / T_eff)
    return chosen_id, accept_prob


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def hybrid_split_decision(
    sample_gains: np.ndarray,
    R: float,
    delta: float,
    n_samples: int,
    temperature: float,
    leader_sig_tokens: Iterable[str],
    candidate_sig_tokens: Iterable[str],
    lam: float = 0.5,
) -> bool:
    """Decide whether to accept a tree split and promote its root to leader.

    The decision follows the combined acceptance rule described in the
    module docstring.
    """
    epsilon, G, delta_E = compute_split_metric(sample_gains, R, delta, n_samples)
    sigma = similarity(signature(leader_sig_tokens), signature(candidate_sig_tokens))
    return simulated_annealing_accept(delta_E, temperature, sigma, lam)


def hybrid_broadcast_decision(
    total_phases: int,
    current_phase: int,
    temperature: float,
    leader_sig_tokens: Iterable[str],
    candidate_sig_tokens: Iterable[str],
) -> bool:
    """Determine whether a node should broadcast in the current phase.

    The basic broadcast probability is modulated by a simulated‑annealing
    acceptance that depends on signature similarity.
    """
    base_p = broadcast_probability(total_phases, current_phase)
    sigma = similarity(signature(leader_sig_tokens), signature(candidate_sig_tokens))
    # Treat σ as a reduction of temperature → more similar nodes broadcast more.
    accept = simulated_annealing_accept(-math.log(base_p + 1e-12), temperature, sigma)
    return accept


def hybrid_regret_leader_step(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    current_leader_tokens: Iterable[str],
    candidate_tokens: Iterable[str],
    temperature: float,
) -> str:
    """Perform one hybrid leader‑selection step and return the new leader id."""
    selected_id, accept_prob = hybrid_leader_selection(
        actions,
        counterfactuals,
        current_leader_tokens,
        candidate_tokens,
        temperature,
    )
    # Stochastically accept the new leader according to the computed probability.
    if random.random() < accept_prob:
        return selected_id
    # Fallback: keep the current leader (assumed to be the first action id)
    return actions[0].id if actions else ""


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy data for split decision
    gains = np.array([0.12, 0.35, 0.07, 0.22])
    eps, G, dE = compute_split_metric(gains, R=1.0, delta=0.05, n_samples=100)
    print(f"Hoeffding ε={eps:.4f}, Tropical G={G:.4f}, ΔE={dE:.4f}")

    decision = hybrid_split_decision(
        sample_gains=gains,
        R=1.0,
        delta=0.05,
        n_samples=100,
        temperature=0.8,
        leader_sig_tokens=["alpha", "beta"],
        candidate_sig_tokens=["gamma", "delta"],
    )
    print(f"Hybrid split accepted? {decision}")

    # Dummy data for broadcast decision
    broadcast = hybrid_broadcast_decision(
        total_phases=10,
        current_phase=3,
        temperature=0.5,
        leader_sig_tokens=["nodeA", "nodeB"],
        candidate_sig_tokens=["nodeC"],
    )
    print(f"Broadcast decision: {broadcast}")

    # Dummy actions for leader selection
    actions = [
        MathAction(id="L1", expected_value=0.4),
        MathAction(id="L2", expected_value=0.6),
        MathAction(id="L3", expected_value=0.5),
    ]
    counterfactuals = [
        MathCounterfactual(action_id="L1", outcome_value=0.5, probability=0.9),
        MathCounterfactual(action_id="L2", outcome_value=0.4, probability=0.7),
        MathCounterfactual(action_id="L3", outcome_value=0.6, probability=0.6),
    ]
    new_leader = hybrid_regret_leader_step(
        actions,
        counterfactuals,
        current_leader_tokens=["L1"],
        candidate_tokens=["L2", "L3"],
        temperature=0.7,
    )
    print(f"New leader selected: {new_leader}")