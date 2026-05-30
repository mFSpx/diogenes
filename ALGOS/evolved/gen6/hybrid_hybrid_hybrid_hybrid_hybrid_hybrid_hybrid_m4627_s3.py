# DARWIN HAMMER — match 4627, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m1867_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_liquid_hybrid_pheromone_hyb_m1131_s2.py (gen5)
# born: 2026-05-29T23:57:01Z

"""Hybrid Algorithm integrating Parent A (bandit + Schoolfield temperature curve) 
and Parent B (MinHash similarity + probabilistic maximal independent set).

Mathematical Bridge:
- Parent B provides a Jaccard‑like similarity matrix S derived from MinHash signatures 
  of token sets associated with graph nodes.
- Parent A updates a probability vector p (propensities of bandit actions) via a linear 
  matrix operation p' = (I + α·S)·p, where α scales the influence of similarity on the 
  bandit’s exploration distribution.
- The temperature‑dependent scaling from the Schoolfield model (ρ(T)) further modulates 
  the updated propensities, yielding p'' = ρ(T)·p'.

Thus the core fusion consists of:
    1) Computing S from graph node token lists (Parent B).
    2) Updating bandit propensities with S (Parent A matrix update).
    3) Applying the Schoolfield temperature performance factor (Parent A) to the 
       resulting propensities.

The module exposes three primary hybrid functions demonstrating this pipeline.
"""

import math
import random
import sys
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable, Set, Mapping, Any, Hashable

import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Immutable description of a bandit arm."""
    action_id: str
    propensity: float  # raw (unnormalized) propensity
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Immutable record of a single interaction."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters of the Schoolfield temperature‑performance curve."""
    rho_25: float = 1.0                # rate at 25 °C (arbitrary units)
    delta_h_activation: float = 12_000.0  # J mol⁻¹
    t_low: float = 283.15              # K  (≈10 °C)
    t_high: float = 307.15             # K  (≈34 °C)
    delta_h_low: float = -45_000.0     # J mol⁻¹
    delta_h_high: float = 65_000.0     # J mol⁻¹
    r_cal: float = 1.987               # cal mol⁻¹ K⁻¹ (≈8.314 J mol⁻¹ K⁻¹)


def schoolfield_rate(T: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Compute the temperature‑dependent rate ρ(T) using the Schoolfield model."""
    R = params.r_cal * 4.184  # convert cal mol⁻¹ K⁻¹ to J mol⁻¹ K⁻¹
    term_activation = math.exp(-params.delta_h_activation / (R * (1.0 / T - 1.0 / 298.15)))
    term_low = 1.0 + math.exp(params.delta_h_low / R * (1.0 / params.t_low - 1.0 / T))
    term_high = 1.0 + math.exp(params.delta_h_high / R * (1.0 / T - 1.0 / params.t_high))
    return params.rho_25 * term_activation / (term_low * term_high)


# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    """Return a MinHash signature of length k for the given token list."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def minhash_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity based on MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def similarity_matrix(
    node_tokens: Mapping[Hashable, List[str]],
    k: int = 128,
) -> np.ndarray:
    """
    Build a symmetric similarity matrix S where S[i, j] = similarity of node i and j.
    The order of nodes follows the iteration order of node_tokens.keys().
    """
    nodes = list(node_tokens.keys())
    n = len(nodes)
    sigs = [minhash_signature(node_tokens[nod], k) for nod in nodes]
    S = np.eye(n)  # diagonal = 1.0 by definition
    for i in range(n):
        for j in range(i + 1, n):
            sim = minhash_similarity(sigs[i], sigs[j])
            S[i, j] = S[j, i] = sim
    return S, nodes


def broadcast_probability(phase: int, steps: int) -> float:
    """
    Return p = 1 / 2**(phase‑steps) clamped to [0, 1].
    When phase ≤ steps the probability is 1 (full broadcast).
    """
    if phase < 1 or steps < 1:
        raise ValueError("phase and steps must be positive")
    exponent = max(0, phase - steps)
    return min(1.0, 1.0 / (2**exponent))


def maximal_independent_set(
    graph: Mapping[Hashable, Set[Hashable]],
    phases: int = 8,
    seed: Any = None,
) -> Set[Hashable]:
    """
    Approximate a maximal independent set using probabilistic local broadcasts.
    The probability schedule follows broadcast_probability(phase, phases).
    """
    rng = random.Random(seed)
    undecided: Set[Hashable] = set(graph)
    leaders: Set[Hashable] = set()
    blocked: Set[Hashable] = set()

    for phase in range(1, phases + 1):
        if not undecided:
            break
        p = broadcast_probability(phase, phases)
        broadcasts = {n for n in undecided if rng.random() < p}
        new_leaders = {
            n
            for n in broadcasts
            if not (graph.get(n, set()) & blocked)
        }
        leaders.update(new_leaders)
        # block neighbours of new leaders
        for n in new_leaders:
            blocked.update(graph.get(n, set()))
        undecided -= (new_leaders | blocked)
    return leaders


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def hybrid_propensity_update(
    actions: List[BanditAction],
    similarity: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Update raw propensities using the similarity matrix.
    p' = (I + α·S)·p where p is the column vector of current propensities.
    Returns a normalized probability distribution (sums to 1).
    """
    if similarity.shape[0] != similarity.shape[1]:
        raise ValueError("Similarity matrix must be square")
    n = similarity.shape[0]
    if len(actions) != n:
        raise ValueError("Number of actions must match similarity matrix size")
    p = np.array([a.propensity for a in actions], dtype=float).reshape((n, 1))
    I = np.eye(n)
    updated = (I + alpha * similarity) @ p
    # Ensure non‑negative and normalize
    updated = np.clip(updated, a_min=0.0, a_max=None)
    total = updated.sum()
    if total == 0.0:
        # fallback to uniform distribution
        return np.full((n, 1), 1.0 / n)
    return updated / total


def hybrid_temperature_scaled_distribution(
    actions: List[BanditAction],
    temperature_K: float,
    similarity: np.ndarray,
    alpha: float = 0.5,
    params: SchoolfieldParams = SchoolfieldParams(),
) -> List[Tuple[str, float]]:
    """
    Combine similarity‑driven propensity update with Schoolfield temperature scaling.
    Returns a list of (action_id, final_probability) pairs.
    """
    # 1) similarity‑adjusted propensities
    prob_vec = hybrid_propensity_update(actions, similarity, alpha)  # column vector

    # 2) temperature scaling factor
    rho_T = schoolfield_rate(temperature_K, params)

    # 3) apply scaling and renormalize
    scaled = prob_vec * rho_T
    total = scaled.sum()
    if total == 0.0:
        final = np.full_like(scaled, 1.0 / len(actions))
    else:
        final = scaled / total

    return [(actions[i].action_id, float(final[i])) for i in range(len(actions))]


def hybrid_maximal_independent_set(
    graph: Mapping[Hashable, Set[Hashable]],
    node_tokens: Mapping[Hashable, List[str]],
    phases: int = 8,
    alpha: float = 0.5,
    temperature_K: float = 298.15,
) -> Set[Hashable]:
    """
    Hybrid MIS:
    1) Compute similarity matrix from node tokens.
    2) Derive a weighting vector via the bandit‑style update (using dummy actions).
    3) Modulate broadcast probabilities of each node by its weight * temperature factor.
    4) Run a probabilistic MIS selection using the modulated probabilities.
    """
    # Build similarity matrix
    S, ordered_nodes = similarity_matrix(node_tokens)

    # Create placeholder BanditAction objects (one per node) with equal base propensities
    base_propensity = 1.0
    dummy_actions = [
        BanditAction(action_id=str(node), propensity=base_propensity,
                     expected_reward=0.0, confidence_bound=0.0, algorithm="hybrid_dummy")
        for node in ordered_nodes
    ]

    # Update propensities with similarity and temperature
    probs = hybrid_temperature_scaled_distribution(
        dummy_actions,
        temperature_K,
        S,
        alpha=alpha,
    )
    # Map node -> probability
    node_to_prob = {node: prob for (node, prob) in probs}

    # Modified MIS that respects node probabilities
    rng = random.Random()
    undecided: Set[Hashable] = set(graph)
    leaders: Set[Hashable] = set()
    blocked: Set[Hashable] = set()

    for phase in range(1, phases + 1):
        if not undecided:
            break
        base_p = broadcast_probability(phase, phases)
        # node‑specific broadcast probability
        broadcasts = {
            n for n in undecided
            if rng.random() < base_p * node_to_prob.get(n, 0.0)
        }
        new_leaders = {
            n for n in broadcasts
            if not (graph.get(n, set()) & blocked)
        }
        leaders.update(new_leaders)
        for n in new_leaders:
            blocked.update(graph.get(n, set()))
        undecided -= (new_leaders | blocked)
    return leaders


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple graph with token lists
    graph_example: Dict[int, Set[int]] = {
        0: {1, 2},
        1: {0, 3},
        2: {0, 3},
        3: {1, 2, 4},
        4: {3},
    }
    token_example: Dict[int, List[str]] = {
        0: ["apple", "banana", "cherry"],
        1: ["banana", "date"],
        2: ["apple", "fig"],
        3: ["grape", "cherry", "date"],
        4: ["fig", "grape"],
    }

    # Create some dummy bandit actions
    actions = [
        BanditAction(action_id="a0", propensity=1.0, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="demo"),
        BanditAction(action_id="a1", propensity=2.0, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="demo"),
        BanditAction(action_id="a2", propensity=1.5, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="demo"),
        BanditAction(action_id="a3", propensity=0.5, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="demo"),
        BanditAction(action_id="a4", propensity=1.2, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="demo"),
    ]

    # Compute similarity matrix
    S, order = similarity_matrix(token_example, k=64)

    # Hybrid probability distribution
    dist = hybrid_temperature_scaled_distribution(
        actions,
        temperature_K=300.0,
        similarity=S,
        alpha=0.3,
    )
    print("Hybrid probability distribution:")
    for aid, prob in dist:
        print(f"  {aid}: {prob:.4f}")

    # Hybrid maximal independent set
    mis = hybrid_maximal_independent_set(
        graph=graph_example,
        node_tokens=token_example,
        phases=6,
        alpha=0.4,
        temperature_K=300.0,
    )
    print("\nHybrid maximal independent set:", mis)