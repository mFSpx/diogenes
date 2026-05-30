# DARWIN HAMMER — match 2082, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_ternar_m333_s0.py (gen3)
# parent_b: hybrid_ternary_router_hybrid_minimum_cost__m36_s2.py (gen2)
# born: 2026-05-29T23:40:52Z

import math
import random
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared constants and helpers
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return weekday index where 0 = Sunday … 6 = Saturday.
    """
    return (date(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Tuple[str, ...], dow: int) -> np.ndarray:
    """
    Produce a normalized weight vector for *groups* based on the weekday ``dow``.
    """
    n = len(groups)
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def _hash(seed: int, token: str) -> int:
    """Hash a token with a 4‑byte seed using a simple mixing function."""
    h = np.uint64(seed)
    for c in token:
        h = np.uint64(h ^ ord(c))
        h = np.uint64(h * 0x100000001B3)
        h &= MAX64
    return int(h)


def minhash_similarity(token_a: str, token_b: str, seed: int = 0xDEADBEEF) -> float:
    """
    Compute a Jaccard‑like similarity from two MinHash signatures.
    The result is in [0, 1].
    """
    ha = _hash(seed, token_a)
    hb = _hash(seed, token_b)
    # Hamming distance on 64‑bit signatures
    diff = bin(ha ^ hb).count("1")
    return 1.0 - diff / 64.0


# ----------------------------------------------------------------------
# Variational Free Energy
# ----------------------------------------------------------------------
def variational_free_energy(q: np.ndarray, p: np.ndarray) -> float:
    """
    Compute the variational free energy between two probability distributions.

    Parameters
    ----------
    q : np.ndarray
        Approximate distribution (must sum to 1, non‑negative).
    p : np.ndarray
        Target distribution (must sum to 1, non‑negative).

    Returns
    -------
    float
        The variational free energy Σ q·log(q/p).
    """
    eps = np.finfo(float).eps
    q_safe = np.clip(q, eps, 1.0)
    p_safe = np.clip(p, eps, 1.0)
    return float(np.sum(q_safe * np.log(q_safe / p_safe)))


# ----------------------------------------------------------------------
# Hybrid LTC – integrates free energy and MinHash similarity
# ----------------------------------------------------------------------
def hybrid_liquid_time_constant(
    gating: float,
    minhash_sim: float,
    free_energy: float,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> float:
    """
    Compute an effective Liquid‑Time‑Constant τ that is modulated by
    variational free energy and MinHash similarity.

    τ = gating * (1 + α·F) * (1 + β·s)

    Parameters
    ----------
    gating : float
        Base gating signal (typically in (0, 1]).
    minhash_sim : float
        Similarity between packet token and group token, in [0, 1].
    free_energy : float
        Variational free energy for the group.
    alpha, beta : float
        Scaling coefficients that control the influence of free energy and
        similarity respectively.

    Returns
    -------
    float
        Effective time constant τ.
    """
    tau = gating * (1.0 + alpha * free_energy) * (1.0 + beta * minhash_sim)
    return _pct(tau)


# ----------------------------------------------------------------------
# Bayesian Minimum‑Cost Tree utilities
# ----------------------------------------------------------------------
def initialize_edge_priors(groups: Tuple[str, ...]) -> Dict[Tuple[str, str], float]:
    """
    Initialise a uniform prior probability for every directed edge
    (source → target) in the complete graph over *groups*.
    """
    n = len(groups)
    uniform = 1.0 / (n - 1)  # exclude self‑loops
    priors: Dict[Tuple[str, str], float] = {}
    for src in groups:
        for dst in groups:
            if src != dst:
                priors[(src, dst)] = uniform
    return priors


def bayesian_update_prior(
    prior: float, likelihood: float, evidence: float
) -> float:
    """
    Apply Bayes' rule for a single edge prior.

    posterior ∝ prior * likelihood
    normalized by evidence.
    """
    if evidence == 0:
        return prior  # avoid division by zero
    posterior = prior * likelihood / evidence
    return max(0.0, min(1.0, posterior))


def normalize_edge_priors(priors: Dict[Tuple[str, str], float]) -> None:
    """
    Renormalise outgoing priors for each source node so that they sum to 1.
    """
    outgoing: Dict[str, float] = {}
    for (src, _), prob in priors.items():
        outgoing[src] = outgoing.get(src, 0.0) + prob

    for (src, dst), prob in list(priors.items()):
        total = outgoing[src]
        if total > 0:
            priors[(src, dst)] = prob / total


# ----------------------------------------------------------------------
# Hybrid Routing
# ----------------------------------------------------------------------
def compute_group_priors(edge_priors: Dict[Tuple[str, str], float]) -> Dict[str, float]:
    """
    Compute group priors by summing edge priors.

    Parameters
    ----------
    edge_priors : Dict[Tuple[str, str], float]
        Edge priors.

    Returns
    -------
    Dict[str, float]
        Group priors.
    """
    group_priors: Dict[str, float] = {}
    for (src, dst), prob in edge_priors.items():
        group_priors[dst] = group_priors.get(dst, 0.0) + prob
    return group_priors


def hybrid_route_packet(
    packet_token: str,
    gating: float = 0.8,
    alpha: float = 0.5,
    beta: float = 0.3,
    edge_priors: Dict[Tuple[str, str], float] = None,
) -> Tuple[str, Dict[Tuple[str, str], float]]:
    """
    Route a packet token to one of the GROUPS using the hybrid cost model.

    Steps
    -----
    1. Compute a weekday‑based weight vector w.
    2. For each group g:
       * similarity s_g = MinHash(packet_token, g)
       * construct a provisional distribution q_g (here a one‑hot on g)
       * prior distribution p_g is derived from outgoing edge priors.
       * free energy F_g = variational_free_energy(q_g, p_g)
       * τ_g = hybrid_liquid_time_constant(gating, s_g, F_g, α, β)
       * cost C_g = π_g * τ_g, where π_g is the sum of priors on edges
         incoming to g (treated as the group's prior weight).
    3. Choose the g with minimal cost.

    Returns
    -------
    Tuple[str, Dict[Tuple[str, str], float]]
        The chosen group and updated edge priors.
    """
    if edge_priors is None:
        edge_priors = initialize_edge_priors(GROUPS)

    dow = doomsday(datetime.now().year, datetime.now().month, datetime.now().day)
    weight_vec = weekday_weight_vector(GROUPS, dow)

    costs = {}
    for group in GROUPS:
        similarity = minhash_similarity(packet_token, group)
        q_g = np.array([1.0 if i == GROUPS.index(group) else 0.0 for i in range(len(GROUPS))])
        p_g = np.array([edge_priors.get((g, group), 0.0) / sum(edge_priors.get((g, group), 0.0) for g in GROUPS if (g, group) in edge_priors) if sum(edge_priors.get((g, group), 0.0) for g in GROUPS if (g, group) in edge_priors) > 0 else 1.0 / len(GROUPS) for i in range(len(GROUPS))])
        free_energy = variational_free_energy(q_g, p_g)
        tau = hybrid_liquid_time_constant(gating, similarity, free_energy, alpha, beta)
        group_priors = compute_group_priors(edge_priors)
        cost = group_priors.get(group, 0.0) * tau
        costs[group] = cost

    chosen_group = min(costs, key=costs.get)

    # Update edge priors using Bayes' rule
    likelihood = 1.0 / costs[chosen_group]
    evidence = sum([edge_priors.get((g, chosen_group), 0.0) * likelihood for g in GROUPS if (g, chosen_group) in edge_priors])
    for g in GROUPS:
        if (g, chosen_group) in edge_priors:
            edge_priors[(g, chosen_group)] = bayesian_update_prior(edge_priors[(g, chosen_group)], likelihood, evidence)

    normalize_edge_priors(edge_priors)

    return chosen_group, edge_priors


# Test the hybrid routing function
if __name__ == "__main__":
    packet_token = "test_packet"
    chosen_group, updated_edge_priors = hybrid_route_packet(packet_token)
    print(f"Chosen group: {chosen_group}")
    print("Updated edge priors:")
    for (src, dst), prob in updated_edge_priors.items():
        print(f"{src} -> {dst}: {prob}")