# DARWIN HAMMER — match 4606, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1465_s2.py (gen5)
# parent_b: hybrid_decreasing_pruning_hybrid_hybrid_minimu_m63_s0.py (gen3)
# born: 2026-05-29T23:57:01Z

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Parent A – Regret‑Weighted Strategy (unchanged, but with tighter docstring)
# ----------------------------------------------------------------------
def compute_regret_weighted_strategy(
    actions: List[int],
    rewards: List[float],
    num_actions: int,
) -> np.ndarray:
    """
    Compute a normalized regret value for each possible action.

    The regret for action *i* is the gap between the best observed average
    reward and the average reward of *i*, scaled to the interval [0, 1].

    Parameters
    ----------
    actions : List[int]
        Sequence of taken actions (integer identifiers in ``range(num_actions)``).
    rewards : List[float]
        Observed reward for each action in ``actions``.
    num_actions : int
        Total number of distinct actions.

    Returns
    -------
    np.ndarray
        Array ``r`` of shape ``(num_actions,)`` with values in ``[0, 1]`` where
        ``r[i]`` is the regret of action *i*.
    """
    if len(actions) != len(rewards):
        raise ValueError("actions and rewards must have the same length")
    totals = np.zeros(num_actions, dtype=float)
    counts = np.zeros(num_actions, dtype=int)
    for a, r in zip(actions, rewards):
        totals[a] += r
        counts[a] += 1
    avg_rewards = np.where(counts > 0, totals / counts, 0.0)
    best_avg = avg_rewards.max()
    # Normalise by the larger of best_avg and 1 to avoid division by zero
    scale = best_avg if best_avg > 0 else 1.0
    regret = (best_avg - avg_rewards) / scale
    return np.clip(regret, 0.0, 1.0)


# ----------------------------------------------------------------------
# Bayesian utilities – now using a Beta conjugate prior
# ----------------------------------------------------------------------
def beta_parameters_from_trust(
    trust: float,
    regret: float,
    concentration: float = 10.0,
) -> Tuple[float, float]:
    """
    Convert a scalar trust value (∈[0,1]) and regret (∈[0,1]) into
    Beta distribution parameters (α, β).

    The regret acts as a confidence dampener: higher regret inflates both
    α and β towards 1, flattening the distribution.
    """
    # Clamp inputs
    trust = max(0.0, min(1.0, trust))
    regret = max(0.0, min(1.0, regret))

    # Effective concentration after regret attenuation
    eff_conc = concentration * (1.0 - regret) + 1.0  # +1 avoids zero concentration

    α = 1.0 + trust * eff_conc
    β = 1.0 + (1.0 - trust) * eff_conc
    return α, β


def beta_update(
    α: float,
    β: float,
    evidence: int,
    false_positive: float = 0.01,
) -> Tuple[float, float]:
    """
    Perform a Bayesian update of a Beta prior given binary evidence.

    ``evidence`` is the number of *positive* observations (0 ≤ evidence ≤ 1).
    ``false_positive`` is interpreted as the probability that a reported
    positive is spurious; it is folded into the update as an extra negative count.
    """
    if not (0 <= evidence <= 1):
        raise ValueError("Evidence must be 0 or 1 for Bernoulli updates")
    # Positive count
    α_new = α + evidence
    # Negative count (including false positives)
    β_new = β + (1 - evidence) * (1 - false_positive) + false_positive
    return α_new, β_new


def beta_mean(α: float, β: float) -> float:
    """Return the mean of a Beta(α,β) distribution."""
    return α / (α + β)


# ----------------------------------------------------------------------
# Hybrid Core Functions – deeper integration
# ----------------------------------------------------------------------
def adjust_trust_with_regret_beta(
    trust: Dict[Edge, float],
    regret_vec: np.ndarray,
    action_map: Dict[int, Edge],
    concentration: float = 10.0,
) -> Dict[Edge, Tuple[float, float]]:
    """
    Produce a Beta prior (α,β) for each edge by blending trust and regret.

    Returns a dictionary mapping each edge to its (α,β) parameters.
    """
    priors: Dict[Edge, Tuple[float, float]] = {}
    for a, edge in action_map.items():
        r = regret_vec[a] if a < len(regret_vec) else 0.0
        τ = trust.get(edge, 0.0)
        priors[edge] = beta_parameters_from_trust(τ, r, concentration)
    return priors


def edge_likelihood_from_geometry(
    p1: Point,
    p2: Point,
    length_scale: float = 1.0,
) -> float:
    """
    Derive a Bernoulli‑like likelihood from geometric length.

    Shorter edges are deemed more plausible. The function returns a value
    in (0,1] that can be interpreted as the probability that the edge is
    truly present given its geometry.
    """
    length = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    length = max(length, 1e-9)  # avoid division by zero
    # Logistic‑style mapping to keep values bounded away from 0
    return 1.0 / (1.0 + (length / length_scale))


def adaptive_prune_probability(
    posterior_mean: float,
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    beta: float = 0.5,
) -> float:
    """
    Time‑dependent pruning probability that also respects posterior confidence.

    ``beta`` controls how aggressively low‑confidence edges are removed.
    """
    if t < 0 or lam < 0 or alpha < 0 or beta < 0:
        raise ValueError("t, lam, alpha and beta must be non‑negative")
    base = lam * math.exp(-alpha * t)          # pure time decay
    confidence_factor = (1.0 - posterior_mean) ** beta
    return min(1.0, base * confidence_factor)


def hybrid_prune_and_flow(
    edges: List[Edge],
    nodes: Dict[str, Point],
    trust: Dict[Edge, float],
    regret_vec: np.ndarray,
    action_map: Dict[int, Edge],
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
    beta: float = 0.5,
    concentration: float = 10.0,
    false_positive: float = 0.01,
    seed: Union[int, str, None] = None,
) -> List[Edge]:
    """
    Full hybrid pipeline with a hierarchical Bayesian model.

    Steps
    -----
    1. Convert trust+regret into Beta priors per edge.
    2. Derive a geometric likelihood for each edge.
    3. Update the Beta prior with the likelihood treated as a single Bernoulli
       observation (rounded to 0/1 for tractability).
    4. Compute posterior means and prune edges using a probability that
       depends on both time and posterior confidence.
    """
    rng = random.Random(seed)

    # 1. Beta priors
    beta_priors = adjust_trust_with_regret_beta(
        trust, regret_vec, action_map, concentration
    )

    # 2‑3. Posterior means after a single pseudo‑observation
    posterior_means: Dict[Edge, float] = {}
    for e in edges:
        p1, p2 = nodes[e[0]], nodes[e[1]]
        likelihood = edge_likelihood_from_geometry(p1, p2)

        # Convert likelihood to a Bernoulli observation (0/1) by stochastic rounding
        obs = 1 if rng.random() < likelihood else 0

        α, β = beta_priors.get(e, (1.0, 1.0))  # default to uniform prior
        α_up, β_up = beta_update(α, β, obs, false_positive)
        posterior_means[e] = beta_mean(α_up, β_up)

    # 4. Prune based on adaptive probability
    kept_edges: List[Edge] = []
    for e in edges:
        p_prune = adaptive_prune_probability(
            posterior_means[e], t, lam, alpha, beta
        )
        # Keep with probability 1 - p_prune
        if rng.random() >= p_prune:
            kept_edges.append(e)

    return kept_edges


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple square graph
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (1.0, 1.0),
        "D": (0.0, 1.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A"), ("A", "C")]
    # Initial trust (some arbitrary values)
    trust = {e: 0.6 for e in edges}
    # Map each action (0‑4) to an edge
    action_map = {i: e for i, e in enumerate(edges)}
    # Simulated actions/rewards for regret computation
    actions = [0, 1, 2, 3, 4, 0, 2, 2]
    rewards = [1.0, 0.5, 0.8, 0.3, 0.9, 1.2, 0.7, 0.6]
    regret_vec = compute_regret_weighted_strategy(actions, rewards, len(edges))

    kept = hybrid_prune_and_flow(
        edges,
        nodes,
        trust,
        regret_vec,
        action_map,
        t=5.0,
        lam=1.0,
        alpha=0.15,
        beta=0.7,
        concentration=15.0,
        false_positive=0.02,
        seed=42,
    )
    print("Kept edges after hybrid update:", kept)