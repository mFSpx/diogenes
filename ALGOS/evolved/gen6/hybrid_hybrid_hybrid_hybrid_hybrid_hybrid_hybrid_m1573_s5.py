# DARWIN HAMMER — match 1573, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_minimum_cost__m994_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m962_s0.py (gen5)
# born: 2026-05-29T23:37:37Z

import numpy as np
import math
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = 1.987  # cal mol^-1 K^-1


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass(frozen=True)
class Point:
    x: float
    y: float


# ----------------------------------------------------------------------
# Global policy store
# ----------------------------------------------------------------------
# _POLICY maps action_id -> [total_reward, count, confidence_bound]
_POLICY: Dict[str, List[float]] = {}


def reset_policy() -> None:
    """Remove all stored statistics."""
    _POLICY.clear()


def _reward(action: str) -> float:
    """Return the empirical mean reward for *action*."""
    total, count, _ = _POLICY.get(action, [0.0, 0.0, 0.0])
    return total / count if count > 0 else 0.0


def _count(action: str) -> int:
    """Return the number of observations for *action*."""
    return int(_POLICY.get(action, [0.0, 0.0, 0.0])[1])


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incrementally update the reward statistics for a batch of actions."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0, 0.0])
        stats[0] += float(u.reward)          # total reward
        stats[1] += 1.0                       # count
        # confidence bound will be recomputed lazily when needed


# ----------------------------------------------------------------------
# Entropy & probability utilities
# ----------------------------------------------------------------------
def _normalize(probs: List[float]) -> List[float]:
    """Return a probability vector that sums to 1 (softmax style)."""
    arr = np.array(probs, dtype=float)
    if arr.size == 0:
        return []
    # avoid overflow
    max_val = np.max(arr)
    exp_arr = np.exp(arr - max_val)
    total = exp_arr.sum()
    return (exp_arr / total).tolist()


def calculate_pheromone_probabilities(surface_key: str, limit: int, db_url: str) -> List[float]:
    """
    Simulated pheromone probabilities.
    Returns a *normalized* probability distribution of length ``limit``.
    """
    rng = np.random.default_rng(seed=hash(surface_key) % (2**32))
    raw = rng.random(limit)  # uniform random numbers in (0,1)
    return _normalize(raw.tolist())


def shannon_entropy(probabilities: List[float]) -> float:
    """Compute Shannon entropy (base‑2) of a proper probability distribution."""
    if not probabilities:
        return 0.0
    probs = np.array(probabilities, dtype=float)
    # Guard against numerical issues
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log2(probs)))


# ----------------------------------------------------------------------
# Bayesian update utilities
# ----------------------------------------------------------------------
def bayesian_update(prior: float, likelihood: float) -> float:
    """
    Return the posterior probability using a simple Bernoulli model:
        posterior = (prior * likelihood) / evidence
    where evidence = prior*likelihood + (1-prior)*(1-likelihood).
    The function clamps inputs to (0,1) to avoid division by zero.
    """
    prior = min(max(prior, 1e-12), 1 - 1e-12)
    likelihood = min(max(likelihood, 1e-12), 1 - 1e-12)
    evidence = prior * likelihood + (1 - prior) * (1 - likelihood)
    return (prior * likelihood) / evidence


# ----------------------------------------------------------------------
# Tree cost utilities
# ----------------------------------------------------------------------
def _euclidean(p1: Point, p2: Point) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Tuple[str, str]],
    root: str,
    path_weight: float = 0.2,
) -> float:
    """
    Compute a weighted sum of Euclidean edge lengths.
    Only edges reachable from *root* are considered (simple BFS).
    """
    if root not in nodes:
        raise ValueError("Root node not present in nodes dictionary.")

    visited = set()
    frontier = [root]
    total = 0.0

    adjacency: Dict[str, List[str]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    while frontier:
        current = frontier.pop()
        visited.add(current)
        for neighbor in adjacency.get(current, []):
            if neighbor in visited:
                continue
            total += path_weight * _euclidean(nodes[current], nodes[neighbor])
            frontier.append(neighbor)

    return total


# ----------------------------------------------------------------------
# Hybrid integration helpers
# ----------------------------------------------------------------------
def decision_hygiene_scores(text: str) -> Dict[str, int]:
    """
    Simulated extraction of decision‑hygiene scores.
    In a real system this would parse *text* with regexes.
    """
    # deterministic stub for reproducibility
    return {"score1": 1, "score2": 2}


def update_bandit_policy_with_pheromone(action_id: str, pheromone_probabilities: List[float]) -> None:
    """
    Update the stored confidence bound for *action_id* using the entropy of the
    pheromone distribution. The propensity (average reward) remains unchanged.
    """
    avg_reward = _reward(action_id)
    entropy = shannon_entropy(pheromone_probabilities)

    # Upper‑confidence bound term (UCB1) with entropy as an extra exploration bonus
    n = _count(action_id)
    total_actions = sum(_count(a) for a in _POLICY) or 1
    ucb_bonus = math.sqrt(2 * math.log(total_actions) / (n + 1e-9))

    confidence = ucb_bonus + entropy
    stats = _POLICY.setdefault(action_id, [0.0, 0.0, 0.0])
    stats[2] = confidence  # replace confidence bound


def update_pheromone_with_bandit(action_id: str, reward: float) -> List[float]:
    """
    Produce a new pheromone distribution that is biased by the bandit reward.
    The returned distribution is normalized.
    """
    # Base distribution (uniform random)
    base = calculate_pheromone_probabilities("surface_key", 10, "db_url")

    # Scale each component by a factor that reflects how well the bandit performed.
    # Higher reward → stronger reinforcement.
    scaling = 1.0 + reward  # simple linear scaling
    biased = [p * scaling for p in base]

    # Incorporate decision‑hygiene likelihood
    scores = decision_hygiene_scores("placeholder")
    likelihood = bayesian_update(_reward(action_id), scores["score1"] / (scores["score1"] + scores["score2"]))
    biased = [p * likelihood for p in biased]

    return _normalize(biased)


def calculate_hybrid_cost(
    nodes: Dict[str, Point],
    edges: List[Tuple[str, str]],
    root: str,
    path_weight: float = 0.2,
) -> float:
    """
    Hybrid cost = tree traversal cost + entropy‑weighted pheromone term.
    The entropy term is multiplied by a factor that reflects the average
    decision‑hygiene score, deepening the mathematical coupling.
    """
    t_cost = tree_cost(nodes, edges, root, path_weight)

    pheromone_probs = calculate_pheromone_probabilities("surface_key", 10, "db_url")
    entropy = shannon_entropy(pheromone_probs)

    # Decision‑hygiene weighting (higher scores increase the impact of entropy)
    scores = decision_hygiene_scores("placeholder")
    avg_score = (scores["score1"] + scores["score2"]) / 2.0
    weighted_entropy = entropy * avg_score

    return t_cost + weighted_entropy


# ----------------------------------------------------------------------
# Example driver (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed numpy's RNG for reproducible demo runs
    np.random.seed(42)

    # 1. Update bandit statistics with a single observation
    update_policy([BanditUpdate("ctx1", "actA", 1.0, 0.5)])

    # 2. Compute pheromone distribution and update bandit confidence
    pher_probs = calculate_pheromone_probabilities("surface_key", 10, "db_url")
    update_bandit_policy_with_pheromone("actA", pher_probs)

    # 3. Adjust pheromones based on the latest bandit reward
    new_pheromone = update_pheromone_with_bandit("actA", 1.0)

    # 4. Evaluate hybrid cost on a tiny graph
    nodes = {
        "n1": Point(1.0, 2.0),
        "n2": Point(3.0, 4.0),
        "n3": Point(5.0, 1.0),
    }
    edges = [("n1", "n2"), ("n2", "n3")]
    hybrid_cost = calculate_hybrid_cost(nodes, edges, "n1")
    print("Hybrid cost:", hybrid_cost)