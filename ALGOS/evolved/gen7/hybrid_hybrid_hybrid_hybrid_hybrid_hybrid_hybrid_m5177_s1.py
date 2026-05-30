# DARWIN HAMMER — match 5177, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1573_s4.py (gen6)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_cockpit_metri_m1578_s0.py (gen4)
# born: 2026-05-30T00:00:17Z

"""Hybrid Fusion of Bandit‑Pheromone Policy (Parent A) and Hyperdimensional MinHash/Binding (Parent B).

Mathematical bridge:
- Each action from the bandit policy is represented as a high‑dimensional random vector
  (Parent B’s `random_vector`).
- The bandit’s propensity (reward average) and the pheromone‑derived entropy
  (Parent A) are combined with hyperdimensional similarity (dot‑product of vectors)
  to form a hybrid utility score.
- Updates bind the current action vector with a pheromone‑derived vector
  (`bind` from Parent B) so that the policy and the hyperdimensional representation
  co‑evolve.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable

import numpy as np

# ---------- Parent A structures ----------
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

_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    _POLICY.clear()

def _reward(action: str) -> float:
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: List[BanditUpdate]) -> None:
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

def shannon_entropy(probabilities: List[float]) -> float:
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def tree_cost(nodes: Dict[str, Point],
              edges: List[Tuple[str, str]],
              root: str,
              path_weight: float = 0.2) -> float:
    return sum(
        math.hypot(nodes[e[0]].x - nodes[e[1]].x,
                   nodes[e[0]].y - nodes[e[1]].y)
        for e in edges) * path_weight

# ---------- Parent B structures ----------
MAX64 = (1 << 64) - 1

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(
        hashlib.blake2b(data, digest_size=8).digest(), "big"
    )

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for i in range(k)) for t in toks]

def random_vector(dim: int = 10000, seed: str | int | None = None) -> List[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]

def bind(a: List[float], b: List[float]) -> List[float]:
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")
    return [x * y for x, y in zip(a, b)]

# ---------- Hybrid Functions ----------
def generate_hyper_vectors(action_ids: Iterable[str],
                           dim: int = 256,
                           seed: int = 42) -> Dict[str, List[float]]:
    """Create a random hyper‑dimensional vector for each action."""
    vectors: Dict[str, List[float]] = {}
    base_rng = random.Random(seed)
    for aid in action_ids:
        vectors[aid] = random_vector(dim, seed=base_rng.randint(0, 1 << 30))
    return vectors

def hybrid_action_score(action_id: str,
                        pheromone_probs: List[float],
                        vectors: Dict[str, List[float]],
                        alpha: float = 0.6) -> float:
    """
    Combine bandit reward, pheromone entropy, and hyper‑dimensional similarity
    into a single utility score.

    score = α * reward
            + (1‑α) * similarity_to_others
            - λ * entropy
    """
    reward = _reward(action_id)
    entropy = shannon_entropy(pheromone_probs)

    vec = np.array(vectors[action_id])
    # similarity to the centroid of all other vectors
    other_vecs = [np.array(v) for k, v in vectors.items() if k != action_id]
    if other_vecs:
        centroid = np.mean(other_vecs, axis=0)
        sim = float(
            np.dot(vec, centroid) /
            (np.linalg.norm(vec) * np.linalg.norm(centroid) + 1e-12)
        )
    else:
        sim = 0.0

    lambda_entropy = 0.1
    return alpha * reward + (1 - alpha) * sim - lambda_entropy * entropy

def update_hybrid_policy(updates: List[BanditUpdate],
                         pheromone_probs: List[float],
                         vectors: Dict[str, List[float]]) -> None:
    """
    Perform the classic bandit update and bind each updated action's vector
    with a pheromone‑derived vector (scaled probabilities).
    """
    update_policy(updates)

    # Transform pheromone probabilities into a vector of the same dimension
    dim = len(next(iter(vectors.values())))
    prob_vec = [p for p in pheromone_probs]
    # repeat or truncate to match dimension
    if len(prob_vec) < dim:
        repeats = dim // len(prob_vec) + 1
        prob_vec = (prob_vec * repeats)[:dim]
    else:
        prob_vec = prob_vec[:dim]

    for u in updates:
        old_vec = vectors.get(u.action_id)
        if old_vec is None:
            continue
        # bind old vector with pheromone probability vector
        new_vec = bind(old_vec, prob_vec)
        vectors[u.action_id] = new_vec

def hybrid_tree_cost(nodes: Dict[str, Point],
                     edges: List[Tuple[str, str]],
                     root: str,
                     vectors: Dict[str, List[float]],
                     path_weight: float = 0.2,
                     vector_weight: float = 0.05) -> float:
    """
    Extend the classic tree cost with a term that penalises high‑norm vectors
    attached to visited nodes, encouraging compact hyper‑dimensional
    representations for frequently traversed branches.
    """
    base = tree_cost(nodes, edges, root, path_weight)

    # accumulate norm of vectors for nodes appearing in edges
    norm_sum = 0.0
    for a, b in edges:
        for nid in (a, b):
            vec = vectors.get(nid)
            if vec is not None:
                norm_sum += np.linalg.norm(vec)

    return base + vector_weight * norm_sum

# ---------- Smoke Test ----------
if __name__ == "__main__":
    # Create dummy actions and updates
    actions = [
        BanditAction("A", 0.5, 1.0, 0.1, "alg1"),
        BanditAction("B", 0.3, 0.8, 0.2, "alg2"),
        BanditAction("C", 0.7, 1.2, 0.05, "alg3"),
    ]

    # Initialise policy with some synthetic rewards
    reset_policy()
    initial_updates = [
        BanditUpdate("ctx1", "A", 1.0, 0.5),
        BanditUpdate("ctx2", "B", 0.5, 0.3),
        BanditUpdate("ctx3", "C", 1.5, 0.7),
    ]
    update_policy(initial_updates)

    # Generate hyper‑vectors for each action
    vecs = generate_hyper_vectors([a.action_id for a in actions], dim=256)

    # Simulate pheromone probabilities (normally obtained from a DB)
    pheromone_probs = [random.random() for _ in range(10)]

    # Compute hybrid scores
    for act in actions:
        score = hybrid_action_score(act.action_id, pheromone_probs, vecs)
        print(f"Hybrid score for {act.action_id}: {score:.4f}")

    # Perform an additional update and bind vectors
    new_updates = [
        BanditUpdate("ctx4", "A", 0.8, 0.5),
        BanditUpdate("ctx5", "B", 0.9, 0.3),
    ]
    update_hybrid_policy(new_updates, pheromone_probs, vecs)

    # Build a tiny graph for tree cost
    nodes = {
        "A": Point(0, 0),
        "B": Point(1, 0),
        "C": Point(0, 1),
    }
    edges = [("A", "B"), ("A", "C")]
    cost = hybrid_tree_cost(nodes, edges, root="A", vectors=vecs)
    print(f"Hybrid tree cost: {cost:.4f}")

    sys.exit(0)