# DARWIN HAMMER — match 2706, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_bandit_router_hybrid_cockpit_metri_m287_s4.py (gen3)
# parent_b: hybrid_hybrid_gini_coeffici_hybrid_tropical_maxp_m1173_s1.py (gen5)
# born: 2026-05-29T23:43:47Z

"""Hybrid Bandit‑Router / Schoolfield + Gini‑RBF‑Tropical Fusion
================================================================

Parents
-------
* **Parent A** – *hybrid_hybrid_bandit_router_hybrid_cockpit_metri_m287_s4.py*  
  Provides a temperature‑dependent activity gate `A(T) ∈ [0,1]` and an honesty
  weight `H ∈ [0,1]`. Their product `G = A·H` scales the context norm for the
  Upper‑Confidence‑Bound (UCB) term and the decay of a pheromone field. Entropy
  of the pheromone distribution is used as a regulariser.

* **Parent B** – *hybrid_hybrid_gini_coeffici_hybrid_tropical_maxp_m1173_s1.py*  
  Supplies a Gini coefficient to measure distribution inequality, an RBF‑based
  similarity matrix between node feature vectors, and a tropical max‑plus
  algebra to combine log‑probabilities into decision‑hygiene scores.

Mathematical Bridge
-------------------
Both parents expose a **scalar gain** that modulates a base quantity:

* `G = A(T)·H`  – temperature‑honesty gain (Parent A)  
* `λ = 1 – G`   – complementary gain

In the fused algorithm `G` is used **as a weighting factor** for the RBF
similarity and for the entropy/Gini regularisation, while `λ` weights the
Gini‑based regulariser. The tropical max‑plus operation receives the same
gain‑scaled similarity as an additive term, thereby unifying the exploration
bias of the bandit with the impurity‑driven split criterion of the Hoeffding
tree.

The core hybrid quantities are:


S_ij   = G * RBF(feat_i, feat_j)                # gain‑scaled similarity
score_i = max_j ( log(p_j) + S_ij )             # tropical max‑plus
Reg    = G * Entropy(pheromone) + λ * Gini(values)


The following module implements this fusion with three public functions that
demonstrate the combined behaviour and a lightweight smoke‑test.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Sequence, Hashable, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A building blocks
# ----------------------------------------------------------------------


def temperature_activity(T: float, T_opt: float = 310.0, k: float = 0.05) -> float:
    """
    Schoolfield‑type activity gate.
    Returns a value in [0, 1] that grows sigmoidally with temperature.
    """
    # Logistic approximation of the Schoolfield curve
    return 1.0 / (1.0 + math.exp(-k * (T - T_opt)))


def honesty_weight(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """
    Anti‑slop ratio H ∈ [0,1].
    """
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def hybrid_gain(T: float, claims_with_evidence: int, total_claims_emitted: int) -> float:
    """
    Joint gain G = A(T) * H.
    """
    return temperature_activity(T) * honesty_weight(claims_with_evidence,
                                                    total_claims_emitted)


def entropy(prob_dist: Iterable[float]) -> float:
    """
    Shannon entropy of a probability distribution (expects non‑negative values).
    """
    probs = np.array(list(prob_dist), dtype=float)
    total = probs.sum()
    if total == 0.0:
        return 0.0
    probs = probs / total
    # avoid log(0)
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))


# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------


def gini_coefficient(values: Iterable[float]) -> float:
    """
    Gini coefficient for non‑negative values.
    """
    xs = sorted(float(v) for v in values)
    if not xs or sum(xs) == 0.0:
        return 0.0
    n = len(xs)
    cumulative = 0.0
    for i, x in enumerate(xs, 1):
        cumulative += (2 * i - n - 1) * x
    return cumulative / (n * sum(xs))


def rbf_similarity(a: Sequence[float],
                   b: Sequence[float],
                   epsilon: float = 1.0) -> float:
    """
    Radial Basis Function similarity (Gaussian kernel).
    """
    if len(a) != len(b):
        raise ValueError("Feature vectors must have equal length")
    r2 = sum((x - y) ** 2 for x, y in zip(a, b))
    return math.exp(-epsilon * r2)


def tropical_max_plus(log_probs: Sequence[float],
                      additive: Sequence[float]) -> float:
    """
    Tropical max‑plus: max_i (log_probs[i] + additive[i]).
    """
    if len(log_probs) != len(additive):
        raise ValueError("Sequences must be of equal length")
    return max(lp + ad for lp, ad in zip(log_probs, additive))


# ----------------------------------------------------------------------
# Hybrid structures
# ----------------------------------------------------------------------


Node = Hashable
FeatureVec = Sequence[float]


@dataclass
class HybridPolicy:
    """
    Holds pheromone levels and node feature vectors.
    """
    pheromone: Dict[Node, float] = field(default_factory=dict)
    features: Dict[Node, FeatureVec] = field(default_factory=dict)

    def normalize_pheromone(self) -> None:
        """Normalises pheromone values to sum to 1 (creates a probability distribution)."""
        total = sum(self.pheromone.values())
        if total == 0.0:
            # initialise uniform distribution if empty
            n = len(self.pheromone)
            if n == 0:
                return
            uniform = 1.0 / n
            for k in self.pheromone:
                self.pheromone[k] = uniform
        else:
            for k in self.pheromone:
                self.pheromone[k] /= total


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def hybrid_compute_gain(T: float,
                        claims_with_evidence: int,
                        total_claims_emitted: int) -> float:
    """
    Compute the joint gain G = A(T)·H.
    """
    return hybrid_gain(T, claims_with_evidence, total_claims_emitted)


def hybrid_node_scores(policy: HybridPolicy,
                       context: FeatureVec,
                       T: float,
                       claims_with_evidence: int,
                       total_claims_emitted: int) -> Dict[Node, float]:
    """
    For each node in `policy.features` compute a tropical max‑plus score that
    combines:
      * a gain‑scaled RBF similarity between the node and the current context,
      * the log‑probability derived from the node's pheromone level.

    Returns a mapping node → score (higher is better).
    """
    G = hybrid_compute_gain(T, claims_with_evidence, total_claims_emitted)
    λ = 1.0 - G

    # Ensure pheromone forms a proper distribution for log‑probability.
    policy.normalize_pheromone()
    pher_vals = np.array([policy.pheromone.get(n, 0.0) for n in policy.features],
                         dtype=float)
    # Guard against zero probabilities.
    safe_vals = np.where(pher_vals > 0, pher_vals, 1e-12)
    log_probs = np.log(safe_vals)

    scores: Dict[Node, float] = {}
    for idx, (node, feat) in enumerate(policy.features.items()):
        # similarity between context and node feature, scaled by G
        sim = rbf_similarity(context, feat) * G
        # tropical max‑plus over all nodes
        additive = np.array([rbf_similarity(context, f) * G for f in policy.features.values()])
        score = tropical_max_plus(log_probs, additive)
        scores[node] = score
    return scores


def hybrid_update_policy(policy: HybridPolicy,
                         selected_node: Node,
                         reward: float,
                         T: float,
                         claims_with_evidence: int,
                         total_claims_emitted: int,
                         decay_rate: float = 0.05) -> None:
    """
    Update pheromone levels after taking `selected_node` and receiving `reward`.

    * Pheromone decay is modulated by the joint gain G (stronger gain → slower decay).
    * An entropy‑Gini regularisation term is added to the pheromone of the
      selected node, encouraging exploration when the distribution is uniform
      (high entropy) and penalising when inequality (high Gini) dominates.
    """
    G = hybrid_compute_gain(T, claims_with_evidence, total_claims_emitted)
    λ = 1.0 - G

    # Decay all pheromones
    for n in policy.pheromone:
        policy.pheromone[n] *= (1.0 - decay_rate * (1.0 - G))

    # Ensure the selected node exists
    if selected_node not in policy.pheromone:
        policy.pheromone[selected_node] = 0.0

    # Base reward addition
    policy.pheromone[selected_node] += reward

    # Regularisation term
    ent = entropy(policy.pheromone.values())
    gini = gini_coefficient(policy.pheromone.values())
    reg = G * ent - λ * gini  # entropy encourages spread, Gini penalises inequality
    policy.pheromone[selected_node] += reg

    # Re‑normalise to keep a probability‑like scale
    policy.normalize_pheromone()


def hybrid_select_action(policy: HybridPolicy,
                         context: FeatureVec,
                         T: float,
                         claims_with_evidence: int,
                         total_claims_emitted: int,
                         exploration_coef: float = 2.0) -> Node:
    """
    Implements a UCB‑style selection where the exploration term is scaled by
    the joint gain G. The underlying value estimate for a node is its pheromone
    level; the exploration bonus uses the gain‑scaled norm of the context.
    """
    G = hybrid_compute_gain(T, claims_with_evidence, total_claims_emitted)

    # Ensure pheromone distribution is normalised
    policy.normalize_pheromone()

    context_norm = math.sqrt(sum(x * x for x in context))
    bonus = exploration_coef * G * context_norm

    # Compute UCB value for each node
    ucb_values: Dict[Node, float] = {}
    for node, prob in policy.pheromone.items():
        ucb = prob + bonus
        ucb_values[node] = ucb

    # Choose node with maximal UCB
    return max(ucb_values, key=ucb_values.get)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a tiny synthetic environment
    random.seed(42)
    np.random.seed(42)

    # Define three nodes with 3‑dimensional feature vectors
    nodes = ["A", "B", "C"]
    features = {
        "A": [0.1, 0.2, 0.3],
        "B": [0.4, 0.5, 0.6],
        "C": [0.7, 0.8, 0.9],
    }

    # Initialise policy
    policy = HybridPolicy(
        pheromone={n: 1.0 for n in nodes},  # uniform start
        features=features,
    )

    # Context vector (e.g., current observation)
    context_vec = [0.35, 0.45, 0.55]

    # Environmental parameters
    temperature = 315.0                # slightly above optimum
    claims_evidence = 8
    total_claims = 10

    # Select an action
    chosen = hybrid_select_action(policy,
                                  context_vec,
                                  temperature,
                                  claims_evidence,
                                  total_claims)

    print(f"Chosen node: {chosen}")

    # Compute scores for inspection
    scores = hybrid_node_scores(policy,
                                context_vec,
                                temperature,
                                claims_evidence,
                                total_claims)
    for n, s in scores.items():
        print(f"Score[{n}] = {s:.4f}")

    # Simulate a reward (e.g., 0.7) and update policy
    reward_obtained = 0.7
    hybrid_update_policy(policy,
                         selected_node=chosen,
                         reward=reward_obtained,
                         T=temperature,
                         claims_with_evidence=claims_evidence,
                         total_claims_emitted=total_claims)

    print("Updated pheromone levels:")
    for n, p in policy.pheromone.items():
        print(f"  {n}: {p:.4f}")

    sys.exit(0)