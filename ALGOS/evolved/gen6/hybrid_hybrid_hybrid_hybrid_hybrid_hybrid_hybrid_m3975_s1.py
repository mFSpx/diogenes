# DARWIN HAMMER — match 3975, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_minimu_m1261_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m2206_s2.py (gen5)
# born: 2026-05-29T23:52:55Z

"""Hybrid Fisher‑Ricci‑Bayes‑Bandit Algorithm
================================================

Parents
-------
* **Parent A** – `hybrid_hybrid_hybrid_bayes__hybrid_hybrid_minimu_m1261_s1.py`  
  Provides a Bayesian edge‑belief update, SSIM‑based similarity scoring and a
  log‑count bandit router.

* **Parent B** – `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m2206_s2.py`  
  Supplies a Fisher‑information based score, a lightweight Ollivier‑Ricci
  curvature estimator and an endpoint‑circuit‑breaker threshold.

Mathematical Bridge
-------------------
Both parents rely on *curvature* as a geometric weight:

* Parent A uses Ollivier‑Ricci curvature `κ(x,y)` to weight node distances.
* Parent B computes a *Fisher score* `I(θ)` that can be interpreted as a
  Riemannian metric density.

The fusion treats the product `w = I(θ)·κ(x,y)` as a **Fisher‑Ricci weight**.
This weight modulates the Bayesian posterior of each edge and consequently
the expected reward used by the bandit router.  The SSIM similarity between
node embeddings is then combined with the weighted expected reward to obtain
the final hybrid score.

The module implements three core functions that demonstrate this unified
system:
1. `compute_fisher_ricci_weight` – builds the geometric weight from Fisher
   information and Ricci curvature.
2. `bayes_edge_posteriors` – vectorised Bayesian update that incorporates the
   Fisher‑Ricci weight and observed rewards.
3. `hybrid_score` – merges SSIM similarity with the weighted expected rewards
   to produce a single scalar score for a pair of nodes.

A small smoke test at the bottom exercises the pipeline on synthetic data.
"""

import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Basic data structures (borrowed from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

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

# ----------------------------------------------------------------------
# Utility functions from Parent B
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Normalized Gaussian beam."""
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ricci_curvature(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Lightweight Ollivier‑Ricci estimator using Euclidean distance."""
    dist = np.linalg.norm(x - y)
    # Prevent division by zero; curvature is bounded between 0 and 1.
    return max(0.0, 1.0 - (dist ** 2) / (4.0 * (eps + 1e-6)))

def angular_representation(dt: datetime) -> float:
    """Map a datetime to an angle θ ∈ [0, 2π)."""
    seconds_in_day = 24 * 3600
    timestamp = dt.timestamp()
    return 2.0 * np.pi * ((timestamp % seconds_in_day) / seconds_in_day)

# ----------------------------------------------------------------------
# SSIM implementation (simplified version from Parent A)
# ----------------------------------------------------------------------
def compute_ssim(x: List[float], y: List[float],
                 dynamic_range: float = 1.0,
                 k1: float = 0.01,
                 k2: float = 0.03) -> float:
    """Structural Similarity Index between two 1‑D signals."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return float(numerator / (denominator + 1e-12))

# ----------------------------------------------------------------------
# Fusion core
# ----------------------------------------------------------------------
def compute_fisher_ricci_weight(edge_feat: Dict) -> float:
    """
    Build the geometric weight w = I(θ) * κ for a single edge.

    Expected keys in ``edge_feat``:
        - 'timestamp' : datetime of the observation
        - 'center'    : float, centre of the Gaussian beam
        - 'width'     : float, width of the Gaussian beam
        - 'vec_a'     : np.ndarray, embedding of node A
        - 'vec_b'     : np.ndarray, embedding of node B
        - 'eps'       : optional float for curvature stability
    """
    theta = angular_representation(edge_feat['timestamp'])
    I = fisher_score(theta,
                     center=edge_feat.get('center', 0.0),
                     width=edge_feat.get('width', 1.0))
    kappa = ricci_curvature(edge_feat['vec_a'],
                            edge_feat['vec_b'],
                            eps=edge_feat.get('eps', 1e-12))
    return I * kappa

def bayes_edge_posteriors(
    edges: List[Edge],
    priors: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
    alpha: float = 1.0
) -> np.ndarray:
    """
    Vectorised Bayesian update for all edges.

    Posterior ∝ prior * (1 + α * weight * reward)

    The result is normalised to sum to 1.
    """
    # Guard against shape mismatches
    if not (len(priors) == len(rewards) == len(weights) == len(edges)):
        raise ValueError("All input arrays must have the same length as edges.")
    likelihood = 1.0 + alpha * weights * rewards
    unnorm = priors * likelihood
    total = unnorm.sum()
    if total == 0:
        # fallback to uniform distribution
        return np.full_like(priors, 1.0 / len(priors))
    return unnorm / total

def hybrid_score(
    node_vec_a: np.ndarray,
    node_vec_b: np.ndarray,
    edge_posterior: float,
    ssim_weight: float = 0.5
) -> float:
    """
    Combine SSIM similarity with the posterior probability of the connecting edge.

    score = ssim_weight * SSIM(a, b) + (1 - ssim_weight) * edge_posterior
    """
    # SSIM expects 1‑D sequences; we flatten the embeddings.
    ssim = compute_ssim(node_vec_a.tolist(), node_vec_b.tolist())
    return ssim_weight * ssim + (1.0 - ssim_weight) * edge_posterior

def hybrid_tree_cost(
    tree_edges: List[Edge],
    edge_posteriors: np.ndarray,
    base_costs: np.ndarray
) -> float:
    """
    Expected cost of a spanning tree where each edge cost is scaled by its
    posterior belief (higher belief → lower effective cost).

    cost = Σ base_cost_i * (1 - posterior_i)
    """
    if not (len(tree_edges) == len(edge_posteriors) == len(base_costs)):
        raise ValueError("Length mismatch among inputs.")
    scaled = base_costs * (1.0 - edge_posteriors)
    return float(scaled.sum())

# ----------------------------------------------------------------------
# Simple demonstration of a bandit action using the hybrid posterior
# ----------------------------------------------------------------------
def select_bandit_action(
    actions: List[BanditAction],
    edge_posteriors: Dict[Edge, float],
    edge_of_interest: Edge
) -> BanditAction:
    """
    Choose the action with the highest upper confidence bound where the
    confidence bound is inflated by the posterior belief of the associated edge.
    """
    posterior = edge_posteriors.get(edge_of_interest, 0.0)
    best = None
    best_score = -math.inf
    for act in actions:
        ucb = act.expected_reward + act.confidence_bound * (1.0 + posterior)
        if ucb > best_score:
            best_score = ucb
            best = act
    return best

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic graph with 3 edges
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    priors = np.array([0.33, 0.33, 0.34], dtype=np.float64)

    # Random rewards observed for each edge
    rewards = np.array([1.0, 0.2, 0.5], dtype=np.float64)

    # Build edge feature dictionaries
    now = datetime.now(timezone.utc)
    edge_features = []
    for (u, v) in edges:
        feat = {
            "timestamp": now - timedelta(minutes=random.randint(0, 60)),
            "center": 0.0,
            "width": 1.0,
            "vec_a": np.random.rand(5),
            "vec_b": np.random.rand(5),
            "eps": 1e-6,
        }
        edge_features.append(feat)

    # Compute Fisher‑Ricci weights
    weights = np.array([compute_fisher_ricci_weight(f) for f in edge_features],
                       dtype=np.float64)

    # Bayesian posterior update
    posteriors = bayes_edge_posteriors(edges, priors, rewards, weights)

    # Example node embeddings for score computation
    node_embeddings = {
        "A": np.random.rand(5),
        "B": np.random.rand(5),
        "C": np.random.rand(5),
    }

    # Compute hybrid scores for each edge
    for (e, post) in zip(edges, posteriors):
        a, b = e
        score = hybrid_score(node_embeddings[a], node_embeddings[b], post)
        print(f"Hybrid score for edge {a}-{b}: {score:.4f}")

    # Tree cost (using the three edges as a cycle, treat as a tree for demo)
    base_costs = np.array([1.0, 1.2, 0.9], dtype=np.float64)
    cost = hybrid_tree_cost(edges, posteriors, base_costs)
    print(f"Hybrid expected tree cost: {cost:.4f}")

    # Bandit actions demo
    actions = [
        BanditAction("act1", 0.5, 0.8, 0.1, "hybrid"),
        BanditAction("act2", 0.3, 0.6, 0.2, "hybrid"),
        BanditAction("act3", 0.2, 0.9, 0.05, "hybrid"),
    ]
    edge_post_dict = {e: p for e, p in zip(edges, posteriors)}
    chosen = select_bandit_action(actions, edge_post_dict, ("A", "B"))
    print(f"Chosen bandit action: {chosen.action_id} with expected reward {chosen.expected_reward:.2f}")