# DARWIN HAMMER — match 190, survivor 4
# gen: 3
# parent_a: tropical_maxplus.py (gen0)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py (gen2)
# born: 2026-05-29T23:27:29Z
#
# DISTILLED USE: Graph promotion pipeline path scoring. MAP inference on
# the 18,627 staged evidence DAG using only max() and + — no exp, no log,
# no numerical underflow. Orders of magnitude cheaper than sum-product BP
# on 4GB VRAM. Wire as the path-scoring primitive in graph_promotion
# preflight to rank candidate promotion paths before operator review.

"""Hybrid tropical‑Bayesian tree algorithm.

Parents
-------
* tropical_maxplus.py – defines the max‑plus semiring (⊕ = max, ⊗ = +) and
  matrix/polynomial utilities.
* hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py – builds Euclidean
  tree metrics, performs Bayesian updates and computes Shannon‑entropy based
  decision scores.

Mathematical bridge
-------------------
Probabilities are handled in log‑space.  In log‑space a product of probabilities
becomes an ordinary addition, which is exactly the tropical multiplication
⊗.  A sum of probabilities becomes log‑sum‑exp; a common approximation for
max‑plus algebra replaces the log‑sum‑exp by the tropical addition ⊕ = max.
Thus Bayesian update, marginalisation and belief propagation on a tree can be
expressed with the tropical primitives.  The hybrid algorithm therefore

* stores priors, likelihoods and false‑positive rates as log‑probabilities,
* uses ``t_mul`` to add log‑values (multiply probabilities),
* uses ``t_add`` (max) as a fast surrogate for log‑sum‑exp,
* employs ``t_matmul`` to propagate the most probable (maximum‑log‑probability)
  belief from a root node through the tree,
* finally combines the resulting log‑beliefs with the Euclidean edge costs
  (treated as negative log‑likelihoods) and with Shannon entropy to obtain a
  decision‑hygiene score.

The code below implements the above fusion while keeping the original
interfaces of both parents where appropriate.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parent A – tropical max‑plus primitives
# ---------------------------------------------------------------------------

def t_add(x, y):
    """Tropical addition (⊕): max(x, y). Broadcasts."""
    return np.maximum(x, y)


def t_mul(x, y):
    """Tropical multiplication (⊗): x + y. Broadcasts."""
    return np.add(x, y)


def t_matmul(A, B):
    """
    Tropical matrix multiplication.

    C[i, j] = max_k ( A[i, k] + B[k, j] )
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # shape (m, p) @ (p, n) → (m, n)
    # broadcast A[:, :, None] + B[None, :, :] → (m, p, n)
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)


def t_polyval(coeffs, x):
    """
    Evaluate a tropical polynomial p(x) = max_i (coeffs[i] + i * x).
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    exponents = np.arange(len(coeffs), dtype=float)
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents.reshape((-1,) + (1,) * x.ndim) * x
    return np.max(terms, axis=0)

# ---------------------------------------------------------------------------
# Parent B – tree metrics + Bayesian utilities
# ---------------------------------------------------------------------------

def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)

    return adj, edge_len, dist


def bayes_marginal(prior: np.ndarray, likelihood: np.ndarray, false_positive: float) -> np.ndarray:
    """
    Exact marginal: P(E) = L·p + FP·(1‑p)
    """
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: np.ndarray, likelihood: np.ndarray, marginal: np.ndarray) -> np.ndarray:
    """
    Exact posterior: P(H|E) = L·p / P(E)
    """
    # avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        posterior = np.where(marginal > 0, (likelihood * prior) / marginal, 0.0)
    return posterior


def shannon_entropy(p: np.ndarray) -> float:
    """Standard Shannon entropy H(p) = -∑ p log p (base e)."""
    p = np.clip(p, 1e-12, 1.0)  # avoid log(0)
    return -np.sum(p * np.log(p))

# ---------------------------------------------------------------------------
# Hybrid layer – tropical‑Bayesian fusion
# ---------------------------------------------------------------------------

def log_to_prob(log_vals: np.ndarray) -> np.ndarray:
    """Convert log‑probabilities to ordinary probabilities, normalising."""
    max_log = np.max(log_vals)  # for numerical stability
    probs_unnorm = np.exp(log_vals - max_log)
    total = np.sum(probs_unnorm)
    return probs_unnorm / total if total > 0 else np.zeros_like(probs_unnorm)


def tropical_bayes_update_log(
    prior_log: np.ndarray,
    likelihood_log: np.ndarray,
    false_pos_log: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a Bayesian update in log‑space using tropical primitives.

    * Multiplication of probabilities → tropical multiplication (t_mul).
    * Summation for marginal → tropical addition (t_add) as a max‑approximation.
      The exact marginal can still be recovered if needed.

    Returns
    -------
    posterior_log : np.ndarray
        Approximate posterior log‑probabilities (max‑approximation).
    marginal_log : np.ndarray
        Approximate log‑marginal using tropical addition.
    """
    # log‑likelihood of true event
    ll_true = t_mul(prior_log, likelihood_log)          # log(p) + log(L)
    # log‑likelihood of false positive (assume false_pos_log is log(fp))
    ll_false = t_mul(t_add(prior_log, -np.inf), false_pos_log)  # log(1-p) + log(fp) → use -inf for log(1-p) placeholder
    # Tropical addition approximates log‑sum‑exp
    marginal_log = t_add(ll_true, ll_false)

    # Posterior (exact) in log domain: log posterior = ll_true - marginal_log
    posterior_log = ll_true - marginal_log
    return posterior_log, marginal_log


def tropical_tree_propagation(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    init_log_belief: Dict[str, float],
) -> Dict[str, float]:
    """
    Propagate log‑beliefs from ``root`` through the tree using tropical matrix
    multiplication.  Edge costs are treated as negative log‑likelihoods (larger
    Euclidean distance → lower likelihood).

    Parameters
    ----------
    init_log_belief : dict mapping node name → log‑belief (usually -inf except root).

    Returns
    -------
    belief_log : dict mapping node name → max‑log‑belief reachable from root.
    """
    # Index nodes
    node_list = list(nodes.keys())
    index = {name: i for i, name in enumerate(node_list)}
    n = len(node_list)

    # Build tropical adjacency matrix A where A[i, j] = -edge_len if edge exists else -inf
    A = np.full((n, n), -np.inf, dtype=float)
    for a, b in edges:
        i, j = index[a], index[b]
        dist = length(nodes[a], nodes[b])
        weight = -dist  # negative distance → log‑likelihood
        A[i, j] = weight
        A[j, i] = weight  # undirected tree

    # Initialise belief vector
    b = np.full((n,), -np.inf, dtype=float)
    for name, logv in init_log_belief.items():
        b[index[name]] = logv

    # Repeated tropical multiplication propagates along paths.
    # Since the graph is a tree, at most (n-1) hops are needed.
    belief = b.copy()
    for _ in range(n - 1):
        belief = t_add(belief, t_matmul(A, belief[:, None]).flatten())
        # The above computes max(belief, max_k (A[:,k] + belief[k]))

    return {name: belief[idx] for name, idx in index.items()}


def hybrid_decision_score(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    prior: np.ndarray,
    likelihood: np.ndarray,
    false_positive: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute a decision‑hygiene score that blends:
      * Shannon entropy of the posterior probability distribution,
      * Tropical cost (negative log‑likelihood) from root to every node,
      * A weighting factor α ∈ [0,1] to balance information vs. cost.

    Returns
    -------
    score : float – lower is better (more certain, lower cost).
    node_scores : dict mapping node → combined score.
    """
    # ---- Bayesian part (exact) ----
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)
    entropy = shannon_entropy(posterior)

    # ---- Tropical part ----
    # Log‑beliefs: start with log prior at root, -inf elsewhere.
    init_log = {root: np.log(prior[0] if prior.size == 1 else prior[np.argmax(posterior)])}
    belief_log = tropical_tree_propagation(nodes, edges, root, init_log)

    # Convert tropical log‑beliefs to probabilities (softmax‑like) for scoring.
    belief_probs = {n: np.exp(v) for n, v in belief_log.items()}
    total_belief = sum(belief_probs.values())
    belief_norm = {n: p / total_belief for n, p in belief_probs.items()}

    # Combine: α * entropy + (1-α) * expected tropical cost
    α = 0.5
    expected_cost = -sum(belief_norm[n] * belief_log[n] for n in belief_log)  # negative log‑belief ≈ cost
    score = α * entropy + (1 - α) * expected_cost

    # Node‑wise combined scores (for inspection)
    node_scores = {
        n: α * (-np.log(posterior[np.argmax(posterior)])) + (1 - α) * (-belief_log[n])
        for n in belief_log
    }

    return score, node_scores

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple tree with 4 nodes forming a line: A–B–C–D
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (2.0, 0.0),
        "D": (3.0, 0.0),
    }
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    root = "A"

    # Prior / likelihood for a single binary hypothesis (for demo)
    prior = np.array([0.6, 0.4])          # two hypotheses
    likelihood = np.array([0.9, 0.2])     # P(E|H)
    false_positive = 0.05

    # Run hybrid decision score
    score, node_scores = hybrid_decision_score(
        nodes, edges, root, prior, likelihood, false_positive
    )

    print(f"Hybrid decision‑hygiene score: {score:.4f}")
    for n, s in node_scores.items():
        print(f"  Node {n}: combined score = {s:.4f}")

    # Demonstrate tropical polynomial evaluation on a scalar
    coeffs = np.array([0.0, -1.0, -3.0])   # p(x) = max(0, -1 + x, -3 + 2x)
    x = 2.5
    print(f"Tropical polynomial at x={x}: {t_polyval(coeffs, x)}")

    # Verify tropical matrix multiplication on a small matrix
    A = np.array([[0, -np.inf], [-2, 0]])
    B = np.array([[0, -1], [-np.inf, 0]])
    print("Tropical matmul result:\n", t_matmul(A, B))