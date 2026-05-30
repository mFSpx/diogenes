# DARWIN HAMMER — match 190, survivor 1
# gen: 3
# parent_a: tropical_maxplus.py (gen0)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py (gen2)
# born: 2026-05-29T23:27:29Z

"""
This module integrates the tropical_maxplus and hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0 algorithms into a single hybrid system.
The bridge between the two structures is the concept of information entropy applied to the decision hygiene scoring system,
which can be represented as a tropical polynomial. The tropical polynomial can be evaluated using the tropical_maxplus algorithm,
and the decision hygiene scoring system can be updated using the Bayesian update from the hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0 algorithm.
"""

import numpy as np
import math
import random
import sys
import pathlib

def t_add(x, y):
    """Tropical addition: max(x, y). Broadcasts."""
    return np.maximum(x, y)

def t_mul(x, y):
    """Tropical multiplication: x + y. Broadcasts."""
    return np.add(x, y)

def t_matmul(A, B):
    """Tropical matrix multiply.

    C[i, j] = max_k( A[i, k] + B[k, j] )

    A: (m, p), B: (p, n) → C: (m, n).
    Handles -inf entries correctly via numpy broadcasting.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    # broadcast: A[i, k, newaxis] + B[newaxis, k, j] then max over k
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def t_polyval(coeffs, x):
    """Evaluate a tropical polynomial at x.

    Tropical polynomial: p(x) = coeffs[0] ⊕ (coeffs[1] ⊗ x) ⊕ ... ⊕ (coeffs[d] ⊗ d*x)
                               = max_i( coeffs[i] + i*x )

    coeffs: 1-D array of length d+1 (tropical coefficients, may include -inf).
    x     : scalar or array broadcastable against (d+1,).
    Returns same shape as x.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    x = np.asarray(x, dtype=float)
    # exponents [0, 1, ..., d] — tropical exponentiation = ordinary multiplication
    exponents = np.arange(len(coeffs), dtype=float)
    # shape: (d+1,) + x.shape
    terms = coeffs.reshape((-1,) + (1,) * x.ndim) + exponents.reshape((-1,) + (1,) * x.ndim) * x
    return np.max(terms, axis=0)

def length(a, b):
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(nodes, edges, root):
    """
    Build adjacency, compute Euclidean edge lengths and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (ordered as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj = {n: [] for n in nodes}
    edge_len = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])

    # BFS/DFS to compute distances from root
    dist = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                # identify the edge direction used for length lookup
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)

    return adj, edge_len, dist

def bayes_marginal(prior, likelihood, false_positive):
    """
    Vectorised marginal:  P(E) = L·p + FP·(1‑p)
    """
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior, likelihood, marginal):
    """
    Vectorised posterior:  P
    """
    return (likelihood * prior) / marginal

def hybrid_tropical_bayes_update(coeffs, x, prior, likelihood, false_positive):
    """
    Hybrid function that combines tropical polynomial evaluation with Bayesian update.

    coeffs: 1-D array of length d+1 (tropical coefficients, may include -inf).
    x     : scalar or array broadcastable against (d+1,).
    prior: numpy array of prior probabilities.
    likelihood: numpy array of likelihoods.
    false_positive: scalar false positive rate.
    """
    tropical_polynomial = t_polyval(coeffs, x)
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)
    return tropical_polynomial, marginal, posterior

def hybrid_tree_tropical_bayes_update(nodes, edges, root, coeffs, x, prior, likelihood, false_positive):
    """
    Hybrid function that combines tree metrics with tropical polynomial evaluation and Bayesian update.

    nodes: dictionary of node coordinates.
    edges: list of edges.
    root: root node.
    coeffs: 1-D array of length d+1 (tropical coefficients, may include -inf).
    x     : scalar or array broadcastable against (d+1,).
    prior: numpy array of prior probabilities.
    likelihood: numpy array of likelihoods.
    false_positive: scalar false positive rate.
    """
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    tropical_polynomial = t_polyval(coeffs, x)
    marginal = bayes_marginal(prior, likelihood, false_positive)
    posterior = bayes_update(prior, likelihood, marginal)
    return adj, edge_len, dist, tropical_polynomial, marginal, posterior

if __name__ == "__main__":
    # Smoke test
    coeffs = np.array([1.0, 2.0, 3.0])
    x = 2.0
    prior = np.array([0.5, 0.5])
    likelihood = np.array([0.8, 0.2])
    false_positive = 0.1
    nodes = {'A': (0.0, 0.0), 'B': (1.0, 1.0), 'C': (2.0, 2.0)}
    edges = [('A', 'B'), ('B', 'C')]
    root = 'A'
    hybrid_tropical_bayes_update(coeffs, x, prior, likelihood, false_positive)
    hybrid_tree_tropical_bayes_update(nodes, edges, root, coeffs, x, prior, likelihood, false_positive)