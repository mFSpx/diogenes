# DARWIN HAMMER — match 190, survivor 0
# gen: 3
# parent_a: tropical_maxplus.py (gen0)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py (gen2)
# born: 2026-05-29T23:27:29Z

"""
This module integrates the tropical_maxplus and hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0 algorithms into a single hybrid system.
The bridge between the two structures is the concept of applying tropical max-plus algebra to the decision hygiene scoring system, 
and the expected cost of the minimum-cost tree computed using Bayesian update.
"""

import numpy as np
import math
import random
import sys
import pathlib

def t_add(x, y):
    return np.maximum(x, y)

def t_mul(x, y):
    return np.add(x, y)

def t_matmul(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def length(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def tree_metrics(nodes, edges, root):
    adj = {n: [] for n in nodes}
    edge_len = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
    dist = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)
    return adj, edge_len, dist

def bayes_marginal(prior, likelihood, false_positive):
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior, likelihood, marginal):
    return likelihood * prior / marginal

def hybrid_tree_metrics(nodes, edges, root, prior, likelihood, false_positive):
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    marginal = bayes_marginal(prior, likelihood, false_positive)
    updated = bayes_update(prior, likelihood, marginal)
    tropical_dist = [t_add(dist[node], updated) for node in dist]
    return adj, edge_len, dict(zip(dist.keys(), tropical_dist))

def tropical_bayes_update(prior, likelihood, marginal):
    return t_mul(likelihood, prior) / marginal

def hybrid_bayes_update(prior, likelihood, marginal, nodes, edges, root):
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    updated = bayes_update(prior, likelihood, marginal)
    tropical_updated = t_mul(updated, marginal)
    return dict(zip(dist.keys(), [t_add(dist[node], tropical_updated) for node in dist]))

if __name__ == "__main__":
    nodes = {'A': (0, 0), 'B': (1, 1), 'C': (2, 2)}
    edges = [('A', 'B'), ('B', 'C')]
    root = 'A'
    prior = np.array([0.5, 0.5])
    likelihood = np.array([0.7, 0.3])
    false_positive = 0.1
    hybrid_tree_metrics(nodes, edges, root, prior, likelihood, false_positive)
    marginal = bayes_marginal(prior, likelihood, false_positive)
    tropical_bayes_update(prior, likelihood, marginal)
    hybrid_bayes_update(prior, likelihood, marginal, nodes, edges, root)