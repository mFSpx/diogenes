# DARWIN HAMMER — match 4404, survivor 0
# gen: 4
# parent_a: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s4.py (gen3)
# parent_b: hybrid_tropical_maxplus_hybrid_hybrid_minimu_m190_s0.py (gen3)
# born: 2026-05-29T23:55:21Z

"""
Module for integrating the DARWIN HAMMER match 11, survivor 4 (physarum network hybrid bandit routing) 
and DARWIN HAMMER match 190, survivor 0 (hybrid tropical max-plus routing with decision hygiene scoring) 
algorithms into a single hybrid system. 
The bridge between the two structures lies in applying tropical max-plus algebra to the decision hygiene scoring system, 
and the expected cost of the minimum-cost tree computed using Bayesian update.
This fusion enables adaptive, learning-based routing in the physarum network, influenced by the bandit's exploration-exploitation trade-offs.
Parent algorithms: physarum_network.py, hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py, 
tropical_maxplus.py, hybrid_hybrid_minimum_cost__hybrid_decision_hygi_m7_s0.py
"""

import numpy as np
import math
import random
import sys
import pathlib

@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit decision."""
    action_id: str
    propensity: float            # interpreted as inflow rate
    expected_reward: float
    confidence_bound: float      # interpreted as outflow rate
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Observed reward for a given action."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


def t_add(x, y):
    return np.maximum(x, y)

def t_matmul(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return np.max(A[:, :, np.newaxis] + B[np.newaxis, :, :], axis=1)

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def integrate_bandit_with_physarum(bandit_action: BanditAction, edge_length: float, pressure_a: float, pressure_b: float, conductance: float, eps: float = 1e-12) -> float:
    """Integrate bandit propensity and confidence bound with physarum flux-based conductance updates."""
    q = bandit_action.propensity - bandit_action.confidence_bound
    flux_value = flux(conductance, edge_length, pressure_a, pressure_b, eps)
    return update_conductance(conductance, q, dt=1.0, gain=1.0, decay=0.05) + flux_value


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


def length(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def integrate_tropical_maxplus_with_bandit(bandit_action: BanditAction, nodes, edges, root, prior, likelihood, false_positive):
    """Integrate bandit propensity and confidence bound with tropical max-plus routing and decision hygiene scoring."""
    adj, edge_len, dist = hybrid_tree_metrics(nodes, edges, root, prior, likelihood, false_positive)
    bandit_q = bandit_action.propensity - bandit_action.confidence_bound
    flux_value = flux(1.0, max(edge_len.values()), 1.0, 0.0)
    return update_conductance(1.0, bandit_q, dt=1.0, gain=1.0, decay=0.05) + flux_value * t_matmul(dist, dist)


def hybrid_hybrid_routing(bandit_action: BanditAction, nodes, edges, root, prior, likelihood, false_positive):
    """Hybrid routing algorithm integrating physarum network with tropical max-plus routing and decision hygiene scoring."""
    return integrate_bandit_with_physarum(bandit_action, max(edge_len.values()), 1.0, 0.0, integrate_tropical_maxplus_with_bandit(bandit_action, nodes, edges, root, prior, likelihood, false_positive))


def tropical_bayes_update(prior, likelihood, marginal):
    return t_mul(likelihood, prior) / marginal


def hybrid_bayes_update(prior, likelihood, marginal, nodes, edges, root):
    adj, edge_len, dist = tree_metrics(nodes, edges, root)
    updated = bayes_update(prior, likelihood, marginal)
    tropical_updated = t_mul(updated, marginal)
    return dict(zip(dist.keys(), [t_add(dist[nod], tropical_updated[nod]) for nod in dist]))


def hybrid_physarum_tropical_maxplus(nodes, edges, root, prior, likelihood, false_positive, bandit_action: BanditAction):
    """Hybrid algorithm integrating physarum network with tropical max-plus routing and decision hygiene scoring."""
    adj, edge_len, dist = hybrid_tree_metrics(nodes, edges, root, prior, likelihood, false_positive)
    marginal = bayes_marginal(prior, likelihood, false_positive)
    updated = bayes_update(prior, likelihood, marginal)
    tropical_updated = t_mul(updated, marginal)
    return dict(zip(dist.keys(), [t_add(dist[nod], tropical_updated[nod]) for nod in dist])), edge_len


if __name__ == "__main__":
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0)}
    edges = [(0, 1), (1, 2)]
    root = 0
    prior = 1.0
    likelihood = 0.5
    false_positive = 0.1
    bandit_action = BanditAction('action_1', 0.5, 1.0, 0.2, 'algorithm_1')

    _, _ = hybrid_physarum_tropical_maxplus(nodes, edges, root, prior, likelihood, false_positive, bandit_action)