# DARWIN HAMMER — match 3680, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_minimum_cost__hybrid_hybrid_bandit_m249_s5.py (gen3)
# parent_b: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_hybrid_m1214_s3.py (gen5)
# born: 2026-05-29T23:51:19Z

"""Hybrid Algorithm: hybrid_hybrid_minimum_cost_tree_bandit_curvature_sketch
Integrates:

* Parent A – “Hybrid Tree‑Bandit‑Sketch Algorithm” (minimum‑cost tree,
  Bayesian edge update, distinct‑count λ via LogLog sketch).
* Parent B – “Hybrid Bayesian‑Krampus‑Ollivier‑Ricci‑CountMin‑Bandit
  Algorithm” (Count‑Min sketch for log‑likelihood, bandit actions with
  propensity/reward, Ollivier‑Ricci curvature influencing edge priors).

Mathematical Bridge
-------------------
1. **Likelihood per edge** `L_e` is obtained from a Count‑Min sketch that
   aggregates log‑likelihood contributions of observed items (tokens).
2. **Prior per edge** `π_e` is derived from a bandit policy: the
   propensity of the action associated with the edge serves as the prior
   probability.
3. **Bayesian posterior** `p_e` follows  
   `p_e = π_e * L_e / Σ_{e'} π_{e'} * L_{e'}` (Eq. 2‑A).
4. **Ollivier‑Ricci curvature** `κ_e` of an edge is computed from the
   graph’s distance matrix; it modulates the posterior as  
   `p_e' = p_e^{1+κ_e}` (higher curvature strengthens belief).
5. **Distinct‑pattern λ** is estimated by a LogLog sketch (Parent A) and
   acts as the RLCT weight in the hybrid cost.
6. **Hybrid cost** (Eq. 3‑A) merges tree geometry and node‑wise
   curvature‑weighted beliefs:  

   `C_h = Σ_e p_e'·ℓ(e) + λ Σ_v q_v·d(v)`

   where `ℓ(e)` is edge length, `d(v)` is distance from root,
   `q_v` is the normalized product of incident edge posteriors.

The code below implements this fused pipeline with three core functions
demonstrating the hybrid operation.
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Sketch utilities (shared by both parents)
# ----------------------------------------------------------------------
def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Count‑Min sketch counting occurrences of *items*."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            idx = hash((item, d)) % width
            table[d][idx] += 1
    return table

def estimate_distinct_loglog(items: Iterable[str], b: int = 12) -> int:
    """Very lightweight LogLog distinct‑count estimator."""
    m = 1 << b                     # number of registers
    registers = [0] * m
    for item in items:
        h = hash(item)
        idx = h & (m - 1)          # low‑b bits as register index
        w = h >> b                 # remaining bits
        # position of leftmost 1 (1‑based); if w==0 use b+1
        rho = (w.bit_length() - w.bit_length() + 1) if w else b + 1
        rho = (w ^ (w - 1)).bit_length()  # count trailing zeros +1
        registers[idx] = max(registers[idx], rho)
    # harmonic mean of 2**(-register)
    Z = sum(2.0 ** -r for r in registers)
    alpha = 0.7213 / (1 + 1.079 / m)  # bias correction for large m
    estimate = int(alpha * m * m / Z)
    return max(estimate, len(set(items)))  # fall back to exact for tiny sets

# ----------------------------------------------------------------------
# Bandit structures (from Parent B)
# ----------------------------------------------------------------------
@dataclass
class BanditAction:
    action_id: str
    propensity: float          # prior probability of taking this action
    expected_reward: float
    confidence_bound: float

def bandit_update_policy(actions: Dict[str, BanditAction],
                         rewards: Dict[str, float],
                         alpha: float = 0.1) -> None:
    """Online update of bandit actions using observed *rewards*."""
    for aid, reward in rewards.items():
        act = actions[aid]
        # Incremental mean update
        new_exp = act.expected_reward + (reward - act.expected_reward) * alpha
        # Shrink confidence bound with more data (simple heuristic)
        new_cb = max(0.0, act.confidence_bound - alpha * 0.05)
        actions[aid] = BanditAction(
            action_id=aid,
            propensity=act.propensity,
            expected_reward=new_exp,
            confidence_bound=new_cb,
        )

# ----------------------------------------------------------------------
# Graph utilities (Parent A)
# ----------------------------------------------------------------------
def tree_metrics(root: str,
                 edges: List[Edge],
                 lengths: Dict[Edge, float]) -> Tuple[Dict[str, float], Dict[str, List[Edge]]]:
    """
    Compute root‑to‑node distances `d(v)` and adjacency list.
    Returns (distances, adjacency).
    """
    adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for u, v in edges:
        w = lengths[(u, v)]
        adj[u].append((v, w))
        adj[v].append((u, w))

    distances: Dict[str, float] = {root: 0.0}
    visited = {root}
    stack = [root]
    while stack:
        node = stack.pop()
        for neigh, w in adj[node]:
            if neigh not in visited:
                distances[neigh] = distances[node] + w
                visited.add(neigh)
                stack.append(neigh)
    # simplify adjacency to edge list per node
    simple_adj: Dict[str, List[Edge]] = defaultdict(list)
    for u, v in edges:
        simple_adj[u].append((u, v))
        simple_adj[v].append((v, u))
    return distances, simple_adj

# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature (Parent B contribution)
# ----------------------------------------------------------------------
def ollivier_ricci_curvature(edge: Edge,
                             distances: Dict[Tuple[str, str], float],
                             epsilon: float = 1e-6) -> float:
    """
    Approximate Ollivier‑Ricci curvature κ_e for *edge* using
    a very simple transport plan:
        κ = 1 - (W1 / d),
    where W1 is the earth mover distance between the unit
    neighbourhood measures of the two endpoints.
    For speed we use the degree‑normalized inverse distance as a proxy.
    """
    u, v = edge
    d_uv = distances.get((u, v), distances.get((v, u), 1.0))
    # degree‑based neighbourhood mass
    deg_u = sum(1 for (a, b) in distances if a == u or b == u)
    deg_v = sum(1 for (a, b) in distances if a == v or b == v)
    if deg_u == 0 or deg_v == 0:
        return 0.0
    transport = abs(1.0 / deg_u - 1.0 / deg_v)
    kappa = 1.0 - transport / (d_uv + epsilon)
    return kappa

# ----------------------------------------------------------------------
# Bayesian edge posterior (Parent A) fused with curvature (Parent B)
# ----------------------------------------------------------------------
def bayes_edge_posteriors(priors: Dict[Edge, float],
                          log_likelihoods: Dict[Edge, float],
                          curvatures: Dict[Edge, float]) -> Dict[Edge, float]:
    """
    Compute curvature‑modulated posterior:
        p_e ∝ (π_e * exp(L_e))^{1+κ_e}
    Returns normalized posteriors.
    """
    unnorm: Dict[Edge, float] = {}
    for e, prior in priors.items():
        ll = log_likelihoods.get(e, -math.inf)
        kappa = curvatures.get(e, 0.0)
        # exponentiate log‑likelihood safely
        likelihood = math.exp(ll) if ll > -700 else 0.0
        base = prior * likelihood
        weight = base ** (1.0 + kappa) if base > 0 else 0.0
        unnorm[e] = weight
    total = sum(unnorm.values()) or 1.0
    return {e: w / total for e, w in unnorm.items()}

# ----------------------------------------------------------------------
# Hybrid cost computation (core of the fusion)
# ----------------------------------------------------------------------
def hybrid_tree_cost(root: str,
                     edges: List[Edge],
                     lengths: Dict[Edge, float],
                     posteriors: Dict[Edge, float],
                     lambda_rlct: float) -> float:
    """
    Compute the hybrid cost:
        C_h = Σ_e p_e·ℓ(e) + λ Σ_v q_v·d(v)
    where q_v = normalized product of incident edge posteriors.
    """
    # Edge contribution
    edge_term = sum(posteriors[e] * lengths[e] for e in edges)

    # Node‑wise products
    distances, adj = tree_metrics(root, edges, lengths)
    node_products: Dict[str, float] = {}
    for node, incident_edges in adj.items():
        prod = 1.0
        for e in incident_edges:
            prod *= posteriors.get(e, 1e-9)
        node_products[node] = prod
    total_prod = sum(node_products.values()) or 1.0
    q = {v: prod / total_prod for v, prod in node_products.items()}

    node_term = sum(q[v] * distances[v] for v in distances)

    return edge_term + lambda_rlct * node_term

# ----------------------------------------------------------------------
# End‑to‑end hybrid pipeline
# ----------------------------------------------------------------------
def compute_hybrid_cost(root: str,
                        edges: List[Edge],
                        lengths: Dict[Edge, float],
                        observations: List[str],
                        bandit_actions: Dict[Edge, BanditAction]) -> float:
    """
    Orchestrates the full hybrid procedure:
      1. Count‑Min sketch → per‑edge log‑likelihood estimate.
      2. LogLog sketch → λ (distinct‑pattern RLCT weight).
      3. Bandit propensities → priors π_e.
      4. Ollivier‑Ricci curvature κ_e.
      5. Bayesian posterior with curvature modulation.
      6. Hybrid cost C_h.
    """
    # 1. Log‑likelihood per edge via Count‑Min (simple proxy)
    sketch = count_min_sketch(observations)
    # Use the minimum count across depth as a surrogate for log‑likelihood
    log_likelihoods: Dict[Edge, float] = {}
    for e in edges:
        # hash edge to a token string
        token = f"{e[0]}->{e[1]}"
        mins = min(sketch[d][hash((token, d)) % len(sketch[d])] for d in range(len(sketch)))
        # Convert count to log‑likelihood (add 1 to avoid log(0))
        log_likelihoods[e] = math.log(mins + 1)

    # 2. Distinct‑pattern estimate λ
    lambda_rlct = float(estimate_distinct_loglog(observations))

    # 3. Priors from bandit propensities
    priors: Dict[Edge, float] = {e: bandit_actions[e].propensity for e in edges}

    # 4. Curvature per edge
    # Build a simple distance dictionary for curvature (undirected)
    pair_dist: Dict[Tuple[str, str], float] = {}
    for (u, v) in edges:
        pair_dist[(u, v)] = lengths[(u, v)]
        pair_dist[(v, u)] = lengths[(u, v)]
    curvatures = {e: ollivier_ricci_curvature(e, pair_dist) for e in edges}

    # 5. Posterior
    posteriors = bayes_edge_posteriors(priors, log_likelihoods, curvatures)

    # 6. Hybrid cost
    cost = hybrid_tree_cost(root, edges, lengths, posteriors, lambda_rlct)
    return cost

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple tree: root A connected to B and C
    root_node = "A"
    edge_list = [("A", "B"), ("A", "C")]
    edge_lengths = {("A", "B"): 1.5, ("A", "C"): 2.0}

    # Synthetic observations (tokens)
    obs = [f"token_{i%5}" for i in range(100)]

    # Initialise bandit actions (one per edge)
    bandit_actions: Dict[Edge, BanditAction] = {
        ("A", "B"): BanditAction(action_id="e_AB", propensity=0.6, expected_reward=0.0, confidence_bound=0.5),
        ("A", "C"): BanditAction(action_id="e_AC", propensity=0.4, expected_reward=0.0, confidence_bound=0.5),
    }

    # Simulate a tiny reward feedback
    simulated_rewards = {"e_AB": 1.0, "e_AC": 0.2}
    bandit_update_policy({e: bandit_actions[e] for e in edge_list}, simulated_rewards)

    # Compute hybrid cost
    hybrid_cost = compute_hybrid_cost(root_node, edge_list, edge_lengths, obs, bandit_actions)
    print(f"Hybrid cost: {hybrid_cost:.4f}")