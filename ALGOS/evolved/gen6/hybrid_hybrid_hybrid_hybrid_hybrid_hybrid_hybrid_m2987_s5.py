# DARWIN HAMMER — match 2987, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1182_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_bayes__hybrid_ternary_route_m812_s1.py (gen4)
# born: 2026-05-29T23:47:13Z

"""Hybrid Algorithm integrating Physarum‑style conductance dynamics with Bayesian
feature‑aware minimum‑cost routing.

Parents:
- hybrid_hybrid_hybrid_physar_hybrid_hybrid_m1182_s0.py (Flux‑based conductance
  update + bandit router with Voronoi partition)
- hybrid_hybrid_hybrid_bayes__hybrid_ternary_route_m812_s1.py (Deterministic
  feature extraction + ternary MST routing with Bayesian posterior)

Mathematical Bridge
------------------
Both parents rely on Euclidean distances:
* Parent A uses distance implicitly in a Voronoi partition to relate contexts
  to actions.
* Parent B builds a Minimum Spanning Tree (MST) with edge cost that is a weighted
  sum of Euclidean distance in the spatial plane and Euclidean distance in a
  semantic feature space.

The hybrid therefore defines a *composite edge cost*  

    c_{ij} = α·‖p_i−p_j‖₂ + β·‖v_i−v_j‖₂ ,

where p_i∈ℝ² is the node position and v_i∈ℝⁿ is the deterministic feature
vector.  The MST built on these costs yields a conductance network.  Conductance
evolution follows the Physarum flux rule  

    Φ_{ij} = g_{ij}/ℓ_{ij}·(π_i−π_j) ,

with g_{ij} the edge conductance, ℓ_{ij}=‖p_i−p_j‖₂ the physical length and
π_i a scalar “pressure” derived from a Bayesian posterior over nodes.
Bandit propensities modulate conductance updates via  

    g_{ij}←max(0, g_{ij}+Δt·(gain·|propensity·reward|−decay·g_{ij})) .

Thus spatial, semantic and decision‑theoretic information are fused into a single
dynamic graph.

The module below implements this hybrid pipeline."""
import sys
import math
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Core primitives from Parent A
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def hybrid_bandit_update(conductance: float, propensity: float, reward: float,
                         dt: float = 1.0, gain: float = 1.0, decay: float = 0.05) -> float:
    q = propensity * reward
    return update_conductance(conductance, q, dt, gain, decay)


# ----------------------------------------------------------------------
# Bandit data structures (lightweight version)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # ∈[0,1]
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


# ----------------------------------------------------------------------
# Feature extraction – deterministic mapping (Parent B)
# ----------------------------------------------------------------------
def _deterministic_hash(text: str) -> int:
    """Stable 64‑bit integer hash for *text* using SHA‑256."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def extract_master_vector(text: str, dim: int = 8) -> np.ndarray:
    """
    Produce a reproducible pseudo‑random feature vector of length *dim*.
    The vector is normalised to unit length.
    """
    seed = _deterministic_hash(text) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ----------------------------------------------------------------------
# Graph / Node representation
# ----------------------------------------------------------------------
Point = Tuple[float, float]


@dataclass
class Node:
    node_id: str
    position: Point                     # p_i ∈ ℝ²
    feature: np.ndarray                 # v_i ∈ ℝⁿ
    pressure: float = 0.0               # scalar π_i (posterior probability)
    edges: Dict[str, float] = field(default_factory=dict)   # neighbor_id → conductance g_ij


# ----------------------------------------------------------------------
# Distance and composite cost utilities
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def feature_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.linalg.norm(v1 - v2)


def composite_edge_cost(node_a: Node, node_b: Node, alpha: float = 0.5, beta: float = 0.5) -> float:
    """c_{ij} = α·‖p_i−p_j‖₂ + β·‖v_i−v_j‖₂."""
    d_spatial = euclidean_distance(node_a.position, node_b.position)
    d_feature = feature_distance(node_a.feature, node_b.feature)
    return alpha * d_spatial + beta * d_feature


# ----------------------------------------------------------------------
# Minimum‑Cost (MST) construction using Prim's algorithm
# ----------------------------------------------------------------------
def build_mst(nodes: List[Node], alpha: float = 0.5, beta: float = 0.5) -> List[Tuple[str, str, float]]:
    """
    Returns a list of edges (id_a, id_b, cost) forming a Minimum Spanning Tree
    over the supplied nodes using the composite edge cost.
    """
    if not nodes:
        return []
    # Initialise
    visited = {nodes[0].node_id}
    edges: List[Tuple[float, str, str]] = []  # (cost, from, to)
    mst: List[Tuple[str, str, float]] = []

    # Pre‑compute cost matrix for efficiency
    id_to_node = {n.node_id: n for n in nodes}
    node_ids = [n.node_id for n in nodes]

    while len(visited) < len(nodes):
        min_edge = None
        min_cost = float('inf')
        for u_id in visited:
            u = id_to_node[u_id]
            for v_id in node_ids:
                if v_id in visited:
                    continue
                v = id_to_node[v_id]
                cost = composite_edge_cost(u, v, alpha, beta)
                if cost < min_cost:
                    min_cost = cost
                    min_edge = (u_id, v_id, cost)
        if min_edge is None:
            break  # disconnected (should not happen)
        mst.append(min_edge)
        visited.add(min_edge[1])

    return mst


# ----------------------------------------------------------------------
# Bayesian posterior based on MST degree (simple likelihood)
# ----------------------------------------------------------------------
def compute_posterior(nodes: List[Node], mst_edges: List[Tuple[str, str, float]]) -> None:
    """
    Updates each node's ``pressure`` attribute to a posterior probability.
    Prior is uniform; likelihood ∝ (degree + 1) to avoid zero.
    The pressures are normalised to sum to 1.
    """
    degree: Dict[str, int] = {n.node_id: 0 for n in nodes}
    for a, b, _ in mst_edges:
        degree[a] += 1
        degree[b] += 1
    likelihood = np.array([degree[n.node_id] + 1 for n in nodes], dtype=float)
    prior = np.ones_like(likelihood) / len(likelihood)
    unnorm_posterior = prior * likelihood
    posterior = unnorm_posterior / unnorm_posterior.sum()
    for n, prob in zip(nodes, posterior):
        n.pressure = prob


# ----------------------------------------------------------------------
# Hybrid update step: conductance evolution driven by bandit actions
# ----------------------------------------------------------------------
def hybrid_update_step(nodes: List[Node],
                       mst_edges: List[Tuple[str, str, float]],
                       actions: Dict[Tuple[str, str], BanditAction],
                       dt: float = 1.0,
                       gain: float = 1.0,
                       decay: float = 0.05) -> None:
    """
    For every edge in the MST:
      * retrieve the associated BanditAction (identified by (src, dst) unordered)
      * draw a stochastic reward ∈{0,1} with probability = expected_reward
      * update the edge conductance using the Physarum‑bandit rule
      * store the new conductance back into both incident nodes
    """
    # Helper to fetch or create conductance entry
    def get_conductance(u: Node, v: Node) -> float:
        return u.edges.get(v.node_id, 1.0)   # default initial conductance = 1.0

    for a_id, b_id, _ in mst_edges:
        node_a = next(n for n in nodes if n.node_id == a_id)
        node_b = next(n for n in nodes if n.node_id == b_id)

        # Unordered key for action lookup
        key = tuple(sorted((a_id, b_id)))
        action = actions.get(key)
        if action is None:
            # If no explicit action, fabricate a neutral one
            action = BanditAction(
                action_id=f"{a_id}<->{b_id}",
                propensity=0.5,
                expected_reward=0.5,
                confidence_bound=0.0,
                algorithm="default"
            )
            actions[key] = action

        # Stochastic reward based on expected_reward
        reward = 1.0 if random.random() < action.expected_reward else 0.0

        # Current conductance (symmetric)
        g_current = get_conductance(node_a, node_b)

        # Update using hybrid bandit rule
        g_new = hybrid_bandit_update(g_current,
                                    propensity=action.propensity,
                                    reward=reward,
                                    dt=dt,
                                    gain=gain,
                                    decay=decay)

        # Store symmetrically
        node_a.edges[node_b.node_id] = g_new
        node_b.edges[node_a.node_id] = g_new


# ----------------------------------------------------------------------
# Example function that computes flux on all MST edges after update
# ----------------------------------------------------------------------
def compute_fluxes(nodes: List[Node],
                   mst_edges: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
    """
    Returns a list of (src_id, dst_id, flux_value) for each MST edge using the
    current pressures and conductances.
    """
    fluxes = []
    for a_id, b_id, _ in mst_edges:
        node_a = next(n for n in nodes if n.node_id == a_id)
        node_b = next(n for n in nodes if n.node_id == b_id)
        length = euclidean_distance(node_a.position, node_b.position)
        conductance = node_a.edges.get(node_b.node_id, 0.0)
        phi = flux(conductance, length, node_a.pressure, node_b.pressure)
        fluxes.append((a_id, b_id, phi))
    return fluxes


# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_route_mst_bayes(nodes: List[Node],
                          alpha: float = 0.5,
                          beta: float = 0.5,
                          dt: float = 1.0,
                          gain: float = 1.0,
                          decay: float = 0.05) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    """
    Executes the full hybrid algorithm:
      1. Build MST using composite costs.
      2. Compute Bayesian posterior pressures from MST degree.
      3. Perform a conductance update step driven by bandit actions.
      4. Return the MST edge list and the resulting fluxes.
    """
    mst = build_mst(nodes, alpha, beta)
    compute_posterior(nodes, mst)

    # Prepare an empty action store; actions will be auto‑generated if missing.
    actions: Dict[Tuple[str, str], BanditAction] = {}

    hybrid_update_step(nodes, mst, actions, dt, gain, decay)

    fluxes = compute_fluxes(nodes, mst)
    return mst, fluxes


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Create a small synthetic dataset
    texts = [
        "alpha", "bravo", "charlie", "delta", "echo"
    ]
    nodes: List[Node] = []
    for i, txt in enumerate(texts):
        pos = (random.uniform(0, 10), random.uniform(0, 10))
        feat = extract_master_vector(txt, dim=8)
        nodes.append(Node(node_id=f"N{i}", position=pos, feature=feat))

    mst_edges, fluxes = hybrid_route_mst_bayes(nodes,
                                               alpha=0.6,
                                               beta=0.4,
                                               dt=1.0,
                                               gain=1.0,
                                               decay=0.05)

    print("MST edges (src, dst, cost):")
    for e in mst_edges:
        print(e)

    print("\nNode pressures (posterior probabilities):")
    for n in nodes:
        print(f"{n.node_id}: {n.pressure:.4f}")

    print("\nFluxes after conductance update:")
    for src, dst, phi in fluxes:
        print(f"{src}<->{dst}: flux = {phi:.6f}")

    # Verify that conductances are non‑negative
    for n in nodes:
        for neigh, g in n.edges.items():
            assert g >= 0.0, "Conductance became negative!"

    print("\nSmoke test completed successfully.")