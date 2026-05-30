# DARWIN HAMMER — match 1182, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_physarum_netw_hybrid_sparse_wta_hy_m336_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_voronoi_parti_m169_s0.py (gen3)
# born: 2026-05-29T23:33:15Z

"""Hybrid algorithm merging Physarum flux conductance dynamics with Voronoi‑enhanced bandit routing.

Parents:
- hybrid_hybrid_physarum_netw_hybrid_sparse_wta_hy_m336_s0.py (Physarum flux, conductance update, sparse WTA)
- hybrid_hybrid_hybrid_bandit_hybrid_voronoi_parti_m169_s0.py (Bandit actions, Euclidean Voronoi geometry, circuit‑breaker)

Mathematical bridge:
Both parents manipulate a *propensity*‑like quantity (bandit propensity ↔ Physarum flux source) and a *distance* measure
(edge length in the flux equation, Euclidean distance in the Voronoi partition).  The hybrid therefore updates
edge conductances using the bandit‑derived “quality” q = propensity·reward while weighting the update by the
geometric distance between the edge’s endpoints.  The resulting conductance vector is projected into a high‑dimensional
sparse winner‑take‑all (WTA) code and compared (Hamming distance) against a reference mask for privacy‑aware
pool evaluation.  A lightweight circuit‑breaker monitors reward quality and can disable edges that repeatedly under‑perform.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Core Physarum‑Bandit primitives
# ----------------------------------------------------------------------


def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    """Flux through an edge (Physarum primitive)."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float = 1.0,
                       gain: float = 1.0, decay: float = 0.05) -> float:
    """Standard conductance ODE step."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def hybrid_bandit_flux_update(conductance: float,
                              propensity: float,
                              reward: float,
                              edge_length: float,
                              euclid_dist: float,
                              dt: float = 1.0) -> float:
    """
    Fuse bandit quality q = propensity·reward with Physarum conductance dynamics.
    The Euclidean distance between the edge endpoints modulates the effective gain:
        gain_eff = exp( - euclid_dist / (edge_length + eps) )
    """
    eps = 1e-12
    q = propensity * reward
    gain_eff = math.exp(-euclid_dist / (edge_length + eps))
    return update_conductance(conductance, q, dt, gain=gain_eff, decay=0.05)


# ----------------------------------------------------------------------
# Voronoi / Geometry helpers (Bandit side)
# ----------------------------------------------------------------------


Point = Tuple[float, float]


def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# Sparse Winner‑Take‑All encoding
# ----------------------------------------------------------------------


def sparse_wta_encode(values: List[float], m: int, top_k: int = 1, salt: str = '') -> np.ndarray:
    """
    Deterministically project a dense vector into an m‑dimensional sparse binary code.
    The `top_k` largest absolute values win; ties are broken by a hash of (index, salt).
    """
    if m <= 0:
        raise ValueError('m must be positive')
    if top_k <= 0:
        raise ValueError('top_k must be positive')
    out = np.zeros(m, dtype=np.int8)

    # Obtain indices sorted by absolute magnitude
    abs_vals = np.abs(values)
    sorted_idx = np.argsort(-abs_vals)  # descending

    # Resolve ties with deterministic hash
    selected = []
    i = 0
    while len(selected) < top_k and i < len(sorted_idx):
        idx = sorted_idx[i]
        # Hash combines index and optional salt to ensure reproducibility across runs
        h = hashlib.sha256(f'{idx}:{salt}'.encode()).hexdigest()
        tie_key = (abs_vals[idx], h)
        selected.append((tie_key, idx))
        i += 1

    # Choose the top_k after tie resolution
    selected.sort(key=lambda x: x[0], reverse=True)
    for _, idx in selected[:top_k]:
        # Map original index to a position in the sparse space via modulo
        pos = idx % m
        out[pos] = 1
    return out


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Binary Hamming distance between two equal‑length vectors."""
    if a.shape != b.shape:
        raise ValueError('Vectors must have the same shape')
    return int(np.sum(a != b))


# ----------------------------------------------------------------------
# Circuit‑breaker (privacy‑aware pool guard)
# ----------------------------------------------------------------------


class CircuitBreaker:
    """Simple failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3, reward_floor: float = 0.0):
        if failure_threshold <= 0:
            raise ValueError('failure_threshold must be positive')
        self.failure_threshold = failure_threshold
        self.reward_floor = reward_floor
        self.failures: int = 0
        self.open: bool = False

    def record(self, reward: float) -> None:
        """Record a reward; open the breaker if failures exceed the threshold."""
        if reward < self.reward_floor:
            self.failures += 1
        else:
            # successful reward resets the counter partially
            self.failures = max(0, self.failures - 1)

        if self.failures >= self.failure_threshold:
            self.open = True

    def reset(self) -> None:
        self.failures = 0
        self.open = False


# ----------------------------------------------------------------------
# Data structures tying together geometry, bandit, and conductance
# ----------------------------------------------------------------------


@dataclass
class Node:
    node_id: str
    position: Point


@dataclass
class Edge:
    edge_id: str
    node_a: Node
    node_b: Node
    conductance: float = 1.0
    length: float = field(init=False)

    def __post_init__(self):
        self.length = euclidean_distance(self.node_a.position, self.node_b.position)


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    position: Point  # Voronoi centre for this action


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------


def propagate_flux(edges: List[Edge],
                  pressures: Dict[str, float]) -> Dict[str, float]:
    """
    Compute flux on every edge given node pressures.
    Returns a dict mapping edge_id → flux value.
    """
    fluxes = {}
    for e in edges:
        p_a = pressures.get(e.node_a.node_id, 0.0)
        p_b = pressures.get(e.node_b.node_id, 0.0)
        fluxes[e.edge_id] = flux(e.conductance, e.length, p_a, p_b)
    return fluxes


def hybrid_step(edges: List[Edge],
                actions: List[BanditAction],
                rewards: Dict[str, float],
                dt: float = 1.0) -> None:
    """
    One hybrid iteration:
      1. For each edge, find the nearest bandit action (Voronoi assignment).
      2. Update the edge conductance using hybrid_bandit_flux_update.
      3. Record reward in a CircuitBreaker tied to the edge.
    """
    # Build a quick lookup for actions by id
    action_by_id = {a.action_id: a for a in actions}
    # Initialise a breaker per edge
    breakers: Dict[str, CircuitBreaker] = {e.edge_id: CircuitBreaker() for e in edges}

    for e in edges:
        # 1️⃣ Voronoi nearest‑action search (brute force, fine for small demos)
        nearest = min(actions,
                      key=lambda a: euclidean_distance(e.node_a.position, a.position))
        # 2️⃣ Retrieve reward (fallback to 0)
        reward = rewards.get(nearest.action_id, 0.0)

        # 3️⃣ Compute Euclidean distance between edge midpoint and action centre
        mid_point = ((e.node_a.position[0] + e.node_b.position[0]) / 2,
                     (e.node_a.position[1] + e.node_b.position[1]) / 2)
        dist = euclidean_distance(mid_point, nearest.position)

        # 4️⃣ Conductance update
        e.conductance = hybrid_bandit_flux_update(
            conductance=e.conductance,
            propensity=nearest.propensity,
            reward=reward,
            edge_length=e.length,
            euclid_dist=dist,
            dt=dt
        )

        # 5️⃣ Circuit‑breaker bookkeeping
        breakers[e.edge_id].record(reward)

        # Optional: disable edge if breaker opened (set conductance to 0)
        if breakers[e.edge_id].open:
            e.conductance = 0.0


def encode_network_state(edges: List[Edge],
                         m: int = 128,
                         top_k: int = 3,
                         salt: str = 'hybrid') -> np.ndarray:
    """
    Project the vector of edge conductances into a sparse WTA code.
    """
    conductances = [e.conductance for e in edges]
    return sparse_wta_encode(conductances, m, top_k=top_k, salt=salt)


def evaluate_pool(edges: List[Edge],
                  reference_mask: np.ndarray,
                  m: int = 128,
                  top_k: int = 3) -> int:
    """
    Encode current network, then compute Hamming distance to a reference mask.
    The distance can be used as a privacy‑preserving similarity metric.
    """
    encoded = encode_network_state(edges, m=m, top_k=top_k)
    return hamming_distance(encoded, reference_mask)


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny geometric graph
    nodes = [Node("A", (0.0, 0.0)),
             Node("B", (1.0, 0.0)),
             Node("C", (0.5, 1.0))]

    edges = [Edge("e1", nodes[0], nodes[1]),
             Edge("e2", nodes[1], nodes[2]),
             Edge("e3", nodes[2], nodes[0])]

    # Define three bandit actions placed near each node
    actions = [BanditAction(action_id="a1", propensity=0.8, expected_reward=1.0,
                            confidence_bound=0.2, position=(0.0, 0.0)),
               BanditAction(action_id="a2", propensity=0.6, expected_reward=0.5,
                            confidence_bound=0.3, position=(1.0, 0.0)),
               BanditAction(action_id="a3", propensity=0.9, expected_reward=0.9,
                            confidence_bound=0.1, position=(0.5, 1.0))]

    # Simulated rewards for each action
    rewards = {"a1": 1.0, "a2": 0.2, "a3": 0.8}

    # Run a few hybrid steps
    for step in range(5):
        hybrid_step(edges, actions, rewards, dt=0.5)

    # Build a random reference mask for evaluation
    ref_mask = np.random.randint(0, 2, size=128, dtype=np.int8)

    # Evaluate privacy‑aware pool similarity
    dist = evaluate_pool(edges, ref_mask, m=128, top_k=2)
    print(f"Hamming distance to reference mask: {dist}")

    # Show final conductances
    for e in edges:
        print(f"Edge {e.edge_id}: conductance = {e.conductance:.4f}")