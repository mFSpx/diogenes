# DARWIN HAMMER — match 681, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s4.py (gen2)
# parent_b: hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s4.py (gen2)
# born: 2026-05-29T23:30:23Z

"""Hybrid Leader‑Election / Hoeffding‑Tree + Sheaf‑Based Tropical Max‑Plus.

Parents:
- ``hybrid_hybrid_distributed_l_hybrid_hoeffding_tre_m24_s4`` (probabilistic
  broadcast, simulated‑annealing acceptance, Hoeffding bound, tropical max‑plus
  split evaluation).
- ``hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s4`` (cellular sheaf
  representation, linear restriction maps, count‑min sketch utilities).

Mathematical bridge:
Both algorithms decide *whether* a structural change (a split, a leader
promotion, etc.) should be kept.  The Hoeffding bound ε plays the role of a
temperature‑like quantity: a larger ε makes acceptance harder.  The tropical
max‑plus evaluation supplies an “energy” E for a candidate split; in the sheaf
setting this energy is obtained from the restriction maps and node sections via
max‑plus algebra:

    G(node) = max_{(u,v)∈E_out(node)}  max_i ( (src_map·s_u)_i  ⊕  (dst_map·s_v)_i )
          = max over edges of the element‑wise max‑plus sum of the restricted
            section vectors.

We then define ΔE = ε – G and use a simulated‑annealing acceptance probability
p = exp(−ΔE / T).  The hybrid algorithm therefore fuses:

1. Hoeffding bound computation (statistics).
2. Tropical max‑plus gain derived from sheaf linear algebra.
3. Simulated‑annealing acceptance controlling leader promotion / split
   adoption.

The code below implements this fusion and provides three core functions that
demonstrate the hybrid operation.  A small smoke test at the end exercises the
pipeline on a synthetic sheaf."""

from __future__ import annotations

import math
import random
import sys
import pathlib
from collections.abc import Mapping, Hashable
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]

# ----------------------------------------------------------------------
# Sheaf definition (from parent B)
# ----------------------------------------------------------------------


class Sheaf:
    """
    Cellular sheaf on a directed graph.

    * Nodes carry a vector space of dimension given by ``node_dims``.
    * Each directed edge ``(u, v)`` carries a pair of linear restriction maps
      ``src_map : ℝ^{dim(u)} → ℝ^{dim(e)}`` and ``dst_map : ℝ^{dim(v)} → ℝ^{dim(e)}``.
    * A *section* assigns a vector to every node.
    """

    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]):
        self.node_dims: Dict[Any, int] = dict(node_dims)
        self.edges: List[Tuple[Any, Any]] = list(edges)
        # (u,v) → (src_map, dst_map)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        # node → section vector
        self._sections: Dict[Any, np.ndarray] = {}

    # -------------------------------------------------------------------
    # Restriction maps
    # -------------------------------------------------------------------
    def set_restriction(
        self,
        edge: Tuple[Any, Any],
        src_map: np.ndarray,
        dst_map: np.ndarray,
    ) -> None:
        """Register the two restriction matrices for a directed edge."""
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float), np.asarray(dst_map, dtype=float))

    # -------------------------------------------------------------------
    # Sections
    # -------------------------------------------------------------------
    def set_section(self, node: Any, value: np.ndarray) -> None:
        """Assign a vector to a node."""
        vec = np.asarray(value, dtype=float)
        if vec.shape[0] != self.node_dims[node]:
            raise ValueError(f"Section dimension for node {node} must be {self.node_dims[node]}")
        self._sections[node] = vec

    def get_section(self, node: Any) -> np.ndarray:
        """Retrieve the section vector for a node (zero vector if absent)."""
        if node not in self._sections:
            # default to zero vector of appropriate dimension
            self._sections[node] = np.zeros(self.node_dims[node], dtype=float)
        return self._sections[node]

    # -------------------------------------------------------------------
    # Utility: outgoing edges of a node
    # -------------------------------------------------------------------
    def outgoing_edges(self, node: Any) -> List[Tuple[Any, Any]]:
        return [e for e in self.edges if e[0] == node]

# ----------------------------------------------------------------------
# Core hybrid functions (demonstrate the fusion)
# ----------------------------------------------------------------------


def hoeffding_bound(num_samples: int, value_range: float, delta: float) -> float:
    """
    Compute Hoeffding bound ε = sqrt( (R^2 * ln(1/δ)) / (2 * n) ).

    Parameters
    ----------
    num_samples: int
        Number of independent observations (n).
    value_range: float
        The range R = max - min of the random variable.
    delta: float
        Desired probability of exceeding the bound (typically 0.05).

    Returns
    -------
    epsilon: float
        Hoeffding bound ε.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if value_range <= 0:
        raise ValueError("value_range must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    epsilon = math.sqrt((value_range ** 2 * math.log(1.0 / delta)) / (2.0 * num_samples))
    return epsilon


def tropical_max_plus_gain(sheaf: Sheaf, node: Any) -> float:
    """
    Compute tropical max‑plus gain G for a candidate split rooted at ``node``.

    For each outgoing edge (node, v) we compute the element‑wise max‑plus sum
    of the restricted section vectors:

        g_edge = max_i ( (src_map @ s_node)_i  +  (dst_map @ s_v)_i )

    The overall gain is the maximum g_edge over all outgoing edges.

    Returns
    -------
    gain: float
        Tropical gain (higher is better, interpreted as lower energy).
    """
    max_gain = -math.inf
    s_node = sheaf.get_section(node)

    for edge in sheaf.outgoing_edges(node):
        u, v = edge
        src_map, dst_map = sheaf._restrictions[edge]
        s_v = sheaf.get_section(v)

        # Linear restriction
        src_vec = src_map @ s_node          # shape (d_e,)
        dst_vec = dst_map @ s_v

        # Max‑plus (tropical) addition is ordinary addition; max is the tropical sum.
        edge_gain = np.max(src_vec + dst_vec)  # scalar
        if edge_gain > max_gain:
            max_gain = edge_gain

    # If node has no outgoing edges, define gain as 0 (neutral).
    return max_gain if max_gain != -math.inf else 0.0


def simulated_annealing_accept(delta_E: float, temperature: float) -> bool:
    """
    Acceptance test based on simulated annealing.

    Probability p = exp(−ΔE / T).  If ΔE ≤ 0 the move is always accepted.

    Parameters
    ----------
    delta_E: float
        Energy difference ΔE = ε − G.
    temperature: float
        Current temperature T (must be > 0).

    Returns
    -------
    accept: bool
        Whether the candidate split is accepted.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if delta_E <= 0:
        return True
    prob = math.exp(-delta_E / temperature)
    return random.random() < prob


def hybrid_split_decision(
    sheaf: Sheaf,
    node: Any,
    num_samples: int,
    value_range: float,
    delta: float,
    temperature: float,
) -> bool:
    """
    Decide whether to accept a split (or promote a leader) at ``node`` by fusing
    Hoeffding statistics, tropical max‑plus gain, and simulated‑annealing
    acceptance.

    Steps
    -----
    1. Compute Hoeffding bound ε.
    2. Compute tropical gain G via sheaf restriction maps.
    3. ΔE = ε − G.
    4. Accept with probability exp(−ΔE / T).

    Returns
    -------
    accepted: bool
        True if the split is kept.
    """
    epsilon = hoeffding_bound(num_samples, value_range, delta)
    gain = tropical_max_plus_gain(sheaf, node)
    delta_E = epsilon - gain
    return simulated_annealing_accept(delta_E, temperature)


def broadcast_probability(total_phases: int, current_phase: int) -> float:
    """
    Probability that a node broadcasts in the current phase (parent A primitive).

    Mirrors the original implementation: p = min(1, 1 / 2^{total_phases - current_phase}).
    """
    if total_phases < 1 or current_phase < 1:
        raise ValueError("phases and phase must be positive")
    exponent = max(0, total_phases - current_phase)
    return min(1.0, 1.0 / (2 ** exponent))


# ----------------------------------------------------------------------
# Simple Count‑Min Sketch (utility from parent B, kept lightweight)
# ----------------------------------------------------------------------


def count_min_sketch(stream: List[int], width: int, depth: int) -> np.ndarray:
    """
    Very small count‑min sketch implementation.

    Returns a (depth, width) array of counts.
    """
    if width <= 0 or depth <= 0:
        raise ValueError("width and depth must be positive")
    sketch = np.zeros((depth, width), dtype=int)
    for item in stream:
        for d in range(depth):
            # deterministic pseudo‑hash using built‑in hash
            idx = (hash((item, d)) % width)
            sketch[d, idx] += 1
    return sketch


def hybrid_info_loss(sheaf: Sheaf, sketch: np.ndarray) -> float:
    """
    Example hybrid metric: combine sheaf‑based tropical gain with an
    information‑loss estimate derived from a count‑min sketch.

    The loss is defined as:

        L = (average sketch count) / (1 + total tropical gain)

    Lower L indicates a more informative (less noisy) configuration.
    """
    total_gain = sum(tropical_max_plus_gain(sheaf, node) for node in sheaf.node_dims)
    avg_count = sketch.mean()
    return avg_count / (1.0 + total_gain)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a tiny directed graph with two nodes and one edge.
    node_dims = {"A": 3, "B": 2}
    edges = [("A", "B")]

    sheaf = Sheaf(node_dims, edges)

    # Random restriction maps (compatible dimensions)
    src_map = np.random.rand(4, node_dims["A"])   # maps A (3) → edge space (4)
    dst_map = np.random.rand(4, node_dims["B"])   # maps B (2) → edge space (4)
    sheaf.set_restriction(("A", "B"), src_map, dst_map)

    # Random sections
    sheaf.set_section("A", np.random.rand(node_dims["A"]))
    sheaf.set_section("B", np.random.rand(node_dims["B"]))

    # Parameters for Hoeffding & annealing
    n_samples = 100
    value_range = 1.0          # assume normalized gain values in [0,1]
    delta = 0.05
    temperature = 0.5

    # Decision
    accept = hybrid_split_decision(
        sheaf,
        "A",
        num_samples=n_samples,
        value_range=value_range,
        delta=delta,
        temperature=temperature,
    )
    print(f"Hybrid split decision for node 'A': {'ACCEPTED' if accept else 'REJECTED'}")

    # Broadcast probability demo
    prob = broadcast_probability(total_phases=5, current_phase=3)
    print(f"Broadcast probability (phase 3 of 5): {prob:.4f}")

    # Count‑min sketch demo
    stream = [random.randint(0, 10) for _ in range(1000)]
    sketch = count_min_sketch(stream, width=20, depth=5)
    loss = hybrid_info_loss(sheaf, sketch)
    print(f"Hybrid information loss metric: {loss:.4f}")