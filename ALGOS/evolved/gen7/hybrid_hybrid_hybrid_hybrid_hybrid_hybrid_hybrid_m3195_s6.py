# DARWIN HAMMER — match 3195, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m1238_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_xgboos_hybrid_hybrid_hybrid_m1553_s2.py (gen6)
# born: 2026-05-29T23:48:36Z

"""Hybrid Algorithm: Fusion of DARWIN HAMMER privacy risk & Ollivier‑Ricci curvature with
Pheromone‑based signal decay and 3‑D Geometric Algebra.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m1238_s3.py (privacy risk + Ollivier‑Ricci)
- hybrid_hybrid_hybrid_xgboos_hybrid_hybrid_hybrid_m1553_s2.py (pheromone decay + 3‑D GA)

Mathematical bridge:
The reconstruction risk score `r_i` of a node is used as a *mass scaling factor* for
both the lazy random‑walk distribution in the Ollivier‑Ricci curvature and for the
strength of pheromone signals attached to the node.  The hybrid transport plan
between neighboring nodes `i` and `j` becomes

    μ_i(v) = α·r_i·δ_{i=v} + (1‑α)·r_i·(1/deg(i))·∑_{u∈N(i)} δ_{u=v}

where `r_i` now also multiplies the pheromone value when aggregating curvature
contributions.  The curvature scalar `κ(i)` is finally lifted into a bivector
`K(i) = κ(i)·e12` using the 3‑D Geometric Algebra representation, allowing downstream
geometric processing while preserving the original scalar meaning.

The module therefore provides:
1. Privacy‑aware risk scoring.
2. Pheromone entry management with exponential decay.
3. Hybrid Ollivier‑Ricci curvature computation weighted by risk.
4. GA‑based curvature embedding.
"""

import sys
import math
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Constants & helpers
# ----------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
_EPS = 1e-8  # regularisation for matrix inverses


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z notation."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Privacy risk utilities (Parent A)
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Normalized reconstruction risk in [0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


# ----------------------------------------------------------------------
# Pheromone entry (Parent B)
# ----------------------------------------------------------------------
class PheromoneEntry:
    """Container for a decaying pheromone signal."""
    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid1())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = float(signal_value)
        self.half_life_seconds = int(half_life_seconds)
        now = datetime.utcnow()
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        """Seconds elapsed since last decay."""
        return (datetime.utcnow() - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Exponential decay factor based on half‑life."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        """Apply decay to the stored value."""
        self.signal_value *= self.decay_factor()
        self.last_decay = datetime.utcnow()


# ----------------------------------------------------------------------
# Simple 3‑D Geometric Algebra (GA) implementation (Parent B)
# ----------------------------------------------------------------------
class Multivector:
    """
    3‑D GA multivector with components:
    [s, e1, e2, e3, e12, e13, e23, e123]
    """
    __slots__ = ("components",)

    def __init__(self, components: List[float] = None):
        if components is None:
            components = [0.0] * 8
        if len(components) != 8:
            raise ValueError("Multivector requires 8 components.")
        self.components = np.array(components, dtype=float)

    def __add__(self, other: "Multivector") -> "Multivector":
        return Multivector(self.components + other.components)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return Multivector(self.components - other.components)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector(self.components * scalar)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        comp = ", ".join(f"{c:.3g}" for c in self.components)
        return f"Multivector([{comp}])"

    @staticmethod
    def bivector_e12(scale: float = 1.0) -> "Multivector":
        """Return scale·e12 bivector."""
        vec = [0.0] * 8
        vec[4] = scale  # e12 position
        return Multivector(vec)


# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass
class Node:
    """Graph node enriched with RAM, privacy risk and pheromones."""
    id: int
    ram_mb: int
    risk_score: float = 0.0
    pheromones: List[PheromoneEntry] = field(default_factory=list)

    def total_pheromone_signal(self) -> float:
        """Sum of current pheromone values (after decay)."""
        return sum(p.signal_value for p in self.pheromones)


# ----------------------------------------------------------------------
# Hybrid transport distribution (risk‑aware)
# ----------------------------------------------------------------------
def hybrid_transport_distribution(nodes: List[Node],
                                  adjacency: np.ndarray,
                                  alpha: float = 0.5) -> Dict[int, Dict[int, float]]:
    """
    Compute μ_i(v) for every node i and every possible destination v.

    Parameters
    ----------
    nodes : List[Node]
        List of graph nodes.
    adjacency : np.ndarray
        Symmetric (N,N) adjacency matrix (0/1 entries).
    alpha : float
        Weight of staying‑in‑place mass.

    Returns
    -------
    Dict[int, Dict[int, float]]
        Nested dict μ[i][v] = probability mass from i to v.
    """
    n = len(nodes)
    deg = adjacency.sum(axis=1)  # degree vector
    mu: Dict[int, Dict[int, float]] = {}

    for i, node_i in enumerate(nodes):
        mu_i: Dict[int, float] = {}
        r_i = node_i.risk_score
        # stay‑in‑place component
        mu_i[node_i.id] = alpha * r_i
        # neighbor component
        if deg[i] > 0:
            neighbor_share = (1.0 - alpha) * r_i / deg[i]
            for j in range(n):
                if adjacency[i, j]:
                    mu_i[nodes[j].id] = mu_i.get(nodes[j].id, 0.0) + neighbor_share
        mu[i] = mu_i
    return mu


# ----------------------------------------------------------------------
# Hybrid Ollivier‑Ricci curvature (risk + pheromone weighting)
# ----------------------------------------------------------------------
def hybrid_ollivier_ricci(nodes: List[Node],
                         adjacency: np.ndarray,
                         alpha: float = 0.5,
                         epsilon: float = 1.0) -> Dict[int, float]:
    """
    Compute privacy‑aware curvature κ(i) for each node.

    The transport plan μ_i(v) is weighted by node risk and the *total pheromone
    signal* of the destination node, giving higher influence to heavily signaled
    neighbours.

    Returns
    -------
    Dict[int, float] – mapping node.id → curvature scalar.
    """
    n = len(nodes)
    mu = hybrid_transport_distribution(nodes, adjacency, alpha)

    curvature: Dict[int, float] = {}
    for i, node_i in enumerate(nodes):
        transport_mass = 0.0
        for v_id, mass in mu[i].items():
            # locate destination node
            dest_node = next(filter(lambda nd: nd.id == v_id, nodes))
            # pheromone‑scaled mass
            pheromone_factor = 1.0 + epsilon * dest_node.total_pheromone_signal()
            transport_mass += mass * pheromone_factor
        deg_i = adjacency[i].sum()
        if deg_i == 0:
            curvature[node_i.id] = 0.0
        else:
            curvature[node_i.id] = 1.0 - transport_mass / deg_i
    return curvature


# ----------------------------------------------------------------------
# Embedding curvature into GA bivectors
# ----------------------------------------------------------------------
def curvature_to_bivector(curvature: Dict[int, float]) -> Dict[int, Multivector]:
    """
    Lift each scalar curvature κ into a bivector κ·e12.
    """
    return {node_id: Multivector.bivector_e12(kappa) for node_id, kappa in curvature.items()}


# ----------------------------------------------------------------------
# Pheromone management utilities
# ----------------------------------------------------------------------
def decay_all_pheromones(nodes: List[Node]) -> None:
    """Apply exponential decay to every pheromone entry of every node."""
    for node in nodes:
        for pher in node.pheromones:
            pher.apply_decay()


def inject_random_pheromones(nodes: List[Node],
                            count_per_node: int = 2,
                            half_life_range: tuple = (30, 120)) -> None:
    """
    Populate each node with a few random pheromone entries.
    """
    for node in nodes:
        for _ in range(count_per_node):
            surface = f"surf_{random.randint(0, 9)}"
            kind = random.choice(["signal", "noise"])
            value = random.random()
            half_life = random.randint(*half_life_range)
            node.pheromones.append(PheromoneEntry(surface, kind, value, half_life))


# ----------------------------------------------------------------------
# Example high‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_step(nodes: List[Node],
                adjacency: np.ndarray,
                alpha: float = 0.5,
                epsilon: float = 1.0) -> Dict[int, Multivector]:
    """
    One complete hybrid iteration:
    1. Decay existing pheromones.
    2. Compute risk‑aware Ollivier‑Ricci curvature.
    3. Embed curvature into GA bivectors.

    Returns
    -------
    Dict[int, Multivector] – node id → curvature bivector.
    """
    decay_all_pheromones(nodes)
    curv = hybrid_ollivier_ricci(nodes, adjacency, alpha, epsilon)
    return curvature_to_bivector(curv)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny random graph of 5 nodes
    rng = np.random.default_rng(seed=42)
    num_nodes = 5
    adj = (rng.random((num_nodes, num_nodes)) > 0.6).astype(int)
    np.fill_diagonal(adj, 0)
    adj = np.maximum(adj, adj.T)  # make symmetric

    # Instantiate nodes with random RAM and risk scores
    nodes: List[Node] = []
    for i in range(num_nodes):
        ram = random.randint(256, 2048)
        uq = random.randint(0, 1000)
        total = random.randint(1000, 5000)
        risk = reconstruction_risk_score(uq, total)
        nodes.append(Node(id=i, ram_mb=ram, risk_score=risk))

    # Inject pheromones
    inject_random_pheromones(nodes, count_per_node=3)

    # Perform a hybrid step
    bivectors = hybrid_step(nodes, adj, alpha=0.6, epsilon=0.8)

    # Simple verification output
    for nid, biv in bivectors.items():
        print(f"Node {nid}: curvature bivector = {biv}")

    sys.exit(0)