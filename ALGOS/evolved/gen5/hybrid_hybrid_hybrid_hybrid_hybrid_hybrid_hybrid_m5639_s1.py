# DARWIN HAMMER — match 5639, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m84_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m960_s2.py (gen4)
# born: 2026-05-30T00:03:42Z

"""Hybrid algorithm merging:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m84_s1.py (Multivector‑modulated pheromone workshare)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m960_s2.py (Pheromone‑weighted Minimum‑Cost Tree)

Mathematical bridge:
The pheromone level of each graph edge is stored as a 1‑vector blade in a
Multivector **P**.  The scalar part of **P** (overall pheromone intensity) and
the StoreState “dance” value act as a dynamic gain *g* that scales every edge
pheromone *pₑ*.  These scaled pheromones weight the edge lengths ℓ(e) in the
minimum‑cost cost function

    C_h = Σₑ (g·pₑ)·ℓ(e) + λ·Σᵥ qᵥ·d(v)                                   (1)

where *qᵥ* is a node belief derived from the propensities of BanditActions
incident to node *v*.  The hybrid reward is

    R_h = Σₐ qₐ·rₐ                                                       (2)

Thus the Multivector structure from parent A provides the pheromone vector,
while the cost aggregation from parent B supplies the tree‑scoring logic.
The StoreState supplies a feedback loop (Δ) that adapts *g* over time,
creating a closed‑loop hybrid system.""" 

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable

# ----------------------------------------------------------------------
# Core data structures (unified definitions)
# ----------------------------------------------------------------------
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

@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Update store level and keep last delta for the 'dance' property."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """Dynamic gain derived from the most recent delta."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))

# ----------------------------------------------------------------------
# Multivector utilities (simplified for 1‑vectors only)
# ----------------------------------------------------------------------
class Multivector:
    """
    Very small subset of geometric algebra sufficient for the hybrid.
    Keys are frozensets of identifiers (blades).  The empty frozenset
    represents the scalar part.
    """
    def __init__(self, components: Dict[frozenset, float] = None, n: int = 0):
        self.n = int(n)
        self.components: Dict[frozenset, float] = {}
        if components:
            for blade, coef in components.items():
                if coef != 0.0:
                    self.components[frozenset(blade)] = float(coef)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) - coef
        return Multivector(result, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({b: c * scalar for b, c in self.components.items()}, self.n)

    __rmul__ = __mul__

    def scalar_part(self) -> float:
        """Return the coefficient of the empty blade."""
        return self.components.get(frozenset(), 0.0)

    def blade_coeff(self, blade: frozenset) -> float:
        """Coefficient of a specific blade (e.g., frozenset({'e1'}))."""
        return self.components.get(blade, 0.0)

    def grade(self, k: int) -> "Multivector":
        """Extract the k‑grade part (blades with exactly k identifiers)."""
        return Multivector(
            {b: c for b, c in self.components.items() if len(b) == k}, self.n
        )

    def __repr__(self) -> str:
        return f"Multivector({self.components})"

# ----------------------------------------------------------------------
# Graph helpers
# ----------------------------------------------------------------------
Node = str
Edge = Tuple[Node, Node]
EdgeID = str

def edge_id(u: Node, v: Node) -> EdgeID:
    """Canonical edge identifier (order‑independent)."""
    return f"{min(u, v)}-{max(u, v)}"

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def build_pheromone_multivector(edge_pheromones: Dict[EdgeID, float]) -> Multivector:
    """
    Convert a mapping edge_id → pheromone value into a Multivector where each
    edge corresponds to a 1‑vector blade.
    """
    comps = {frozenset([eid]): val for eid, val in edge_pheromones.items() if val != 0.0}
    # optional scalar part can encode total pheromone mass
    total = sum(edge_pheromones.values())
    comps[frozenset()] = total
    return Multivector(comps, n=len(edge_pheromones))

def compute_node_beliefs(
    actions: Iterable[BanditAction],
    incident_map: Dict[Node, List[EdgeID]],
) -> Dict[Node, float]:
    """
    Aggregate bandit propensities into a belief q_v for each node.
    q_v = Σ_{a incident to v} propensity_a / (1 + Σ propensity_a)
    """
    node_sum: Dict[Node, float] = {v: 0.0 for v in incident_map}
    for a in actions:
        # Assume action_id encodes the edge it belongs to: "e1" etc.
        eid = a.action_id
        for node, edges in incident_map.items():
            if eid in edges:
                node_sum[node] += a.propensity
    beliefs = {}
    for node, s in node_sum.items():
        beliefs[node] = s / (1.0 + s) if s > 0 else 0.0
    return beliefs

def hybrid_cost(
    multivector: Multivector,
    edges: Dict[EdgeID, float],
    node_distances: Dict[Node, float],
    store: StoreState,
    lambda_param: float = 1.0,
) -> float:
    """
    Compute the hybrid cost C_h defined in the module docstring.

    - multivector: pheromone representation (provides p_e)
    - edges: mapping edge_id → base length ℓ(e)
    - node_distances: d(v) for each node
    - store: supplies dynamic gain g = store.dance
    - lambda_param: weight of the node‑belief term
    """
    g = store.dance
    # Edge term Σ_e (g·p_e)·ℓ(e)
    edge_term = 0.0
    for eid, length in edges.items():
        p_e = multivector.blade_coeff(frozenset([eid]))
        edge_term += (g * p_e) * length

    # Node belief term Σ_v q_v·d(v)
    # Build incident map for beliefs
    incident_map: Dict[Node, List[EdgeID]] = {}
    for eid in edges:
        u, v = eid.split("-")
        incident_map.setdefault(u, []).append(eid)
        incident_map.setdefault(v, []).append(eid)

    # Dummy BanditAction list for belief computation – in practice the caller
    # provides real actions; here we only need propensities, so we mock.
    dummy_actions: List[BanditAction] = []  # will be supplied externally
    # For flexibility, we allow caller to set a global variable later.
    if hasattr(hybrid_cost, "_cached_actions"):
        dummy_actions = hybrid_cost._cached_actions  # type: ignore

    q = compute_node_beliefs(dummy_actions, incident_map)
    node_term = sum(q.get(v, 0.0) * node_distances.get(v, 0.0) for v in node_distances)

    return edge_term + lambda_param * node_term

def hybrid_update_cycle(
    actions: List[BanditAction],
    edge_pheromones: Dict[EdgeID, float],
    store: StoreState,
    inflow: List[float],
    outflow: List[float],
    lambda_param: float = 1.0,
) -> Tuple[float, Multivector, StoreState]:
    """
    Perform one iteration of the hybrid algorithm:
    1. Update the StoreState using supplied inflow/outflow.
    2. Build a Multivector from updated pheromones.
    3. Compute the hybrid cost.
    4. Return (cost, multivector, updated store).
    """
    # 1. Store dynamics
    store.update(inflow, outflow)

    # 2. Multivector construction
    mv = build_pheromone_multivector(edge_pheromones)

    # Cache actions for hybrid_cost (simple global‑ish trick)
    hybrid_cost._cached_actions = actions  # type: ignore

    # 3. Define a toy graph (edges and node distances) – in real use these
    # would be arguments; we keep them minimal for the demo.
    edges = {eid: random.uniform(1.0, 5.0) for eid in edge_pheromones}
    nodes = set()
    for eid in edges:
        u, v = eid.split("-")
        nodes.update([u, v])
    node_distances = {n: random.uniform(0.5, 3.0) for n in nodes}

    # 4. Compute cost
    cost = hybrid_cost(mv, edges, node_distances, store, lambda_param=lambda_param)

    return cost, mv, store

# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple graph with three nodes and two edges
    edge_pheromones = {
        edge_id("A", "B"): 0.8,
        edge_id("B", "C"): 0.3,
    }

    # Create a few BanditActions (propensity values arbitrary)
    actions = [
        BanditAction(action_id=edge_id("A", "B"), propensity=0.6,
                     expected_reward=1.0, confidence_bound=0.2, algorithm="bandit"),
        BanditAction(action_id=edge_id("B", "C"), propensity=0.4,
                     expected_reward=0.8, confidence_bound=0.1, algorithm="bandit"),
    ]

    store = StoreState(level=2.0, alpha=1.2, beta=0.9, dt=0.5,
                      base=1.0, gain=0.5, limit=5.0)

    inflow = [0.5, 0.2]   # example resource inflow
    outflow = [0.1]       # example resource outflow

    cost, mv, updated_store = hybrid_update_cycle(
        actions=actions,
        edge_pheromones=edge_pheromones,
        store=store,
        inflow=inflow,
        outflow=outflow,
        lambda_param=0.7,
    )

    print("Hybrid cost:", cost)
    print("Multivector:", mv)
    print("Updated StoreState:", updated_store)
    print("Store dance (gain factor):", updated_store.dance)