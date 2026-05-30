# DARWIN HAMMER — match 258, survivor 4
# gen: 3
# parent_a: hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s2.py (gen2)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s7.py (gen1)
# born: 2026-05-29T23:27:53Z

"""Hybrid Semantic‑Morphology Fractional Tree Cost (HSM‑FTC)

This module fuses two distinct lineages:

* **Parent A** – *semantic_neighbors / hybrid_endpoint_circuit* – provides a
  morphology model (length, width, height, mass) together with geometric
  indices (sphericity, flatness) and a derived *recovery priority* that
  quantifies how quickly a physical endpoint can self‑right.

* **Parent B** – *caputo_fractional / minimum_cost_tree* – supplies the
  Caputo fractional‑memory kernel ϕ(t;α)=t^{α‑1}/Γ(α) and the machinery
  for weighting a sequence of incremental edge‑cost contributions in a
  tree.

**Mathematical bridge**

When a new edge is inserted into a growing tree, the *incremental cost*
is split into a *material* part (derived from the endpoint’s morphology)
and a *path* part (distance‑weighted).  The *recovery priority* `ρ` of the
attached node modulates the material term, reflecting that more
self‑rightable endpoints are cheaper to integrate.  The sequence of
incremental costs `{c_k}` is then combined with the Caputo kernel to
produce a *fractional‑memory tree cost*


C_α = Σ_{k=0}^{T‑1} w_k · c_k ,   w_k = ϕ(T‑1‑k;α) / Σ_j ϕ(T‑1‑j;α)


Thus geometry‑driven recovery priority becomes the physical weight in a
history‑aware fractional cost evaluation.

The resulting hybrid algorithm can be used for topology‑aware design
problems where both structural morphology and long‑range construction
memory matter (e.g. evolving robotic morphologies, phylogenetic network
scoring, or incremental layout optimisation)."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Morphology & Recovery Priority
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity: cube‑root of volume divided by longest side."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness = (length+width)/(2·height)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    """Physical time needed for self‑righting, derived from flatness."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalised priority in [0,1]; higher means faster recovery."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def _cos(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Cosine similarity – retained from the original neighbour code."""
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if den == 0 else sum(x * y for x, y in zip(a, b)) / den

# ----------------------------------------------------------------------
# Parent B – Caputo fractional kernel utilities
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def gamma_lanczos(z: float) -> float:
    """Lanczos approximation for the Gamma function (real z>0)."""
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    z -= 1
    x = _LANCZOS_C[0]
    for i in range(1, len(_LANCZOS_C)):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def caputo_weights(alpha: float, T: int) -> np.ndarray:
    """
    Compute normalized Caputo weights w_k = ϕ(T‑1‑k;α) / Σ_j ϕ(T‑1‑j;α)
    where ϕ(t;α) = t^{α‑1} / Γ(α) for t>0, and ϕ(0;α)=1/Γ(α) (α>0).
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0,1]")
    if T <= 0:
        raise ValueError("T must be positive")
    gamma_a = gamma_lanczos(alpha)
    times = np.arange(T - 1, -1, -1, dtype=float)  # T‑1‑k
    phi = np.where(times == 0.0, 1.0, times ** (alpha - 1.0)) / gamma_a
    w = phi / phi.sum()
    return w

# ----------------------------------------------------------------------
# Hybrid data structures
# ----------------------------------------------------------------------
@dataclass
class Node:
    id: int
    morph: Morphology
    position: Tuple[float, float]  # 2‑D coordinates for distance calculations

@dataclass
class Edge:
    parent: int
    child: int
    length: float  # Euclidean distance between parent and child

class FractionalMorphTree:
    """Tree that records insertion order and can compute fractional‑memory cost."""
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []          # insertion order matters
        self.root_id: int | None = None

    # ------------------------------------------------------------------
    # Core hybrid operations
    # ------------------------------------------------------------------
    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node {node.id} already exists")
        self.nodes[node.id] = node
        if self.root_id is None:
            self.root_id = node.id

    def add_edge(self, parent_id: int, child_id: int) -> None:
        if parent_id not in self.nodes or child_id not in self.nodes:
            raise KeyError("Both endpoints must be existing nodes")
        p = self.nodes[parent_id].position
        c = self.nodes[child_id].position
        length = math.hypot(p[0] - c[0], p[1] - c[1])
        self.edges.append(Edge(parent=parent_id, child=child_id, length=length))

    def _incremental_cost(self, edge: Edge) -> float:
        """
        Hybrid incremental cost for a single edge.

        material_cost  = volume * (1 - ρ)   where ρ = recovery_priority(node)
        path_cost      = edge.length * ρ
        total          = material_cost + path_cost
        """
        child_node = self.nodes[edge.child]
        # Volume as proxy for material usage
        volume = child_node.morph.length * child_node.morph.width * child_node.morph.height
        rho = recovery_priority(child_node.morph)
        material_cost = volume * (1.0 - rho)
        path_cost = edge.length * rho
        return material_cost + path_cost

    def fractional_cost(self, alpha: float = 0.7) -> float:
        """
        Compute the Caputo‑weighted sum of incremental costs over the
        insertion sequence.
        """
        T = len(self.edges)
        if T == 0:
            return 0.0
        weights = caputo_weights(alpha, T)
        inc_costs = np.array([self._incremental_cost(e) for e in self.edges], dtype=float)
        return float((weights * inc_costs).sum())

# ----------------------------------------------------------------------
# Helper functions demonstrating the hybrid operation
# ----------------------------------------------------------------------
def random_morphology() -> Morphology:
    """Generate a random but physically plausible morphology."""
    length = random.uniform(0.2, 2.0)
    width  = random.uniform(0.2, 2.0)
    height = random.uniform(0.2, 2.0)
    mass   = random.uniform(0.5, 5.0)
    return Morphology(length, width, height, mass)

def build_random_tree(num_nodes: int = 7, seed: int = 42) -> FractionalMorphTree:
    """
    Construct a random rooted tree with `num_nodes` nodes.
    Nodes receive random morphologies and random 2‑D positions.
    Edges are added in a breadth‑first fashion to guarantee a tree.
    """
    random.seed(seed)
    tree = FractionalMorphTree()
    # create nodes
    for i in range(num_nodes):
        pos = (random.uniform(-5, 5), random.uniform(-5, 5))
        node = Node(id=i, morph=random_morphology(), position=pos)
        tree.add_node(node)
    # connect nodes ensuring a tree (simple parent = floor((i‑1)/2))
    for child_id in range(1, num_nodes):
        parent_id = (child_id - 1) // 2
        tree.add_edge(parent_id, child_id)
    return tree

def hybrid_cost_demo(alpha: float = 0.7, nodes: int = 10) -> Tuple[float, List[float]]:
    """
    Build a random tree, compute its fractional cost, and also return the
    list of raw incremental costs for inspection.
    """
    tree = build_random_tree(num_nodes=nodes)
    raw = [tree._incremental_cost(e) for e in tree.edges]
    cost = tree.fractional_cost(alpha=alpha)
    return cost, raw

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple demonstration with default parameters
    alpha_val = 0.6
    node_count = 8
    cost, raw_costs = hybrid_cost_demo(alpha=alpha_val, nodes=node_count)
    print(f"Hybrid fractional cost (α={alpha_val:.2f}, nodes={node_count}): {cost:.4f}")
    print("Raw incremental costs per edge:", ["{:.3f}".format(c) for c in raw_costs])
    # Verify that the weights sum to 1
    T = len(raw_costs)
    w = caputo_weights(alpha_val, T)
    print("Caputo weights sum check:", w.sum())
    # Ensure no exception on empty tree
    empty_tree = FractionalMorphTree()
    print("Cost of empty tree (should be 0.0):", empty_tree.fractional_cost(alpha=0.9))