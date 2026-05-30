# DARWIN HAMMER — match 3876, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_fold_c_m580_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_omni_chaotic__m1545_s2.py (gen4)
# born: 2026-05-29T23:52:12Z

"""
Hybrid Voronoi‑Sheaf / LTC‑Chaotic Engine

Parent A:  hybrid_hybrid_voronoi_parti_hybrid_hybrid_fold_c_m580_s0.py  
Parent B:  hybrid_hybrid_hybrid_hybrid_hybrid_omni_chaotic__m1545_s2.py  

Mathematical bridge
-------------------
Both parents operate on a set of *points* that are interpreted as elements of a
metric space.  In the Voronoi‑sheaf side the points are seeds that induce a
partition; restriction maps between neighbouring cells are linear operators.
In the LTC‑chaotic side the same points are used as *states* whose temporal
evolution is modulated by an effective time‑constant τ(t) and by a chaotic
weight w(z).  The fusion consists of attaching a scalar τ·w to every sheaf node
and letting the restriction maps propagate this scalar multiplicatively.  The
resulting system is a sheaf whose sections are one‑dimensional “activity”
values that evolve both spatially (via Voronoi adjacency) and temporally
(via the LTC‑chaotic dynamics).

The implementation below
* builds a Voronoi‑sheaf from a random seed set,
* defines τ(t) (LTC effective time constant) and w(z) (logistic‑map chaos),
* updates a global policy by multiplying τ·w with a fold‑change signal and
  propagates the result through the sheaf’s restriction maps.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Basic geometric utilities (from Parent A)
# ----------------------------------------------------------------------
class Point(tuple):
    """Immutable 2‑D point used as a tuple (x, y)."""
    pass

def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    """Return index of the seed closest to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

# ----------------------------------------------------------------------
# Sheaf structure (from Parent A)
# ----------------------------------------------------------------------
class Sheaf:
    """A very small sheaf: nodes have a vector dimension, edges carry
    a pair of linear restriction maps (src→shared, dst→shared)."""
    def __init__(self, node_dims: dict[int, int], edges: list[tuple[int, int]]):
        self.node_dims = dict(node_dims)          # node → dimension
        self.edges = list(edges)                  # list of (u, v)
        self._restrictions: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._sections: dict[int, np.ndarray] = {}   # node → current section (vector)

    def set_restriction(self, edge: tuple[int, int],
                        src_map: np.ndarray,
                        dst_map: np.ndarray) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("restriction maps must share the same row dimension")
        self._restrictions[edge] = (src_map, dst_map)

    def set_section(self, node: int, vec: np.ndarray) -> None:
        """Assign a section (vector) to a node; dimension is checked."""
        if vec.shape[0] != self.node_dims[node]:
            raise ValueError("section dimension mismatch")
        self._sections[node] = vec.copy()

    def get_section(self, node: int) -> np.ndarray:
        """Return the current section of *node* (zero vector if undefined)."""
        if node not in self._sections:
            self._sections[node] = np.zeros(self.node_dims[node])
        return self._sections[node]

    def propagate(self) -> None:
        """One synchronous pass: for every edge (u,v) average the
        transformed sections and write back to both endpoints."""
        new_sections: dict[int, np.ndarray] = {}
        for (u, v), (Su, Sv) in self._restrictions.items():
            sec_u = self.get_section(u)
            sec_v = self.get_section(v)
            # map to the shared space
            shared_u = Su @ sec_u
            shared_v = Sv @ sec_v
            # simple averaging in the shared space
            shared = 0.5 * (shared_u + shared_v)
            # pull back to each node (pseudo‑inverse via transpose)
            # (We assume Su, Sv are square and invertible for simplicity)
            new_u = np.linalg.pinv(Su) @ shared
            new_v = np.linalg.pinv(Sv) @ shared
            new_sections[u] = new_u if u not in new_sections else new_sections[u] + new_u
            new_sections[v] = new_v if v not in new_sections else new_sections[v] + new_v

        # replace sections (averaging over contributions if a node appears in several edges)
        for node, vec in new_sections.items():
            count = sum(1 for (a, b) in self.edges if a == node or b == node)
            self._sections[node] = vec / max(count, 1)

# ----------------------------------------------------------------------
# Global policy (from Parent A)
# ----------------------------------------------------------------------
_POLICY: dict[int, float] = {}

def reset_policy() -> None:
    """Clear the policy dictionary."""
    _POLICY.clear()

# ----------------------------------------------------------------------
# LTC effective time constant (from Parent B)
# ----------------------------------------------------------------------
def effective_tau(day_of_week: int, amplitude: float = 0.3) -> float:
    """
    Compute the Liquid‑Time‑Constant τ(t) for a given day.
    day_of_week ∈ {0,…,6} (Monday=0).  The base constant is 1,
    modulated by a sinusoid of period 7.
    """
    theta = 2 * math.pi * (day_of_week % 7) / 7.0
    return 1.0 + amplitude * math.sin(theta)

# ----------------------------------------------------------------------
# Chaotic weighting – logistic map (from Parent B)
# ----------------------------------------------------------------------
def chaotic_weight(z: float, r: float = 3.7) -> float:
    """
    One iteration of the logistic map, 0 < z < 1.
    The parameter r is chosen in the chaotic regime.
    """
    if not 0.0 < z < 1.0:
        raise ValueError("z must lie in (0,1)")
    return r * z * (1.0 - z)

# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def generate_seeds(n: int, scale: float = 10.0) -> list[Point]:
    """Uniformly sample *n* 2‑D seed points inside a square of side *scale*."""
    return [Point(random.uniform(0, scale), random.uniform(0, scale)) for _ in range(n)]

def build_sheaf(seeds: list[Point]) -> Sheaf:
    """
    Construct a Sheaf whose nodes correspond to Voronoi cells of *seeds*.
    Each node is given dimension 2.  Edges connect each seed to its immediate
    neighbour in the sorted order (a simple ring topology) – sufficient for
    demonstrating restriction‑map propagation.
    """
    n = len(seeds)
    node_dims = {i: 2 for i in range(n)}
    edges = [(i, (i + 1) % n) for i in range(n)]  # ring adjacency

    sheaf = Sheaf(node_dims, edges)

    # Random linear restrictions that are square and invertible.
    rng = np.random.default_rng()
    for (u, v) in edges:
        A = rng.normal(size=(2, 2))
        while np.linalg.cond(A) > 1e3:          # avoid near‑singular matrices
            A = rng.normal(size=(2, 2))
        B = rng.normal(size=(2, 2))
        while np.linalg.cond(B) > 1e3:
            B = rng.normal(size=(2, 2))
        sheaf.set_restriction((u, v), A, B)

    # Initialise each node with its seed coordinates as a column vector.
    for i, pt in enumerate(seeds):
        sheaf.set_section(i, np.array([pt[0], pt[1]]))
    return sheaf

def hybrid_step(sheaf: Sheaf,
                point: Point,
                day_of_week: int,
                z: float,
                fold_change: float = 1.0) -> None:
    """
    Perform a single hybrid update:
      1. Locate the Voronoi cell (node) that contains *point*.
      2. Compute the temporal factor τ = effective_tau(day_of_week).
      3. Compute the chaotic factor w = chaotic_weight(z).
      4. Update the global policy for the node with τ·w·fold_change.
      5. Inject the same scalar into the node’s sheaf section (as a uniform
         scaling of its current vector) and run a sheaf propagation pass.
    """
    node = nearest(point, list(sheaf.node_dims.keys()))  # seeds are indexed 0..n‑1
    tau = effective_tau(day_of_week)
    w = chaotic_weight(z)
    contribution = tau * w * fold_change

    # ----- policy update -------------------------------------------------
    _POLICY[node] = _POLICY.get(node, 0.0) + contribution

    # ----- sheaf section update -----------------------------------------
    current = sheaf.get_section(node)
    # Scale the vector uniformly by (1 + contribution) – keeps dimensionality.
    new_section = current * (1.0 + contribution)
    sheaf.set_section(node, new_section)

    # Propagate the influence through the adjacency restrictions.
    sheaf.propagate()

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(0)

    # 1. Build the spatial component.
    seeds = generate_seeds(6, scale=20.0)
    sheaf = build_sheaf(seeds)

    # 2. Run a few hybrid steps with synthetic data.
    reset_policy()
    for step in range(5):
        # Random observation point inside the same domain.
        pt = Point(random.uniform(0, 20), random.uniform(0, 20))
        day = step % 7
        z = random.random()  # chaotic seed in (0,1)
        hybrid_step(sheaf, pt, day, z, fold_change=0.5)

    # 3. Print a concise summary to verify that everything executed.
    print("Policy values per node:")
    for node, val in sorted(_POLICY.items()):
        print(f"  node {node}: {val:.4f}")

    print("\nFinal sheaf sections (first two nodes shown):")
    for i in range(min(2, len(seeds))):
        vec = sheaf.get_section(i)
        print(f"  node {i}: [{vec[0]:.3f}, {vec[1]:.3f}]")