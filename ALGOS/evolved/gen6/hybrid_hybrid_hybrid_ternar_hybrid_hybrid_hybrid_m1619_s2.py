# DARWIN HAMMER — match 1619, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_ternary_lens__hybrid_hybrid_sheaf__m72_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1016_s0.py (gen5)
# born: 2026-05-29T23:37:56Z

"""Hybrid Sheaf‑Clifford Bandit Algorithm
=====================================

Parents
-------
* **Parent A** – *hybrid_hybrid_ternary_lens__hybrid_hybrid_sheaf__m72_s1.py*  
  Provides a sheaf data structure over a graph with restriction maps and
  section vectors.  Pruning is applied to sections via a probability function.

* **Parent B** – *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1016_s0.py*  
  Introduces a Clifford‑geometric product (implemented here as a simple
  multivector product), an epsilon‑greedy bandit router driven by the day‑of‑week,
  and a count‑min sketch for frequency estimation.

Mathematical Bridge
-------------------
The bridge consists of treating every sheaf section as a **multivector**.
The geometric product from Parent B is used to transform sections, while the
pruning probability from Parent A decides which transformed sections survive.
The surviving sections influence the bandit’s estimated rewards through the
count‑min sketch, closing the loop between sheaf cohomology consistency and
bandit‑driven resource allocation.

The core hybrid operations are:
1. **Geometric transformation** – `section ← G ⋅ section` where `G` is a
   multivector (geometric product).
2. **Probability‑driven pruning** – each transformed section is kept with
   probability `p` (the pruning probability).
3. **Bandit feedback** – the surviving sections update a count‑min sketch that
   estimates the frequency of the bandit’s actions; this frequency modulates
   a VRAM‑allocation rule that also depends on a sheaf consistency score.

The following module implements this fused system.
"""

import math
import random
import sys
from datetime import datetime, timezone, date
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

class Sheaf:
    """Simple sheaf over a directed graph.

    Nodes store section vectors (numpy arrays).  Edges store linear restriction
    maps represented as two matrices: source‑to‑edge and edge‑to‑target.
    """

    def __init__(self, node_dims, edge_list):
        """
        Parameters
        ----------
        node_dims : dict[int, int]
            Mapping from node identifier to dimension of its section vector.
        edge_list : list[tuple[int, int]]
            Directed edges (u, v) of the graph.
        """
        self.node_dims = dict(node_dims)
        self.edges = list(edge_list)
        self._restrictions = {}   # (u,v) -> (src_map, dst_map)
        self._sections = {}       # node -> np.ndarray

    def set_restriction(self, edge, src_map, dst_map):
        """Define restriction maps for an edge."""
        u, v = edge
        src_map = np.array(src_map, dtype=float)
        dst_map = np.array(dst_map, dtype=float)
        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node, value):
        """Assign a section vector to a node."""
        dim = self.node_dims.get(node)
        if dim is None:
            raise KeyError(f"Node {node} not declared in node_dims")
        arr = np.array(value, dtype=float)
        if arr.shape != (dim,):
            raise ValueError(f"Section for node {node} must have shape ({dim},)")
        self._sections[node] = arr

    def get_section(self, node):
        """Retrieve the section vector of a node (zero vector if undefined)."""
        dim = self.node_dims[node]
        return self._sections.get(node, np.zeros(dim, dtype=float))

    def consistency_score(self):
        """Return a scalar in [0,1] measuring how many restrictions are satisfied.

        For each edge (u,v) we check whether dst_map @ (src_map @ s_u) ≈ s_v.
        The score is the fraction of edges that satisfy the equality within a
        tolerance.
        """
        if not self.edges:
            return 1.0
        tol = 1e-6
        satisfied = 0
        for (u, v) in self.edges:
            if (u, v) not in self._restrictions:
                continue
            src_map, dst_map = self._restrictions[(u, v)]
            s_u = self.get_section(u)
            s_v = self.get_section(v)
            transformed = dst_map @ (src_map @ s_u)
            if np.allclose(transformed, s_v, atol=tol):
                satisfied += 1
        return satisfied / len(self.edges)


class Multivector:
    """Very lightweight Clifford‑algebra element.

    For demonstration we restrict to grade‑1 vectors stored as a NumPy array.
    The geometric product is approximated by element‑wise multiplication,
    which preserves the spirit of a bilinear map while staying cheap.
    """

    def __init__(self, components):
        self.vec = np.array(components, dtype=float)

    def geometric_product(self, other):
        """Return the geometric product of two multivectors."""
        if self.vec.shape != other.vec.shape:
            raise ValueError("Multivectors must have the same dimension")
        # Element‑wise product as a placeholder for the full GA product
        return Multivector(self.vec * other.vec)

    def scale(self, scalar):
        """Scalar multiplication."""
        return Multivector(self.vec * scalar)

    def as_array(self):
        return self.vec.copy()


class CountMinSketch:
    """Count‑Min Sketch with simple 2‑hash scheme."""

    def __init__(self, width=1000, depth=5, seed=0):
        self.width = width
        self.depth = depth
        self.tables = np.zeros((depth, width), dtype=int)
        random.seed(seed)
        # Generate pairwise‑independent hash coefficients
        self._hash_params = [(random.randint(1, 2**31 - 1),
                              random.randint(0, 2**31 - 1))
                             for _ in range(depth)]

    def _hash(self, x, i):
        a, b = self._hash_params[i]
        return (a * hash(x) + b) % self.width

    def update(self, item, increment=1):
        for i in range(self.depth):
            idx = self._hash(item, i)
            self.tables[i, idx] += increment

    def estimate(self, item):
        """Return the minimum count across hash rows."""
        return min(self.tables[i, self._hash(item, i)] for i in range(self.depth))


class BanditRouter:
    """Epsilon‑greedy contextual bandit using the day‑of‑week as context."""

    def __init__(self, actions, epsilon=0.1, seed=0):
        self.actions = list(actions)
        self.epsilon = epsilon
        self.random = random.Random(seed)
        # Value estimates are stored in a Count‑Min Sketch
        self.sketch = CountMinSketch(width=2000, depth=4, seed=seed)

    def _context_feature(self, day):
        """Scale day‑of‑week (Monday=0 … Sunday=6) to [0,1]."""
        return day.weekday() / 6.0

    def select_action(self, day):
        """Return an action identifier."""
        if self.random.random() < self.epsilon:
            return self.random.choice(self.actions)
        # Greedy: pick action with highest estimated count (frequency)
        estimates = [(a, self.sketch.estimate(a)) for a in self.actions]
        best_action = max(estimates, key=lambda tup: tup[1])[0]
        return best_action

    def update(self, action, reward=1):
        """Update the sketch with observed reward (default reward=1)."""
        self.sketch.update(action, increment=reward)


# ---------------------------------------------------------------------------
# Hybrid Operations
# ---------------------------------------------------------------------------

def apply_geometric_transform(sheaf, G):
    """Apply a geometric product with multivector G to every sheaf section.

    The transformed section replaces the original one.
    """
    for node in sheaf.node_dims:
        sec = sheaf.get_section(node)
        mv_sec = Multivector(sec)
        transformed = G.geometric_product(mv_sec)
        sheaf.set_section(node, transformed.as_array())


def prune_sections(sheaf, prob):
    """Probabilistically zero out each section with probability (1‑prob)."""
    for node in sheaf.node_dims:
        if random.random() > prob:
            dim = sheaf.node_dims[node]
            sheaf.set_section(node, np.zeros(dim, dtype=float))


def hybrid_step(sheaf, router, day, G, prune_prob):
    """Perform one hybrid iteration for a calendar day.

    1. Transform sheaf sections with the geometric product `G`.
    2. Prune sections according to `prune_prob`.
    3. Use the bandit router (context = day) to select an action.
    4. Update the router’s sketch with the selected action.
    5. Return a dictionary with diagnostics.
    """
    # 1‑2: sheaf update
    apply_geometric_transform(sheaf, G)
    prune_sections(sheaf, prune_prob)

    # 3‑4: bandit routing
    action = router.select_action(day)
    router.update(action, reward=1)

    # 5: diagnostics
    consistency = sheaf.consistency_score()
    freq_est = router.sketch.estimate(action)
    vram_alloc = compute_vram_allocation(consistency, freq_est, prune_prob)

    return {
        "day": day.isoformat(),
        "action": action,
        "consistency": consistency,
        "freq_estimate": freq_est,
        "vram_allocation": vram_alloc,
    }


def compute_vram_allocation(consistency, freq_estimate, prune_prob,
                            base_vram=1024.0):
    """Map hybrid signals to a VRAM budget.

    - `consistency` ∈ [0,1] scales the allocation linearly.
    - Higher `freq_estimate` (more popular actions) increase allocation.
    - Larger `prune_prob` (more sections kept) also increase allocation.

    The formula is purely illustrative.
    """
    freq_factor = 1.0 + (freq_estimate / 100.0)   # damped growth
    prune_factor = 0.5 + 0.5 * prune_prob        # between 0.5 and 1.0
    return base_vram * consistency * freq_factor * prune_factor


# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Build a tiny graph sheaf (3 nodes, 2 edges)
    node_dims = {0: 3, 1: 3, 2: 3}
    edges = [(0, 1), (1, 2)]
    sheaf = Sheaf(node_dims, edges)

    # Random initial sections
    for n in node_dims:
        sheaf.set_section(n, np.random.randn(node_dims[n]))

    # Simple identity restrictions (no actual transformation)
    I = np.eye(3)
    for e in edges:
        sheaf.set_restriction(e, I, I)

    # 2. Define a multivector G (acts like a scaling matrix)
    G = Multivector([1.2, 0.8, 1.0])

    # 3. Instantiate bandit router with 4 possible actions
    router = BanditRouter(actions=[0, 1, 2, 3], epsilon=0.2, seed=42)

    # 4. Run the hybrid for a week
    start = date.today()
    results = []
    for offset in range(7):
        day = start.fromordinal(start.toordinal() + offset)
        diag = hybrid_step(
            sheaf=sheaf,
            router=router,
            day=day,
            G=G,
            prune_prob=0.7  # keep 70 % of sections on average
        )
        results.append(diag)

    # 5. Print a concise summary
    for r in results:
        print(f"{r['day']}: action={r['action']}, "
              f"consistency={r['consistency']:.3f}, "
              f"freq_est={r['freq_estimate']}, "
              f"vram={r['vram_allocation']:.1f} MB")
    sys.exit(0)