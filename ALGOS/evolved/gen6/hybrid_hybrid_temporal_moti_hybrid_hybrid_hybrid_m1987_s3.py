# DARWIN HAMMER — match 1987, survivor 3
# gen: 6
# parent_a: hybrid_temporal_motifs_hybrid_hybrid_hybrid_m867_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m538_s1.py (gen5)
# born: 2026-05-29T23:40:18Z

"""Hybrid Temporal‑Motif Sheaf‑RBF Optimizer
================================================

This module fuses the two parent algorithms:

* **Parent A** – ``temporal_motifs.py`` and a sheaf‑based dense associative memory.
* **Parent B** – a decision‑hygiene bandit coupled with an RBF surrogate optimizer.

**Mathematical bridge**

A sheaf’s *section* at each node is interpreted as the **resource vector**
``e_i = [d_i, p_i, s_i]`` used in Parent B.  
The section is linearly transformed by the weight matrix **W** (Parent B) to
produce ``x_i = W @ e_i``.  The collection of motif patterns extracted by
Parent A is hashed into numeric vectors and stored as the **centres** of the
RBF surrogate (Parent B).  The surrogate predicts a reward ``r̂_i`` for each
node; this reward drives:

1. the VRAM store ODE (global scalar ``z``) and the learning‑rate modulation,
2. the update of ``W`` (gradient‑like rule from Parent B), and
3. the update of the sheaf’s sections (closing the sheaf‑motif loop).

Thus the temporal‑motif mining supplies the surrogate’s geometry while the
sheaf supplies the resource vectors, yielding a fully coupled hybrid system.

The implementation below provides three demonstration functions and a smoke
test that runs without error.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BurstSignal:
    key: str
    count: int
    z_score: float

@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int

# ----------------------------------------------------------------------
# Sheaf class (Parent A)
# ----------------------------------------------------------------------
class Sheaf:
    """A simple cellular sheaf with linear restriction maps and sections."""
    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]):
        self.node_dims = dict(node_dims)          # node → dimension
        self.edges = list(edges)                  # list of (src, dst)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[Any, Any],
                        src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float),
                                      np.asarray(dst_map, dtype=float))

    def set_section(self, node: Any, value: np.ndarray) -> None:
        if value.shape[0] != self.node_dims[node]:
            raise ValueError("Section dimension must match node dimension")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: Any) -> np.ndarray:
        return self._sections.get(node, np.zeros(self.node_dims[node]))

    def all_sections_matrix(self) -> np.ndarray:
        """Return a matrix (n_nodes × max_dim) with padded zeros."""
        max_dim = max(self.node_dims.values())
        mat = np.zeros((len(self.node_dims), max_dim))
        for idx, (node, dim) in enumerate(self.node_dims.items()):
            sec = self.get_section(node)
            mat[idx, :dim] = sec
        return mat

# ----------------------------------------------------------------------
# RBF surrogate (Parent B)
# ----------------------------------------------------------------------
class RBFSurrogate:
    """RBF surrogate model with online centre/weight management."""
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = float(epsilon)
        self.centers: List[np.ndarray] = []   # list of x_k vectors
        self.weights: np.ndarray = np.array([])   # w_k coefficients

    def _kernel(self, x: np.ndarray, c: np.ndarray) -> float:
        diff = x - c
        return math.exp(-self.epsilon ** 2 * np.dot(diff, diff))

    def predict(self, x: np.ndarray) -> float:
        if not self.centers:
            return 0.0
        kernels = np.array([self._kernel(x, c) for c in self.centers])
        return float(np.dot(self.weights, kernels))

    def add_observation(self, x: np.ndarray, y: float) -> None:
        """Add a new centre x with target y and recompute all weights."""
        self.centers.append(np.asarray(x, dtype=float))
        n = len(self.centers)
        K = np.empty((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                K[i, j] = self._kernel(self.centers[i], self.centers[j])
        # Solve K w = y_vec where y_vec = [y_1,...,y_n] (here we use the
        # latest observation for all entries – a simple online rule).
        y_vec = np.full(n, y, dtype=float)
        try:
            self.weights = np.linalg.solve(K, y_vec)
        except np.linalg.LinAlgError:
            # In case K is singular, fall back to least‑squares.
            self.weights, *_ = np.linalg.lstsq(K, y_vec, rcond=None)

# ----------------------------------------------------------------------
# Temporal‑motif utilities (simplified version of Parent A)
# ----------------------------------------------------------------------
def sessionize_events(events: List[Dict[str, Any]],
                     gap_seconds: float = 1800.0) -> List[List[Dict[str, Any]]]:
    """Group events into sessions separated by a gap larger than *gap_seconds*."""
    if not events:
        return []
    # Assume events are sorted by timestamp (key 'ts' in seconds).
    sessions: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [events[0]]
    for ev in events[1:]:
        if ev['ts'] - current[-1]['ts'] > gap_seconds:
            sessions.append(current)
            current = [ev]
        else:
            current.append(ev)
    sessions.append(current)
    return sessions

def extract_motifs_from_sessions(sessions: List[List[Dict[str, Any]]],
                                 min_support: int = 2) -> List[TemporalMotif]:
    """Very coarse motif extractor: treat ordered event type strings as patterns."""
    pattern_counter: Counter = Counter()
    for sess in sessions:
        pattern = tuple(ev['type'] for ev in sess)
        pattern_counter[pattern] += 1
    motifs = [TemporalMotif(pattern=p, support=c)
              for p, c in pattern_counter.items() if c >= min_support]
    return motifs

def motif_to_vector(motif: TemporalMotif, dim: int) -> np.ndarray:
    """Hash each token of the motif into a numeric vector of length *dim*."""
    vec = np.zeros(dim, dtype=float)
    for idx, token in enumerate(motif.pattern):
        h = hash(token) % dim
        vec[h] += 1.0
    # Normalise by support to keep magnitude comparable.
    if motif.support > 0:
        vec /= motif.support
    return vec

# ----------------------------------------------------------------------
# Hybrid dynamics (bridge between the two parents)
# ----------------------------------------------------------------------
class HybridSystem:
    """Encapsulates the coupled sheaf‑RBF‑bandit dynamics."""
    def __init__(self,
                 sheaf: Sheaf,
                 motifs: List[TemporalMotif],
                 dim_resource: int = 3,
                 epsilon: float = 1.0,
                 base_eta: float = 0.01,
                 alpha: float = 0.1,
                 beta: float = 0.01):
        self.sheaf = sheaf
        self.motifs = motifs
        self.dim_resource = dim_resource               # d_in
        self.base_eta = float(base_eta)
        self.alpha = float(alpha)                       # ODE drive term
        self.beta = float(beta)                         # ODE decay term
        self.z = 0.0                                     # VRAM store
        # Initialise weight matrix W (d_out × d_in).  Choose d_out = dim_resource.
        self.W = np.eye(dim_resource, dim_resource, dtype=float)
        # Initialise RBF surrogate.
        self.rbf = RBFSurrogate(epsilon=epsilon)

        # Pre‑populate surrogate centres with motif vectors.
        for motif in motifs:
            vec = motif_to_vector(motif, dim_resource)
            # Use the motif support as a proxy reward for initialisation.
            self.rbf.add_observation(vec, float(motif.support))

    def _resource_from_section(self, sec: np.ndarray) -> np.ndarray:
        """Map a sheaf section to a 3‑dimensional resource vector."""
        # Simple projection: take first three components (pad with zeros).
        out = np.zeros(self.dim_resource, dtype=float)
        length = min(self.dim_resource, sec.shape[0])
        out[:length] = sec[:length]
        return out

    def step(self, dt: float = 1.0) -> None:
        """Perform one hybrid update across all nodes."""
        # 1. Gather sections → resource vectors.
        resources: List[np.ndarray] = []
        nodes = list(self.sheaf.node_dims.keys())
        for node in nodes:
            sec = self.sheaf.get_section(node)
            resources.append(self._resource_from_section(sec))

        # 2. Linear transformation x = W @ e for each node.
        xs = [self.W @ e for e in resources]

        # 3. RBF surrogate prediction r̂ for each transformed vector.
        preds = [self.rbf.predict(x) for x in xs]

        # 4. Aggregate predictions (mean) to drive the global store.
        r_hat_mean = float(np.mean(preds)) if preds else 0.0

        # 5. Store dynamics (Euler integration).
        self.z += dt * (self.alpha * (r_hat_mean - self.z) - self.beta * self.z)

        # 6. Learning‑rate modulation.
        eta = self.base_eta * (1.0 + self.z)

        # 7. Update weight matrix W (gradient‑like rule).
        #   W ← W + η·r̂·(1ᵀ ⊗ e)  → outer product of ones(d_out) and e.
        for e, r in zip(resources, preds):
            outer = np.outer(np.ones(self.W.shape[0]), e)   # d_out × d_in
            self.W += eta * r * outer

        # 8. Update sheaf sections using the same reward signal.
        for node, e, r in zip(nodes, resources, preds):
            sec = self.sheaf.get_section(node)
            # Simple additive update proportional to reward.
            update = eta * r * np.ones_like(sec)
            self.sheaf.set_section(node, sec + update)

        # 9. Add new observations to the surrogate (online learning).
        for x, r in zip(xs, preds):
            self.rbf.add_observation(x, r)

# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def demo_build_sheaf(num_nodes: int = 5,
                     max_dim: int = 4) -> Sheaf:
    """Create a random sheaf with *num_nodes* nodes."""
    node_dims = {f'v{i}': random.randint(2, max_dim) for i in range(num_nodes)}
    edges = [(f'v{i}', f'v{i+1}') for i in range(num_nodes - 1)]
    sheaf = Sheaf(node_dims, edges)

    # Initialise random sections.
    for node, dim in node_dims.items():
        vec = np.random.randn(dim)
        sheaf.set_section(node, vec)
    return sheaf

def demo_generate_events(num_events: int = 30) -> List[Dict[str, Any]]:
    """Generate synthetic timestamped events with a few types."""
    types = ['A', 'B', 'C', 'D']
    ts = 0.0
    events = []
    for _ in range(num_events):
        ts += random.expovariate(1/300.0)   # average gap 300 s
        events.append({'ts': ts, 'type': random.choice(types)})
    return events

def demo_hybrid_run(steps: int = 5) -> None:
    """Run the hybrid system for a few steps and print diagnostics."""
    # 1. Build sheaf.
    sheaf = demo_build_sheaf()

    # 2. Generate events and extract motifs.
    events = demo_generate_events()
    sessions = sessionize_events(events)
    motifs = extract_motifs_from_sessions(sessions)

    # 3. Initialise hybrid system.
    hybrid = HybridSystem(sheaf, motifs)

    print("Initial weight matrix W:\n", hybrid.W)
    print("Initial VRAM store z =", hybrid.z)

    for t in range(steps):
        hybrid.step(dt=1.0)
        print(f"\n--- Step {t+1} ---")
        print("z (VRAM store) =", hybrid.z)
        print("Learning rate η =", hybrid.base_eta * (1.0 + hybrid.z))
        print("Weight matrix W (norm) =", np.linalg.norm(hybrid.W))
        # Show a sample section.
        sample_node = list(sheaf.node_dims.keys())[0]
        print(f"Section at node {sample_node} (first 3 entries):",
              hybrid.sheaf.get_section(sample_node)[:3])

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_hybrid_run(steps=3)