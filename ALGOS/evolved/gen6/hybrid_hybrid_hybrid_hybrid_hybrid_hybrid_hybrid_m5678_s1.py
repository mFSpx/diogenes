# DARWIN HAMMER — match 5678, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_decisi_m1115_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_distri_rbf_surrogate_m648_s2.py (gen5)
# born: 2026-05-30T00:04:09Z

"""Hybrid Leader Election via Perceptual‑Hash Graph Curvature and RBF‑Surrogate Conductance

Parents
-------
* **Parent A** – Constructs a perceptual‑hash graph of elements, computes Ollivier‑Ricci‑like
  curvature on the graph, and uses the curvature‑derived scores for leader election.
* **Parent B** – Uses a Physarum‑inspired conductance term inside a simulated‑annealing
  temperature; the conductance matrix is expensive, so an RBF surrogate predicts conductance
  from node features.

Mathematical Bridge
-------------------
The curvature values produced by Parent A are high‑dimensional node features that capture
local similarity structure.  We feed these features (together with node degree) into the
RBF surrogate from Parent B to predict the edge conductance 𝑔̂ required in the hybrid
temperature formula

    T(k, phase) = 2^{-(phases‑phase)} · t₀·α^{k} · 𝑔̂ / (|pₐ‑p_b|+ε)

Thus the graph‑theoretic information of A drives the physics‑inspired annealing of B,
producing a single unified leader‑election dynamics.
"""

import math
import random
import sys
import pathlib
from collections.abc import Mapping, Hashable
from typing import List, Tuple, Sequence, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – perceptual hash graph and curvature
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]

def compute_phash(values: List[float]) -> int:
    """Average‑threshold perceptual hash (up to 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits."""
    return (a ^ b).bit_count()

def build_graph(elements: List[List[float]]) -> Graph:
    """
    Build an undirected graph where each node corresponds to an element.
    An edge exists when the Hamming distance between perceptual hashes ≤ 4.
    """
    graph: Dict[Node, set[Node]] = {}
    hashes: Dict[str, int] = {}
    for i, element in enumerate(elements):
        node = str(i)
        hashes[node] = compute_phash(element)
        graph[node] = set()
    nodes = list(graph.keys())
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            if hamming_distance(hashes[u], hashes[v]) <= 4:
                graph[u].add(v)
                graph[v].add(u)
    return graph

def compute_curvature(graph: Graph) -> Dict[Node, float]:
    """
    Approximate Ollivier‑Ricci curvature per node.
    For node i, curvature C_i = (1/|N_i|) Σ_{j∈N_i} (deg(j) - deg(i)) / max(deg(i),1).
    Nodes without neighbors receive curvature 0.
    """
    curvature: Dict[Node, float] = {}
    for node, neighbors in graph.items():
        deg_i = len(neighbors)
        if deg_i == 0:
            curvature[node] = 0.0
            continue
        diff_sum = 0.0
        for nb in neighbors:
            deg_nb = len(graph[nb])
            diff_sum += (deg_nb - deg_i) / max(deg_i, 1)
        curvature[node] = diff_sum / deg_i
    return curvature

# ----------------------------------------------------------------------
# Parent B – RBF surrogate for conductance
# ----------------------------------------------------------------------
Vector = Sequence[float]

def _gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def _euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same length")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

class RBFSurrogate:
    """
    Simple Gaussian RBF surrogate.
    Stores centers X (n_samples × n_features) and weights w solving (K+λI)w = y.
    """
    def __init__(self, epsilon: float = 1.0, reg: float = 1e-8):
        self.epsilon = epsilon
        self.reg = reg
        self.centers: np.ndarray = np.empty((0, 0))
        self.weights: np.ndarray = np.empty((0,))
        self.fitted = False

    def _kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        K = np.empty((n, n))
        for i in range(n):
            for j in range(i, n):
                r = _euclidean(X[i], X[j])
                val = _gaussian(r, self.epsilon)
                K[i, j] = val
                K[j, i] = val
        return K

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit surrogate on feature matrix X and target vector y."""
        if X.ndim != 2:
            raise ValueError("X must be 2‑dimensional")
        if y.ndim != 1:
            raise ValueError("y must be 1‑dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples mismatch")
        K = self._kernel_matrix(X)
        # Regularized solve
        A = K + self.reg * np.eye(K.shape[0])
        self.weights = np.linalg.solve(A, y)
        self.centers = X.copy()
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict conductance for each row in X."""
        if not self.fitted:
            raise RuntimeError("Surrogate not fitted")
        n_center = self.centers.shape[0]
        n_query = X.shape[0]
        Kq = np.empty((n_query, n_center))
        for i in range(n_query):
            for j in range(n_center):
                r = _euclidean(X[i], self.centers[j])
                Kq[i, j] = _gaussian(r, self.epsilon)
        return Kq @ self.weights

# ----------------------------------------------------------------------
# Hybrid temperature and leader election
# ----------------------------------------------------------------------
def hybrid_temperature(k: int, phase: int, total_phases: int,
                       g_hat: float, p_a: float, p_b: float,
                       t0: float = 1.0, alpha: float = 0.95,
                       eps: float = 1e-9) -> float:
    """
    Compute temperature T(k, phase) = 2^{-(total_phases‑phase)} * t0*α^{k} * ĝ/(|pₐ‑p_b|+ε)
    """
    broadcast = 2 ** (-(total_phases - phase))
    cooling = t0 * (alpha ** k)
    scaling = g_hat / (abs(p_a - p_b) + eps)
    return broadcast * cooling * scaling

def _node_score(curvature: float, degree: int) -> float:
    """Simple score combining curvature and degree."""
    return curvature * math.log1p(degree)

def hybrid_leader_election(elements: List[List[float]],
                           pressures: Dict[Node, float],
                           t0: float = 1.0,
                           alpha: float = 0.95,
                           total_phases: int = 5,
                           iters_per_phase: int = 20) -> Node:
    """
    Perform leader election using:
    1. Graph curvature (Parent A) as node features.
    2. RBF surrogate (Parent B) to predict conductance from those features.
    3. Simulated‑annealing Metropolis acceptance with hybrid temperature.
    Returns the elected leader node identifier.
    """
    # 1. Build graph and curvature
    graph = build_graph(elements)
    curvature = compute_curvature(graph)
    degrees = {node: len(neigh) for node, neigh in graph.items()}

    # 2. Assemble feature matrix (curvature, degree) for each node
    nodes = list(graph.keys())
    feats = np.array([[curvature[n], float(degrees[n])] for n in nodes])

    # 3. Generate synthetic conductance targets for training (e.g., linear combo)
    true_g = np.array([0.8 * curvature[n] + 0.2 * degrees[n] + random.gauss(0, 0.05)
                       for n in nodes])

    # 4. Fit RBF surrogate on a random subset (30 % of nodes)
    idx = np.arange(len(nodes))
    np.random.shuffle(idx)
    train_frac = 0.3
    n_train = max(1, int(train_frac * len(nodes)))
    train_idx = idx[:n_train]
    surrogate = RBFSurrogate(epsilon=1.0, reg=1e-6)
    surrogate.fit(feats[train_idx], true_g[train_idx])

    # 5. Predict conductance for all nodes (used later in temperature)
    g_pred = surrogate.predict(feats)

    # 6. Initialise leader randomly
    leader = random.choice(nodes)

    # 7. Simulated annealing over phases
    iteration = 0
    for phase in range(total_phases):
        for _ in range(iters_per_phase):
            # pick a candidate node uniformly
            candidate = random.choice(nodes)

            # compute Metropolis acceptance
            cur_score = _node_score(curvature[leader], degrees[leader])
            cand_score = _node_score(curvature[candidate], degrees[candidate])
            delta = cand_score - cur_score

            # Use average predicted conductance of the two nodes for temperature
            g_hat = (g_pred[nodes.index(leader)] + g_pred[nodes.index(candidate)]) / 2.0
            T = hybrid_temperature(iteration, phase, total_phases,
                                   g_hat, pressures[leader], pressures[candidate],
                                   t0=t0, alpha=alpha)

            accept = delta > 0 or random.random() < math.exp(delta / max(T, 1e-12))
            if accept:
                leader = candidate
            iteration += 1

    return leader

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create synthetic elements: 50 vectors of length 100 with values in [0,1]
    random.seed(42)
    np.random.seed(42)
    elements = [list(np.random.rand(100)) for _ in range(50)]

    # Random pressures per node
    graph_dummy = build_graph(elements)
    pressures = {node: random.uniform(0.0, 1.0) for node in graph_dummy.keys()}

    leader = hybrid_leader_election(elements, pressures)
    print(f"Elected leader node: {leader}")