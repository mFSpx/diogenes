# DARWIN HAMMER — match 877, survivor 0
# gen: 3
# parent_a: hybrid_hybrid_distributed_l_chelydrid_ambush_m42_s0.py (gen2)
# parent_b: hybrid_rbf_surrogate_tri_algo_conduit_m8_s1.py (gen1)
# born: 2026-05-29T23:31:31Z

"""Hybrid Algorithm: Distributed Leader Deduplication + RBF Surrogate Conduit

Parents:
- **hybrid_hybrid_distributed_l_chelydrid_ambush_m42_s0.py** – builds a perceptual‑hash graph,
  finds connected components and elects a leader (representative) per cluster.
- **hybrid_rbf_surrogate_tri_algo_conduit_m8_s1.py** – defines a radial‑basis‑function (RBF)
  surrogate model whose weights are obtained by solving a linear system; the surrogate
  consumes signal/noise scores and predicts a scalar output.

Mathematical Bridge:
The bridge is the *cluster feature vector* extracted from each connected component of
the hash‑graph.  The feature vector (size, mean, variance, average hash distance) is fed
as the input **x** to the RBF surrogate.  The surrogate’s prediction is interpreted as a
*burst‑propensity* which, together with simple work‑cost‑urgency physics, decides whether
the “ambush” action is taken for that cluster.

The resulting module therefore:
1. Constructs a perceptual‑hash graph from raw elements.
2. Finds connected components and elects a leader per component.
3. Derives a numeric feature vector per component.
4. Trains an RBF surrogate on synthetic (feature, target) pairs.
5. Predicts a burst propensity for each component.
6. Applies a physics‑inspired decision rule to emit the final ambush decision.
"""

from __future__ import annotations
import math
import random
import sys
import pathlib
from collections.abc import Mapping, Hashable, Sequence
from dataclasses import dataclass
from typing import List, Tuple, Set, Dict

import numpy as np

# ----------------------------------------------------------------------
# Helper functions from Parent A
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
Vector = Sequence[float]


def compute_phash(values: List[float]) -> int:
    """Perceptual hash: 1 bit per value (up to 64) indicating >= average."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two integers."""
    return (a ^ b).bit_count()


def build_graph(elements: List[List[float]]) -> Tuple[Graph, Dict[Node, int]]:
    """
    Build an undirected graph where each node corresponds to an element index.
    Two nodes are connected if the Hamming distance between their perceptual hashes
    is ≤ 4.
    Returns the adjacency mapping and a dict of node → hash (for later use).
    """
    hashes: Dict[Node, int] = {str(i): compute_phash(elem) for i, elem in enumerate(elements)}
    graph: Dict[Node, Set[Node]] = {node: set() for node in hashes}
    nodes = list(hashes.keys())
    for i, ni in enumerate(nodes):
        for nj in nodes[i + 1 :]:
            if hamming_distance(hashes[ni], hashes[nj]) <= 4:
                graph[ni].add(nj)
                graph[nj].add(ni)
    return graph, hashes


def _dfs(node: Node, graph: Graph, visited: Set[Node], component: Set[Node]) -> None:
    """Depth‑first search to collect a connected component."""
    visited.add(node)
    component.add(node)
    for neigh in graph[node]:
        if neigh not in visited:
            _dfs(neigh, graph, visited, component)


def connected_components(graph: Graph) -> List[Set[Node]]:
    """Return a list of connected components (each as a set of node identifiers)."""
    visited: Set[Node] = set()
    components: List[Set[Node]] = []
    for node in graph:
        if node not in visited:
            comp: Set[Node] = set()
            _dfs(node, graph, visited, comp)
            components.append(comp)
    return components


def elect_leaders(components: List[Set[Node]], hashes: Dict[Node, int]) -> Dict[Node, Node]:
    """
    For each component pick the node with the smallest perceptual hash as leader.
    Returns a mapping component‑representative node → leader node.
    """
    leader_map: Dict[Node, Node] = {}
    for comp in components:
        leader = min(comp, key=lambda n: hashes[n])
        # Use an arbitrary representative (first element) as key
        rep = next(iter(comp))
        leader_map[rep] = leader
    return leader_map


# ----------------------------------------------------------------------
# Helper functions from Parent B (RBF surrogate)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    """
    Solve Ax = b via Gauss‑Jordan elimination.
    Raises ValueError if the matrix is singular.
    """
    n = len(b)
    # Augment matrix
    M = [row[:] + [rhs] for row, rhs in zip(A, b)]

    for col in range(n):
        # Pivot selection
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        M[col], M[pivot] = M[pivot], M[col]

        # Normalize pivot row
        div = M[col][col]
        M[col] = [v / div for v in M[col]]

        # Eliminate other rows
        for row in range(n):
            if row == col:
                continue
            factor = M[row][col]
            M[row] = [v - factor * p for v, p in zip(M[row], M[col])]

    return [row[-1] for row in M]


@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Linear combination of Gaussian RBFs centered at `centers`."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


# ----------------------------------------------------------------------
# Hybrid core: feature extraction, surrogate training, ambush decision
# ----------------------------------------------------------------------
def cluster_features(
    elements: List[List[float]],
    component: Set[Node],
    hashes: Dict[Node, int],
) -> Tuple[float, float, float, float]:
    """
    Compute a 4‑dimensional feature vector for a component:
    1. cluster size (float)
    2. mean of the first numeric dimension across all members
    3. variance of that dimension
    4. average Hamming distance of members to the component leader
    """
    size = float(len(component))
    indices = [int(node) for node in component]
    first_dim = [elements[i][0] for i in indices if elements[i]]
    mean = float(np.mean(first_dim)) if first_dim else 0.0
    var = float(np.var(first_dim)) if first_dim else 0.0

    # leader hash (smallest)
    leader_hash = min(hashes[n] for n in component)
    avg_hamming = float(
        sum(hamming_distance(hashes[n], leader_hash) for n in component) / size
    )
    return (size, mean, var, avg_hamming)


def train_rbf_surrogate(
    feature_list: List[Tuple[float, ...]], epsilon: float = 1.0
) -> RBFSurrogate:
    """
    Train an RBF surrogate on synthetic targets.
    Targets are generated as a linear combination of features plus small noise,
    ensuring a solvable system.
    """
    # Synthetic target generation
    true_coefs = np.random.uniform(-1, 1, size=len(feature_list[0]))
    targets = [
        sum(c * f for c, f in zip(true_coefs, feat)) + random.gauss(0, 0.05)
        for feat in feature_list
    ]

    # Build the RBF kernel matrix K_ij = φ(||c_i - c_j||)
    n = len(feature_list)
    K = [[gaussian(euclidean(feature_list[i], feature_list[j]), epsilon) for j in range(n)] for i in range(n)]

    # Solve K * w = targets for weights w
    weights = solve_linear(K, targets)
    return RBFSurrogate(centers=feature_list, weights=weights, epsilon=epsilon)


def burst_decision(
    propensity: float,
    work_value: float,
    cost_drag: float,
    urgency_force: float,
) -> bool:
    """
    Physics‑inspired ambush rule.
    - `propensity` is the surrogate prediction (higher → more likely to act).
    - `work_value` is an external scalar (here we reuse propensity).
    - `cost_drag` penalises large clusters.
    - `urgency_force` (0‑1) boosts the decision when urgency is high.
    The decision is True if the adjusted work exceeds the drag.
    """
    adjusted_work = work_value * (1.0 + urgency_force)
    threshold = cost_drag * (1.0 - urgency_force)
    return adjusted_work > threshold and propensity > 0.0


def hybrid_process(elements: List[List[float]]) -> Dict[Node, bool]:
    """
    End‑to‑end hybrid pipeline.
    Returns a mapping from component representative node → ambush decision (bool).
    """
    # 1. Graph construction & leader election
    graph, hashes = build_graph(elements)
    comps = connected_components(graph)
    leaders = elect_leaders(comps, hashes)

    # 2. Feature extraction per component
    feats: List[Tuple[float, ...]] = []
    comp_order: List[Set[Node]] = []  # preserve order for later lookup
    for comp in comps:
        f = cluster_features(elements, comp, hashes)
        feats.append(f)
        comp_order.append(comp)

    # 3. Train RBF surrogate on the extracted features
    surrogate = train_rbf_surrogate(feats, epsilon=1.0)

    # 4. Predict propensity and apply ambush rule
    decisions: Dict[Node, bool] = {}
    for comp, feat in zip(comp_order, feats):
        rep = next(iter(comp))  # representative key for output dict
        propensity = surrogate.predict(feat)

        # Derive simple physics parameters from the feature vector
        work = propensity
        cost = feat[0]  # cluster size
        urgency = max(0.0, min(1.0, 1.0 - feat[3] / 4.0))  # invert avg hamming (more similar → higher urgency)

        decisions[rep] = burst_decision(propensity, work, cost, urgency)

    return decisions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate a synthetic dataset: 50 elements, each 5‑dimensional float vector
    random.seed(42)
    np.random.seed(42)
    elements = [list(np.random.rand(5) * 10) for _ in range(50)]

    # Run the hybrid pipeline
    result = hybrid_process(elements)

    # Simple sanity output
    print("Ambush decisions per component representative:")
    for rep, decision in result.items():
        print(f"  Component {rep}: {'ENGAGE' if decision else 'hold'}")
    print(f"Total components: {len(result)}")
    sys.exit(0)