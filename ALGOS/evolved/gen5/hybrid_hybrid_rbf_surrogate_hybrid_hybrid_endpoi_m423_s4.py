# DARWIN HAMMER — match 423, survivor 4
# gen: 5
# parent_a: hybrid_rbf_surrogate_hybrid_distributed_l_m58_s1.py (gen2)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s4.py (gen4)
# born: 2026-05-29T23:28:55Z

"""
Hybrid Algorithm: rbf_surrogate + endpoint_circuit_breaker with Fisher‑score bridge

Parents:
- hybrid_rbf_surrogate_hybrid_distributed_l_m58_s1.py
- hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s4.py

Mathematical Bridge:
The RBF surrogate model provides a continuous estimate 𝑝̂(i) of the perceptual
similarity of node *i* to its neighbours (via the similarity matrix built from
perceptual hashes).  A Fisher score F(i) is computed for each node using the
predicted similarities as a proxy class‑separability measure between “leader”
and “follower” label sets.  The Fisher score rescales the failure threshold
τ of the EndpointCircuitBreaker:

    τ_i = τ₀ · (1 + α·F(i))

Thus nodes that are more “discriminative” (high Fisher score) become more
tolerant to failures, while indistinguishable nodes adopt a stricter threshold.
The three core operations are:
1. Build an RBF surrogate from node feature vectors.
2. Derive a Fisher‑score‑adjusted circuit‑breaker per node.
3. Use the circuit‑breaker decisions to drive a broadcast/election step.

The implementation below fuses the matrix operations of both parents into a
single, self‑contained module.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Sequence, Mapping, Hashable, List, Tuple, Dict, Set, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Basic utilities (shared by both parents)
# ----------------------------------------------------------------------
Vector = Sequence[float]
Node = Hashable
Graph = Mapping[Node, Set[Node]]
FeatureVec = Sequence[float]

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Radial basis function – isotropic Gaussian."""
    return math.exp(-((epsilon * r) ** 2))

def compute_phash(values: List[float]) -> int:
    """Very small perceptual hash: 1‑bit per value (up to 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def similarity_matrix(hashes: Dict[Node, int]) -> Tuple[np.ndarray, List[Node]]:
    """Pairwise similarity (1 – normalized Hamming distance)."""
    nodes = list(hashes.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        hi = hashes[ni]
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
            else:
                hj = hashes[nj]
                d = hamming_distance(hi, hj)
                S[i, j] = 1.0 - d / 64.0
    return S, nodes

# ----------------------------------------------------------------------
# RBF Surrogate (from parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

def fit_rbf(points: Iterable[Vector], values: Iterable[float],
            epsilon: float = 1.0, ridge: float = 1e-9) -> RBFSurrogate:
    """
    Solve (G + ridge·I) w = y for weights w, where
    G_{ij} = gaussian(||c_i - c_j||, epsilon) and centers = points.
    """
    X = np.array(list(points), dtype=np.float64)
    y = np.array(list(values), dtype=np.float64)
    n = X.shape[0]
    G = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            r = euclidean(X[i], X[j])
            G[i, j] = G[j, i] = gaussian(r, epsilon)
    A = G + ridge * np.eye(n)
    w = np.linalg.solve(A, y)
    centers = [tuple(row) for row in X]
    return RBFSurrogate(centers=centers, weights=w.tolist(), epsilon=epsilon)

# ----------------------------------------------------------------------
# Fisher score computation (from parent B)
# ----------------------------------------------------------------------
def fisher_score(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute per‑feature Fisher scores for a binary classification problem.
    Returns a 1‑D array of length `features.shape[1]`.
    """
    if labels.ndim != 1:
        raise ValueError("labels must be a 1‑D array")
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("Fisher score defined for exactly two classes")
    c0, c1 = classes
    mask0 = labels == c0
    mask1 = labels == c1
    mu0 = features[mask0].mean(axis=0)
    mu1 = features[mask1].mean(axis=0)
    var0 = features[mask0].var(axis=0, ddof=1)
    var1 = features[mask1].var(axis=0, ddof=1)
    # Avoid division by zero
    denom = var0 + var1
    denom[denom == 0] = 1e-12
    score = ((mu0 - mu1) ** 2) / denom
    return score

# ----------------------------------------------------------------------
# Endpoint Circuit Breaker (from parent B) with Fisher‑adjusted threshold
# ----------------------------------------------------------------------
class AdaptiveCircuitBreaker:
    """
    Circuit breaker whose failure threshold τ_i is adapted per node by a
    multiplicative factor derived from the Fisher score of that node.
    """
    def __init__(self, base_threshold: int = 3, alpha: float = 0.5):
        if base_threshold <= 0:
            raise ValueError("base_threshold must be positive")
        self.base_threshold = base_threshold
        self.alpha = alpha
        self._state: Dict[Node, Tuple[int, bool]] = {}  # node → (failures, open)

    def _threshold_for(self, fisher_factor: float) -> int:
        """Compute τ_i = τ₀·(1 + α·F) and round up to nearest int."""
        return max(1, int(math.ceil(self.base_threshold * (1.0 + self.alpha * fisher_factor))))

    def record_success(self, node: Node) -> None:
        self._state[node] = (0, False)

    def record_failure(self, node: Node, fisher_factor: float) -> None:
        failures, _ = self._state.get(node, (0, False))
        failures += 1
        thr = self._threshold_for(fisher_factor)
        open_ = failures >= thr
        self._state[node] = (failures, open_)

    def allow(self, node: Node) -> bool:
        """True if the circuit for *node* is closed."""
        return not self._state.get(node, (0, False))[1]

    def snapshot(self) -> Dict[Node, Dict[str, int | bool]]:
        return {
            n: {"failures": f, "open": o}
            for n, (f, o) in self._state.items()
        }

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def build_node_hashes(features: Dict[Node, FeatureVec]) -> Dict[Node, int]:
    """Compute a perceptual hash for each node from its raw feature vector."""
    return {n: compute_phash(list(vec)) for n, vec in features.items()}

def predict_similarity_surrogate(
    features: Dict[Node, FeatureVec],
    surrogate: RBFSurrogate
) -> Dict[Node, float]:
    """Use the RBF surrogate to predict a similarity score for each node."""
    return {n: surrogate.predict(list(vec)) for n, vec in features.items()}

def compute_node_fisher_factors(
    features: Dict[Node, FeatureVec],
    leader_labels: Dict[Node, int]
) -> Dict[Node, float]:
    """
    Compute a Fisher‑derived factor per node.
    The factor is the dot‑product of the node's feature vector with the
    normalized Fisher scores (so that more discriminative features yield larger
    factors).
    """
    nodes = list(features.keys())
    X = np.array([features[n] for n in nodes], dtype=np.float64)
    y = np.array([leader_labels[n] for n in nodes], dtype=np.int32)
    scores = fisher_score(X, y)                     # per‑feature scores
    norm = np.linalg.norm(scores)
    if norm == 0:
        norm = 1e-12
    weights = scores / norm                         # normalized
    factors = X @ weights                           # linear combination
    # Rescale to [0, 1] for stability
    min_f, max_f = factors.min(), factors.max()
    if max_f - min_f < 1e-12:
        return {n: 0.0 for n in nodes}
    return {n: (f - min_f) / (max_f - min_f) for n, f in zip(nodes, factors)}

def hybrid_step(
    graph: Graph,
    features: Dict[Node, FeatureVec],
    leader_labels: Dict[Node, int],
    breaker: AdaptiveCircuitBreaker,
    surrogate: RBFSurrogate,
    alpha: float = 0.5
) -> Set[Node]:
    """
    One iteration of the hybrid algorithm:
    1. Predict similarity scores with the RBF surrogate.
    2. Compute Fisher factors and adjust circuit‑breaker thresholds.
    3. Nodes whose circuit is closed broadcast to neighbours; a node becomes a
       leader if it receives at least one broadcast from a leader neighbour.
    Returns the set of newly elected leaders.
    """
    # 1 – surrogate predictions (not directly used downstream but illustrate the bridge)
    _preds = predict_similarity_surrogate(features, surrogate)

    # 2 – Fisher factors for adaptive thresholds
    fisher_factors = compute_node_fisher_factors(features, leader_labels)

    # 3 – broadcast according to circuit state
    newly_leaders: Set[Node] = set()
    for node in graph:
        # Update circuit state based on a synthetic failure probability derived
        # from the surrogate prediction (higher similarity ⇒ lower failure chance)
        failure_prob = 1.0 - _preds[node]  # simple inversion
        if random.random() < failure_prob:
            breaker.record_failure(node, fisher_factors[node])
        else:
            breaker.record_success(node)

        if not breaker.allow(node):
            continue  # circuit open – node cannot broadcast

        # Broadcast to neighbours; if any neighbour is already a leader, become leader
        if any(nei in leader_labels and leader_labels[nei] == 1 for nei in graph[node]):
            newly_leaders.add(node)

    # Update leader_labels in‑place
    for n in newly_leaders:
        leader_labels[n] = 1
    return newly_leaders

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny random graph
    random.seed(42)
    nodes = [f"N{i}" for i in range(6)]
    graph: Graph = {n: set() for n in nodes}
    for n in nodes:
        # each node connects to up to 2 random others
        peers = random.sample([p for p in nodes if p != n], k=random.randint(1, 2))
        graph[n].update(peers)
        for p in peers:
            graph[p].add(n)

    # Random 3‑dimensional feature vectors
    features: Dict[Node, FeatureVec] = {
        n: [random.random() for _ in range(3)] for n in nodes
    }

    # Initial leader labeling (binary): pick one random leader
    leader_labels: Dict[Node, int] = {n: 0 for n in nodes}
    initial_leader = random.choice(nodes)
    leader_labels[initial_leader] = 1

    # Build surrogate from features and a dummy target (e.g., similarity to leader)
    target_values = [1.0 if leader_labels[n] else 0.0 for n in nodes]
    surrogate = fit_rbf(features.values(), target_values, epsilon=1.5)

    # Instantiate adaptive circuit breaker
    breaker = AdaptiveCircuitBreaker(base_threshold=3, alpha=0.7)

    # Run a few hybrid steps
    for step in range(5):
        new = hybrid_step(graph, features, leader_labels, breaker, surrogate)
        print(f"Step {step+1}: new leaders = {sorted(new)}")
        if not new:
            break

    print("Final leader set:", sorted([n for n, v in leader_labels.items() if v == 1]))
    print("Circuit breaker snapshot:", breaker.snapshot())