# DARWIN HAMMER — match 4645, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_rbf_surrogate_hybrid_hybrid_endpoi_m423_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2245_s0.py (gen6)
# born: 2026-05-29T23:58:33Z

"""Hybrid algorithm combining:
- Parent A (hybrid_rbf_surrogate …): radial‑basis‑function surrogate for perceptual similarity,
  phash generation and hamming‑based similarity matrix.
- Parent B (hybrid_hybrid_hybrid …): fractional Caputo‑type store dynamics and
  Shannon‑entropy‑weighted cue integration.

Mathematical bridge:
The RBF surrogate yields a similarity cue vector 𝑠∈ℝⁿ for a set of graph nodes.
An entropy‑derived weight vector 𝑤(𝑠) = −∑p_i log p_i  (with p_i = s_i/∑s) quantifies the
information content of the cues.  The fractional Caputo derivative of the store level
modulates the decay factor applied to these entropy‑weighted cues, i.e.

    ΔL = α·inflow − β·outflow + 𝒟^{α_f}L_{t−1}

and the effective cue contribution to the store update is

    η = (1 − entropy(𝑤))·ΔL .

Thus the memory of the store (via the Caputo derivative) influences how strongly the
RBF‑derived similarity cues affect the system state, while the cue distribution (entropy)
feeds back to adjust the store dynamics.  The functions below expose this fused behaviour.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence, Mapping, Hashable, List, Dict, Set, Tuple

import numpy as np

Vector = Sequence[float]
Node = Hashable
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Parent A components (RBF surrogate, phash, similarity)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def similarity_matrix_from_hashes(hashes: Dict[Node, int]) -> Tuple[np.ndarray, List[Node]]:
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

def build_rbf_surrogate(samples: List[FeatureVec], epsilon: float = 1.0) -> RBFSurrogate:
    """Very simple surrogate: equal weights normalized to sum to 1."""
    if not samples:
        raise ValueError("samples must not be empty")
    n = len(samples)
    weights = [1.0 / n] * n
    centers = [tuple(s) for s in samples]
    return RBFSurrogate(centers=centers, weights=weights, epsilon=epsilon)

def rbf_similarity_matrix(surrogate: RBFSurrogate, nodes: List[Node], features: Mapping[Node, FeatureVec]) -> Tuple[np.ndarray, List[Node]]:
    """Compute an NxN similarity matrix where entry (i,j) = surrogate.predict(|f_i-f_j|)."""
    n = len(nodes)
    S = np.empty((n, n), dtype=np.float64)
    for i, ni in enumerate(nodes):
        vi = np.array(features[ni])
        for j, nj in enumerate(nodes):
            if j < i:
                S[i, j] = S[j, i]
            else:
                vj = np.array(features[nj])
                diff = np.abs(vi - vj).tolist()
                S[i, j] = surrogate.predict(diff)
    return S, nodes

# ----------------------------------------------------------------------
# Parent B components (fractional store, entropy cue weighting)
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0          # inflow scaling
    beta: float = 1.0           # outflow scaling
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _history: List[float] = field(default_factory=list, init=False, repr=False)

    def update(self, inflow: float, outflow: float, frac_alpha: float) -> Tuple[float, float]:
        """Update store level using a simple Caputo‑type fractional derivative."""
        prev = self.level
        self._history.append(prev)

        delta_classical = self.alpha * inflow - self.beta * outflow

        if len(self._history) >= 2:
            caputo = (self._history[-1] - self._history[-2]) / (self.dt ** frac_alpha) / math.gamma(1 - frac_alpha)
        else:
            caputo = 0.0

        delta_total = delta_classical + caputo
        self.level = max(0.0, self.level + self.dt * delta_total)
        self.level = min(self.level, self.limit)
        return self.level, delta_total

def shannon_entropy(probs: Sequence[float]) -> float:
    """Standard Shannon entropy, assumes probs sum to 1 and are non‑negative."""
    eps = 1e-12
    return -sum(p * math.log(p + eps) for p in probs)

def entropy_weights(cues: Sequence[float]) -> List[float]:
    """Convert raw cue magnitudes into probability‑like weights and return their entropy‑based scaling."""
    total = sum(cues) + 1e-12
    probs = [c / total for c in cues]
    ent = shannon_entropy(probs)
    # Scale each cue by (1 - normalized entropy) to reward low‑entropy (more informative) distributions
    norm_ent = ent / math.log(len(cues) + 1e-12)  # normalized to [0,1]
    scale = 1.0 - norm_ent
    return [c * scale for c in cues]

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_hybrid_cues(
    graph: Mapping[Node, Set[Node]],
    features: Mapping[Node, FeatureVec],
    surrogate: RBFSurrogate,
) -> Tuple[np.ndarray, List[Node]]:
    """
    Produce a similarity cue matrix using the RBF surrogate.
    Nodes are taken from the graph keys; the returned matrix is symmetric.
    """
    nodes = list(graph.keys())
    S, _ = rbf_similarity_matrix(surrogate, nodes, features)
    # Normalise rows to [0,1] for entropy processing
    row_sums = S.sum(axis=1, keepdims=True) + 1e-12
    S_norm = S / row_sums
    return S_norm, nodes

def update_store_from_cues(
    store: StoreState,
    cues: np.ndarray,
    frac_alpha: float = 0.5,
) -> Tuple[float, np.ndarray]:
    """
    For each node i, treat the i‑th row of the cue matrix as a set of
    similarity cues. Convert the row to entropy‑scaled weights and feed the
    aggregate (mean) as the inflow to the store. Outflow is taken as the mean
    of the previous inflow (simple momentum).
    """
    if cues.size == 0:
        raise ValueError("cues matrix is empty")
    # Compute entropy‑scaled cues per row
    scaled_rows = np.apply_along_axis(entropy_weights, 1, cues)
    # Aggregate to a single scalar inflow/outflow pair
    inflow = float(np.mean(scaled_rows))
    outflow = inflow * 0.3  # arbitrary decay proportion
    level, delta = store.update(inflow, outflow, frac_alpha)
    return level, scaled_rows

def endpoint_circuit_breaker(
    similarity: np.ndarray,
    store_level: float,
    base_threshold: float = 0.6,
) -> np.ndarray:
    """
    Adjust a binary break‑condition matrix based on current store level.
    Higher store level raises the failure threshold, making the breaker more tolerant.
    """
    dynamic_thresh = base_threshold + 0.3 * (store_level / 10.0)  # store_level capped at limit=10
    breaker = (similarity < dynamic_thresh).astype(int)
    return breaker

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny random graph with feature vectors
    random.seed(42)
    np.random.seed(42)

    nodes = [f"n{i}" for i in range(5)]
    graph: Dict[Node, Set[Node]] = {n: set() for n in nodes}
    # connect each node to two random others (undirected)
    for n in nodes:
        others = [o for o in nodes if o != n]
        partners = random.sample(others, 2)
        graph[n].update(partners)
        for p in partners:
            graph[p].add(n)

    # random 8‑dimensional feature vectors
    features: Dict[Node, FeatureVec] = {
        n: np.random.rand(8).tolist() for n in nodes
    }

    # Build RBF surrogate from the feature set
    surrogate = build_rbf_surrogate(list(features.values()), epsilon=0.8)

    # Hybrid cue computation
    cues, ordered_nodes = compute_hybrid_cues(graph, features, surrogate)

    # Initialise store
    store = StoreState(level=2.0, limit=10.0, alpha=1.2, beta=0.8, dt=1.0)

    # Update store using cues
    level, scaled = update_store_from_cues(store, cues, frac_alpha=0.4)
    print(f"Updated store level: {level:.3f}")

    # Apply endpoint circuit breaker
    breaker = endpoint_circuit_breaker(cues, level)
    print("Circuit breaker matrix (1 = break, 0 = pass):")
    print(breaker)