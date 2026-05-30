# DARWIN HAMMER — match 423, survivor 5
# gen: 5
# parent_a: hybrid_rbf_surrogate_hybrid_distributed_l_m58_s1.py (gen2)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s4.py (gen4)
# born: 2026-05-29T23:28:55Z

"""Hybrid RBF‑Surrogate / Endpoint‑Circuit‑Breaker with Fisher‑Score Bridge

Parents:
- hybrid_rbf_surrogate_hybrid_distributed_l_m58_s1.py (RBF surrogate + perceptual hash similarity)
- hybrid_hybrid_endpoint_circ_hybrid_hybrid_fisher_m268_s4.py (EndpointCircuitBreaker + Fisher score + SSIM on Morphology)

Mathematical bridge:
The RBF surrogate predicts a continuous “reliability” score 𝑦̂ for each node from its feature vector.
Using the perceptual‑hash similarity matrix S we obtain a binary class label for every node
(high‑similarity vs low‑similarity).  The Fisher score 𝔽, computed from the surrogate
predictions across the two classes, quantifies class separability.  𝔽 is then used to
adapt the failure threshold τ of an EndpointCircuitBreaker:
 τ = τ₀ + ⌊α·𝔽⌋,
where τ₀ is a base threshold and α a scaling factor.  Nodes whose surrogate prediction
exceeds the current τ are allowed to broadcast; otherwise the breaker records a failure.
The SSIM‑like function on Morphology objects provides an auxiliary similarity check that
can be combined with the hash‑based similarity if desired.

The module implements this fused pipeline and provides three core functions demonstrating
the hybrid operation.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Mapping, Hashable, List, Dict, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Vector = Sequence[float]
Node = Hashable
Graph = Mapping[Node, Set[Node]]
FeatureVec = Sequence[float]

# ----------------------------------------------------------------------
# Parent A – RBF surrogate & perceptual hash utilities
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    """Simple perceptual hash: 1‑bit per value compared to mean, up to 64 bits."""
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
    """Pairwise similarity (1‑hamming/64) matrix from perceptual hashes."""
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
        """RBF interpolation prediction for a single point."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

def fit_rbf(
    points: Iterable[Vector],
    values: Iterable[float],
    epsilon: float = 1.0,
    ridge: float = 1e-9,
) -> RBFSurrogate:
    """Linear‑solve the RBF system (with ridge regularisation) and return a surrogate."""
    P = list(points)
    V = np.asarray(list(values), dtype=np.float64)
    m = len(P)
    # Build kernel matrix K_{ij} = φ(||p_i - p_j||)
    K = np.empty((m, m), dtype=np.float64)
    for i, pi in enumerate(P):
        for j, pj in enumerate(P):
            K[i, j] = gaussian(euclidean(pi, pj), epsilon)
    # Ridge regularisation
    K += ridge * np.eye(m)
    # Solve for weights
    weights = np.linalg.solve(K, V)
    return RBFSurrogate(centers=P, weights=weights.tolist(), epsilon=epsilon)

# ----------------------------------------------------------------------
# Parent B – EndpointCircuitBreaker, Fisher score & SSIM on Morphology
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float

    def as_vector(self) -> Tuple[float, float, float, float]:
        return (self.length, self.width, self.height, self.mass)

class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }

def fisher_score(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the classic Fisher discriminant score:
        F = (μ₁ - μ₂)² / (σ₁² + σ₂²)
    where μ and σ are class means and standard deviations.
    """
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape")
    class0 = predictions[labels == 0]
    class1 = predictions[labels == 1]
    if len(class0) == 0 or len(class1) == 0:
        return 0.0
    mu0, mu1 = class0.mean(), class1.mean()
    sigma0, sigma1 = class0.var(), class1.var()
    denominator = sigma0 + sigma1
    if denominator == 0:
        return 0.0
    return ((mu1 - mu0) ** 2) / denominator

def ssim_morphology(m1: Morphology, m2: Morphology) -> float:
    """
    A lightweight SSIM‑like similarity for Morphology vectors.
    Normalises each attribute to [0,1] using the global min/max of the pair,
    then computes the mean structural similarity:
        SSIM = (2μ₁μ₂ + C₁)(2σ₁₂ + C₂) / ((μ₁²+μ₂² + C₁)(σ₁²+σ₂² + C₂))
    For simplicity we set C₁=C₂=1e-4 and use variance as σ².
    """
    C1 = C2 = 1e-4
    v1 = np.array(m1.as_vector(), dtype=np.float64)
    v2 = np.array(m2.as_vector(), dtype=np.float64)
    mins = np.minimum(v1, v2)
    maxs = np.maximum(v1, v2)
    # Avoid division by zero
    denom = np.where(maxs - mins == 0, 1.0, maxs - mins)
    n1 = (v1 - mins) / denom
    n2 = (v2 - mins) / denom
    mu1, mu2 = n1.mean(), n2.mean()
    sigma1_sq, sigma2_sq = n1.var(), n2.var()
    sigma12 = ((n1 - mu1) * (n2 - mu2)).mean()
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return numerator / denominator if denominator != 0 else 0.0

# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def hybrid_fisher_adjusted_breaker(
    surrogate: RBFSurrogate,
    features: Dict[Node, FeatureVec],
    hashes: Dict[Node, int],
    base_threshold: int = 3,
    alpha: float = 5.0,
) -> EndpointCircuitBreaker:
    """
    Build an EndpointCircuitBreaker whose failure threshold is modulated by the
    Fisher score computed from surrogate predictions and a binary class derived
    from the perceptual‑hash similarity matrix.
    """
    # 1. Predict reliability for each node
    preds = np.array([surrogate.predict(f) for f in features.values()], dtype=np.float64)

    # 2. Derive binary class labels from similarity matrix:
    #    nodes with average similarity > 0.5 are class 1, else class 0.
    S, nodes = similarity_matrix(hashes)
    avg_sim = S.mean(axis=1)
    labels = np.array([1 if s > 0.5 else 0 for s in avg_sim], dtype=np.int8)

    # 3. Compute Fisher score
    F = fisher_score(preds, labels)

    # 4. Adjust threshold
    new_threshold = max(1, base_threshold + int(alpha * F))
    return EndpointCircuitBreaker(failure_threshold=new_threshold)

def hybrid_broadcast_decision(
    graph: Graph,
    features: Dict[Node, FeatureVec],
    morphologies: Dict[Node, Morphology],
    surrogate: RBFSurrogate,
    base_threshold: int = 3,
    alpha: float = 5.0,
    ssim_weight: float = 0.3,
) -> Set[Node]:
    """
    Decide which nodes are permitted to broadcast.
    - Nodes whose surrogate prediction exceeds the dynamically adjusted breaker threshold
      are tentatively allowed.
    - For each tentative node, compute SSIM with its immediate neighbours' morphologies.
      If the mean neighbour SSIM weighted by `ssim_weight` falls below 0.4, treat it as a
      failure (record_failure); otherwise record_success.
    Returns the set of nodes whose circuit breaker remains closed.
    """
    # Build hashes from feature vectors (using simple mean‑based hash)
    hashes = {n: compute_phash(list(f)) for n, f in features.items()}

    # Create adaptive breaker
    breaker = hybrid_fisher_adjusted_breaker(
        surrogate, features, hashes, base_threshold=base_threshold, alpha=alpha
    )

    allowed: Set[Node] = set()
    for node, vec in features.items():
        pred = surrogate.predict(vec)
        if pred >= breaker.failure_threshold:
            # Evaluate SSIM with neighbours
            neighbour_ssims = []
            for nb in graph.get(node, []):
                if nb in morphologies and node in morphologies:
                    neighbour_ssims.append(ssim_morphology(morphologies[node], morphologies[nb]))
            mean_ssim = sum(neighbour_ssims) / len(neighbour_ssims) if neighbour_ssims else 0.0
            # Combine surrogate prediction and SSIM into a simple score
            combined_score = pred * (1 - ssim_weight) + mean_ssim * ssim_weight
            if combined_score >= breaker.failure_threshold:
                breaker.record_success()
                allowed.add(node)
            else:
                breaker.record_failure()
        else:
            breaker.record_failure()
    return allowed

def build_example_surrogate() -> RBFSurrogate:
    """
    Construct a toy RBF surrogate on 5 random 3‑D points with synthetic target values.
    This helper is used in the smoke test.
    """
    random.seed(0)
    points = [[random.random() for _ in range(3)] for _ in range(5)]
    # Synthetic target: sum of coordinates + small noise
    values = [sum(p) + random.gauss(0, 0.05) for p in points]
    return fit_rbf(points, values, epsilon=1.5, ridge=1e-6)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small synthetic graph
    G: Graph = {
        "A": {"B", "C"},
        "B": {"A", "D"},
        "C": {"A"},
        "D": {"B"},
        "E": set(),
    }

    # Random feature vectors (3‑dimensional)
    features: Dict[Node, FeatureVec] = {
        n: [random.random() for _ in range(3)] for n in G
    }

    # Random morphologies
    morphologies: Dict[Node, Morphology] = {
        n: Morphology(
            length=random.uniform(0.5, 2.0),
            width=random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass=random.uniform(1.0, 5.0),
        )
        for n in G
    }

    surrogate = build_example_surrogate()
    allowed_nodes = hybrid_broadcast_decision(
        graph=G,
        features=features,
        morphologies=morphologies,
        surrogate=surrogate,
        base_threshold=2,
        alpha=3.0,
        ssim_weight=0.2,
    )
    print("Nodes allowed to broadcast:", sorted(allowed_nodes))