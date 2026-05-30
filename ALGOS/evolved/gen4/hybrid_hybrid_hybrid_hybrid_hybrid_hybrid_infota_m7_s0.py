# DARWIN HAMMER — match 7, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_rbf_su_hybrid_capybara_opti_m52_s0.py (gen3)
# parent_b: hybrid_hybrid_infotaxis_min_hybrid_hybrid_distri_m20_s4.py (gen3)
# born: 2026-05-29T23:26:18Z

"""Hybrid Algorithm: RBF‑Surrogate + Entropic MinHash Drag Dynamics

Parents:
- hybrid_hybrid_hybrid_rbf_su_hybrid_capybara_opti_m52_s0.py (Radial‑Basis Function surrogate)
- hybrid_hybrid_infotaxis_min_hybrid_hybrid_distri_m20_s4.py (Entropic MinHash + Chelydrid drag dynamics)

Mathematical Bridge:
The MinHash signature of a token set is interpreted as a high‑dimensional coordinate
vector.  Euclidean distances between two signatures feed the Gaussian kernel of the
RBF surrogate.  The surrogate learns a mapping from a feature vector that contains
entropy, drag‑limited peak velocity (obtained by integrating a force series derived
from the signature) and the raw Jaccard‑like similarity to a final hybrid similarity
score in [0, 1].  Thus the linear system of the RBF surrogate and the physics‑based
drag integration are fused into a single predictive model."""
import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Utilities from Parent A (RBF Surrogate)
# ----------------------------------------------------------------------


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b with simple Gauss‑Jordan elimination (no pivoting beyond max row)."""
    n = len(b)
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular surrogate system")
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for row in range(n):
            if row == col:
                continue
            factor = m[row][col]
            m[row] = [v - factor * p for v, p in zip(m[row], m[col])]
    return [row[-1] for row in m]


@dataclass(frozen=True)
class RBFSurrogate:
    """Gaussian RBF surrogate model."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


# ----------------------------------------------------------------------
# Utilities from Parent B (Entropic MinHash + Drag Dynamics)
# ----------------------------------------------------------------------


def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    """Shannon entropy of a probability mass function."""
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("positive probability mass required")
    return -sum(
        (p / total) * math.log(max(p / total, eps))
        for p in probabilities
        if p > 0.0
    )


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash used by MinHash."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Standard MinHash signature of length k."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity between two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have same length")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def minhash_to_force_series(sig: List[int], scale: float = 1e-6) -> np.ndarray:
    """
    Convert a MinHash signature into a force time‑series.
    Each integer is scaled to a force magnitude; the series is
    normalised to have zero mean to emulate alternating thrust.
    """
    forces = np.array(sig, dtype=float) * scale
    forces -= forces.mean()
    return forces


def integrate_drag(
    forces: np.ndarray,
    mass: float = 1.0,
    drag_coeff: float = 0.1,
    dt: float = 0.01,
) -> float:
    """
    Simple explicit Euler integration of a drag‑limited 1‑D motion:
        dv/dt = (F - c*v) / m
    Returns the peak velocity reached during the simulation.
    """
    v = 0.0
    v_peak = 0.0
    for f in forces:
        a = (f - drag_coeff * v) / mass
        v += a * dt
        if abs(v) > abs(v_peak):
            v_peak = v
    return abs(v_peak)


# ----------------------------------------------------------------------
# Hybrid Construction
# ----------------------------------------------------------------------


def build_feature_vector(
    data: bytes,
    tokens: Iterable[str],
    prob_dist: List[float],
) -> List[float]:
    """
    Create a feature vector that will be consumed by the RBF surrogate.
    Features:
        0 – Shannon entropy of the probability distribution.
        1 – Peak velocity from drag integration of the MinHash‑derived force series.
        2 – Normalised size of the data payload (bytes / 1e6).
    """
    # Entropy
    ent = entropy(prob_dist)

    # MinHash signature → forces → peak velocity
    sig = signature(tokens)
    forces = minhash_to_force_series(sig)
    v_peak = integrate_drag(forces)

    # Normalised data size (to keep values roughly O(1))
    size_norm = len(data) / 1_000_000.0

    return [ent, v_peak, size_norm]


def train_rbf_surrogate(
    feature_vectors: List[List[float]],
    target_scores: List[float],
    epsilon: float = 1.0,
) -> RBFSurrogate:
    """
    Fit an RBF surrogate to map feature_vectors → target_scores.
    The centers are the provided feature vectors; the weights are solved from
    the linear system A w = y where A_ij = gaussian(||c_i - c_j||, epsilon).
    """
    if len(feature_vectors) != len(target_scores):
        raise ValueError("feature_vectors and target_scores must have same length")
    n = len(feature_vectors)
    # Build kernel matrix
    A = [
        [gaussian(euclidean(feature_vectors[i], feature_vectors[j]), epsilon) for j in range(n)]
        for i in range(n)
    ]
    # Solve for weights
    w = solve_linear(A, target_scores)
    # Convert centers to tuples for hashability
    centers = [tuple(vec) for vec in feature_vectors]
    return RBFSurrogate(centers=centers, weights=w, epsilon=epsilon)


def hybrid_similarity(
    data_a: bytes,
    tokens_a: Iterable[str],
    prob_a: List[float],
    data_b: bytes,
    tokens_b: Iterable[str],
    prob_b: List[float],
    surrogate: RBFSurrogate,
) -> float:
    """
    Compute a hybrid similarity score between two objects.

    1. Build independent feature vectors for each object.
    2. Concatenate the two vectors and the raw MinHash similarity into a
       single joint vector.
    3. Feed the joint vector to the trained RBF surrogate, yielding a
       dimension‑less score in [0, 1].
    """
    fv_a = build_feature_vector(data_a, tokens_a, prob_a)
    fv_b = build_feature_vector(data_b, tokens_b, prob_b)

    # Raw MinHash similarity (independent of the surrogate)
    sig_a = signature(tokens_a)
    sig_b = signature(tokens_b)
    raw_sim = similarity(sig_a, sig_b)

    # Joint feature vector
    joint = fv_a + fv_b + [raw_sim]

    # Surrogate prediction (clamped to [0,1] for safety)
    pred = surrogate.predict(joint)
    return max(0.0, min(1.0, pred))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic samples
    random.seed(42)
    samples = []
    targets = []
    for i in range(10):
        # Random byte payload
        data = bytes(random.getrandbits(8) for _ in range(random.randint(10_000, 100_000)))
        # Random token set (simulating words)
        tokens = [f"tok{random.randint(0, 500)}" for _ in range(random.randint(5, 20))]
        # Random probability distribution (10 bins)
        probs = np.random.dirichlet(alpha=np.ones(10)).tolist()
        fv = build_feature_vector(data, tokens, probs)
        samples.append(fv)

        # Define a synthetic target: higher entropy & higher peak velocity → higher score
        target = (fv[0] + fv[1]) / 2.0  # simple proxy, already roughly in [0,~5]
        # Normalise to [0,1] using an ad‑hoc scaling
        targets.append(min(1.0, target / 5.0))

    # Train surrogate
    surrogate = train_rbf_surrogate(samples, targets, epsilon=0.5)

    # Compare two fresh random objects
    data_a = bytes(random.getrandbits(8) for _ in range(50_000))
    tokens_a = [f"tok{random.randint(0, 500)}" for _ in range(12)]
    prob_a = np.random.dirichlet(alpha=np.ones(10)).tolist()

    data_b = bytes(random.getrandbits(8) for _ in range(80_000))
    tokens_b = [f"tok{random.randint(0, 500)}" for _ in range(15)]
    prob_b = np.random.dirichlet(alpha=np.ones(10)).tolist()

    score = hybrid_similarity(data_a, tokens_a, prob_a, data_b, tokens_b, prob_b, surrogate)
    print(f"Hybrid similarity score: {score:.4f}")