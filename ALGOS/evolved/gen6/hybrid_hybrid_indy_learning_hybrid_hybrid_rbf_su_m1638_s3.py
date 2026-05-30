# DARWIN HAMMER — match 1638, survivor 3
# gen: 6
# parent_a: hybrid_indy_learning_vector_hybrid_hybrid_geomet_m113_s0.py (gen3)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hybrid_endpoi_m423_s2.py (gen5)
# born: 2026-05-29T23:38:10Z

import re
import math
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Callable, Any

import numpy as np

# ----------------------------------------------------------------------
# Global constants and utilities
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

WORD_RE = re.compile(r"\S+")

DEFAULT_TERMS = (
    "ENTITY", "ATTRIBUTE", "RELATIONSHIP", "ACTION", "EVENT", "TIME", "EVIDENCE",
    "CLAIM", "HYPOTHESIS", "SIGNAL", "PATTERN", "TOOL", "ALGORITHM", "BOOK",
    "SOURCE", "LEAD", "LOCATION", "LAW", "RULE",
)

def sha256_json(value: Any) -> str:
    """Deterministic SHA‑256 hash of a JSON‑serialisable object."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


# ----------------------------------------------------------------------
# Geometric algebra utilities (kept for future extensions)
# ----------------------------------------------------------------------
def geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a simple concatenation of dot and wedge (cross) products."""
    dot = np.dot(a, b)
    wedge = np.cross(a, b)
    return np.concatenate((np.atleast_1d(dot), wedge))


# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature (enhanced)
# ----------------------------------------------------------------------
def pairwise_distances(vectors: np.ndarray) -> np.ndarray:
    """Efficient pairwise Euclidean distance matrix."""
    diff = vectors[:, None, :] - vectors[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def ollivier_ricci_curvature(
    region_vectors: List[np.ndarray],
    region_centroids: List[np.ndarray],
    alpha: float = 0.5,
) -> float:
    """
    A more faithful (yet still lightweight) Ollivier‑Ricci curvature estimator.

    The curvature κ between two probability measures μ_i, μ_j supported on
    region_vectors is approximated by

        κ_ij = 1 - (W_1(μ_i, μ_j) / d(c_i, c_j))

    where W_1 is the 1‑Wasserstein distance (here approximated by the
    Euclidean distance between the vectors) and d is the distance between
    centroids. The final curvature is the average of κ_ij over all i < j.

    Parameters
    ----------
    region_vectors : list of np.ndarray
        Feature vectors belonging to each Voronoi region.
    region_centroids : list of np.ndarray
        Corresponding seed (centroid) vectors.
    alpha : float
        Smoothing factor that blends the raw transport cost with a small
        regularisation term to avoid division by zero.

    Returns
    -------
    float
        Estimated average Ollivier‑Ricci curvature.
    """
    n = len(region_vectors)
    if n < 2:
        return 0.0

    # Stack for vectorised ops
    vecs = np.stack(region_vectors)
    cents = np.stack(region_centroids)

    # Pairwise Euclidean distances between centroids
    d_c = pairwise_distances(cents)

    # Pairwise distances between region vectors (used as a proxy for W_1)
    d_v = pairwise_distances(vecs)

    # Compute κ_ij for i < j, ignoring the diagonal
    curvature_sum = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            denom = max(d_c[i, j], 1e-12)  # avoid division by zero
            transport = d_v[i, j]
            kappa = 1.0 - (transport / denom)
            # Optional smoothing
            curvature_sum += alpha * kappa + (1 - alpha) * (1.0 / (1.0 + transport))
            count += 1

    return curvature_sum / count if count else 0.0


# ----------------------------------------------------------------------
# RBF surrogate utilities
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass(frozen=True)
class RBFSurrogate:
    """
    Simple RBF surrogate model.

    The model stores centers and corresponding weights.  Prediction is a
    weighted sum of Gaussian kernels.
    """
    centers: Tuple[Tuple[float, ...], ...]
    weights: Tuple[float, ...]
    epsilon: float = 1.0

    @staticmethod
    def from_vectors(vectors: List[np.ndarray],
                     weight_factory: Callable[[np.ndarray], float] | None = None,
                     epsilon: float = 1.0) -> "RBFSurrogate":
        """Convenient constructor that builds uniform or custom weights."""
        if weight_factory is None:
            weights = tuple(1.0 for _ in vectors)
        else:
            weights = tuple(weight_factory(v) for v in vectors)
        centers = tuple(tuple(v.tolist()) for v in vectors)
        return RBFSurrogate(centers=centers, weights=weights, epsilon=epsilon)

    def predict(self, x: Sequence[float]) -> float:
        """Evaluate the surrogate at point ``x``."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )


# ----------------------------------------------------------------------
# Core hybrid algorithm
# ----------------------------------------------------------------------
def _text_to_frequency_vector(
    text: str,
    ontology_terms: Tuple[str, ...],
) -> np.ndarray:
    """Map a text string to a term‑frequency vector."""
    vec = np.zeros(len(ontology_terms), dtype=float)
    for word in WORD_RE.findall(text):
        try:
            idx = ontology_terms.index(word)
        except ValueError:
            continue
        vec[idx] += 1.0
    return vec


def hybrid_operation(
    texts: List[str],
    ontology_terms: List[str],
    seed_vectors: List[np.ndarray],
    curvature_alpha: float = 0.5,
    rbf_epsilon: float = 1.0,
) -> Tuple[float, float]:
    """
    Execute the hybrid pipeline.

    Returns
    -------
    similarity_mean : float
        Average pairwise similarity derived from the curvature‑aware RBF kernel.
    curvature : float
        Estimated Ollivier‑Ricci curvature of the Voronoi partition.
    """
    if not texts:
        raise ValueError("texts list must not be empty")
    if len(seed_vectors) < 2:
        raise ValueError("need at least two seed vectors for a Voronoi partition")

    ontology_terms_tuple = tuple(ontology_terms)

    # 1️⃣  Text → frequency vectors
    frequency_vectors = [_text_to_frequency_vector(t, ontology_terms_tuple) for t in texts]

    # 2️⃣  Voronoi assignment (hard nearest‑seed)
    region_vectors: List[np.ndarray] = []
    region_centroids: List[np.ndarray] = []
    seed_matrix = np.stack(seed_vectors)

    for vec in frequency_vectors:
        distances = np.linalg.norm(seed_matrix - vec, axis=1)
        nearest_idx = int(np.argmin(distances))
        region_vectors.append(vec)
        region_centroids.append(seed_vectors[nearest_idx])

    # 3️⃣  Curvature estimation
    curvature = ollivier_ricci_curvature(
        region_vectors, region_centroids, alpha=curvature_alpha
    )

    # 4️⃣  Build curvature‑aware RBF surrogate
    #    The weight of each centre is modulated by its local curvature contribution.
    def weight_factory(v: np.ndarray) -> float:
        # Higher curvature → stronger influence; map κ∈[-1,1] → weight∈[0.5,1.5]
        local_curv = ollivier_ricci_curvature([v], [v], alpha=curvature_alpha)
        return 1.0 + 0.5 * local_curv

    surrogate = RBFSurrogate.from_vectors(
        vectors=region_centroids,
        weight_factory=weight_factory,
        epsilon=rbf_epsilon,
    )

    # 5️⃣  Pairwise similarity matrix using the surrogate
    n = len(region_centroids)
    similarity_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            # Kernel product captures joint similarity; curvature scales the result.
            sim = surrogate.predict(region_centroids[i]) * surrogate.predict(region_centroids[j])
            sim *= (1.0 + curvature)  # integrate global curvature
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    # The diagonal is trivially 1 (self‑similarity) – optional but often useful.
    np.fill_diagonal(similarity_matrix, 1.0)

    similarity_mean = float(np.mean(similarity_matrix))

    return similarity_mean, curvature


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
def smoke_test() -> None:
    texts = [
        "This is a test text.",
        "This is another test text.",
        "Entity relationship action event.",
    ]
    ontology_terms = list(DEFAULT_TERMS)
    # Fixed random seed for reproducibility
    rng = np.random.default_rng(42)
    seed_vectors = [rng.random(len(ontology_terms)) for _ in range(5)]

    similarity, curvature = hybrid_operation(
        texts=texts,
        ontology_terms=ontology_terms,
        seed_vectors=seed_vectors,
        curvature_alpha=0.6,
        rbf_epsilon=0.8,
    )
    print(f"Mean similarity: {similarity:.6f}")
    print(f"Ollivier‑Ricci curvature: {curvature:.6f}")


if __name__ == "__main__":
    smoke_test()