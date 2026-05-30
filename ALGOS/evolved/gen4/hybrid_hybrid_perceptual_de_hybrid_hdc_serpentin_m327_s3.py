# DARWIN HAMMER — match 327, survivor 3
# gen: 4
# parent_a: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s3.py (gen3)
# parent_b: hybrid_hdc_serpentina_self_righ_m50_s0.py (gen1)
# born: 2026-05-29T23:28:20Z

"""Hybrid Perceptual-Hyperdimensional RBF Model
Parents:
- hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s3.py (perceptual hashing + RBF surrogate)
- hybrid_hdc_serpentina_self_righ_m50_s0.py (hyperdimensional vectors influenced by morphology)

Mathematical Bridge:
The bridge is the shared notion of *distance* in two spaces:
1. In the perceptual domain, Hamming distance between perceptual hashes groups
   similar feature vectors (parent A).
2. In the hyperdimensional domain, Euclidean distance between morphology‑influenced
   bipolar vectors defines a kernel matrix for a radial‑basis‑function (RBF) surrogate
   (parent B).

We first cluster data by perceptual hash, then map each cluster to a single
hyperdimensional vector whose seed is derived from the cluster’s average
morphology (sphericity index).  The RBF kernel built from Euclidean distances
between these cluster vectors is solved against a target derived from the
morphology‑based recovery priority.  Prediction for a new sample combines
its perceptual hash (to locate a cluster) with the hyperdimensional binding
of a symbol vector and the morphology‑influenced vector, and finally evaluates
the surrogate RBF model.

The implementation below fuses the governing equations of both parents:
- Gaussian RBF kernel (`gaussian`) and linear system solve (`solve_linear`).
- Morphology metrics (`sphericity_index`, `flatness_index`, `recovery_priority`).
- Hyperdimensional vector generation (`morphology_influenced_vector`,
  `symbol_vector`, `bind`).
- Perceptual hashing and clustering (`compute_phash`, `cluster_by_phash`).
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, List, Dict

import numpy as np

Vector = Sequence[float]

# ----------------------------------------------------------------------
# Parent A utilities (perceptual hashing, RBF kernel, linear solve)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def compute_phash(values: List[float]) -> int:
    """Average‑based perceptual hash (max 64 bits)."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def cluster_by_phash(hashes: Dict[str, int], max_distance: int = 4) -> List[List[str]]:
    """Agglomerative clustering based on Hamming distance of perceptual hashes."""
    clusters: List[List[str]] = []
    for key, h in hashes.items():
        placed = False
        for c in clusters:
            if hamming_distance(h, hashes[c[0]]) <= max_distance:
                c.append(key)
                placed = True
                break
        if not placed:
            clusters.append([key])
    return clusters

def solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b using NumPy's linear solver (equivalent to Gaussian elimination)."""
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)
    x = np.linalg.solve(A_np, b_np)
    return x.tolist()

# ----------------------------------------------------------------------
# Parent B utilities (morphology, hyperdimensional vectors, binding)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1] based on righting time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def random_vector(dim: int = 10000, seed: str | int | None = None) -> np.ndarray:
    rng = random.Random(seed)
    return np.array([1 if rng.getrandbits(1) else -1 for _ in range(dim)], dtype=int)

def symbol_vector(symbol: str, dim: int = 10000) -> np.ndarray:
    seed_bytes = hashlib.sha256(symbol.encode('utf-8')).digest()[:8]
    seed = int.from_bytes(seed_bytes, 'big')
    return random_vector(dim, seed)

def morphology_influenced_vector(m: Morphology, dim: int = 10000) -> np.ndarray:
    si = sphericity_index(m.length, m.width, m.height)
    seed = int(si * 1_000_000)  # scale to get variability
    return random_vector(dim, seed)

def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Binding for bipolar vectors: element‑wise multiplication."""
    if a.shape != b.shape:
        raise ValueError('vectors must have equal shape')
    return a * b

# ----------------------------------------------------------------------
# Hybrid data structures and core algorithms
# ----------------------------------------------------------------------
@dataclass
class DataPoint:
    """Container for a single observation."""
    id: str
    features: List[float]          # raw numeric descriptor
    morphology: Morphology
    symbol: str                    # optional symbolic tag

def build_hybrid_model(data: List[DataPoint],
                       dim: int = 10000,
                       epsilon: float = 1.0) -> Dict:
    """
    Train the hybrid surrogate:
    1. Compute perceptual hashes of feature vectors.
    2. Cluster by hash (Hamming distance ≤ 4).
    3. For each cluster compute:
       - Representative feature (mean of features).
       - Representative morphology (mean of each field).
       - Hyperdimensional prototype = bind(symbol_vector, morphology_influenced_vector).
    4. Assemble RBF kernel K_ij = gaussian(||p_i - p_j||) where p_i are the prototypes.
    5. Target y_i = recovery_priority(mean_morphology_i).
    6. Solve K w = y for weights w.
    Returns a dictionary containing all intermediate structures needed for prediction.
    """
    # 1. Hashes
    hashes = {pt.id: compute_phash(pt.features) for pt in data}
    # 2. Clustering
    clusters = cluster_by_phash(hashes)

    # Containers for per‑cluster aggregates
    proto_vectors: List[np.ndarray] = []
    targets: List[float] = []
    cluster_ids: List[List[str]] = []

    for cluster in clusters:
        cluster_pts = [next(p for p in data if p.id == pid) for pid in cluster]
        # Representative feature (mean)
        mean_feat = [sum(p.features[i] for p in cluster_pts) / len(cluster_pts)
                     for i in range(len(cluster_pts[0].features))]
        # Representative morphology (mean of each field)
        mean_morph = Morphology(
            length=sum(p.morphology.length for p in cluster_pts) / len(cluster_pts),
            width=sum(p.morphology.width for p in cluster_pts) / len(cluster_pts),
            height=sum(p.morphology.height for p in cluster_pts) / len(cluster_pts),
            mass=sum(p.morphology.mass for p in cluster_pts) / len(cluster_pts),
        )
        # Symbol for the cluster: concatenate ids (deterministic)
        cluster_symbol = "_".join(sorted(cluster))
        # Hyperdimensional prototype
        proto = bind(symbol_vector(cluster_symbol, dim),
                     morphology_influenced_vector(mean_morph, dim))
        proto_vectors.append(proto.astype(float))  # cast to float for Euclidean distance
        # Target from morphology
        targets.append(recovery_priority(mean_morph))
        cluster_ids.append(cluster)

    # 4. RBF kernel matrix
    n = len(proto_vectors)
    K = [[gaussian(euclidean(proto_vectors[i], proto_vectors[j]), epsilon)
          for j in range(n)] for i in range(n)]

    # 5 & 6. Solve for weights
    weights = solve_linear(K, targets)

    # Package model
    model = {
        "proto_vectors": proto_vectors,
        "weights": weights,
        "clusters": cluster_ids,
        "hashes": hashes,
        "epsilon": epsilon,
        "dim": dim,
    }
    return model

def hybrid_predict(model: Dict, point: DataPoint) -> float:
    """
    Predict the recovery priority for a new DataPoint.
    Steps:
    1. Compute its perceptual hash and locate the nearest cluster (Hamming distance).
    2. Build its hyperdimensional representation via binding.
    3. Evaluate the RBF surrogate using the trained prototypes and weights.
    Returns the predicted priority (float in [0,1]).
    """
    # 1. Find nearest cluster by hash
    point_hash = compute_phash(point.features)
    min_dist = sys.maxsize
    nearest_idx = -1
    for idx, cluster in enumerate(model["clusters"]):
        # compare against the first member of each cluster (representative)
        rep_id = cluster[0]
        dist = hamming_distance(point_hash, model["hashes"][rep_id])
        if dist < min_dist:
            min_dist = dist
            nearest_idx = idx

    # 2. Hyperdimensional representation of the point
    proto = bind(symbol_vector(point.id, model["dim"]),
                 morphology_influenced_vector(point.morphology, model["dim"])).astype(float)

    # 3. RBF evaluation
    eps = model["epsilon"]
    k_vals = [gaussian(euclidean(proto, p), eps) for p in model["proto_vectors"]]
    prediction = sum(w * k for w, k in zip(model["weights"], k_vals))
    # Clamp to [0,1] for safety
    return max(0.0, min(1.0, prediction))

def hybrid_demo() -> None:
    """Simple demonstration constructing a model and predicting on a fresh point."""
    # Synthetic dataset
    rng = random.Random(42)
    data = []
    for i in range(20):
        morph = Morphology(
            length=rng.uniform(0.5, 2.0),
            width=rng.uniform(0.5, 2.0),
            height=rng.uniform(0.5, 2.0),
            mass=rng.uniform(0.1, 5.0),
        )
        features = [rng.random() for _ in range(10)]
        dp = DataPoint(
            id=f"pt{i}",
            features=features,
            morphology=morph,
            symbol=f"S{i}"
        )
        data.append(dp)

    model = build_hybrid_model(data, dim=2048, epsilon=0.5)

    # New point (similar to pt0)
    test_point = DataPoint(
        id="test",
        features=data[0].features[:],  # copy of first point's features
        morphology=data[0].morphology,
        symbol="TEST"
    )
    pred = hybrid_predict(model, test_point)
    print(f"Predicted recovery priority for test point: {pred:.4f}")

if __name__ == "__main__":
    hybrid_demo()