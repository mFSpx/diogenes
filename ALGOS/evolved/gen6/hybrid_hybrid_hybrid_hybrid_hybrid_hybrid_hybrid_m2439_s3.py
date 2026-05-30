# DARWIN HAMMER — match 2439, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c_m633_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_sketch_m1023_s0.py (gen4)
# born: 2026-05-29T23:42:19Z

"""Hybrid Voronoi‑Decision‑Hygiene Algorithm
Parents:
* hybrid_hybrid_hybrid_vorono_hybrid_liquid_time_c_m633_s0.py
* hybrid_hybrid_hybrid_decisi_hybrid_hybrid_sketch_m1023_s0.py

Mathematical bridge:
The 2‑D space of points is partitioned by a Voronoi diagram.  Each region r obtains a
region‑level hidden state h_r that evolves according to a liquid‑time‑constant ODE  

    dh_r/dt = -h_r / τ_r + f_r ,

where τ_r is an input‑dependent time constant.  τ_r is derived from a hyper‑dimensional
binding between the region centroid c_r and a compact sketch summary s_r (produced by a
count‑min sketch of the feature vectors that fall inside the region).  The binding is
implemented as an element‑wise product of binary hyper‑vectors; the cosine similarity of
the bound vector with the centroid vector yields a scaling factor for τ_r.

For each data point the decision‑hygiene score is a Shannon‑entropy weighted sum of its
features.  Feature weights are adjusted by a Hoeffding bound that uses the sketch counts
as the empirical frequencies, thus coupling the dimensionality‑reduction side to the
entropy‑based scoring side.

The three core functions below demonstrate this fused mathematics:
1. `voronoi_assign` – builds the Voronoi partition.
2. `update_region_state` – liquid‑time‑constant ODE with τ derived from hyper‑dimensional
   binding of centroid and sketch.
3. `hygiene_score_with_sketch` – entropy weighting combined with Hoeffding‑adjusted
   feature importance. """

import math
import random
import sys
import pathlib
import numpy as np
from collections import defaultdict
import hashlib

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = tuple[float, float]          # 2‑D coordinate
FeatureVec = np.ndarray              # 1‑D float array
HyperVector = np.ndarray             # binary hyper‑vector (uint8)

# ----------------------------------------------------------------------
# Voronoi utilities
# ----------------------------------------------------------------------
def _euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _nearest_region(p: Point, seeds: list[Point]) -> int:
    """Return index of the nearest seed (ties broken by index)."""
    if not seeds:
        raise ValueError("seed list is empty")
    return min(range(len(seeds)), key=lambda i: (_euclidean(p, seeds[i]), i))

def voronoi_assign(points: list[Point], seeds: list[Point]) -> dict[int, list[int]]:
    """
    Assign each point index to the nearest seed.
    Returns a mapping region_id -> list of point indices.
    """
    regions: dict[int, list[int]] = {i: [] for i in range(len(seeds))}
    for idx, p in enumerate(points):
        r = _nearest_region(p, seeds)
        regions[r].append(idx)
    return regions

# ----------------------------------------------------------------------
# Count‑Min Sketch (integer counts)
# ----------------------------------------------------------------------
def count_min_sketch(items: list[int], width: int = 64, depth: int = 4) -> list[list[int]]:
    """
    Simple count‑min sketch.
    `items` are integer identifiers (e.g. hashed feature bins).
    Returns a 2‑D list `depth × width` of counts.
    """
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16)
            table[d][h % width] += 1
    return table

def sketch_to_vector(sketch: list[list[int]]) -> np.ndarray:
    """
    Collapse a sketch into a dense vector by averaging across hash rows.
    The result is a real‑valued vector suitable for binding.
    """
    arr = np.array(sketch, dtype=np.float64)
    return arr.mean(axis=0)  # shape (width,)

# ----------------------------------------------------------------------
# Hyper‑dimensional binding utilities
# ----------------------------------------------------------------------
def random_hypervector(dim: int = 1024, density: float = 0.5) -> HyperVector:
    """Generate a random binary hyper‑vector."""
    bits = np.random.rand(dim) < density
    return bits.astype(np.uint8)

def bind(v1: HyperVector, v2: HyperVector) -> HyperVector:
    """Element‑wise XOR binding (commutative)."""
    return np.bitwise_xor(v1, v2)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two real vectors."""
    if a.size == 0 or b.size == 0:
        return 0.0
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if den == 0 else num / den

# ----------------------------------------------------------------------
# Liquid‑time‑constant region dynamics
# ----------------------------------------------------------------------
def compute_tau(
    centroid: Point,
    sketch_vec: np.ndarray,
    base_tau: float = 1.0,
    dim: int = 1024,
) -> float:
    """
    Derive an input‑dependent time constant τ.
    - Convert centroid to a binary hyper‑vector via hashing.
    - Bind centroid vector with sketch summary.
    - Scale base_tau by (1 + similarity) ∈ [1,2].
    """
    # hash centroid to integer then to binary hyper‑vector
    cx = int(hashlib.sha256(f"{centroid[0]:.6f},{centroid[1]:.6f}".encode()).hexdigest(), 16)
    rnd = np.random.RandomState(cx % (2**32))
    centroid_hv = (rnd.rand(dim) < 0.5).astype(np.uint8)

    # bind with sketch summary (treated as binary by thresholding)
    sketch_hv = (sketch_vec > sketch_vec.mean()).astype(np.uint8)
    bound = bind(centroid_hv, sketch_hv)

    # similarity with original centroid hyper‑vector
    sim = cosine_similarity(centroid_hv.astype(np.float64), bound.astype(np.float64))
    return base_tau * (1.0 + sim)  # τ ∈ [τ, 2τ]

def update_region_state(
    h_prev: float,
    input_signal: float,
    tau: float,
    dt: float = 0.1,
) -> float:
    """
    One Euler integration step of dh/dt = -h/τ + input_signal.
    """
    dh = (-h_prev / tau + input_signal) * dt
    return h_prev + dh

# ----------------------------------------------------------------------
# Decision‑hygiene scoring with entropy & Hoeffding bound
# ----------------------------------------------------------------------
def shannon_entropy(col: np.ndarray) -> float:
    """Entropy of a discrete distribution given by column counts."""
    probs = col / col.sum() if col.sum() > 0 else np.zeros_like(col)
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log2(probs)))

def hygiene_score_with_sketch(
    features: np.ndarray,               # shape (n_samples, n_features)
    sketch: list[list[int]],
    delta: float = 0.05,
) -> np.ndarray:
    """
    Compute a hygiene score for each sample.
    Steps:
    1. Column‑wise entropy → base weights w_i.
    2. Convert sketch to frequency estimates f_i.
    3. Hoeffding bound ε_i = sqrt( (1/(2*N_i)) * log(2/delta) ),
       where N_i is the total count for feature i in the sketch.
    4. Adjust weights: w_i' = w_i * (1 - ε_i)  (down‑weight uncertain features).
    5. Score_i = sum_j w'_j * feature_{i,j}.
    Returns a 1‑D array of scores.
    """
    n_samples, n_feat = features.shape
    # 1. entropy weights
    entropies = np.apply_along_axis(shannon_entropy, 0, features)
    base_weights = entropies / (entropies.sum() + 1e-12)

    # 2. sketch frequencies per feature (approximate by column sums of sketch)
    sketch_arr = np.array(sketch, dtype=np.float64)  # depth × width
    freq_est = sketch_arr.mean(axis=0)  # length = width
    # Map width to feature dimension (simple modulo)
    N_i = np.array([freq_est[i % freq_est.size] for i in range(n_feat)]) + 1e-9

    # 3. Hoeffding epsilon
    eps = np.sqrt((1.0 / (2.0 * N_i)) * np.log(2.0 / delta))

    # 4. adjusted weights
    adj_weights = base_weights * (1.0 - eps)
    adj_weights = np.clip(adj_weights, 0, None)  # no negative weights
    adj_weights /= (adj_weights.sum() + 1e-12)

    # 5. scores
    scores = features @ adj_weights
    return scores

# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_process(
    points: list[Point],
    seeds: list[Point],
    feature_matrix: np.ndarray,
    dt: float = 0.1,
    base_tau: float = 1.0,
) -> dict[int, dict]:
    """
    Execute the fused algorithm.
    Returns a dict region_id -> {
        'hidden_state': final h,
        'centroid': region centroid,
        'scores': hygiene scores of points belonging to the region
    }.
    """
    # 1. Voronoi assignment
    regions = voronoi_assign(points, seeds)

    # 2. Prepare output container
    results: dict[int, dict] = {}

    # 3. Process each region independently
    for r_id, idx_list in regions.items():
        if not idx_list:
            # empty region – keep default state
            results[r_id] = {
                'hidden_state': 0.0,
                'centroid': seeds[r_id],
                'scores': np.array([]),
            }
            continue

        # Region centroid (average of point coordinates)
        pts = np.array([points[i] for i in idx_list], dtype=np.float64)
        centroid = tuple(pts.mean(axis=0))

        # Sketch of feature vectors (hash each feature value to an int bucket)
        # Simple binning: cast float to int after scaling.
        feature_subset = feature_matrix[idx_list]
        hashed_items = (feature_subset * 1e3).astype(np.int64).ravel().tolist()
        sketch = count_min_sketch(hashed_items)

        # Convert sketch to vector for τ computation
        sketch_vec = sketch_to_vector(sketch)

        # Compute τ for this region
        tau = compute_tau(centroid, sketch_vec, base_tau=base_tau)

        # Initialise hidden state (zero) and run a single update using mean feature norm as input
        input_signal = float(np.linalg.norm(feature_subset.mean(axis=0)))
        h = update_region_state(0.0, input_signal, tau, dt=dt)

        # Hygiene scores for points in the region
        scores = hygiene_score_with_sketch(feature_subset, sketch)

        results[r_id] = {
            'hidden_state': h,
            'centroid': centroid,
            'scores': scores,
        }

    return results

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # generate synthetic data
    random.seed(0)
    np.random.seed(0)

    # 20 random 2‑D points
    points = [(random.random(), random.random()) for _ in range(20)]

    # 4 random seeds (Voronoi generators)
    seeds = [(random.random(), random.random()) for _ in range(4)]

    # 20 samples, 8 features each, values in [0,1)
    feature_matrix = np.random.rand(20, 8)

    # run the hybrid algorithm
    out = hybrid_process(points, seeds, feature_matrix)

    # simple sanity check: each region should have a hidden_state float and scores array
    for rid, info in out.items():
        assert isinstance(info['hidden_state'], float)
        assert isinstance(info['centroid'], tuple) and len(info['centroid']) == 2
        assert isinstance(info['scores'], np.ndarray)
    print("Hybrid process completed successfully. Region summary:")
    for rid, info in out.items():
        print(f"Region {rid}: hidden_state={info['hidden_state']:.4f}, "
              f"num_points={len(info['scores'])}, avg_score={info['scores'].mean() if info['scores'].size else float('nan'):.4f}")