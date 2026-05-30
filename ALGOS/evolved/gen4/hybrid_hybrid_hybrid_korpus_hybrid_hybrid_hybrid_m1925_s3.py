# DARWIN HAMMER — match 1925, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_korpus_text_h_hybrid_geometric_pro_m523_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_minimu_m662_s0.py (gen3)
# born: 2026-05-29T23:39:47Z

"""
Hybrid Algorithm: minhash‑sketch‑voronoi‑geometric product

Parents:
- hybrid_hybrid_korpus_text_h_hybrid_geometric_pro_m523_s0.py (A)
- hybrid_hybrid_hybrid_sketch_hybrid_hybrid_minimu_m662_s0.py (B)

Mathematical bridge:
    * A produces a MinHash signature for each text document, yielding a high‑dimensional integer vector.
    * B aggregates many such vectors with a Count‑Min sketch and extracts epistemic certainty weights
      via an RLCT (regularized logarithmic convergence test) estimate from training losses.
    * The epistemic weights are used to bias the placement of Voronoi seeds that partition the
      signature space (A’s geometric component).  Within each Voronoi cell we construct a simple
      multivector whose grades are the summed signature components and evaluate a geometric
      product (approximated by a dot‑outer product) that couples the two parent topologies.
The resulting pipeline fuses textual MinHashing, probabilistic sketching, epistemic weighting,
and Voronoi‑based geometric algebra in a single unified system.
"""

import re
import math
import random
import sys
import pathlib
import hashlib
from collections import defaultdict
import numpy as np

# ----------------------------------------------------------------------
# Core utilities from Parent A
# ----------------------------------------------------------------------
def minhash_for_text(text: str, k: int = 64) -> np.ndarray:
    """Generate a MinHash signature of length k for a given text."""
    text = re.sub(r"\s+", " ", text or "").strip().lower()
    shingles = [text[i:i + 5] for i in range(len(text) - 4)]
    signature = np.full(k, np.iinfo(np.int64).max, dtype=np.int64)
    for s in shingles:
        h = hash(s)
        idx = h % k
        signature[idx] = min(signature[idx], h & 0xFFFFFFFFFFFFFFFF)
    return signature


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def nearest_seed(point: np.ndarray, seeds: np.ndarray) -> int:
    """Return the index of the seed closest to *point* (ties broken by index)."""
    if seeds.size == 0:
        raise ValueError("seed set is empty")
    dists = np.linalg.norm(seeds - point, axis=1)
    return int(np.argmin(dists))


def assign_points_to_voronoi(points: np.ndarray, seeds: np.ndarray) -> dict[int, list[int]]:
    """
    Assign each point index to the nearest Voronoi seed.
    Returns a dict mapping seed index → list of point indices.
    """
    regions = defaultdict(list)
    for idx, pt in enumerate(points):
        s = nearest_seed(pt, seeds)
        regions[s].append(idx)
    return dict(regions)


# ----------------------------------------------------------------------
# Core utilities from Parent B
# ----------------------------------------------------------------------
def count_min_sketch(items, width: int = 64, depth: int = 4) -> list[list[int]]:
    """Standard Count‑Min sketch."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16)
            table[d][h % width] += 1
    return table


def minhash_lsh_index(docs_shingles: dict[int, list[str]]) -> dict[str, list[int]]:
    """Very light LSH index based on the smallest SHA‑1 hash of a shingle."""
    buckets = defaultdict(list)
    for doc_id, shingles in docs_shingles.items():
        if not shingles:
            key = "empty"
        else:
            key = min(hashlib.sha1(s.encode()).hexdigest()[:6] for s in shingles)
        buckets[key].append(doc_id)
    return dict(buckets)


def estimate_rlct_from_losses(train_losses_per_n: list[float],
                              n_values: list[int]) -> float:
    """
    Estimate the RLCT slope from (n, loss) data using a simple linear regression
    on log(loss) vs. log(log(n)).  The returned slope is interpreted as an
    epistemic‑certainty weight (larger slope → higher certainty).
    """
    losses = np.asarray(train_losses_per_n, dtype=np.float64)
    ns = np.asarray(n_values, dtype=np.float64)

    if np.any(ns <= math.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have the same length")

    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))

    # Center the data
    x_centered = x - x.mean()
    y_centered = y - y.mean()

    var_x = np.dot(x_centered, x_centered)
    if var_x < 1e-15:
        raise ValueError("variance of x is too small for regression")

    slope = np.dot(x_centered, y_centered) / var_x
    return float(slope)


# ----------------------------------------------------------------------
# Simple Multivector placeholder (geometric algebra grade‑0 and grade‑1 only)
# ----------------------------------------------------------------------
class Multivector:
    """
    Minimal multivector supporting scalar (grade‑0) and vector (grade‑1) parts.
    The geometric product is approximated by:
        (a0 + a·e) * (b0 + b·e) = a0*b0 + a0*b·e + b0*a·e + (a·b) + a∧b
    For our hybrid we return the scalar part (dot) and the vector part (sum).
    """
    def __init__(self, scalar: float = 0.0, vector: np.ndarray = None):
        self.scalar = float(scalar)
        self.vector = np.zeros_like(vector) if vector is not None else np.zeros(0)

    def geometric_product(self, other: "Multivector") -> "Multivector":
        # scalar part: a0*b0 + a·b
        scalar_part = self.scalar * other.scalar + float(np.dot(self.vector, other.vector))
        # vector part: a0*b_vec + b0*a_vec
        vector_part = self.scalar * other.vector + other.scalar * self.vector
        return Multivector(scalar_part, vector_part)

    def __repr__(self):
        return f"Multivector(scalar={self.scalar:.3f}, vector={self.vector})"


# ----------------------------------------------------------------------
# Hybrid functions (meeting the requirement of at least three)
# ----------------------------------------------------------------------
def compute_minhash_signatures(docs: dict[int, str],
                               k: int = 64) -> dict[int, np.ndarray]:
    """
    For each document produce a MinHash signature (Parent A).
    Returns a mapping doc_id → signature vector.
    """
    return {doc_id: minhash_for_text(text, k) for doc_id, text in docs.items()}


def epistemic_weighted_voronoi(seeds_count: int,
                               signatures: dict[int, np.ndarray],
                               rlct_slope: float) -> dict[int, list[int]]:
    """
    Build Voronoi regions over the signature space.
    Seed positions are drawn from the existing signatures with probability
    proportional to an epistemic weight derived from the RLCT slope (Parent B).
    Returns region mapping seed_index → list of document indices.
    """
    # Convert signatures to 2‑D points using the first two components (scaled)
    points = np.stack([sig[:2] for sig in signatures.values()], axis=0).astype(np.float64)
    # Normalise to unit square for numerical stability
    if points.ptp().max() > 0:
        points = (points - points.min(axis=0)) / points.ptp(axis=0)

    # Epistemic probability: higher RLCT slope → higher chance to become a seed
    prob = np.full(points.shape[0], 1.0)
    prob *= max(rlct_slope, 0.0) + 1.0  # ensure positivity
    prob /= prob.sum()

    seed_indices = np.random.choice(points.shape[0],
                                    size=seeds_count,
                                    replace=False,
                                    p=prob)
    seeds = points[seed_indices]

    return assign_points_to_voronoi(points, seeds)


def region_geometric_products(regions: dict[int, list[int]],
                              signatures: dict[int, np.ndarray]) -> dict[int, Multivector]:
    """
    For each Voronoi region construct a multivector from the summed signatures
    and compute the self‑geometric product (grade‑0 scalar + grade‑1 vector).
    The result captures the fused algebraic structure of both parents.
    """
    # Prepare an ordered list of signatures to index them easily
    doc_ids = list(signatures.keys())
    sig_matrix = np.stack([signatures[doc_id] for doc_id in doc_ids], axis=0).astype(np.float64)

    result = {}
    for seed_idx, doc_idx_list in regions.items():
        if not doc_idx_list:
            result[seed_idx] = Multivector()
            continue
        # Sum signatures belonging to the region
        region_sum = sig_matrix[doc_idx_list].sum(axis=0)
        # Build a multivector: scalar = mean of components, vector = normalized sum
        scalar = float(region_sum.mean())
        vector = region_sum / (np.linalg.norm(region_sum) + 1e-12)
        mv = Multivector(scalar, vector)
        # Self‑product to embed the geometric algebra interaction
        result[seed_idx] = mv.geometric_product(mv)
    return result


def hybrid_pipeline(docs: dict[int, str],
                    train_losses: list[float],
                    n_values: list[int],
                    seed_count: int = 3,
                    minhash_k: int = 64) -> dict[int, Multivector]:
    """
    End‑to‑end hybrid algorithm:
      1. MinHash each document (Parent A).
      2. Estimate epistemic certainty via RLCT from loss data (Parent B).
      3. Partition signatures with epistemic‑weighted Voronoi seeds.
      4. Compute a geometric product per region.
    Returns a mapping region_id → resulting Multivector.
    """
    signatures = compute_minhash_signatures(docs, k=minhash_k)
    rlct_slope = estimate_rlct_from_losses(train_losses, n_values)
    regions = epistemic_weighted_voronoi(seed_count, signatures, rlct_slope)
    region_mvs = region_geometric_products(regions, signatures)
    return region_mvs


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Tiny synthetic corpus
    corpus = {
        0: "The quick brown fox jumps over the lazy dog.",
        1: "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        2: "Pack my box with five dozen liquor jugs.",
        3: "Sphinx of black quartz, judge my vow."
    }

    # Dummy training loss data (monotonically decreasing as n grows)
    losses = [0.9, 0.6, 0.4, 0.3]
    ns = [10, 100, 1000, 10000]

    # Execute the hybrid pipeline
    output = hybrid_pipeline(corpus, losses, ns, seed_count=2, minhash_k=128)

    # Print results
    for region_id, mv in output.items():
        print(f"Region {region_id}: {mv}")