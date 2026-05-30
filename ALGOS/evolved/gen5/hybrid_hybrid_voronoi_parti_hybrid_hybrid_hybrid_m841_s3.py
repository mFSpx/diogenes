# DARWIN HAMMER — match 841, survivor 3
# gen: 5
# parent_a: hybrid_voronoi_partition_hybrid_hybrid_hybrid_m325_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m264_s0.py (gen4)
# born: 2026-05-29T23:31:20Z

"""Hybrid Voronoi‑RBF‑Associative Memory

Parents:
- `voronoi_dense_associative_sheaf.py` (Voronoi partition + dense associative memory)
- `hybrid_hybrid_hybrid_hybrid_perceptual_de_m264_s0.py` (regex feature extraction + RBF surrogate)

Mathematical bridge:
Both parents rely on Euclidean distances.  The Voronoi step assigns a query point to
seed‑centroids using the same distance metric that the radial‑basis function (RBF)
uses to compute similarity.  We therefore compute RBF weights from the distances
to *all* seeds, modulate those weights by a scalar derived from regex‑based feature
scores, and finally blend the linear associative‑memory readouts stored in a
`Sheaf` (one memory matrix per seed).  The resulting vector is a distance‑aware,
feature‑aware retrieval from a distributed associative memory."""

import numpy as np
import math
import random
import sys
import pathlib
import re
from typing import List, Tuple, Dict, Any

# ----------------------------------------------------------------------
# Geometry utilities (from Parent A)
# ----------------------------------------------------------------------
def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def nearest(point: np.ndarray, seeds: np.ndarray) -> int:
    """Index of the closest seed to *point*."""
    if not seeds.size:
        raise ValueError('seeds required')
    return np.argmin(np.apply_along_axis(lambda x: distance(point, x), 1, seeds))


def assign(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Return a binary region matrix R of shape (n_seeds, n_points) where
    R[i, j] == 1 iff point j belongs to the Voronoi cell of seed i.
    """
    n_seeds = seeds.shape[0]
    n_pts = points.shape[0]
    regions = np.zeros((n_seeds, n_pts), dtype=int)
    for j, p in enumerate(points):
        i = nearest(p, seeds)
        regions[i, j] = 1
    return regions


# ----------------------------------------------------------------------
# Radial‑basis utilities (from Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """Euclidean distance for plain tuples (used by the original RBF code)."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_feature_scores(text: str) -> Dict[str, float]:
    """
    Regex‑based feature extraction.  Returns a dict mapping feature names to
    integer counts (treated as scores).
    """
    feature_scores: Dict[str, float] = {}
    feature_scores["evidence"] = len(
        re.findall(
            r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
            text,
            re.I,
        )
    )
    feature_scores["planning"] = len(
        re.findall(
            r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
            text,
            re.I,
        )
    )
    feature_scores["delay"] = len(
        re.findall(r"\b(?:delay|postpone|defer|wait|stall|hold|slow)\b", text, re.I)
    )
    return feature_scores


# ----------------------------------------------------------------------
# Dense Associative Memory expressed as a Sheaf (simplified)
# ----------------------------------------------------------------------
class Sheaf:
    """
    A minimal sheaf where each node (identified by a seed index) stores a
    dense associative memory matrix `W` of shape (dim, dim).  A *section* is a
    vector stored at the node; retrieval multiplies the stored matrix by a cue.
    """

    def __init__(self, node_ids: List[int], dim: int):
        self.dim = dim
        self.node_ids = list(node_ids)
        # Randomly initialise associative matrices; in a real system they would be trained.
        self._weights: Dict[int, np.ndarray] = {
            nid: np.random.randn(dim, dim) for nid in self.node_ids
        }
        # Sections (optional stored vectors) – initialise to zeros.
        self._sections: Dict[int, np.ndarray] = {
            nid: np.zeros(dim) for nid in self.node_ids
        }

    def set_section(self, node: int, value: np.ndarray) -> None:
        """Store a vector at a node."""
        if value.shape != (self.dim,):
            raise ValueError("section must have shape (dim,)")
        self._sections[node] = np.asarray(value, dtype=float)

    def retrieve(self, node: int, cue: np.ndarray) -> np.ndarray:
        """
        Linear associative readout:  output = W_node @ cue .
        """
        if cue.shape != (self.dim,):
            raise ValueError("cue must have shape (dim,)")
        W = self._weights[node]
        return W @ cue

    def blend_retrievals(self, cues: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Blend associative reads from all nodes.
        *cues* – (n_nodes, dim) array of cue vectors (typically the same cue repeated).
        *weights* – (n_nodes,) non‑negative scalars that sum to 1 (or are normalised internally).
        Returns a single (dim,) vector.
        """
        if cues.shape != (len(self.node_ids), self.dim):
            raise ValueError("cues shape mismatch")
        if weights.shape != (len(self.node_ids),):
            raise ValueError("weights shape mismatch")
        # Normalise weights to avoid numerical blow‑up.
        w = weights / (weights.sum() + 1e-12)
        out = np.zeros(self.dim)
        for idx, nid in enumerate(self.node_ids):
            out += w[idx] * self.retrieve(nid, cues[idx])
        return out


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def build_hybrid_sheaf(points: np.ndarray, seeds: np.ndarray, dim: int) -> Sheaf:
    """
    Construct a Sheaf whose nodes correspond to the Voronoi seeds.
    Each node receives a random associative matrix.
    Optionally, we store as a section the centroid of its Voronoi cell
    (useful as a cue).
    """
    sheaf = Sheaf(node_ids=list(range(seeds.shape[0])), dim=dim)

    # Compute Voronoi assignment matrix.
    region_mat = assign(points, seeds)

    # For each seed, compute the centroid of its assigned points (if any)
    # and store it as the node's section.
    for i in range(seeds.shape[0]):
        mask = region_mat[i].astype(bool)
        if mask.any():
            centroid = points[mask].mean(axis=0)
        else:
            centroid = seeds[i]  # fallback to the seed itself
        sheaf.set_section(i, centroid)
    return sheaf


def rbf_weights_from_seeds(query: np.ndarray, seeds: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Compute Gaussian RBF weights for a query point with respect to all seeds.
    Returns a (n_seeds,) array.
    """
    dists = np.linalg.norm(seeds - query, axis=1)
    return np.vectorize(lambda r: gaussian(r, epsilon))(dists)


def feature_modulation_factor(text: str) -> float:
    """
    Convert regex feature scores into a scalar multiplier.
    The factor is 1 + (total_score / (total_score + 1)), i.e. lies in (1, 2).
    """
    scores = compute_feature_scores(text)
    total = sum(scores.values())
    return 1.0 + (total / (total + 1.0))


def query_hybrid(
    sheaf: Sheaf,
    points: np.ndarray,
    seeds: np.ndarray,
    query_point: np.ndarray,
    text: str,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Perform a hybrid retrieval:

    1. Compute RBF similarity of the query to each seed.
    2. Modulate the similarities by a factor derived from regex feature scores.
    3. Use the (modulated) similarities as blending weights for the associative
       memory reads stored in the sheaf.
    4. The cue supplied to each node is the node's stored section (its centroid).

    Returns a (dim,) vector representing the fused memory output.
    """
    # 1. Base RBF weights
    base_weights = rbf_weights_from_seeds(query_point, seeds, epsilon)

    # 2. Feature‑driven modulation
    mod_factor = feature_modulation_factor(text)
    mod_weights = base_weights * mod_factor

    # 3. Prepare cues (centroids) for each node
    cues = np.stack([sheaf._sections[nid] for nid in sheaf.node_ids])

    # 4. Blend associative reads
    result = sheaf.blend_retrievals(cues, mod_weights)
    return result


def evaluate_partition(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper that returns the Voronoi region matrix.
    """
    return assign(points, seeds)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Seed reproducibility
    random.seed(0)
    np.random.seed(0)

    # Synthetic data
    dim = 5
    n_points = 200
    n_seeds = 7

    points = np.random.randn(n_points, dim)
    seeds = np.random.randn(n_seeds, dim)

    # Build the hybrid sheaf
    sheaf = build_hybrid_sheaf(points, seeds, dim)

    # Random query and text
    query = np.random.randn(dim)
    sample_text = (
        "The evidence confirms the plan, but there is a delay due to scheduling issues."
    )

    # Hybrid query
    out_vec = query_hybrid(sheaf, points, seeds, query, sample_text, epsilon=0.8)

    # Simple sanity checks
    assert out_vec.shape == (dim,)
    region_mat = evaluate_partition(points, seeds)
    assert region_mat.shape == (n_seeds, n_points)

    print("Hybrid output vector:", out_vec)
    print("Voronoi region matrix shape:", region_mat.shape)
    print("Smoke test completed successfully.")