# DARWIN HAMMER — match 841, survivor 4
# gen: 5
# parent_a: hybrid_voronoi_partition_hybrid_hybrid_hybrid_m325_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m264_s0.py (gen4)
# born: 2026-05-29T23:31:20Z

import numpy as np
import math
import re
from typing import List, Tuple, Dict, Any

# ----------------------------------------------------------------------
# Geometry utilities (shared)
# ----------------------------------------------------------------------
def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def nearest(point: np.ndarray, seeds: np.ndarray) -> int:
    """Index of the closest seed to *point*."""
    if seeds.size == 0:
        raise ValueError("seeds array is empty")
    return int(np.argmin(np.linalg.norm(seeds - point, axis=1)))


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
# Feature extraction (regex based)
# ----------------------------------------------------------------------
_FEATURE_PATTERNS: Dict[str, re.Pattern] = {
    "evidence": re.compile(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
        re.I,
    ),
    "planning": re.compile(
        r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
        re.I,
    ),
    "delay": re.compile(r"\b(?:delay|postpone|defer|wait|stall|hold|slow)\b", re.I),
}


def compute_feature_scores(text: str) -> Dict[str, float]:
    """
    Count occurrences of each regex pattern in *text*.
    Returns a dict mapping feature names to counts.
    """
    scores: Dict[str, float] = {}
    for name, pat in _FEATURE_PATTERNS.items():
        scores[name] = float(len(pat.findall(text)))
    return scores


def normalize_feature_vector(vec: np.ndarray) -> np.ndarray:
    """L2‑normalize a feature vector; returns zeros if norm is zero."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def text_to_feature_vector(text: str, feature_order: List[str]) -> np.ndarray:
    """
    Convert *text* into a fixed‑order, L2‑normalized feature vector.
    """
    scores = compute_feature_scores(text)
    raw = np.array([scores.get(name, 0.0) for name in feature_order], dtype=float)
    return normalize_feature_vector(raw)


# ----------------------------------------------------------------------
# Dense Associative Memory expressed as a Sheaf (simplified)
# ----------------------------------------------------------------------
class Sheaf:
    """
    Minimal sheaf where each node (identified by a seed index) stores:
      * a dense associative matrix W of shape (dim, dim)
      * a *section* vector (e.g., the centroid of its Voronoi cell)
    Retrieval blends per‑node linear reads using supplied weights.
    """

    def __init__(self, node_ids: List[int], dim: int):
        self.dim = dim
        self.node_ids = list(node_ids)
        self._weights: Dict[int, np.ndarray] = {
            nid: np.random.randn(dim, dim) for nid in self.node_ids
        }
        self._sections: Dict[int, np.ndarray] = {
            nid: np.zeros(dim, dtype=float) for nid in self.node_ids
        }

    def set_section(self, node: int, value: np.ndarray) -> None:
        """Store a vector at a node."""
        if value.shape != (self.dim,):
            raise ValueError("section must have shape (dim,)")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: int) -> np.ndarray:
        """Retrieve the stored section vector."""
        return self._sections[node]

    def retrieve(self, node: int, cue: np.ndarray) -> np.ndarray:
        """
        Linear associative readout:  output = W_node @ cue .
        """
        if cue.shape != (self.dim,):
            raise ValueError("cue must have shape (dim,)")
        return self._weights[node] @ cue

    def blend(self, cues: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Blend associative reads from all nodes.
        *cues* – (n_nodes, dim) array of cue vectors.
        *weights* – (n_nodes,) non‑negative scalars; will be normalised internally.
        Returns a single (dim,) vector.
        """
        n_nodes = len(self.node_ids)
        if cues.shape != (n_nodes, self.dim):
            raise ValueError("cues shape mismatch")
        if weights.shape != (n_nodes,):
            raise ValueError("weights shape mismatch")
        w = weights / (weights.sum() + 1e-12)  # safe normalisation
        out = np.zeros(self.dim, dtype=float)
        for idx, nid in enumerate(self.node_ids):
            out += w[idx] * self.retrieve(nid, cues[idx])
        return out


# ----------------------------------------------------------------------
# RBF utilities (vectorised, normalised)
# ----------------------------------------------------------------------
def gaussian_rbf(r: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Isotropic Gaussian RBF applied element‑wise to a distance array.
    Returns values in (0, 1].
    """
    return np.exp(-((epsilon * r) ** 2))


def rbf_weights(query: np.ndarray, seeds: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Compute normalised Gaussian RBF weights for *query* w.r.t. *seeds*.
    The returned array sums to 1.
    """
    dists = np.linalg.norm(seeds - query, axis=1)
    raw = gaussian_rbf(dists, epsilon)
    return raw / (raw.sum() + 1e-12)


# ----------------------------------------------------------------------
# Hybrid construction utilities
# ----------------------------------------------------------------------
def build_hybrid_sheaf(
    points: np.ndarray,
    seeds: np.ndarray,
    dim: int,
    feature_order: List[str],
) -> Tuple[Sheaf, np.ndarray]:
    """
    Construct a Sheaf whose nodes correspond to the Voronoi seeds.
    Returns (sheaf, seed_feature_embeddings) where the latter is a
    (n_seeds, n_features) matrix mapping textual features to per‑seed
    modulation factors.
    """
    n_seeds = seeds.shape[0]
    sheaf = Sheaf(node_ids=list(range(n_seeds)), dim=dim)

    # Voronoi assignment and centroid sections
    region_mat = assign(points, seeds)
    for i in range(n_seeds):
        mask = region_mat[i].astype(bool)
        centroid = points[mask].mean(axis=0) if mask.any() else seeds[i]
        sheaf.set_section(i, centroid)

    # Random seed‑specific feature embeddings (learnable in a real system)
    seed_feature_embeddings = np.random.randn(n_seeds, len(feature_order))
    # Normalise rows to keep modulation factors bounded
    seed_feature_embeddings /= np.linalg.norm(seed_feature_embeddings, axis=1, keepdims=True) + 1e-12

    return sheaf, seed_feature_embeddings


# ----------------------------------------------------------------------
# Core hybrid query
# ----------------------------------------------------------------------
def query_hybrid(
    sheaf: Sheaf,
    seeds: np.ndarray,
    query_point: np.ndarray,
    text: str,
    seed_feature_embeddings: np.ndarray,
    feature_order: List[str],
    epsilon: float = 1.0,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Perform a deep hybrid retrieval:

    1. Compute base RBF weights for the geometric query.
    2. Convert *text* into a normalised feature vector.
    3. Modulate each seed's weight by a dot‑product between the text feature
       vector and the seed‑specific embedding, scaled by *alpha*.
    4. Normalise the resulting weights.
    5. Use each seed's stored section (centroid) as the cue for its associative
       memory readout.
    6. Blend the per‑seed reads according to the modulated weights.

    Parameters
    ----------
    sheaf : Sheaf
        The associative‑memory sheaf built from the same *seeds*.
    seeds : np.ndarray, shape (n_seeds, d)
        Seed coordinates used for the Voronoi partition.
    query_point : np.ndarray, shape (d,)
        The geometric query.
    text : str
        Free‑form text whose regex‑derived features influence retrieval.
    seed_feature_embeddings : np.ndarray, shape (n_seeds, n_features)
        Per‑seed embeddings that map textual features to modulation scalars.
    feature_order : List[str]
        Ordered list of feature names matching the columns of *seed_feature_embeddings*.
    epsilon : float, default 1.0
        RBF shape parameter.
    alpha : float, default 0.5
        Strength of textual modulation (0 → pure geometry, 1 → strong text influence).

    Returns
    -------
    np.ndarray, shape (dim,)
        The blended associative‑memory output vector.
    """
    # 1. Base geometric RBF weights (already normalised)
    base_weights = rbf_weights(query_point, seeds, epsilon)  # (n_seeds,)

    # 2. Textual feature vector
    txt_vec = text_to_feature_vector(text, feature_order)  # (n_features,)

    # 3. Per‑seed modulation via dot‑product, shifted to (0, 1] range
    raw_mod = 1.0 + alpha * (seed_feature_embeddings @ txt_vec)  # (n_seeds,)
    mod_weights = base_weights * raw_mod

    # 4. Normalise final weights
    final_weights = mod_weights / (mod_weights.sum() + 1e-12)  # (n_seeds,)

    # 5. Assemble cues: each node uses its stored section (centroid)
    cues = np.stack([sheaf.get_section(nid) for nid in sheaf.node_ids], axis=0)  # (n_seeds, dim)

    # 6. Blend associative reads
    return sheaf.blend(cues, final_weights)


# ----------------------------------------------------------------------
# Example usage (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic geometry
    rng = np.random.default_rng(42)
    dim_space = 5
    n_points = 200
    n_seeds = 8
    points = rng.normal(size=(n_points, dim_space))
    seeds = rng.normal(size=(n_seeds, dim_space))

    # Feature ordering (must match embeddings)
    feature_names = list(_FEATURE_PATTERNS.keys())

    # Build sheaf and seed‑feature embeddings
    sheaf, seed_feat_emb = build_hybrid_sheaf(points, seeds, dim=dim_space, feature_order=feature_names)

    # Query
    q_point = rng.normal(size=dim_space)
    q_text = "The evidence confirms the plan but there is a delay in the schedule."

    result = query_hybrid(
        sheaf=sheaf,
        seeds=seeds,
        query_point=q_point,
        text=q_text,
        seed_feature_embeddings=seed_feat_emb,
        feature_order=feature_names,
        epsilon=1.2,
        alpha=0.7,
    )
    print("Hybrid retrieval vector:", result)