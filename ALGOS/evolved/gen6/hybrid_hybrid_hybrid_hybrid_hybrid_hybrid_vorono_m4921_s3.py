# DARWIN HAMMER — match 4921, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_hybrid_xgboos_m642_s4.py (gen5)
# parent_b: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m841_s4.py (gen5)
# born: 2026-05-29T23:58:49Z

"""Hybrid Leader‑Tree / Voronoi‑Partition Algorithm (HLTVP)

This module fuses the two parent algorithms:

* **Parent A** – *Hybrid Leader‑Tree* (tropical max‑plus broadcast,
  Hoeffding‑bound split test, simulated‑annealing acceptance).
* **Parent B** – *Hybrid Voronoi Partition* (Euclidean‑style Voronoi
  region assignment, regex‑driven feature extraction).

**Mathematical bridge**

The tropical max‑plus product produces a *similarity* matrix  
`S[i,j] = max_k (P[i,k] + C[j,k])` between data points `P` and seed
vectors `C`.  Interpreting `S` as a *margin* `m` for a binary logistic loss
allows us to reuse the gradient/hessian machinery of Parent A.  The
region assignment of Parent B is then performed on the *distance*
`d = -S`, i.e. the point is attached to the seed with the largest
tropical similarity (equivalently the smallest tropical distance).  The
logistic gradient is further regularised with a MinHash‑derived similarity
`s` and the Shannon entropy `H` of the token‑set distribution – exactly
the regularisation introduced in Parent A.  The resulting gain is
evaluated with a Hoeffding bound; a Metropolis‑style simulated‑annealing
step finally decides whether the new leader configuration is kept.

The three public functions below illustrate the integrated workflow:
`tropical_matmul`, `voronoi_assign_tropical`, and `hybrid_leader_voronoi_step`."""

from __future__ import annotations

import math
import random
import sys
import pathlib
import re
from collections import Counter
from typing import Mapping, Hashable, Set, Tuple, List, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, Set[Node]]
TokenSet = Set[str]

# ----------------------------------------------------------------------
# Parent A core – tropical (max‑plus) matrix multiplication
# ----------------------------------------------------------------------
def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Max‑plus matrix multiplication.

    (A ⊗ B)[i, j] = max_k (A[i, k] + B[k, j])

    Parameters
    ----------
    A : (n, m) ndarray
    B : (m, p) ndarray

    Returns
    -------
    C : (n, p) ndarray
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError("inner dimensions must agree")
    n, m = A.shape
    p = B.shape[1]
    # broadcasting trick: (n, m, 1) + (1, m, p) -> (n, m, p)
    A_exp = A[:, :, np.newaxis]          # (n, m, 1)
    B_exp = B[np.newaxis, :, :]          # (1, m, p)
    sum_mat = A_exp + B_exp              # (n, m, p)
    C = np.max(sum_mat, axis=1)          # max over k -> (n, p)
    return C

# ----------------------------------------------------------------------
# Parent B core – Voronoi region assignment (Euclidean style)
# ----------------------------------------------------------------------
def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def nearest(point: np.ndarray, seeds: np.ndarray) -> int:
    """Index of the closest seed to *point* (ordinary Euclidean metric)."""
    if seeds.size == 0:
        raise ValueError("seeds array is empty")
    return int(np.argmin(np.linalg.norm(seeds - point, axis=1)))


def assign(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Binary region matrix R of shape (n_seeds, n_points) where
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
# Feature extraction (regex based) – reused from Parent B
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
    """L2‑normalize a feature vector (guard against zero norm)."""
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm

# ----------------------------------------------------------------------
# Helper utilities for the hybrid regularisation
# ----------------------------------------------------------------------
def minhash_jaccard(set_a: TokenSet, set_b: TokenSet) -> float:
    """Approximate Jaccard similarity using MinHash (exact here for simplicity)."""
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union != 0 else 0.0


def shannon_entropy(counter: Counter) -> float:
    """Shannon entropy of a frequency counter."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for cnt in counter.values():
        p = cnt / total
        ent -= p * math.log(p, 2)
    return ent

# ----------------------------------------------------------------------
# Parent A core – adjusted logistic gradient/hessian with similarity‑entropy regularisation
# ----------------------------------------------------------------------
def adjusted_grad_hess(
    margin: float,
    label: int,
    similarity: float,
    entropy: float,
    alpha: float = 0.1,
) -> Tuple[float, float]:
    """
    Logistic gradient and hessian augmented with a similarity‑entropy term.

    Logistic loss: ℓ = log(1 + exp(-y * m)),  y ∈ {+1, -1}
    Gradient:  g = -y * σ(-y*m)   where σ is the sigmoid
    Hessian:   h = σ(m) * σ(-m)

    Regularisation adds `α * s * H` to the loss, therefore
    g ← g + α * s * dH/dm (we approximate dH/dm ≈ 0) and
    h ← h (unchanged).  For simplicity we just add the term to the gradient.
    """
    y = 1 if label == 1 else -1
    # sigmoid(-y*m) = 1 / (1 + exp(y*m))
    sigmoid = 1.0 / (1.0 + math.exp(y * margin))
    grad = -y * sigmoid + alpha * similarity * entropy
    # Hessian of logistic loss (without regularisation)
    sigma = 1.0 / (1.0 + math.exp(-margin))
    hess = sigma * (1.0 - sigma)
    return grad, hess

# ----------------------------------------------------------------------
# Hybrid operation -------------------------------------------------------
# ----------------------------------------------------------------------
def voronoi_assign_tropical(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Voronoi assignment using tropical (max‑plus) similarity as the distance metric.

    The tropical similarity matrix S = max_k (points_i,k + seeds_j,k)
    is turned into a distance d = -S.  Each point is assigned to the seed
    with the *largest* similarity (i.e. smallest tropical distance).

    Returns a binary region matrix R of shape (n_seeds, n_points).
    """
    # points: (n_pts, d), seeds: (n_seeds, d)
    # Compute similarity via tropical matmul of points and seeds.T
    similarity = tropical_matmul(points, seeds.T)          # (n_pts, n_seeds)
    # Assign each point to seed with max similarity
    assignments = np.argmax(similarity, axis=1)            # (n_pts,)
    n_seeds = seeds.shape[0]
    n_pts = points.shape[0]
    R = np.zeros((n_seeds, n_pts), dtype=int)
    for j, i in enumerate(assignments):
        R[i, j] = 1
    return R

def hybrid_leader_voronoi_step(
    graph: Graph,
    token_sets: Mapping[Node, TokenSet],
    points: np.ndarray,
    seeds: np.ndarray,
    labels: Mapping[Node, int],
    alpha: float = 0.1,
    delta: float = 0.05,
    temperature: float = 1.0,
) -> Tuple[Set[Node], np.ndarray]:
    """
    One iteration of the fused algorithm.

    1. Compute tropical Voronoi regions.
    2. For each region compute a logistic margin (mean tropical similarity).
    3. Augment gradient/hessian with MinHash similarity to a reference token set
       (the union of all tokens) and entropy of the region's token distribution.
    4. Apply a Hoeffding bound test to decide whether the region's gain
       justifies electing its seed as a leader.
    5. Simulated‑annealing acceptance decides the final leader set.

    Returns
    -------
    leaders : set of nodes chosen as leaders (seed indices as nodes)
    region_matrix : binary region matrix (n_seeds, n_points)
    """
    # 1. Voronoi assignment via tropical similarity
    region_matrix = voronoi_assign_tropical(points, seeds)   # (n_seeds, n_pts)

    n_seeds = seeds.shape[0]
    leaders: Set[Node] = set()
    # Reference token set: union of all tokens
    reference_tokens: TokenSet = set().union(*token_sets.values())
    # Global token frequency for entropy baseline
    global_counter = Counter()
    for ts in token_sets.values():
        global_counter.update(ts)

    for seed_idx in range(n_seeds):
        # Points belonging to this seed
        point_indices = np.where(region_matrix[seed_idx] == 1)[0]
        if point_indices.size == 0:
            continue

        # 2. Margin = average tropical similarity for the region
        # similarity matrix computed earlier (points vs seeds)
        similarity = tropical_matmul(points, seeds.T)        # (n_pts, n_seeds)
        margins = similarity[point_indices, seed_idx]
        margin = float(np.mean(margins))

        # 3. Regularisation terms
        # Aggregate token set for the region
        region_tokens: TokenSet = set()
        for node in graph:
            # Assume each node corresponds to a point index (bijection)
            # For the demo we map node to point index via enumeration order
            if node in token_sets and node in labels:
                idx = int(node)  # simplistic mapping for test harness
                if idx in point_indices:
                    region_tokens.update(token_sets[node])

        similarity_reg = minhash_jaccard(region_tokens, reference_tokens)
        entropy_reg = shannon_entropy(Counter(region_tokens))

        # Retrieve label for the seed node (if any)
        seed_node = seed_idx  # using integer node ids for simplicity
        label = labels.get(seed_node, 0)

        grad, hess = adjusted_grad_hess(
            margin, label, similarity_reg, entropy_reg, alpha=alpha
        )

        # 4. Hoeffding bound test (gain approximated by -grad)
        # R is range of gradient; we bound it by [-1, 1] for logistic loss
        R_range = 2.0
        n = point_indices.size
        epsilon = math.sqrt((R_range ** 2 * math.log(2.0 / delta)) / (2 * n))
        gain = -grad  # higher gain = more negative gradient
        if gain > epsilon:
            # 5. Simulated‑annealing Metropolis acceptance
            prob = math.exp(min(0, (gain - epsilon) / temperature))
            if random.random() < prob:
                leaders.add(seed_node)

    return leaders, region_matrix

# ----------------------------------------------------------------------
# Smoke test -------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small synthetic example
    random.seed(42)
    np.random.seed(42)

    # 10 points in 3‑dimensional space
    points = np.random.randn(10, 3)

    # 3 seeds (potential leaders)
    seeds = np.random.randn(3, 3)

    # Simple graph: each point is a node, edges are empty (unused)
    graph: Graph = {i: set() for i in range(10)}

    # Random token sets per node
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon"]
    token_sets: Dict[int, TokenSet] = {
        i: set(random.sample(vocab, random.randint(1, 3))) for i in range(10)
    }

    # Random binary labels for seeds (0/1)
    labels: Dict[int, int] = {i: random.randint(0, 1) for i in range(3)}

    leaders, R = hybrid_leader_voronoi_step(
        graph,
        token_sets,
        points,
        seeds,
        labels,
        alpha=0.05,
        delta=0.1,
        temperature=0.9,
    )

    print("Leaders selected (seed indices):", leaders)
    print("Region matrix shape:", R.shape)
    print("Region matrix (seed × point):\n", R)