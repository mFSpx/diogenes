# DARWIN HAMMER — match 3081, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_possum_hybrid_hybrid_hybrid_m1533_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_hybrid_hybrid_m424_s2.py (gen5)
# born: 2026-05-29T23:47:44Z

"""Hybrid Morphology‑NLMS‑LSM Minimum‑Cost Tree
================================================

This module fuses the two parent algorithms:

* **Parent A** – geometric/morphological calculations on `Entity` objects
  (sphericity, flatness, right‑ing‑time, multivector utilities).
* **Parent B** – Normalised Least‑Mean‑Squares (NLMS) adaptive filtering,
  Latent‑Semantic‑Model (LSM) vector extraction, and a hybrid edge‑weight
  definition that feeds a Prim‑MST construction.

**Mathematical bridge**

Both parents operate on a collection of *nodes* (entities).  
Parent A supplies a *morphology vector*  


m_i = [ sphericity_i , flatness_i , righting_time_i ]


Parent B supplies a *prediction score* `p_i` (NLMS) and a *semantic vector*
`v_i` (LSM).  

The hybrid edge weight for an unordered pair `e = (i, j)` is defined as


d_geo_ij   = haversine distance between (lat,lon) of i and j
d_morph_ij = || m_i – m_j ||_2                     (morphology distance)

d_ij = α * d_geo_ij + β * d_morph_ij               (combined geometry)

π_ij   = (p_i + p_j) / (p_i + p_j + ε)             (Bayesian prior)
ℓ_ij   = cosine_similarity(v_i, v_j)              (likelihood)
ϕ_ij   = c_ij * 0.1                                 (false‑positive term)

m_ij   = bayes_marginal(π_ij, 1-ℓ_ij, ϕ_ij)        (marginal error prob)
w_ij   = d_ij * (1 - m_ij) + ε                     (final hybrid weight)


The resulting scalar `w_ij` is fed to a classic Prim‑MST algorithm, yielding a
minimum‑cost spanning tree that respects geometry, morphology, adaptive‑filter
information and semantic similarity.

The implementation below provides:

* Morphology utilities (`sphericity_index`, `flatness_index`,
  `righting_time_index`).
* A minimal NLMS predictor / updater.
* A deterministic LSM vector extractor.
* The hybrid edge‑weight calculation and MST construction.

All functions are pure NumPy / std‑lib and can be used independently.  A tiny
smoke‑test is provided at the bottom of the file."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

import numpy as np

# ----------------------------------------------------------------------
# Entity and Morphology Dataclasses (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    """Geometric + descriptive record."""
    id: str
    lat: float          # degrees
    lon: float          # degrees
    category: str
    score: float = 0.0  # NLMS decision score (filled later)
    address_signature: str = ""  # free‑form text for LSM
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    mass: float = 0.0


# ----------------------------------------------------------------------
# Morphology calculations (Parent A)
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """(V)^(1/3) / max(dim).  Returns a value in (0,1]."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    volume = length * width * height
    return volume ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """(length+width) / (2*height).  Larger → flatter."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Entity,
                        b: float = 1.0 / 3.0,
                        k: float = 0.35,
                        neck_lever: float = 1.0) -> float:
    """Physical‑style estimate of the time needed for an object to right itself."""
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (k / (b * fi)) * (m.mass ** (1.0 / 3.0)) * neck_lever


def compute_morphology_vector(entity: Entity) -> np.ndarray:
    """
    Returns the 3‑dimensional morphology vector
    [sphericity, flatness, righting_time] for *entity*.
    """
    sph = sphericity_index(entity.length, entity.width, entity.height)
    flt = flatness_index(entity.length, entity.width, entity.height)
    rgt = righting_time_index(entity)
    return np.array([sph, flt, rgt], dtype=np.float64)


# ----------------------------------------------------------------------
# NLMS core (Parent B)
# ----------------------------------------------------------------------
def nlms_predict(x: np.ndarray, w: np.ndarray, eps: float = 1e-8) -> float:
    """
    Normalised Least‑Mean‑Squares prediction.
    y = (w·x) / (eps + ||x||²)
    """
    norm = eps + np.dot(x, x)
    return float(np.dot(w, x) / norm)


def nlms_update(w: np.ndarray,
                x: np.ndarray,
                d: float,
                mu: float = 0.1,
                eps: float = 1e-8) -> np.ndarray:
    """
    NLMS weight update.
    w ← w + μ·e·x / (ε + ||x||²)   where e = d – y
    Returns the updated weight vector.
    """
    y = nlms_predict(x, w, eps)
    e = d - y
    norm = eps + np.dot(x, x)
    w_new = w + (mu * e / norm) * x
    return w_new


# ----------------------------------------------------------------------
# LSM vector extraction (Parent B)
# ----------------------------------------------------------------------
def _hash_text_to_vector(text: str, dim: int = 50) -> np.ndarray:
    """
    Deterministic pseudo‑embedding: each character contributes to a slot
    based on its ordinal value.  The result is L2‑normalised.
    """
    vec = np.zeros(dim, dtype=np.float64)
    for ch in text:
        idx = (ord(ch) * 31) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def lsm_vector(text: str, dim: int = 50) -> np.ndarray:
    """
    Public wrapper for the deterministic LSM embedding.
    """
    return _hash_text_to_vector(text, dim)


# ----------------------------------------------------------------------
# Hybrid edge weight & MST (Parent B + Morphology)
# ----------------------------------------------------------------------
def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """Great‑circle distance in metres."""
    R = 6371000.0  # Earth radius in metres
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)

    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [‑1, 1]."""
    if a.size == 0 or b.size == 0:
        return 0.0
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return dot / norm if norm != 0 else 0.0


def bayes_marginal(prior: float, one_minus_likelihood: float,
                   false_positive: float) -> float:
    """
    Simple Bayesian marginal error model:

        m = prior * (1‑ℓ) + ϕ

    The function is deliberately lightweight – it captures the spirit of the
    original parent without pulling in external probabilistic libraries.
    """
    return prior * one_minus_likelihood + false_positive


def hybrid_edge_weight(e1: Entity,
                       e2: Entity,
                       morph_vecs: Dict[str, np.ndarray],
                       lsm_vecs: Dict[str, np.ndarray],
                       c_ij: float = 0.0,
                       α: float = 0.7,
                       β: float = 0.3,
                       eps: float = 1e-8) -> float:
    """
    Compute the hybrid weight w_ij for the unordered pair (e1, e2).

    Parameters
    ----------
    e1, e2 : Entity
        Nodes of the graph.
    morph_vecs : dict[id → np.ndarray]
        Pre‑computed morphology vectors.
    lsm_vecs : dict[id → np.ndarray]
        Pre‑computed LSM vectors.
    c_ij : float, optional
        Epistemic‑uncertainty factor supplied by the user (0 ≤ c ≤ 1).
    α, β : float, optional
        Relative importance of geographic vs. morphology distance.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    w_ij : float
        Hybrid edge weight.
    """
    # 1️⃣ Geographic distance
    d_geo = haversine_distance(e1.lat, e1.lon, e2.lat, e2.lon)

    # 2️⃣ Morphology distance
    m1, m2 = morph_vecs[e1.id], morph_vecs[e2.id]
    d_morph = float(np.linalg.norm(m1 - m2))

    # 3️⃣ Combined distance
    d_ij = α * d_geo + β * d_morph

    # 4️⃣ NLMS prior (use stored scores; fallback to 0)
    p_i, p_j = e1.score, e2.score
    pi_ij = (p_i + p_j) / (p_i + p_j + eps)

    # 5️⃣ LSM likelihood
    v_i, v_j = lsm_vecs[e1.id], lsm_vecs[e2.id]
    ell_ij = cosine_similarity(v_i, v_j)  # in [‑1, 1]

    # 6️⃣ False‑positive term
    phi_ij = c_ij * 0.1

    # 7️⃣ Marginal error probability
    m_ij = bayes_marginal(pi_ij, 1 - ell_ij, phi_ij)

    # 8️⃣ Final hybrid weight
    w_ij = d_ij * (1 - m_ij) + eps
    return w_ij


def prim_mst(nodes: List[Entity],
            weight_func,
            **weight_kwargs) -> List[Tuple[str, str, float]]:
    """
    Classic Prim algorithm returning a list of edges (id_a, id_b, weight)
    that form a minimum‑cost spanning tree over *nodes*.

    Parameters
    ----------
    nodes : list[Entity]
        Graph vertices.
    weight_func : callable
        Function that accepts two Entity objects and returns a scalar weight.
    **weight_kwargs
        Additional arguments forwarded to *weight_func*.

    Returns
    -------
    mst_edges : list[tuple(str, str, float)]
        Edge list of the MST.
    """
    if not nodes:
        return []

    # Initialise structures
    visited: Set[str] = set()
    edge_heap: List[Tuple[float, str, str]] = []  # (weight, from_id, to_id)
    mst_edges: List[Tuple[str, str, float]] = []

    # Start from the first node
    start = nodes[0]
    visited.add(start.id)

    # Populate initial frontier
    for node in nodes[1:]:
        w = weight_func(start, node, **weight_kwargs)
        edge_heap.append((w, start.id, node.id))
    edge_heap.sort(key=lambda x: x[0])  # cheap priority queue substitute

    while len(visited) < len(nodes):
        # Extract minimum weight edge that connects to an unvisited node
        while edge_heap:
            w, frm, to = edge_heap.pop(0)
            if to not in visited:
                break
        else:
            raise RuntimeError("Graph is disconnected; MST cannot be formed.")

        # Add edge to MST
        mst_edges.append((frm, to, w))
        visited.add(to)

        # Expand frontier from the newly visited node
        new_node = next(e for e in nodes if e.id == to)
        for node in nodes:
            if node.id not in visited:
                w_new = weight_func(new_node, node, **weight_kwargs)
                edge_heap.append((w_new, to, node.id))
        edge_heap.sort(key=lambda x: x[0])

    return mst_edges


# ----------------------------------------------------------------------
# High‑level hybrid pipeline (demonstrates integration)
# ----------------------------------------------------------------------
def hybrid_pipeline(entities: List[Entity],
                    nlms_mu: float = 0.05,
                    c_matrix: Dict[Tuple[str, str], float] = None,
                    α: float = 0.7,
                    β: float = 0.3) -> List[Tuple[str, str, float]]:
    """
    End‑to‑end routine:

    1. Compute morphology and LSM vectors for every entity.
    2. Run a single NLMS adaptation pass on a synthetic feature vector
       (illustrative – real data would supply a stream of observations).
    3. Build a hybrid MST using the combined edge weight.

    Returns the MST edge list.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Pre‑compute feature vectors
    # ------------------------------------------------------------------
    morph_vec