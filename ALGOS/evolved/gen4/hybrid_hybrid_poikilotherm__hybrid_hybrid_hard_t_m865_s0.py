# DARWIN HAMMER — match 865, survivor 0
# gen: 4
# parent_a: hybrid_poikilotherm_schoolf_hybrid_hard_truth_ma_m76_s1.py (gen3)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s2.py (gen3)
# born: 2026-05-29T23:31:19Z

"""Hybrid Poikilotherm‑Stylometric Voronoi Tree Module

Parents
-------
* **Parent A** – *hybrid_poikilotherm_schoolf_hybrid_hard_truth_ma_m76_s1.py*  
  Provides a temperature‑dependent physiological scaling factor ρ(T) and a
  Bayesian tree‑cost functional.

* **Parent B** – *hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s2.py*  
  Supplies a simple Latent‑Semantic‑Metric (LSM) stylometric vectoriser and
  Voronoi‑style nearest‑centroid partitioning.

Mathematical Bridge
-------------------
The bridge is the **temperature‑scaled feature space**.  
For a document *d* we compute an LSM vector **v(d)** (Parent B).  
The poikilotherm developmental rate ρ(T) (Parent A) is interpreted as a
global physiological gain that modulates the magnitude of every feature,
while the normalized activity a(T)∈[0,1] acts as a confidence weight.
Thus the temperature‑aware representation is

\[
\tilde{v}(d;T)=\rho(T)\,a(T)\,v(d).
\]

All geometric operations (edge lengths, Voronoi assignments, Bayesian
posteriors) are performed on these scaled vectors, yielding a unified
temperature‑aware Bayesian‑geometric cost for a document‑tree.

The module implements:
* `developmental_rate` & `normalized_activity` – Poikilotherm utilities.
* `lsm_vector` – simple term‑frequency based stylometric embedding.
* `temperature_scaled_vector` – applies ρ(T)·a(T) to an LSM vector.
* `pairwise_distances` – Euclidean edge lengths between scaled vectors.
* `bayes_edge_posteriors` – Bayesian posterior on each edge using a
  likelihood proportional to exp(‑distance).
* `voronoi_partition` – assigns each document to the nearest seed centroid.
* `hybrid_tree_cost` – temperature‑aware hybrid cost combining edge posteriors
  and node beliefs derived from Voronoi clusters.

The resulting system can be used wherever thermal conditions influence
text‑based similarity structures (e.g., temperature‑sensitive sensor logs,
biological sequence annotation under varying conditions, etc.).
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Poikilotherm rate utilities
# ----------------------------------------------------------------------


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(
    T: float,
    E: float = 0.65,
    Eh: float = 2.0,
    Th: float = 303.15,
    Tl: float = 283.15,
    k: float = 8.617e-5,
) -> float:
    """
    Schoolfield–Rollinson temperature dependent developmental rate ρ(T).

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    E, Eh, Th, Tl, k : float
        Model parameters (default values are typical biological constants).

    Returns
    -------
    float
        Developmental rate ρ(T).
    """
    # avoid division by zero
    if T <= 0:
        return 0.0
    num = math.exp(-E / (k * T))
    den = 1 + math.exp(Eh / k * (1 / Th - 1 / T)) + math.exp(Eh / k * (1 / Tl - 1 / T))
    return num / den


def normalized_activity(T: float, T_opt: float = 295.0, sigma: float = 10.0) -> float:
    """
    Gaussian activity gate a(T)∈[0,1] centred at optimal temperature.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    T_opt : float
        Optimal temperature (default 295 K).
    sigma : float
        Width of the activity curve.

    Returns
    -------
    float
        Normalized activity.
    """
    return math.exp(-0.5 * ((T - T_opt) / sigma) ** 2)


# ----------------------------------------------------------------------
# Parent B – Stylometric LSM vectoriser (simplified)
# ----------------------------------------------------------------------


def words(text: str) -> List[str]:
    """Return a list of lowercase alphabetic words."""
    import re

    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str, vocab: List[str] | None = None) -> np.ndarray:
    """
    Compute a simple term‑frequency vector for *text*.

    Parameters
    ----------
    text : str
        Input document.
    vocab : list of str, optional
        Fixed vocabulary. If None a vocabulary is built from the text itself.

    Returns
    -------
    np.ndarray
        Normalised frequency vector of shape (|vocab|,).
    """
    tokens = words(text)
    if not tokens:
        return np.zeros(0, dtype=float)

    if vocab is None:
        vocab = sorted(set(tokens))
    idx = {w: i for i, w in enumerate(vocab)}
    vec = np.zeros(len(vocab), dtype=float)
    for w in tokens:
        if w in idx:
            vec[idx[w]] += 1.0
    if vec.sum() > 0:
        vec /= vec.sum()  # L1‑normalise
    return vec


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------


def temperature_scaled_vector(vec: np.ndarray, T_K: float) -> np.ndarray:
    """
    Apply the poikilotherm scaling ρ(T)·a(T) to a stylometric vector.

    Parameters
    ----------
    vec : np.ndarray
        Original LSM vector.
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    np.ndarray
        Scaled vector (same shape as *vec*).
    """
    rho = developmental_rate(T_K)
    a = normalized_activity(T_K)
    scale = rho * a
    return vec * scale


def pairwise_distances(vectors: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric matrix of Euclidean distances between rows of *vectors*.

    Parameters
    ----------
    vectors : np.ndarray, shape (n, d)

    Returns
    -------
    np.ndarray, shape (n, n)
        Distance matrix where D[i, j] = ||vectors[i]‑vectors[j]||₂.
    """
    diff = vectors[:, None, :] - vectors[None, :, :]
    D = np.sqrt(np.sum(diff ** 2, axis=2))
    return D


def bayes_edge_posteriors(distances: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
    """
    Compute Bayesian posteriors for edges of a complete graph.

    The likelihood for edge *e* is taken as L_e = exp(-distance_e).
    Prior π_e can be supplied; otherwise a uniform prior is assumed.

    Parameters
    ----------
    distances : np.ndarray, shape (n, n)
        Symmetric distance matrix.
    prior : np.ndarray, shape (n, n), optional
        Prior belief matrix (must be symmetric and non‑negative).

    Returns
    -------
    np.ndarray, shape (n, n)
        Posterior matrix p_e that sums to 1 over the upper‑triangular entries.
    """
    n = distances.shape[0]
    if prior is None:
        prior = np.ones_like(distances)
    # Zero diagonal (no self‑edges)
    np.fill_diagonal(prior, 0.0)

    likelihood = np.exp(-distances)
    unnorm = likelihood * prior
    total = np.sum(np.triu(unnorm, k=1))
    if total == 0:
        return np.zeros_like(unnorm)
    posterior = np.triu(unnorm, k=1) / total
    # Mirror to lower triangle for convenience
    posterior = posterior + posterior.T
    return posterior


def voronoi_partition(vectors: np.ndarray, seeds_idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each vector to the nearest seed (Voronoi cell) and compute cell centroids.

    Parameters
    ----------
    vectors : np.ndarray, shape (n, d)
        Feature vectors.
    seeds_idx : list of int
        Indices of seed vectors (must be a subset of ``range(n)``).

    Returns
    -------
    assignments : np.ndarray, shape (n,)
        Integer label of the seed each vector is assigned to.
    centroids : np.ndarray, shape (k, d)
        Centroid of each Voronoi cell (k = len(seeds_idx)).
    """
    seeds = vectors[seeds_idx]  # shape (k, d)
    # Compute distances from every point to every seed
    diff = vectors[:, None, :] - seeds[None, :, :]  # (n, k, d)
    dists = np.linalg.norm(diff, axis=2)  # (n, k)
    assignments = np.argmin(dists, axis=1)  # (n,)

    k = len(seeds_idx)
    d = vectors.shape[1]
    centroids = np.zeros((k, d), dtype=float)
    for i in range(k):
        members = vectors[assignments == i]
        if len(members) > 0:
            centroids[i] = members.mean(axis=0)
        else:
            centroids[i] = seeds[i]  # fallback to seed itself
    return assignments, centroids


def hybrid_tree_cost(
    posteriors: np.ndarray,
    distances: np.ndarray,
    assignments: np.ndarray,
    lambda_path: float = 0.5,
) -> float:
    """
    Temperature‑aware hybrid cost.

    Edge term:
        C_e = Σ_e p_e * ℓ(e) / Σ_e |p_e|
    Node term:
        For each Voronoi cell c, define belief q_c = Σ_{e∈∂c} p_e
        and distance d_c = mean root‑to‑node distance (here approximated by
        average distance of members to the global centroid).

    The total cost is
        C_h = C_e + λ * ( Σ_c q_c * d_c / Σ_c |q_c| ).

    Parameters
    ----------
    posteriors : np.ndarray, shape (n, n)
        Symmetric posterior matrix p_e.
    distances : np.ndarray, shape (n, n)
        Edge lengths ℓ(e).
    assignments : np.ndarray, shape (n,)
        Voronoi cell label for each node.
    lambda_path : float
        Weight of the node‑term (default 0.5).

    Returns
    -------
    float
        Hybrid cost value.
    """
    # Edge term (upper triangular only)
    mask = np.triu(np.ones_like(posteriors, dtype=bool), k=1)
    p_vals = posteriors[mask]
    l_vals = distances[mask]
    if np.abs(p_vals).sum() == 0:
        edge_term = 0.0
    else:
        edge_term = np.sum(p_vals * l_vals) / np.abs(p_vals).sum()

    # Node term
    n = posteriors.shape[0]
    q = np.zeros(n, dtype=float)  # belief per node
    for i in range(n):
        q[i] = np.sum(posteriors[i, :])  # sum of incident posterior weights

    # Approximate root‑to‑node distance by average distance to global centroid
    centroid = distances.mean()
    d_node = np.full(n, centroid)

    # Aggregate per Voronoi cell
    unique_cells = np.unique(assignments)
    q_cells = []
    d_cells = []
    for cell in unique_cells:
        mask_cell = assignments == cell
        q_cells.append(np.sum(q[mask_cell]))
        d_cells.append(np.mean(d_node[mask_cell]))
    q_cells = np.array(q_cells)
    d_cells = np.array(d_cells)

    if np.abs(q_cells).sum() == 0:
        node_term = 0.0
    else:
        node_term = np.sum(q_cells * d_cells) / np.abs(q_cells).sum()

    return edge_term + lambda_path * node_term


# ----------------------------------------------------------------------
# End‑to‑end example
# ----------------------------------------------------------------------


def temperature_dependent_cost(
    texts: List[str],
    temperature_c: float,
    seed_indices: List[int] | None = None,
) -> float:
    """
    Full pipeline:
    1. Build a common vocabulary from all texts.
    2. Compute LSM vectors.
    3. Scale vectors by ρ(T)·a(T).
    4. Build distance matrix, Bayesian posteriors.
    5. Perform Voronoi partition (seeds chosen randomly if not supplied).
    6. Return the hybrid cost.

    Parameters
    ----------
    texts : list of str
        Corpus of documents.
    temperature_c : float
        Temperature in Celsius.
    seed_indices : list of int, optional
        Indices of seed documents for Voronoi partition. If ``None`` three
        random seeds are selected.

    Returns
    -------
    float
        Hybrid temperature‑aware cost.
    """
    if not texts:
        raise ValueError("Empty text list")

    # 1. Vocabulary
    vocab = sorted({w for txt in texts for w in words(txt)})

    # 2. LSM vectors
    vectors = np.vstack([lsm_vector(txt, vocab) for txt in texts])

    # 3. Temperature scaling
    T_K = c_to_k(temperature_c)
    scaled_vectors = np.apply_along_axis(temperature_scaled_vector, 1, vectors, T_K)

    # 4. Geometry & Bayesian posteriors
    D = pairwise_distances(scaled_vectors)
    post = bayes_edge_posteriors(D)

    # 5. Voronoi seeds
    n = len(texts)
    if seed_indices is None:
        seed_indices = random.sample(range(n), k=min(3, n))
    assignments, _ = voronoi_partition(scaled_vectors, seed_indices)

    # 6. Hybrid cost
    cost = hybrid_tree_cost(post, D, assignments, lambda_path=0.5)
    return cost


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaped above a sleepy canine.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "In a distant galaxy, the stars shimmered brightly.",
        "Temperature affects metabolic rates in ectothermic organisms."
    ]
    temp_c = 25.0  # typical room temperature
    try:
        cost_value = temperature_dependent_cost(sample_texts, temp_c)
        print(f"Hybrid temperature‑aware cost: {cost_value:.6f}")
    except Exception as e:
        print(f"Smoke test failed: {e}", file=sys.stderr)
        sys.exit(1)