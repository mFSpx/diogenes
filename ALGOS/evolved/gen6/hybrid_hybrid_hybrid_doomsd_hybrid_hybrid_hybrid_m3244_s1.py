# DARWIN HAMMER — match 3244, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_doomsday_cale_hybrid_fold_change_d_m1433_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m448_s0.py (gen5)
# born: 2026-05-29T23:48:41Z

"""Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_doomsday_cale_hybrid_fold_change_d_m1433_s0.py (NLMS with RLCT‑adjusted μ and date features)
- Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m448_s0.py (stylometry‑driven weighted graph, ternary lens audit, Ollivier‑Ricci curvature)

Mathematical Bridge:
The NLMS error signal `e_t` (a scalar time‑series) is used to scale the
edge‑weights of the stylometry graph.  Specifically, for a pair of texts
i,j we compute a base similarity `s_ij` from their stylometric vectors.
The final weight is

    w_ij = s_ij * (1 + σ·|e_t|)

where σ is the RLCT‑adjusted learning‑rate μ̂ derived from the current NLMS
weight norm.  This couples the adaptive filter dynamics (Parent A) to the
graph‑theoretic structure (Parent B), allowing prediction errors to reshape
connectivity, after which a ternary‑lens audit and an Ollivier‑Ricci‑like
curvature are computed on the resulting graph.

The module implements three core hybrid functions:
1. `nlms_train_dates` – trains an NLMS filter on chronological date features.
2. `stylometric_vector` – extracts a normalized stylometric feature vector.
3. `hybrid_graph` – builds the weighted graph using the bridge formula and
   returns adjacency, ternary audit, and curvature matrices.
"""

import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ---------- Parent A core components ----------

def doomsday(year: int, month: int, day: int) -> int:
    """Weekday index where 0=Sunday … 6=Saturday."""
    return (int((dt.date(year, month, day).weekday() + 1) % 7))

def date_features(year: int, month: int, day: int) -> np.ndarray:
    """Normalized calendar features."""
    year_scaled = (year - 1900) / (2100 - 1900)
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    day_sin = math.sin(2 * math.pi * day / 31)
    day_cos = math.cos(2 * math.pi * day / 31)
    dow = doomsday(year, month, day) / 6.0  # normalised weekday
    return np.array([year_scaled, month_sin, month_cos, day_sin, day_cos, dow], dtype=float)

def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear NLMS prediction y = w·x."""
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """One NLMS weight update; returns new weights and error."""
    y = nlms_predict(weights, x)
    e = target - y
    norm_x = float(x @ x) + eps
    delta = (mu / norm_x) * e * x
    new_weights = weights + delta
    return new_weights, e

def rlct_adjusted_mu(weights: np.ndarray, base_mu: float = 0.5) -> float:
    """
    RLCT‑inspired learning‑rate adjustment:
        μ̂ = base_mu / (1 + log(1 + ||w||₂))
    """
    norm = np.linalg.norm(weights)
    return base_mu / (1.0 + math.log1p(norm))

def nlms_train_dates(dates: List[Tuple[int, int, int]], epochs: int = 5) -> Tuple[np.ndarray, List[float]]:
    """
    Train NLMS on a sequence of date feature vectors.
    The target at step t is the scalar projection of the next date's feature onto the current one.
    Returns final weights and the list of absolute errors per update (used later as scaling signal).
    """
    if len(dates) < 2:
        raise ValueError("At least two dates required for training.")
    # Initialise weights with small random values
    dim = 6
    w = np.random.randn(dim) * 0.01
    errors: List[float] = []
    for _ in range(epochs):
        for i in range(len(dates) - 1):
            x = date_features(*dates[i])
            # Simple target: dot product of consecutive feature vectors
            target = float(x @ date_features(*dates[i + 1]))
            mu = rlct_adjusted_mu(w)
            w, e = nlms_update(w, x, target, mu=mu)
            errors.append(abs(e))
    return w, errors

# ---------- Parent B core components ----------

FUNCTION_CATS = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}

def tokenize(text: str) -> List[str]:
    """Very simple whitespace / punctuation tokenizer."""
    return [t.strip(".,!?;:\"'()[]{}") for t in text.lower().split() if t]

def stylometric_vector(text: str) -> np.ndarray:
    """
    Compute a normalized stylometric feature vector based on FUNCTION_CATS.
    The vector length equals the number of categories; each entry is the
    proportion of tokens belonging to that category.
    """
    tokens = tokenize(text)
    total = len(tokens) or 1
    vec = []
    for cat in FUNCTION_CATS.values():
        count = sum(1 for t in tokens if t in cat)
        vec.append(count / total)
    return np.array(vec, dtype=float)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two non‑zero vectors."""
    if np.allclose(a, 0) or np.allclose(b, 0):
        return 0.0
    return float(a @ b) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)))

def ternary_lens_audit(adj: np.ndarray, median_degree: float) -> np.ndarray:
    """
    For each node compute:
        -1 if degree < median,
         0 if degree == median,
        +1 if degree > median.
    Returns a column vector of the ternary labels.
    """
    degrees = adj.sum(axis=1)
    audit = np.where(degrees < median_degree, -1,
                     np.where(degrees > median_degree, 1, 0))
    return audit[:, None]  # column vector

def ollivier_ricci_curvature(adj: np.ndarray) -> np.ndarray:
    """
    Simplified Ollivier‑Ricci curvature approximation.
    For edge (i,j):
        κ_ij = 1 - (w_ij / (mean_neighbor_weight_i + mean_neighbor_weight_j))
    Returns a matrix κ with zeros on the diagonal.
    """
    n = adj.shape[0]
    κ = np.zeros((n, n))
    mean_weights = adj.sum(axis=1) / np.maximum(adj.astype(bool).sum(axis=1), 1)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                continue
            denom = mean_weights[i] + mean_weights[j]
            if denom == 0:
                curv = 0.0
            else:
                curv = 1.0 - (adj[i, j] / denom)
            κ[i, j] = κ[j, i] = curv
    return κ

# ---------- Hybrid integration ----------

def hybrid_graph(texts: List[str], date_errors: List[float], sigma: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a weighted graph where:
        base similarity s_ij = cosine(stylometric_i, stylometric_j)
        scaling factor s = 1 + sigma * |e_t|
    The error list `date_errors` is cycled if shorter than the number of edges.
    Returns adjacency matrix, ternary audit vector, and curvature matrix.
    """
    n = len(texts)
    if n == 0:
        raise ValueError("No texts provided.")
    # Stylometric vectors
    styl_vectors = [stylometric_vector(t) for t in texts]
    # Base similarity matrix
    base_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = cosine_similarity(styl_vectors[i], styl_vectors[j])
            base_sim[i, j] = base_sim[j, i] = sim
    # Prepare scaling factors from errors
    if not date_errors:
        scaling_factors = np.ones((n, n))
    else:
        # Cycle errors over the upper‑triangular edge list
        scaling_factors = np.ones((n, n))
        edge_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                e = date_errors[edge_idx % len(date_errors)]
                scale = 1.0 + sigma * abs(e)
                scaling_factors[i, j] = scaling_factors[j, i] = scale
                edge_idx += 1
    # Final adjacency
    adj = base_sim * scaling_factors
    # Ternary lens audit
    median_deg = np.median(adj.sum(axis=1))
    audit = ternary_lens_audit(adj, median_deg)
    # Curvature
    curvature = ollivier_ricci_curvature(adj)
    return adj, audit, curvature

# ---------- Smoke test ----------

if __name__ == "__main__":
    # Sample texts
    sample_texts = [
        "I love programming, and I love coffee.",
        "The quick brown fox jumps over the lazy dog.",
        "In the midst of chaos, there is also opportunity.",
        "She sells sea shells on the sea shore.",
    ]
    # Sample dates (year, month, day)
    sample_dates = [
        (2022, 1, 15),
        (2022, 2, 20),
        (2022, 3, 25),
        (2022, 4, 30),
        (2022, 5, 5),
    ]

    # Train NLMS on dates to obtain error signal
    _, errors = nlms_train_dates(sample_dates, epochs=3)

    # Build hybrid graph
    adjacency, audit_vec, curvature_mat = hybrid_graph(sample_texts, errors, sigma=0.3)

    # Simple sanity prints
    print("Adjacency matrix (rounded):")
    print(np.round(adjacency, 3))
    print("\nTernary lens audit (per node):")
    print(audit_vec.ravel())
    print("\nCurvature matrix (rounded):")
    print(np.round(curvature_mat, 3))

    # Verify dimensions
    assert adjacency.shape == (len(sample_texts), len(sample_texts))
    assert audit_vec.shape == (len(sample_texts), 1)
    assert curvature_mat.shape == adjacency.shape
    print("\nHybrid graph construction succeeded.")