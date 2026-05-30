# DARWIN HAMMER — match 2306, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m413_s0.py (gen4)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_hybrid_m105_s2.py (gen4)
# born: 2026-05-29T23:41:44Z

"""
Hybrid Algorithm: Fusion of Parent A (semantic‑neighbor Bayesian core) and
Parent B (NLMS adaptive learning with hygiene‑entropy scoring).

Mathematical Bridge
------------------
* Parent A provides a Bayesian update where the *likelihood* is supplied by
  semantic‑neighbour distances `d_i ∈ [0,1]`.  The marginal `P(E)` is
  computed from a prior `π` and a false‑positive rate `α`.

* Parent B supplies an NLMS weight‑adaptation rule and a *hygiene* score
  `h(x)` derived from Shannon entropy of a feature vector `x`.  

The fusion proceeds as follows:

1. For a given document `text` we extract a feature vector `x` (Parent B).
2. The hygiene score `h = hybrid_hygiene_score(x)` is used together with the
   Bayesian posterior `p = bayes_update(π, L, P_E)` to form an **effective
   learning rate**  

   `μ_eff = μ_base * p * (1 + h)`  

   (the posterior modulates confidence, the hygiene term rewards
   well‑distributed features).
3. The NLMS update `nlms_update` is executed with `μ_eff`.  Thus the weight
   adaptation is driven simultaneously by probabilistic relevance (Bayes)
   and information‑theoretic hygiene (entropy).
4. Edge weights for a minimum‑cost tree are built from the label score
   `ℓ = label_score(text, label)` and the posterior `p`, yielding  

   `w_edge = ℓ * p`.

The three core functions below demonstrate this integrated workflow.
"""

import math
import random
import sys
import pathlib
import hashlib
from typing import Tuple, List

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric utilities (from Parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Bayesian primitives (shared definition)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = L·π + α·(1‑π)"""
    if not all(0.0 <= v <= 1.0 for v in (prior, likelihood, false_positive)):
        raise ValueError("All probabilities must lie in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = π·L / P(E)"""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# Placeholder semantic‑neighbour and label scoring (Parent A)
# ----------------------------------------------------------------------
def semantic_neighbors(doc_id: str, k: int = 5) -> List[Tuple[str, float]]:
    """
    Return a list of ``k`` neighbours with a synthetic distance in (0,1].
    In a real system this would query an embedding index.
    """
    random.seed(hash(doc_id))
    return [(f"nbr_{i}", random.random()) for i in range(k)]

def label_score(text: str, label: str) -> float:
    """
    Very simple literal fallback: proportion of characters of ``label``
    that appear in ``text``.
    """
    if not label:
        return 0.0
    match = sum(ch in text for ch in label)
    return match / len(label)

# ----------------------------------------------------------------------
# NLMS core (from Parent B)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Dot‑product prediction."""
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Normalised LMS weight update.
    Returns (new_weights, error).
    """
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in (0, 2)")
    y = nlms_predict(weights, x)
    error = target - y
    power = float(x @ x) + eps
    delta = mu * error * x / power
    new_weights = weights + delta
    return new_weights, error

# ----------------------------------------------------------------------
# Feature extraction and hygiene scoring (Parent B)
# ----------------------------------------------------------------------
def extract_features(text: str) -> np.ndarray:
    """9‑dimensional normalized digit count vector."""
    counts = np.array([text.count(str(i)) for i in range(9)], dtype=float)
    total = counts.sum()
    return counts / total if total > 0 else counts

def hybrid_hygiene_score(features: np.ndarray) -> float:
    """
    Combine mean feature magnitude with Shannon entropy.
    Returns a scalar in [0, 2].
    """
    mean_val = np.mean(features)
    entropy = -np.sum(features * np.log2(features + 1e-9))
    entropy_max = np.log2(features.size)
    return mean_val * (1.0 + entropy / entropy_max)

# ----------------------------------------------------------------------
# 1️⃣ Hybrid Bayesian‑NLMS prediction
# ----------------------------------------------------------------------
def hybrid_predict(
    doc_id: str,
    text: str,
    prior: float,
    false_positive: float,
    target: float,
    init_weights: np.ndarray,
    mu_base: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """
    Perform a single hybrid prediction / weight update.

    Steps
    -----
    1. Extract features → `x`.
    2. Compute hygiene `h`.
    3. Obtain a synthetic likelihood `L` from the *closest* semantic neighbour.
    4. Compute marginal `P_E` and posterior `p`.
    5. Scale learning rate: `μ_eff = μ_base * p * (1 + h)`.
    6. Run NLMS update with `μ_eff`.

    Returns
    -------
    new_weights, error
    """
    # 1. Feature vector
    x = extract_features(text)

    # 2. Hygiene score
    h = hybrid_hygiene_score(x)

    # 3. Likelihood from semantic neighbours (use the smallest distance as proxy)
    neighbours = semantic_neighbors(doc_id, k=5)
    # Transform distance d∈[0,1] to likelihood L = 1‑d (closer ⇒ higher likelihood)
    likelihoods = [1.0 - d for _, d in neighbours]
    L = max(likelihoods)  # most plausible neighbour

    # 4. Bayesian quantities
    marginal = bayes_marginal(prior, L, false_positive)
    posterior = bayes_update(prior, L, marginal)

    # 5. Effective learning rate
    mu_eff = mu_base * posterior * (1.0 + h)

    # 6. NLMS adaptation
    new_w, err = nlms_update(init_weights, x, target, mu=mu_eff)

    return new_w, err

# ----------------------------------------------------------------------
# 2️⃣ Edge‑weight construction for a minimum‑cost tree
# ----------------------------------------------------------------------
def edge_weight(
    text: str,
    label: str,
    prior: float,
    false_positive: float,
    doc_id: str,
) -> float:
    """
    Compute a composite edge weight:
        w = label_score(text, label) * posterior
    where posterior comes from a Bayesian step that uses semantic neighbours.
    """
    # Likelihood from neighbours (same transformation as above)
    neighbours = semantic_neighbors(doc_id, k=3)
    L = max(1.0 - d for _, d in neighbours)

    marginal = bayes_marginal(prior, L, false_positive)
    posterior = bayes_update(prior, L, marginal)

    ℓ = label_score(text, label)
    return ℓ * posterior

# ----------------------------------------------------------------------
# 3️⃣ Full hybrid routine: predict & evaluate a tiny graph
# ----------------------------------------------------------------------
def hybrid_process_graph(
    nodes: List[Tuple[str, str]],  # list of (doc_id, text)
    edges: List[Tuple[str, str, str]],  # (src_id, dst_id, label)
    prior: float = 0.5,
    false_positive: float = 0.1,
    target: float = 1.0,
) -> Tuple[np.ndarray, dict]:
    """
    Run the hybrid predictor on each node, then compute edge weights.
    Returns the final weight vector (averaged over nodes) and a dict
    mapping edge identifiers to their computed weights.
    """
    dim = 9  # feature dimension
    agg_weights = np.zeros(dim)
    for doc_id, txt in nodes:
        w0 = np.zeros(dim)  # start from zero for each node
        w_new, _ = hybrid_predict(
            doc_id=doc_id,
            text=txt,
            prior=prior,
            false_positive=false_positive,
            target=target,
            init_weights=w0,
        )
        agg_weights += w_new

    avg_weights = agg_weights / max(1, len(nodes))

    edge_dict = {}
    for src, dst, lbl in edges:
        # use src as the reference document for neighbour info
        w = edge_weight(
            text=next(t for d, t in nodes if d == src),
            label=lbl,
            prior=prior,
            false_positive=false_positive,
            doc_id=src,
        )
        edge_dict[(src, dst, lbl)] = w

    return avg_weights, edge_dict

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal synthetic dataset
    nodes = [
        ("docA", "The quick brown fox jumps over 123456789."),
        ("docB", "Lorem ipsum dolor sit amet, consectetur 0123456789."),
    ]
    edges = [
        ("docA", "docB", "related"),
        ("docB", "docA", "reference"),
    ]

    avg_w, edge_w = hybrid_process_graph(nodes, edges)

    print("Average NLMS weights after hybrid updates:")
    print(avg_w)
    print("\nComputed edge weights:")
    for e, w in edge_w.items():
        print(f"{e}: {w:.4f}")

    # Simple sanity check: weights should be finite numbers
    assert np.all(np.isfinite(avg_w)), "Weight vector contains non‑finite values"
    assert all(np.isfinite(v) for v in edge_w.values()), "Edge weight contains non‑finite values"

    print("\nSmoke test passed.")