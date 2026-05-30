# DARWIN HAMMER — match 2306, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_minimu_m413_s0.py (gen4)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_hybrid_m105_s2.py (gen4)
# born: 2026-05-29T23:41:44Z

"""Hybrid Algorithm combining Semantic‑Neighbor Bayesian inference (Parent A) 
and Adaptive NLMS filtering with Hygiene‑entropy scoring (Parent B).

Mathematical Bridge
-------------------
* The *likelihood* in the Bayesian update is taken from the semantic‑neighbor
  distance distribution returned by ``semantic_neighbors`` (Parent A).
* The *feature vector* fed to the Normalised Least‑Mean‑Squares (NLMS) predictor
  (Parent B) is built from three components:
    1. Normalised character‑count features of the raw text (``extract_features``).
    2. The hygiene‑entropy scalar (``hybrid_hygiene_score``).
    3. The semantic‑neighbor likelihoods (treated as additional dimensions).
* The posterior probability from the Bayesian step is used as the *target* for the
  NLMS weight adaptation, thereby closing a feedback loop between the probabilistic
  reasoning and the adaptive linear model.
* Edge weights for a minimum‑cost spanning tree are formed by blending the
  label‑score (Parent A) with Euclidean distances (Parent A) and the current NLMS
  prediction, yielding a unified cost metric.
"""

import math
import random
import sys
import pathlib
import hashlib
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

# ----------------------------------------------------------------------
# Parent A utilities
# ----------------------------------------------------------------------
def length(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """P(E) = P(E|H)·P(H) + P(E|¬H)·P(¬H)."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P(E|H)·P(H) / P(E)."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal

def label_score(text: str, label: str) -> float:
    """Placeholder literal fallback scoring – returns a deterministic pseudo‑score."""
    # deterministic hash based score in [0,1]
    h = int(hashlib.sha256((text + label).encode()).hexdigest(), 16)
    return (h % 10_000) / 10_000.0

def semantic_neighbors(doc_id: str, k: int = 5) -> List[Tuple[str, float]]:
    """
    Placeholder returning ``k`` synthetic neighbors with decreasing likelihoods.
    In a real system this would query an embedding index.
    """
    random.seed(hash(doc_id))
    neighbors = []
    for i in range(k):
        neighbor_id = f"{doc_id}_nbr{i}"
        likelihood = max(0.0, 1.0 - i * 0.1 + random.uniform(-0.02, 0.02))
        neighbors.append((neighbor_id, likelihood))
    return neighbors

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")

# ----------------------------------------------------------------------
# Parent B utilities
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction w·x."""
    return float(weights @ x)

def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """One Normalised LMS update step."""
    if not (0.0 < mu < 2.0):
        raise ValueError("mu must be in (0,2)")
    y = nlms_predict(weights, x)
    error = target - y
    power = float(x @ x) + eps
    delta = mu * error * x / power
    new_weights = weights + delta
    return new_weights, error

def extract_features(text: str) -> np.ndarray:
    """9‑dimensional normalised digit count vector."""
    counts = np.array([text.count(str(i)) for i in range(9)], dtype=float)
    total = counts.sum()
    return counts / total if total > 0 else counts

def hybrid_hygiene_score(features: np.ndarray) -> float:
    """Mean * (1 + normalised Shannon entropy)."""
    s = np.mean(features)
    H = -np.sum(features * np.log2(features + 1e-9))
    H_max = np.log2(features.size)
    return s * (1.0 + H / H_max)

# ----------------------------------------------------------------------
# Fusion core – three demonstrative functions
# ----------------------------------------------------------------------
def hybrid_feature_vector(text: str, doc_id: str) -> np.ndarray:
    """
    Build a composite feature vector:
        [digit_features (9), hygiene_score (1), semantic_likelihoods (k)]
    """
    digit_feat = extract_features(text)                     # 9
    hygiene = np.array([hybrid_hygiene_score(digit_feat)]) # 1
    neigh = semantic_neighbors(doc_id, k=3)                # choose 3 neighbours
    likelihoods = np.array([lk for _, lk in neigh])        # 3
    return np.concatenate([digit_feat, hygiene, likelihoods])

def hybrid_predict_posterior(
    text: str,
    doc_id: str,
    label: str,
    prior: float,
    weights: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    1. Compute likelihoods from semantic neighbours.
    2. Perform Bayesian update to obtain posterior.
    3. Feed the posterior as target to NLMS predictor (using the composite feature vector).
    Returns (posterior, prediction).
    """
    # Step 1 – pick the highest neighbour likelihood as the event likelihood
    neigh = semantic_neighbors(doc_id, k=1)
    likelihood = neigh[0][1] if neigh else 0.0
    false_pos = 1.0 - prior  # simple complement as false‑positive rate
    marginal = bayes_marginal(prior, likelihood, false_pos)
    posterior = bayes_update(prior, likelihood, marginal)

    # Step 2 – NLMS prediction
    x = hybrid_feature_vector(text, doc_id)
    prediction = nlms_predict(weights, x)

    return posterior, prediction

def hybrid_update_weights(
    text: str,
    doc_id: str,
    label: str,
    prior: float,
    weights: np.ndarray,
    mu: float = 0.5,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute posterior (as in ``hybrid_predict_posterior``) and use it as the
    NLMS target to adapt the weight vector.
    Returns (new_weights, error, posterior).
    """
    posterior, _ = hybrid_predict_posterior(text, doc_id, label, prior, weights)
    x = hybrid_feature_vector(text, doc_id)
    new_weights, error = nlms_update(weights, x, target=posterior, mu=mu)
    return new_weights, error, posterior

def minimum_cost_spanning_tree(
    nodes: List[str],
    positions: Dict[str, Point],
    label: str,
    weights: np.ndarray,
) -> List[Edge]:
    """
    Build a minimum‑cost tree where each edge cost is:
        cost = length(p_i, p_j) * (1 - label_score) + nlms_predict
    The NLMS prediction is evaluated on a feature vector that concatenates the
    two node identifiers (hashed) with the current weight vector.
    """
    # Helper for Union‑Find
    parent = {n: n for n in nodes}
    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            return True
        return False

    # Build all possible edges with their costs
    edge_list: List[Tuple[float, Edge]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            p_a, p_b = positions[a], positions[b]
            euclid = length(p_a, p_b)

            # label‑score component (same for both directions)
            lbl_sc = label_score(a + b, label)  # deterministic pseudo‑score
            base_cost = euclid * (1.0 - lbl_sc)

            # NLMS component – construct a tiny feature vector from hashes
            ha = _hash(0, a)
            hb = _hash(0, b)
            hash_feat = np.array([ha, hb], dtype=float) / (2**32 - 1)
            nlms_part = nlms_predict(weights, hash_feat)

            total_cost = base_cost + nlms_part
            edge_list.append((total_cost, (a, b)))

    # Kruskal's algorithm
    edge_list.sort(key=lambda x: x[0])
    tree: List[Edge] = []
    for cost, e in edge_list:
        if union(*e):
            tree.append(e)
        if len(tree) == len(nodes) - 1:
            break
    return tree

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic seed
    random.seed(42)

    # Initialise a random weight vector matching the hybrid feature dimension (9+1+3 = 13)
    w = np.random.rand(13)

    # Example data
    txt = "The quick brown fox jumps over 1234567890 lazy dogs."
    doc = "doc_001"
    lbl = "animal"

    # Prior belief
    prior_prob = 0.3

    # Hybrid prediction & update
    post, pred = hybrid_predict_posterior(txt, doc, lbl, prior_prob, w)
    print(f"Posterior={post:.4f}, NLMS prediction={pred:.4f}")

    w_new, err, post2 = hybrid_update_weights(txt, doc, lbl, prior_prob, w, mu=0.6)
    print(f"Weight update error={err:.4f}, New posterior={post2:.4f}")

    # Build a tiny graph for MST
    nodes = [f"node_{i}" for i in range(5)]
    positions = {n: (random.random(), random.random()) for n in nodes}
    tree = minimum_cost_spanning_tree(nodes, positions, lbl, w_new)
    print("MST edges:", tree)