# DARWIN HAMMER — match 2765, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_bayes_update__m1014_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m760_s0.py (gen5)
# born: 2026-05-29T23:45:52Z

"""
Hybrid Decision‑Hygiene, Bayesian‑Ricci & VRAM‑Fisher‑HDC Fusion

Parents:
- hybrid_hybrid_hybrid_decisi_hybrid_bayes_update__m1014_s0.py (Decision‑Hygiene + Bayesian update)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m760_s0.py (VRAM scheduler + hyperdimensional Fisher‑JEPA)

Mathematical Bridge:
The feature histogram extracted from free‑form text is interpreted as a **prior probability
distribution** over graph nodes (entropy → scalar prior).  The tree built from VRAM‑budget
coordinates supplies a **distance metric**; the inverse distance serves as a **Fisher‑score
likelihood** (information density).  Bayesian marginalisation then fuses prior and
likelihood, while a hyper‑dimensional encoding maps the resulting posterior vector into
a high‑dimensional binary space for downstream similarity or retrieval tasks.
"""

import math
import random
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# 1. Decision‑Hygiene feature extraction and entropy (Parent A)
# ----------------------------------------------------------------------
def extract_features(text: str) -> Dict[str, int]:
    """Extract evidence‑type tokens from *text* using a compiled regex."""
    evidence_re = re.compile(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|"
        r"receipt|sha256|screenshot|record|log|document|proof|fact|facts|check|checked)\b",
        flags=re.IGNORECASE,
    )
    counts: Dict[str, int] = {}
    for m in evidence_re.finditer(text):
        token = m.group().lower()
        counts[token] = counts.get(token, 0) + 1
    return counts


def compute_shannon_entropy(features: Dict[str, int]) -> float:
    """Shannon entropy H = - Σ p_i log p_i for the feature count distribution."""
    total = sum(features.values())
    if total == 0:
        return 0.0
    probs = [v / total for v in features.values()]
    return -sum(p * math.log(p) for p in probs)


# ----------------------------------------------------------------------
# 2. Tree metrics and distance‑based Fisher score (Parent B)
# ----------------------------------------------------------------------
def _euclidean_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """
    Build adjacency, compute Euclidean edge lengths, and root‑to‑node distances.

    Returns
    -------
    adj : dict mapping node → list of neighbours
    edge_len : dict mapping edge (as supplied) → length
    dist : dict mapping node → distance from *root* (sum of edge lengths along the unique path)
    """
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = _euclidean_length(nodes[a], nodes[b])
        edge_len[(b, a)] = edge_len[(a, b)]  # undirected convenience

    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    visited = {root}
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in visited:
                dist[nxt] = dist[cur] + edge_len[(cur, nxt)]
                visited.add(nxt)
                stack.append(nxt)
    return adj, edge_len, dist


def fisher_likelihood(distances: Dict[str, float], epsilon: float = 1e-6) -> Dict[str, float]:
    """
    Convert root‑to‑node distances into a Fisher‑score‑like likelihood.

    The inverse distance (1 / (d + ε)) is used as a proxy for information density;
    the values are then normalised to the unit interval.
    """
    inv = {node: 1.0 / (d + epsilon) for node, d in distances.items()}
    max_inv = max(inv.values()) if inv else 1.0
    return {node: v / max_inv for node, v in inv.items()}


# ----------------------------------------------------------------------
# 3. Bayesian marginalisation (Parent A) with Fisher likelihood (bridge)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """
    Compute the marginal probability P(E) for a single hypothesis:

        P(E) = likelihood * prior + false_positive * (1 - prior)

    All arguments are assumed to lie in [0, 1].
    """
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probability arguments must be in [0, 1].")
    return likelihood * prior + false_positive * (1.0 - prior)


def posterior_update(prior: float, likelihood: float, false_positive: float = 0.01) -> float:
    """
    Return the posterior probability P(H|E) using Bayes rule:

        posterior = (likelihood * prior) / marginal
    """
    marginal = bayes_marginal(prior, likelihood, false_positive)
    if marginal == 0.0:
        return 0.0
    return (likelihood * prior) / marginal


# ----------------------------------------------------------------------
# 4. Hyper‑dimensional encoding (new primitive)
# ----------------------------------------------------------------------
def generate_random_hypervector(dim: int = 10_000) -> np.ndarray:
    """Create a binary (+1 / -1) hypervector of given dimensionality."""
    return np.where(np.random.rand(dim) > 0.5, 1, -1).astype(np.int8)


def hyperdimensional_encode(posterior_dict: Dict[str, float], dim: int = 10_000) -> np.ndarray:
    """
    Encode a dictionary of node → posterior probability into a single hypervector.

    Each node receives a random seed hypervector; the posterior weight scales the
    hypervector before a component‑wise sum.  The final vector is binarised
    (sign function) to obtain a high‑dimensional binary representation.
    """
    rng = np.random.default_rng(42)  # deterministic seeds for reproducibility
    node_vectors = {
        node: rng.choice([-1, 1], size=dim).astype(np.int8) for node in posterior_dict
    }

    aggregate = np.zeros(dim, dtype=np.int32)
    for node, prob in posterior_dict.items():
        aggregate += prob * node_vectors[node]

    # Binarise: positive → 1, non‑positive → -1
    return np.where(aggregate >= 0, 1, -1).astype(np.int8)


# ----------------------------------------------------------------------
# 5. Hybrid pipeline demonstrating the fused mathematics
# ----------------------------------------------------------------------
def hybrid_process(
    text: str,
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    End‑to‑end hybrid routine:

    1. Extract evidence features from *text* and compute Shannon entropy.
    2. Derive a scalar prior from the entropy (higher entropy → lower prior confidence).
    3. Build the tree, obtain root‑to‑node distances, and turn them into Fisher likelihoods.
    4. Perform Bayesian posterior update for each node.
    5. Encode the posterior map into a hyper‑dimensional vector.

    Returns
    -------
    posterior_map : dict node → posterior probability
    hd_vector     : high‑dimensional binary vector encoding the posterior map
    """
    # ---- Step 1 & 2 ----------------------------------------------------
    feats = extract_features(text)
    entropy = compute_shannon_entropy(feats)

    # Map entropy ∈ [0, log|V|] to a prior ∈ (0,1]; simple monotonic transform:
    prior = 1.0 / (1.0 + entropy)  # ensures prior ∈ (0,1]

    # ---- Step 3 ---------------------------------------------------------
    _, _, distances = tree_metrics(nodes, edges, root)
    likelihoods = fisher_likelihood(distances)

    # ---- Step 4 ---------------------------------------------------------
    posterior_map = {
        node: posterior_update(prior, likelihoods.get(node, 0.0))
        for node in nodes
    }

    # ---- Step 5 ---------------------------------------------------------
    hd_vector = hyperdimensional_encode(posterior_map)

    return posterior_map, hd_vector


# ----------------------------------------------------------------------
# 6. Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The evidence was verified and the source was confirmed. "
        "A screenshot and log file were attached as proof."
    )

    # Simple planar tree representing a toy VRAM‑budget layout
    sample_nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
        "E": (0.5, 1.5),
    }
    sample_edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
    root_node = "A"

    posteriors, hd_vec = hybrid_process(sample_text, sample_nodes, sample_edges, root_node)

    print("Posterior probabilities per node:")
    for n, p in posteriors.items():
        print(f"  {n}: {p:.4f}")

    print("\nHyper‑dimensional vector (first 20 components):")
    print(hd_vec[:20])
    print("\nVector shape:", hd_vec.shape)