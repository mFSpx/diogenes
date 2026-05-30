# DARWIN HAMMER — match 5068, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_semant_hybrid_hybrid_hybrid_m1918_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1456_s1.py (gen6)
# born: 2026-05-29T23:59:36Z

"""Hybrid Fusion Module
Combines the graph‑centric, pheromone‑driven semantics of
`hybrid_hybrid_hybrid_semant_hybrid_hybrid_hybrid_m1918_s1.py` (Parent A)
with the high‑dimensional morphological encoding and similarity
machinery of `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1456_s1.py`
(Parent B).

Mathematical Bridge
-------------------
* Parent A supplies a probabilistic weighting `p_i` (pheromone
  probabilities), a curvature term `κ_i` (Ollivier‑Ricci) and a
  logarithmic count prior `ℓ_i = log(count_i)`.
* Parent B supplies a deterministic embedding `e_i = hybrid_encode(m_i,
  t)` for each node `i` (morphology `m_i`, shared query text `t`) and a
  similarity measure (cosine similarity, optionally refined by SSIM).

The fused score for node *i* is defined as  

    S_i = p_i · κ_i · ℓ_i · cos(e_i, e_q)

where `e_q` is the embedding of the query text.  This product
mathematically merges the stochastic search pressure of A with the
deterministic high‑dimensional representation of B, yielding a single
ranking that respects both pheromone dynamics and morphological
similarity.
"""

import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Shared utilities (derived from Parent A)
# ----------------------------------------------------------------------
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    den = math.sqrt(float(np.dot(a, a))) * math.sqrt(float(np.dot(b, b)))
    return 0.0 if den == 0 else float(np.dot(a, b)) / den


def pheromone_probabilities(pheromones):
    total = sum(pheromones)
    if total == 0:
        raise ValueError("Pheromone list must contain positive values")
    return [p / total for p in pheromones]


def log_count_statistics(nodes, edges):
    """Count incident edges per node and return log‑counts."""
    log_counts = defaultdict(int)
    for a, b in edges:
        log_counts[a] += 1
        log_counts[b] += 1
    return {node: math.log(count) if count > 0 else 0.0
            for node, count in log_counts.items()}


def ollivier_ricci_curvature(nodes, edges):
    """Very lightweight curvature proxy: average Euclidean distance to neighbours."""
    curvature = {}
    for node in nodes:
        neighbors = [b for a, b in edges if a == node] + [a for a, b in edges if b == node]
        if not neighbors:
            curvature[node] = 0.0
            continue
        dists = [math.hypot(nodes[node][0] - nodes[n][0],
                           nodes[node][1] - nodes[n][1]) for n in neighbors]
        curvature[node] = sum(dists) / len(dists)
    return curvature


# ----------------------------------------------------------------------
# Morphology‑based encoding (derived from Parent B)
# ----------------------------------------------------------------------
@dataclass
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def morphology_vector(morphology: Morphology, dim: int = 10000) -> np.ndarray:
    """Repeat the 4‑D morphology to fill a high‑dimensional vector."""
    base = np.array([morphology.length,
                     morphology.width,
                     morphology.height,
                     morphology.mass], dtype=np.float32)
    repeats = dim // 4 + 1
    vec = np.tile(base, repeats)[:dim]
    return vec


def min_hash(text: str, dim: int = 10000) -> np.ndarray:
    """Binary min‑hash of a string."""
    return np.array([hash(text + str(i)) & 1 for i in range(dim)], dtype=np.float32)


def hybrid_encode(morphology: Morphology, text: str, dim: int = 10000) -> np.ndarray:
    """
    Combine a morphology vector with a binary hash of the text.
    The hash is scaled by the mean of the morphology vector,
    yielding a deterministic high‑dimensional embedding.
    """
    morph_vec = morphology_vector(morphology, dim)
    text_hash = min_hash(text, dim)
    prior = float(np.mean(morph_vec))
    return text_hash * prior


# ----------------------------------------------------------------------
# Fusion core (three public functions)
# ----------------------------------------------------------------------
def compute_fused_embeddings(node_ids, morphologies, query_text, dim=2000):
    """
    Produce an embedding for every node using Parent B's `hybrid_encode`,
    and also an embedding for the query text (using a dummy morphology
    that represents the query itself).
    Returns a dict ``{node_id: embedding}`` and the query embedding.
    """
    # Dummy morphology for the pure query (all ones) – it does not bias the hash
    dummy_morph = Morphology(1.0, 1.0, 1.0, 1.0)
    query_emb = hybrid_encode(dummy_morph, query_text, dim)

    node_embeddings = {}
    for nid in node_ids:
        node_embeddings[nid] = hybrid_encode(morphologies[nid], query_text, dim)
    return node_embeddings, query_emb


def fused_node_score(node_id, node_emb, query_emb,
                     prob_weight, curvature, log_count):
    """
    Compute the fused score S_i = p_i · κ_i · ℓ_i · cos(e_i, e_q)
    as described in the module docstring.
    """
    cosine = _cos(node_emb, query_emb)
    return prob_weight * curvature * log_count * cosine


def hybrid_fused_search(query_text, nodes, edges, pheromones,
                        morphologies, dim=2000, top_k=3):
    """
    End‑to‑end fused search:
      1. Derive probabilities, curvature and log‑counts (Parent A).
      2. Encode every node and the query (Parent B).
      3. Score each node with the fused formula.
      4. Return the top‑k node identifiers sorted by descending score.
    """
    node_ids = list(nodes.keys())
    if len(node_ids) != len(pheromones):
        raise ValueError("Number of pheromone values must match number of nodes")

    # 1. Stochastic & geometric priors
    probs = pheromone_probabilities(pheromones)
    curvature = ollivier_ricci_curvature(nodes, edges)
    log_counts = log_count_statistics(nodes, edges)

    # 2. High‑dimensional embeddings
    node_embs, query_emb = compute_fused_embeddings(node_ids,
                                                    morphologies,
                                                    query_text,
                                                    dim)

    # 3. Scoring
    scores = {}
    for idx, nid in enumerate(node_ids):
        p = probs[idx]
        κ = curvature.get(nid, 0.0)
        ℓ = log_counts.get(nid, 0.0)
        e_i = node_embs[nid]
        scores[nid] = fused_node_score(nid, e_i, query_emb, p, κ, ℓ)

    # 4. Ranking
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


# ----------------------------------------------------------------------
# Auxiliary function from Parent B (kept for completeness)
# ----------------------------------------------------------------------
def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Structural Similarity Index Measure (SSIM) for two 1‑D arrays."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return float(numerator / denominator) if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple planar graph with 4 nodes
    nodes = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (0.0, 1.0),
        3: (1.0, 1.0)
    }
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 2)]

    # Pheromone levels (arbitrary positive numbers)
    pheromones = [10.0, 5.0, 8.0, 3.0]

    # Assign a distinct morphology to each node
    morphologies = {
        0: Morphology(1.2, 0.5, 0.3, 2.0),
        1: Morphology(0.9, 0.7, 0.4, 1.5),
        2: Morphology(1.0, 0.6, 0.5, 1.8),
        3: Morphology(1.1, 0.55, 0.45, 1.9)
    }

    query = "sample query text for hybrid search"

    top = hybrid_fused_search(query, nodes, edges, pheromones,
                              morphologies, dim=2000, top_k=4)

    print("Top ranked nodes (id, score):")
    for nid, score in top:
        print(f"Node {nid}: {score:.6f}")

    # Verify auxiliary SSIM works without error
    a = np.random.rand(100)
    b = np.random.rand(100)
    print("SSIM of random vectors:", compute_ssim(a, b))