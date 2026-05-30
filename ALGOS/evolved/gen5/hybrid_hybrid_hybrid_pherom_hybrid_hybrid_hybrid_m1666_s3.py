# DARWIN HAMMER — match 1666, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_pheromone_hyb_hybrid_minimum_cost__m792_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_bayes_claim_k_m9_s3.py (gen4)
# born: 2026-05-29T23:38:06Z

"""Hybrid Pheromone‑Tree‑Bayesian‑Entropy Algorithm
Parents:
- hybrid_hybrid_pheromone_hyb_hybrid_minimum_cost__m792_s2.py (perceptual hashing, Hamming‑based likelihood, Bayesian edge update)
- hybrid_hybrid_hybrid_pherom_hybrid_bayes_claim_k_m9_s3.py (pheromone probability extraction, decision‑hygiene scoring, Shannon entropy)

Mathematical Bridge:
1. Each node is represented by a numeric feature vector.  The vector is reduced to a
   64‑bit perceptual hash (Parent A).  The normalized Hamming similarity
   `sim = 1 - d/64` between two node hashes is interpreted as a data‑driven
   likelihood `L` that an edge between the nodes is relevant.
2. The edge prior `π` (e.g. a physical‑distance‑based cost) is updated with the
   Bayesian marginal `π' = bayes_marginal(π, L, ε)` where `ε` is a configurable
   false‑positive rate (Parent A).
3. From the surface‑usage side (Parent B) we obtain a pheromone probability
   vector `p`.  Its Shannon entropy `H(p)` measures the uncertainty of surface
   usage.  Separately, a free‑form text is parsed into decision‑hygiene scores;
   the entropy of the score distribution `H_hyg` captures the “cleanliness” of
   the decision context.
4. Both entropies are combined into a scaling factor `σ = 1 + α·H(p) + β·H_hyg`
   (α,β are tunable).  The final hybrid edge weight is `w = σ·π'`.  This weight
   simultaneously respects physical cost, pheromone evidence, Bayesian belief,
   and information‑theoretic confidence.

The module implements the above pipeline and provides a minimal spanning‑tree
construction using the hybrid edge weights.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Perceptual hashing utilities (Parent A)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Return a 64‑bit perceptual hash of a list of floats."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    bits <<= max(0, 64 - len(values))  # pad with zeros if needed
    return bits & ((1 << 64) - 1)


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64‑bit integers."""
    return (a ^ b).bit_count()


def hamming_similarity(a: int, b: int) -> float:
    """Normalized similarity in [0,1] derived from Hamming distance."""
    return 1.0 - hamming_distance(a, b) / 64.0


# ----------------------------------------------------------------------
# Bayesian utilities (Parent A)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """
    Compute posterior P(H|E) given:
        prior          = P(H)
        likelihood     = P(E|H)
        false_positive = P(E|¬H)
    Using Bayes rule:
        P(H|E) = P(E|H)P(H) / [P(E|H)P(H) + P(E|¬H)P(¬H)]
    """
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0,1].")
    numerator = likelihood * prior
    denominator = numerator + false_positive * (1.0 - prior)
    return numerator / denominator if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Pheromone probability extraction (Parent B – mocked for offline use)
# ----------------------------------------------------------------------
def calculate_pheromone_probabilities(surface_key: str, limit: int) -> List[float]:
    """
    Mocked version: generate `limit` random positive values and normalize them.
    In a real deployment this would query a PostgreSQL DB.
    """
    random.seed(hash(surface_key) & 0xffffffff)
    raw = [random.random() for _ in range(limit)]
    total = sum(raw) or 1.0
    return [v / total for v in raw]


# ----------------------------------------------------------------------
# Decision hygiene scoring and entropy (Parent B)
# ----------------------------------------------------------------------
def decision_hygiene_scores(text: str) -> Dict[str, int]:
    """Calculate simple hygiene scores based on keyword occurrence."""
    # Keywords are deliberately shortened for the mock implementation.
    keywords = {
        "evidence": ["evidence", "verify", "verified", "confirm", "confirmed",
                     "source", "sourced", "citation", "receipt", "hash",
                     "sha256", "screenshot", "record", "log", "document",
                     "proof", "fact", "facts", "check", "checked", "audit"],
        "planning": ["plan", "checklist", "steps", "step", "sequence", "timeline",
                     "roadmap", "phase", "priority", "prioritize", "triage",
                     "criteria", "protocol", "procedure", "schedule", "budget",
                     "scope", "test", "smoke"],
        "delay": ["pause", "sleep", "wait", "tomorrow", "later", "hold",
                  "cool down", "de-escalate", "not now", "first", "after",
                  "review"]
    }
    scores = {cat: 0 for cat in keywords}
    lowered = text.lower()
    for cat, words in keywords.items():
        for w in words:
            if w in lowered:
                scores[cat] += 1
    return scores


def shannon_entropy(probs: List[float]) -> float:
    """Standard Shannon entropy (base‑e)."""
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def hygiene_entropy(text: str) -> float:
    """Entropy of the normalized hygiene scores."""
    scores = decision_hygiene_scores(text)
    total = sum(scores.values()) or 1.0
    probs = [v / total for v in scores.values()]
    return shannon_entropy(probs)


# ----------------------------------------------------------------------
# Hybrid edge weight computation
# ----------------------------------------------------------------------
def hybrid_edge_weight(
    node_a_vals: List[float],
    node_b_vals: List[float],
    prior: float,
    false_positive: float,
    surface_key: str,
    hygiene_text: str,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """
    Compute a hybrid edge weight that fuses:
        * Physical / prior cost (`prior`)
        * Pheromone evidence via perceptual‑hash similarity (likelihood)
        * Bayesian posterior update
        * Entropic scaling from pheromone distribution and decision hygiene
    """
    # 1. Perceptual hashes and similarity → likelihood
    h_a = compute_phash(node_a_vals)
    h_b = compute_phash(node_b_vals)
    likelihood = hamming_similarity(h_a, h_b)

    # 2. Bayesian posterior
    posterior = bayes_marginal(prior, likelihood, false_positive)

    # 3. Entropy scaling
    pheromone_probs = calculate_pheromone_probabilities(surface_key, limit=10)
    H_pheromone = shannon_entropy(pheromone_probs)

    H_hygiene = hygiene_entropy(hygiene_text)

    sigma = 1.0 + alpha * H_pheromone + beta * H_hygiene

    # 4. Final hybrid weight (lower is better for a minimum‑cost tree)
    return sigma * posterior


# ----------------------------------------------------------------------
# Simple Minimum‑Cost Spanning Tree using hybrid edge weights (Prim's algorithm)
# ----------------------------------------------------------------------
def prim_mst(
    nodes: List[str],
    features: Dict[str, List[float]],
    priors: Dict[Tuple[str, str], float],
    false_positive: float,
    surface_key: str,
    hygiene_text: str,
) -> List[Tuple[str, str, float]]:
    """
    Build a Minimum Spanning Tree (MST) over `nodes` using hybrid edge weights.
    Returns a list of edges (u, v, weight) belonging to the MST.
    """
    if not nodes:
        return []

    in_tree = {nodes[0]}
    edges = []

    # Pre‑compute all possible edge weights lazily
    while len(in_tree) < len(nodes):
        best_edge = None
        best_weight = math.inf

        for u in in_tree:
            for v in nodes:
                if v in in_tree:
                    continue
                weight = hybrid_edge_weight(
                    node_a_vals=features[u],
                    node_b_vals=features[v],
                    prior=priors.get((u, v), priors.get((v, u), 1.0)),
                    false_positive=false_positive,
                    surface_key=surface_key,
                    hygiene_text=hygiene_text,
                )
                if weight < best_weight:
                    best_weight = weight
                    best_edge = (u, v, weight)

        if best_edge is None:
            raise RuntimeError("Graph disconnected; no connecting edge found.")
        u, v, w = best_edge
        edges.append(best_edge)
        in_tree.add(v)

    return edges


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic graph
    node_names = ["A", "B", "C", "D"]
    # Random feature vectors (length 20) for each node
    random.seed(42)
    features = {n: [random.random() for _ in range(20)] for n in node_names}
    # Prior costs based on Euclidean distance (simulated)
    priors = {}
    for i, u in enumerate(node_names):
        for v in node_names[i + 1 :]:
            dist = math.dist(features[u], features[v])
            priors[(u, v)] = dist  # smaller distance → lower prior cost

    mst = prim_mst(
        nodes=node_names,
        features=features,
        priors=priors,
        false_positive=0.1,
        surface_key="demo_surface",
        hygiene_text="We have verified the source and recorded the log.",
    )

    print("Hybrid MST edges (u, v, weight):")
    for e in mst:
        print(e)