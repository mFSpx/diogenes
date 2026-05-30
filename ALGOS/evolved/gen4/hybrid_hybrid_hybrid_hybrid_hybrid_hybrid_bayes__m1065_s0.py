# DARWIN HAMMER — match 1065, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_semantic_neighbors_m298_s1.py (gen3)
# parent_b: hybrid_hybrid_bayes_claim_k_hybrid_bandit_router_m133_s0.py (gen3)
# born: 2026-05-29T23:32:36Z

"""Hybrid Minimum-Cost Semantic Tree with Bayesian‑Bandit Store

Parents:
- **hybrid_hybrid_hybrid_minimu_semantic_neighbors_m298_s1.py** – provides
  Euclidean distance, Bayesian marginal/update, and label scoring used to
  weight edges of a minimum‑cost tree enriched with cosine similarity of
  associated documents.
- **hybrid_hybrid_bayes_claim_k_hybrid_bandit_router_m133_s0.py** – supplies a
  coupled Bayesian‑pruning / contextual‑bandit mechanism whose scalar store
  `S(t)` is driven by pruning probabilities.

Mathematical Bridge:
Both parents expose a scalar probability that can modulate a shared resource.
We treat the *pruning probability* `p_i = bayes_marginal(prior, likelihood,
false_positive)` from the Bayesian side as the inflow to the store `S` of the
bandit side:


S(t) = S(t‑Δt) + α * Σ_i (1 - p_i)            (1)


Edge weights `w_ij` of the tree are built from a convex combination of
physical distance `d_ij`, semantic similarity `c_ij` (cosine), label score
`ℓ_ij`, and the Bayesian marginal `p_ij`:


w_ij = β1 * d_ij + β2 * (1 - c_ij) + β3 * (1 - ℓ_ij) + β4 * (1 - p_ij)   (2)


The resulting weight feeds a contextual bandit that selects the next edge to
explore. The bandit’s confidence bound is scaled by the current store level,
linking the two subsystems into a single hybrid dynamical system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Vector = np.ndarray


@dataclass(frozen=True)
class Document:
    """A simple document representation: identifier and embedding vector."""
    id: str
    embedding: Vector


@dataclass(frozen=True)
class Node:
    """Graph node containing spatial and semantic information."""
    id: str
    point: Point
    document: Document
    label: str


@dataclass
class Edge:
    """Edge between two nodes together with dynamic probabilities."""
    src: Node
    dst: Node
    prior: float               # prior probability for Bayesian update
    likelihood: float          # likelihood of evidence on this edge
    false_positive: float      # false‑positive rate for marginal
    weight: float = 0.0        # computed via (2)
    posterior: float = 0.0     # Bayesian posterior after update


# ----------------------------------------------------------------------
# Helper mathematics (Parent A)
# ----------------------------------------------------------------------
def euclidean_length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def cosine_similarity(v1: Vector, v2: Vector) -> float:
    """Cosine similarity in [0, 1] (clipped for numerical safety)."""
    if v1.size == 0 or v2.size == 0:
        return 0.0
    dot = float(np.dot(v1, v2))
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return max(0.0, min(1.0, dot / norm))


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability P(E) = L·P + FP·(1‑P)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0, 1].")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Posterior P(H|E) = P·L / P(E)."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0.")
    return prior * likelihood / marginal


def label_score(text: str, label: str) -> float:
    """
    Very simple deterministic label scoring:
    proportion of characters shared between `text` and `label`.
    Returns a value in [0, 1].
    """
    if not text or not label:
        return 0.0
    shared = set(text.lower()) & set(label.lower())
    return len(shared) / max(len(set(text.lower())), len(set(label.lower())))


# ----------------------------------------------------------------------
# Store dynamics (Parent B)
# ----------------------------------------------------------------------
def update_store(S: float, pruning_probs: List[float], alpha: float) -> float:
    """
    Equation (1): S(t) = S(t‑Δt) + α * Σ_i (1 - p_i)
    where p_i are pruning probabilities (marginals).
    """
    inflow = sum(1.0 - p for p in pruning_probs)
    return S + alpha * inflow


def confidence_bound(base_conf: float, store: float, scale: float = 1.0) -> float:
    """
    Scale the bandit's confidence term by the current store level.
    Larger store → more exploration (higher bound).
    """
    return base_conf + scale * math.sqrt(store)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def compute_edge_weight(
    edge: Edge,
    beta: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> float:
    """
    Implements equation (2). Returns a non‑negative weight.
    """
    b1, b2, b3, b4 = beta

    # physical distance (scaled)
    d = euclidean_length(edge.src.point, edge.dst.point)

    # semantic cosine similarity
    c = cosine_similarity(edge.src.document.embedding, edge.dst.document.embedding)

    # label relevance
    ℓ = label_score(edge.src.label, edge.dst.label)

    # Bayesian marginal (pruning probability)
    p = bayes_marginal(edge.prior, edge.likelihood, edge.false_positive)

    # Combine – note that higher similarity / higher label score / higher
    # marginal should *reduce* the cost, therefore we use (1 - …) where needed.
    weight = b1 * d + b2 * (1.0 - c) + b3 * (1.0 - ℓ) + b4 * (1.0 - p)
    return max(0.0, weight)


def bayesian_prune_and_update(edges: List[Edge]) -> List[float]:
    """
    For each edge compute its marginal (pruning probability) and then update its
    posterior using Bayes rule. Returns the list of marginals for store update.
    """
    marginals = []
    for e in edges:
        m = bayes_marginal(e.prior, e.likelihood, e.false_positive)
        post = bayes_update(e.prior, e.likelihood, m)
        e.posterior = post
        e.prior = post          # cascade posterior as next prior
        marginals.append(m)
    return marginals


def bandit_select_edge(
    edges: List[Edge],
    store: float,
    exploration_coeff: float = 1.0,
) -> Edge:
    """
    Contextual bandit that picks the edge with the lowest (weight - confidence).
    The confidence term is scaled by the current store level.
    """
    best_edge = None
    best_score = float("inf")
    for e in edges:
        # Simple base confidence proportional to posterior (higher confidence for
        # more probable edges)
        base_conf = math.sqrt(e.posterior)
        conf = confidence_bound(base_conf, store, scale=exploration_coeff)
        score = e.weight - conf
        if score < best_score:
            best_score = score
            best_edge = e
    return best_edge


def hybrid_iteration(
    edges: List[Edge],
    store: float,
    alpha: float,
    beta: Tuple[float, float, float, float],
    exploration_coeff: float = 1.0,
) -> Tuple[Edge, float]:
    """
    Executes one hybrid cycle:
    1. Compute edge weights (2).
    2. Perform Bayesian pruning + posterior update.
    3. Update the store `S` using (1).
    4. Select an edge via the bandit policy.
    Returns the selected edge and the new store value.
    """
    # 1. weight computation
    for e in edges:
        e.weight = compute_edge_weight(e, beta)

    # 2. Bayesian prune & update
    marginals = bayesian_prune_and_update(edges)

    # 3. Store dynamics
    new_store = update_store(store, marginals, alpha)

    # 4. Bandit selection
    chosen = bandit_select_edge(edges, new_store, exploration_coeff)

    return chosen, new_store


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create two simple nodes with deterministic embeddings
    doc_a = Document(id="docA", embedding=np.array([1.0, 0.0, 0.0]))
    doc_b = Document(id="docB", embedding=np.array([0.0, 1.0, 0.0]))
    node_a = Node(id="A", point=(0.0, 0.0), document=doc_a, label="alpha")
    node_b = Node(id="B", point=(3.0, 4.0), document=doc_b, label="beta")

    # Initialise a single edge with arbitrary Bayesian parameters
    edge = Edge(
        src=node_a,
        dst=node_b,
        prior=0.6,
        likelihood=0.7,
        false_positive=0.2,
    )

    # Hyper‑parameters
    alpha = 0.5                     # store inflow scaling
    beta = (0.3, 0.3, 0.2, 0.2)      # weighting of distance, cosine, label, marginal
    store = 1.0                     # initial store level

    # Run a few hybrid iterations
    for step in range(5):
        chosen_edge, store = hybrid_iteration(
            edges=[edge],
            store=store,
            alpha=alpha,
            beta=beta,
            exploration_coeff=0.5,
        )
        print(
            f"Step {step+1}: chosen edge {chosen_edge.src.id}->{chosen_edge.dst.id}, "
            f"weight={chosen_edge.weight:.4f}, posterior={chosen_edge.posterior:.4f}, "
            f"store={store:.4f}"
        )

    sys.exit(0)