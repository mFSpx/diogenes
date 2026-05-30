# DARWIN HAMMER — match 1705, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bayes__m1065_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_ternar_m374_s0.py (gen3)
# born: 2026-05-29T23:38:28Z

"""Hybrid Minimum‑Cost Semantic‑Tree + Pheromone‑Modulated Bandit Router

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bayes__m1065_s0.py (Algorithm A)
- hybrid_hybrid_hybrid_pherom_hybrid_hybrid_ternar_m374_s0.py (Algorithm B)

Mathematical Bridge:
Algorithm A provides a scalar store `S(t)` updated from Bayesian pruning
probabilities `p_i`. Algorithm B supplies a pheromone signal `Φ_k` that
decays over time.  The fusion treats the pheromone signal as a multiplicative
modifier of the store inflow and of the semantic similarity term in the edge
weight:

    S′(t) = S(t‑Δt) + α·Σ_i (1‑p_i)·(1 + γ·Φ_i)                (1′)

    w_ij = β₁·d_ij + β₂·(1‑c_ij·(1+γ·Φ_ij)) + β₃·(1‑ℓ_ij) + β₄·(1‑p_ij)   (2′)

where `γ` controls pheromone influence.  The resulting weights feed a
contextual bandit whose confidence bound is scaled by the current store
level `S′(t)`, yielding a single unified dynamical system.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (shared)
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Vector = np.ndarray


@dataclass(frozen=True)
class Document:
    """Identifier and embedding vector."""
    id: str
    embedding: Vector


@dataclass(frozen=True)
class Node:
    """Graph node containing spatial, semantic and label information."""
    id: str
    point: Point
    document: Document
    label: str


@dataclass
class Edge:
    """Edge between two nodes with Bayesian parameters."""
    src: Node
    dst: Node
    prior: float          # Bayesian prior
    likelihood: float     # Bayesian likelihood
    false_positive: float # Bayesian false‑positive rate


# ----------------------------------------------------------------------
# Pheromone subsystem (from Algorithm B)
# ----------------------------------------------------------------------
class HybridPheromoneSystem:
    """Tracks decaying pheromone signals for arbitrary surface keys."""
    def __init__(self):
        self.pheromones: Dict[str, Dict] = {}

    def calculate_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        """Create or update a pheromone entry, returning the current signal value."""
        now = datetime.now(timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {
                "signal_kind": signal_kind,
                "signal_value": signal_value,
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            }
        else:
            entry = self.pheromones[surface_key]
            elapsed = (now - entry["created_time"]).total_seconds()
            decay_factor = 0.5 ** (elapsed / entry["half_life_seconds"])
            decayed_value = entry["signal_value"] * decay_factor
            # Blend old and new value (simple average for illustration)
            entry["signal_value"] = (decayed_value + signal_value) / 2.0
            entry["created_time"] = now
        return self.pheromones[surface_key]["signal_value"]

    @staticmethod
    def calculate_entropy(probabilities: List[float], eps: float = 1e-12) -> float:
        """Shannon entropy of a discrete distribution."""
        total = sum(probabilities)
        if total <= 0:
            raise ValueError("positive probability mass required")
        probs = [p / total for p in probabilities if p > 0]
        return -sum(p * math.log(max(p, eps)) for p in probs)


# ----------------------------------------------------------------------
# Bayesian‑pruning helper (from Algorithm A)
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """
    Compute posterior probability p = P(H|E) assuming binary hypothesis.
    Simple formulation: p = (likelihood * prior) /
                         (likelihood * prior + false_positive * (1‑prior))
    """
    numerator = likelihood * prior
    denominator = numerator + false_positive * (1.0 - prior)
    return numerator / denominator if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Hybrid dynamics
# ----------------------------------------------------------------------
def update_store(
    edges: List[Edge],
    pheromone_system: HybridPheromoneSystem,
    alpha: float,
    gamma: float,
    S_prev: float,
) -> float:
    """
    Hybrid store update (Eq. 1′).

    For each edge we compute its Bayesian marginal p_i and obtain a pheromone
    signal Φ_i keyed by the edge id.  The inflow term (1‑p_i) is scaled by
    (1 + γ·Φ_i) before being summed.
    """
    inflow = 0.0
    for e in edges:
        p_i = bayes_marginal(e.prior, e.likelihood, e.false_positive)
        # Use a deterministic key for reproducibility
        key = f"{e.src.id}->{e.dst.id}"
        phi_i = pheromone_system.calculate_pheromone_signal(
            surface_key=key,
            signal_kind="edge",
            signal_value=random.random(),      # placeholder stochastic signal
            half_life_seconds=30.0,
        )
        inflow += (1.0 - p_i) * (1.0 + gamma * phi_i)
    S_new = S_prev + alpha * inflow
    return S_new


def cosine_similarity(vec_a: Vector, vec_b: Vector) -> float:
    """Standard cosine similarity in [‑1, 1]."""
    dot = float(np.dot(vec_a, vec_b))
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_edge_weight(
    edge: Edge,
    beta: Tuple[float, float, float, float],
    gamma: float,
    pheromone_system: HybridPheromoneSystem,
) -> float:
    """
    Hybrid edge weight (Eq. 2′).

    Components:
    - d_ij : Euclidean distance between node points.
    - c_ij : cosine similarity of document embeddings.
    - ℓ_ij : label agreement score (1 if same label else 0).
    - p_ij : Bayesian marginal.
    The semantic term (1‑c_ij) is scaled by (1+γ·Φ_ij).
    """
    β1, β2, β3, β4 = beta

    # Euclidean distance
    x1, y1 = edge.src.point
    x2, y2 = edge.dst.point
    d_ij = math.hypot(x2 - x1, y2 - y1)

    # Cosine similarity
    c_ij = cosine_similarity(edge.src.document.embedding, edge.dst.document.embedding)

    # Pheromone signal for this edge
    key = f"{edge.src.id}->{edge.dst.id}"
    phi_ij = pheromone_system.calculate_pheromone_signal(
        surface_key=key,
        signal_kind="edge_weight",
        signal_value=random.random(),
        half_life_seconds=30.0,
    )

    # Label agreement (binary)
    ℓ_ij = 1.0 if edge.src.label == edge.dst.label else 0.0

    # Bayesian marginal
    p_ij = bayes_marginal(edge.prior, edge.likelihood, edge.false_positive)

    # Hybrid weight
    weight = (
        β1 * d_ij
        + β2 * (1.0 - c_ij * (1.0 + gamma * phi_ij))
        + β3 * (1.0 - ℓ_ij)
        + β4 * (1.0 - p_ij)
    )
    return weight


def select_edge_bandit(edges: List[Edge], store_level: float, gamma: float) -> Edge:
    """
    Contextual bandit selector.

    For each edge we compute its hybrid weight, then construct an Upper
    Confidence Bound (UCB) score:

        UCB_i = w_i - store_level * sqrt(ln(T) / (n_i + 1))

    where T is the total number of selections so far (here approximated by
    len(edges)) and n_i is a placeholder visit count (here 0).  The edge with
    the minimal UCB (i.e. most promising) is returned.
    """
    pheromone_system = HybridPheromoneSystem()
    beta = (1.0, 1.0, 1.0, 1.0)  # simple equal weighting for demo
    T = max(len(edges), 1)
    best_edge = None
    best_score = float("inf")
    for e in edges:
        w = compute_edge_weight(e, beta, gamma, pheromone_system)
        # Simplified exploration term (n_i = 0)
        exploration = math.sqrt(math.log(T + 1))
        ucb = w - store_level * exploration
        if ucb < best_score:
            best_score = ucb
            best_edge = e
    return best_edge


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
def _build_dummy_graph(num_nodes: int = 5) -> List[Edge]:
    """Create a small fully‑connected graph with random data for testing."""
    nodes: List[Node] = []
    for i in range(num_nodes):
        point = (random.uniform(0, 10), random.uniform(0, 10))
        embedding = np.random.rand(8)  # 8‑dimensional embedding
        doc = Document(id=f"doc{i}", embedding=embedding)
        label = random.choice(["A", "B"])
        nodes.append(Node(id=f"n{i}", point=point, document=doc, label=label))

    edges: List[Edge] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            prior = random.random()
            likelihood = random.random()
            false_positive = random.random() * 0.2  # keep small
            edges.append(Edge(src=nodes[i], dst=nodes[j],
                              prior=prior, likelihood=likelihood,
                              false_positive=false_positive))
    return edges


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Build test graph
    edges = _build_dummy_graph()

    # Initialize pheromone system and store
    pheromone_system = HybridPheromoneSystem()
    S = 0.0

    # Hyper‑parameters
    alpha = 0.5
    gamma = 0.3
    beta = (1.0, 1.0, 1.0, 1.0)

    # Perform a single hybrid iteration
    S = update_store(edges, pheromone_system, alpha, gamma, S)
    chosen_edge = select_edge_bandit(edges, S, gamma)

    print(f"Updated store level S = {S:.4f}")
    print(f"Chosen edge: {chosen_edge.src.id} -> {chosen_edge.dst.id}")
    print(f"Edge weight: {compute_edge_weight(chosen_edge, beta, gamma, pheromone_system):.4f}")