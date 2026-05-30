# DARWIN HAMMER — match 2768, survivor 8
# gen: 4
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_decision_hygi_m37_s1.py (gen2)
# parent_b: hybrid_hybrid_rbf_surrogate_hybrid_hard_truth_ma_m93_s3.py (gen3)
# born: 2026-05-29T23:45:54Z

import re
import math
import random
import sys
from pathlib import Path
from typing import Mapping, Sequence, Hashable, Set, List, Dict, Tuple

import numpy as np

Vector = Sequence[float]
Node = Hashable
Graph = Mapping[Node, Set[Node]]
FeatureVec = Sequence[float]


def calculate_pheromone_probabilities(
    surface_key: str, limit: int, db_url: str | None = None
) -> np.ndarray:
    """
    Return a normalized probability vector derived from recent pheromone signals.
    If ``db_url`` is ``None`` a synthetic uniform distribution is returned (useful for testing).

    The original implementation queried a PostgreSQL table; this version falls back to
    a deterministic mock to keep the module self‑contained.
    """
    if db_url is None:
        # mock: uniform probabilities
        probs = np.full(limit, 1.0 / limit, dtype=float)
        return probs

    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT signal_value
                FROM lucidota_runtime.surface_pheromone
                WHERE surface_key=%s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (surface_key, limit),
            )
            rows = cur.fetchall()
            values = np.array([r["signal_value"] for r in rows], dtype=float)
            if values.size == 0:
                raise ValueError("No pheromone records found.")
            total = values.sum()
            return values / total


def decision_hygiene_scores(text: str) -> Dict[str, int]:
    """
    Count occurrences of predefined lexical categories in *text*.
    Returns a mapping ``category -> count``.
    """
    EVIDENCE_RE = re.compile(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
        re.I,
    )
    PLANNING_RE = re.compile(
        r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
        re.I,
    )
    DELAY_RE = re.compile(
        r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|first|after|review)\b",
        re.I,
    )
    SUPPORT_RE = re.compile(
        r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
        re.I,
    )
    BOUNDARY_RE = re.compile(
        r"\b(?:boundary|boundaries|wall|limit|restriction|rule|policy|protocol|norm)\b", re.I
    )

    categories = {
        "evidence": len(EVIDENCE_RE.findall(text)),
        "planning": len(PLANNING_RE.findall(text)),
        "delay": len(DELAY_RE.findall(text)),
        "support": len(SUPPORT_RE.findall(text)),
        "boundary": len(BOUNDARY_RE.findall(text)),
    }
    return categories


def decision_hygiene_entropy(text: str) -> float:
    """
    Compute the Shannon entropy of the decision‑hygiene category distribution.
    The entropy is normalised by log(k) where k is the number of categories,
    yielding a value in [0, 1].
    """
    counts = decision_hygiene_scores(text)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts.values()], dtype=float)
    # avoid log(0) by masking zero entries
    mask = probs > 0
    H = -np.sum(probs[mask] * np.log2(probs[mask]))
    H_norm = H / math.log2(len(probs))  # normalise
    return H_norm


def euclidean(a: Vector, b: Vector) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """RBF Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


def rbf_surrogate_similarity(
    features: Mapping[Node, FeatureVec],
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Build the symmetric similarity matrix S where
        S_{ij} = exp(-ε² ‖f_i - f_j‖²).

    Returns a 2‑D NumPy array indexed by the order of ``features.keys()``.
    """
    nodes = list(features.keys())
    n = len(nodes)
    S = np.empty((n, n), dtype=float)

    for i in range(n):
        S[i, i] = 1.0  # distance zero → kernel = 1
        for j in range(i + 1, n):
            dist = euclidean(features[nodes[i]], features[nodes[j]])
            sim = gaussian(dist, epsilon)
            S[i, j] = S[j, i] = sim
    return S


def hybrid_broadcast_probabilities(
    graph: Graph,
    features: Mapping[Node, FeatureVec],
    pheromone_probs: Mapping[Node, float],
    hygiene_entropy: float,
    epsilon: float = 1.0,
) -> Dict[Node, float]:
    """
    Compute hybrid broadcast probabilities for each node.

    For node i:
        π_i = p_i · (1‑H) · mean_{j∈N(i)} K_{ij}

    where:
        p_i          – pheromone prior (from ``pheromone_probs``)
        H            – normalised Shannon entropy (0 ≤ H ≤ 1)
        K_{ij}       – Gaussian RBF similarity between feature vectors of i and j
        N(i)         – neighbour set of i in ``graph``

    The resulting vector is normalised to sum to 1.
    """
    # Map node -> index for matrix lookup
    node_list = list(features.keys())
    idx_of = {node: idx for idx, node in enumerate(node_list)}

    # Compute full similarity matrix once
    S = rbf_surrogate_similarity(features, epsilon)

    raw_probs: Dict[Node, float] = {}
    for node, neighbours in graph.items():
        if node not in idx_of:
            raise KeyError(f"Node {node!r} missing from features.")
        i = idx_of[node]
        if not neighbours:
            neighbor_sim = 0.0
        else:
            sims = [
                S[i, idx_of[nbr]]
                for nbr in neighbours
                if nbr in idx_of
            ]
            neighbor_sim = sum(sims) / len(sims) if sims else 0.0

        prior = pheromone_probs.get(node, 0.0)
        raw = prior * (1.0 - hygiene_entropy) * neighbor_sim
        raw_probs[node] = raw

    total = sum(raw_probs.values())
    if total == 0.0:
        # fallback to uniform distribution
        uniform = 1.0 / len(raw_probs) if raw_probs else 0.0
        return {node: uniform for node in raw_probs}
    return {node: val / total for node, val in raw_probs.items()}


def kl_divergence(p: Dict[Node, float], q: Dict[Node, float]) -> float:
    """
    Compute the KL divergence between two probability distributions.

    Args:
        p: A probability distribution.
        q: A probability distribution.

    Returns:
        The KL divergence between p and q.
    """
    kl_div = 0.0
    for node in p:
        if p[node] > 0:
            kl_div += p[node] * math.log(p[node] / q[node])
    return kl_div


def regularized_hybrid_broadcast_probabilities(
    graph: Graph,
    features: Mapping[Node, FeatureVec],
    pheromone_probs: Mapping[Node, float],
    hygiene_entropy: float,
    epsilon: float = 1.0,
    alpha: float = 0.1,
) -> Dict[Node, float]:
    """
    Compute regularized hybrid broadcast probabilities for each node.

    For node i:
        π_i = (1 - alpha) * p_i · (1‑H) · mean_{j∈N(i)} K_{ij} + alpha * uniform

    where:
        p_i          – pheromone prior (from ``pheromone_probs``)
        H            – normalised Shannon entropy (0 ≤ H ≤ 1)
        K_{ij}       – Gaussian RBF similarity between feature vectors of i and j
        N(i)         – neighbour set of i in ``graph``
        alpha        – regularization parameter

    The resulting vector is normalised to sum to 1.
    """
    probs = hybrid_broadcast_probabilities(
        graph, features, pheromone_probs, hygiene_entropy, epsilon
    )
    uniform_probs = {node: 1.0 / len(probs) for node in probs}
    return {
        node: (1 - alpha) * probs[node] + alpha * uniform_probs[node]
        for node in probs
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic graph with 5 nodes
    nodes = ["A", "B", "C", "D", "E"]
    graph: Graph = {
        "A": {"B", "C"},
        "B": {"A", "D"},
        "C": {"A", "D"},
        "D": {"B", "C", "E"},
        "E": {"D"},
    }

    features = {
        "A": [1.0, 2.0],
        "B": [2.0, 3.0],
        "C": [1.0, 1.0],
        "D": [3.0, 2.0],
        "E": [4.0, 1.0],
    }

    pheromone_probs = {
        "A": 0.2,
        "B": 0.3,
        "C": 0.1,
        "D": 0.2,
        "E": 0.2,
    }

    hygiene_entropy = 0.5

    probs = regularized_hybrid_broadcast_probabilities(
        graph, features, pheromone_probs, hygiene_entropy
    )

    print(probs)