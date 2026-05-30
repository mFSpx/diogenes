# DARWIN HAMMER — match 5632, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_minimum_cost__hybrid_gliner_zero_s_m185_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2508_s0.py (gen6)
# born: 2026-05-30T00:03:47Z

import math
import numpy as np
from typing import List, Tuple


class MathAction:
    """
    Represents an action with an identifier and associated economic parameters.
    The identifier is used to generate a deterministic 64‑dimensional ternary vector
    for geometric‑algebra blending.
    """
    def __init__(self, id: str, expected_value: float, cost: float = 0.0, risk: float = 0.0):
        self.id = id
        self.expected_value = expected_value
        self.cost = cost
        self.risk = risk

    def ternary_vector(self) -> np.ndarray:
        """Generate a 64‑bit ternary (0/1) vector from the hash of the identifier."""
        h = hash(self.id) & ((1 << 64) - 1)          # 64‑bit unsigned hash
        bits = np.unpackbits(np.array([h], dtype='>u8').view(np.uint8))
        return bits.astype(np.float64)               # shape (64,)


class MathCounterfactual:
    def __init__(self, action_id: str, outcome_value: float, probability: float = 1.0):
        self.action_id = action_id
        self.outcome_value = outcome_value
        self.probability = probability


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _hash(i: int, token: str) -> int:
    """Simple deterministic hash used by `signature`."""
    return hash((i, token)) & ((1 << 64) - 1)


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """Min‑hash style signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity of two min‑hash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def ternary_vector_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Proportion of equal components in two ternary vectors."""
    if vec_a.shape != vec_b.shape:
        raise ValueError('vectors must have equal shape')
    return np.mean(vec_a == vec_b)


def _blend_vectors(v_a: np.ndarray, v_b: np.ndarray, weight: float) -> np.ndarray:
    """
    Geometric‑algebra inspired blend:
    result = (1‑weight) * v_a + weight * v_b
    The weight is derived from regret (higher regret → more influence of v_b).
    """
    return (1.0 - weight) * v_a + weight * v_b


def hybrid_action_space(actions: List[MathAction]) -> np.ndarray:
    """
    Returns an (n, n) matrix where entry (i, j) is the blended similarity
    between actions i and j, weighted by the regret of action j.
    """
    n = len(actions)
    if n == 0:
        return np.empty((0, 0))

    # Regret‑based scalar weights in [0,1]
    regrets = np.array([a.expected_value - a.cost - a.risk for a in actions])
    max_regret = np.max(np.abs(regrets)) if np.any(regrets) else 1.0
    weights = (regrets - np.min(regrets)) / (max_regret + 1e-12)   # normalised to [0,1]

    # Pre‑compute ternary vectors
    vectors = np.stack([a.ternary_vector() for a in actions])      # shape (n,64)

    # Compute pairwise blended similarities
    blended = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            sim = ternary_vector_similarity(vectors[i], vectors[j])
            blended[i, j] = _blend_vectors(np.array([sim]), np.array([sim]), weights[j])[0]
    return blended


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """
    Compute the marginal probability P(E) = P(E|H)P(H) + P(E|¬H)P(¬H).
    `false_positive` is interpreted as P(E|¬H).
    """
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Bayesian posterior P(H|E) = P(E|H)P(H) / P(E)."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0):
        raise ValueError("Prior and likelihood must be in [0,1]")
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal


def label_score(text: str, label: str) -> float:
    """
    Simple token‑overlap score between `text` and `label`.
    Returns a value in [0,1] representing Jaccard similarity.
    """
    text_tokens = set(text.lower().split())
    label_tokens = set(label.lower().split())
    if not text_tokens and not label_tokens:
        return 1.0
    intersection = text_tokens & label_tokens
    union = text_tokens | label_tokens
    return len(intersection) / len(union)


def hybrid_tree_cost(
    nodes: List[Tuple[str, str]],
    edges: List[Tuple[float, float]],
    false_positive: float = 0.5,
) -> float:
    """
    Compute the total cost of a directed path defined by `nodes` and `edges`.

    * `nodes[i]` = (node_label, node_text)
    * `edges[i]` = (edge_cost, prior_probability)

    For each edge we:
        1. Score the target node's text against the source node's label.
        2. Form a marginal using the prior, the label score as likelihood,
           and a configurable false‑positive rate.
        3. Update the prior via Bayes' rule.
        4. Multiply the updated probability by the edge cost and accumulate.
    """
    if len(nodes) - 1 != len(edges):
        raise ValueError("Number of edges must be one less than number of nodes")

    total_cost = 0.0
    for i, (cost, prior) in enumerate(edges):
        src_label, _ = nodes[i]
        _, tgt_text = nodes[i + 1]

        likelihood = label_score(tgt_text, src_label)          # P(E|H)
        marginal = bayes_marginal(prior, likelihood, false_positive)
        posterior = bayes_update(prior, likelihood, marginal) # P(H|E)

        total_cost += cost * posterior
    return total_cost


if __name__ == "__main__":
    # Example usage ---------------------------------------------------------
    actions = [
        MathAction(id="A1", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="B2", expected_value=8.0, cost=1.5, risk=0.5),
        MathAction(id="C3", expected_value=12.0, cost=3.0, risk=2.0),
    ]

    print("Hybrid action space matrix:")
    print(hybrid_action_space(actions))

    # Tree example -----------------------------------------------------------
    nodes = [
        ("apple", "red fruit sweet"),
        ("fruit", "yellow fruit sour"),
        ("citrus", "orange citrus sour"),
    ]
    edges = [
        (1.2, 0.7),   # cost, prior probability for edge 0→1
        (0.8, 0.6),   # cost, prior probability for edge 1→2
    ]

    print("\nHybrid tree cost:")
    print(hybrid_tree_cost(nodes, edges, false_positive=0.4))