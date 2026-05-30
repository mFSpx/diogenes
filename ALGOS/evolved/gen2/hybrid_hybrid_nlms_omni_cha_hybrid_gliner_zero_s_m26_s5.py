# DARWIN HAMMER — match 26, survivor 5
# gen: 2
# parent_a: hybrid_nlms_omni_chaotic_sprint_m59_s1.py (gen1)
# parent_b: hybrid_gliner_zero_shot_ext_minimum_cost_tree_m27_s0.py (gen1)
# born: 2026-05-29T23:26:33Z

"""Hybrid NLMS‑Graph‑Tree Fusion

This module fuses two parent algorithms:

* **Parent A** – NLMS adaptive filter (nlms.py) which updates a weight vector
  `w` using the normalized least‑mean‑squares rule.
* **Parent B** – Span extraction + Minimum‑Cost Tree (gliner_zero_shot_extractor +
  minimum_cost_tree) which builds a graph whose nodes are text spans and whose
  edges encode similarity.

**Mathematical bridge**

The similarity matrix `S` of the span‑graph is a collection of feature vectors
`x_i`.  In the original NLMS algorithm a scalar target `d_i` is predicted by
`y_i = w·x_i`.  Here we treat the *desired importance* of each span (its
`score`) as the target `d_i`.  The NLMS update is applied to a global weight
vector `w` that linearly combines the similarity features into a cost
estimate for each edge.  The adapted weights are then used as multiplicative
factors in the edge‑cost definition of the minimum‑cost spanning tree, yielding
a tree that reflects both learned relevance (via NLMS) and intrinsic similarity
(via the graph).

The result is a single unified system that learns to weight graph edges
adaptively while still solving the classic minimum‑cost tree problem.
"""

import json
import math
import random
import sys
import time
import hashlib
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# NLMS core (Parent A)
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS adaptation step.

    Args:
        weights: Current weight vector (shape (n,)).
        x: Input feature vector (shape (n,)).
        target: Desired scalar output.
        mu: Step‑size (0 < mu ≤ 1).
        eps: Small constant to avoid division by zero.

    Returns:
        (new_weights, error) where error = target - y.
    """
    y = predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    new_weights = weights + (mu * error / power) * x
    return new_weights, error


# ----------------------------------------------------------------------
# Span extraction (Parent B – simplified)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str


def _char_frequency_vector(text: str) -> np.ndarray:
    """Return a 26‑dim vector of lowercase alphabet frequencies."""
    vec = np.zeros(26, dtype=float)
    for ch in text.lower():
        if 'a' <= ch <= 'z':
            vec[ord(ch) - ord('a')] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def extract_spans(text: str) -> List[Span]:
    """
    Very simple zero‑shot “extractor”: each whitespace‑separated token becomes a span.
    A random score in [0.5, 1.0] simulates a confidence value.
    """
    spans: List[Span] = []
    pos = 0
    for token in re_finditer(r'\S+', text):
        start = token.start()
        end = token.end()
        span_text = token.group()
        score = random.uniform(0.5, 1.0)
        spans.append(
            Span(
                start=start,
                end=end,
                text=span_text,
                label="WORD",
                score=score,
                backend="simple_tokenizer",
            )
        )
    return spans


def re_finditer(pattern: str, string: str):
    """Thin wrapper around re.finditer to avoid importing the whole module at top."""
    import re

    return re.finditer(pattern, string)


# ----------------------------------------------------------------------
# Graph construction & similarity (bridge)
# ----------------------------------------------------------------------
def build_similarity_matrix(spans: List[Span]) -> np.ndarray:
    """
    Build a symmetric similarity matrix S where S[i, j] = cosine similarity
    between the character‑frequency vectors of span i and span j.
    """
    n = len(spans)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    feats = np.stack([_char_frequency_vector(s.text) for s in spans])
    # Cosine similarity = dot product because vectors are L2‑normalized
    S = np.clip(feats @ feats.T, 0.0, 1.0)
    return S


# ----------------------------------------------------------------------
# Adaptive weighting of edges via NLMS (core fusion)
# ----------------------------------------------------------------------
def adapt_edge_weights(
    similarity: np.ndarray,
    span_scores: np.ndarray,
    mu: float = 0.4,
    epochs: int = 5,
) -> np.ndarray:
    """
    Learn a global weight vector w that maps similarity rows to the
    corresponding span scores using NLMS.  The learned w is then used to
    compute edge costs as 1 - (w·x).

    Args:
        similarity: (n, n) matrix, each row is a feature vector x_i.
        span_scores: (n,) target vector d_i.
        mu: NLMS step size.
        epochs: Number of passes over the data.

    Returns:
        Weight vector w (shape (n,)).
    """
    n = similarity.shape[0]
    if n == 0:
        return np.array([], dtype=float)

    # Initialise w to uniform values
    w = np.ones(n, dtype=float) / n

    for _ in range(epochs):
        for i in range(n):
            x_i = similarity[i]
            d_i = float(span_scores[i])
            w, _ = nlms_update(w, x_i, d_i, mu=mu)
    return w


def edge_cost_matrix(similarity: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Convert similarity matrix into a cost matrix for the tree algorithm.
    Cost_ij = 1 - (w·x_ij) where x_ij is the similarity row for node i
    (using the same weight vector for all edges).
    """
    # Ensure weights shape matches similarity columns
    if similarity.shape[1] != weights.shape[0]:
        raise ValueError("Weight dimension mismatch.")
    # Linear combination of each row with w
    combined = similarity @ weights
    cost = 1.0 - combined
    # Symmetrize and zero diagonal
    cost = (cost + cost.T) / 2.0
    np.fill_diagonal(cost, 0.0)
    return cost


# ----------------------------------------------------------------------
# Minimum‑Cost Spanning Tree (Parent B)
# ----------------------------------------------------------------------
def prim_mst(cost: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Compute a Minimum‑Cost Spanning Tree using Prim's algorithm.
    Returns a list of edges (u, v, cost_uv).
    """
    n = cost.shape[0]
    if n == 0:
        return []

    in_mst = np.zeros(n, dtype=bool)
    edge_to = np.full(n, -1, dtype=int)
    min_cost = np.full(n, np.inf, dtype=float)

    min_cost[0] = 0.0
    result: List[Tuple[int, int, float]] = []

    for _ in range(n):
        # select the vertex not yet in MST with smallest key
        u = int(np.argmin(np.where(in_mst, np.inf, min_cost)))
        in_mst[u] = True

        if edge_to[u] != -1:
            result.append((edge_to[u], u, cost[edge_to[u], u]))

        # update keys of adjacent vertices
        for v in range(n):
            if not in_mst[v] and cost[u, v] < min_cost[v]:
                min_cost[v] = cost[u, v]
                edge_to[v] = u

    return result


# ----------------------------------------------------------------------
# High‑level hybrid pipeline
# ----------------------------------------------------------------------
def hybrid_process(text: str) -> Tuple[List[Span], List[Tuple[int, int, float]]]:
    """
    End‑to‑end processing:
    1. Extract spans.
    2. Build similarity matrix.
    3. Adapt NLMS weights using span scores.
    4. Derive edge costs and compute a minimum‑cost spanning tree.

    Returns:
        (spans, mst_edges)
    """
    spans = extract_spans(text)
    if not spans:
        return [], []

    similarity = build_similarity_matrix(spans)
    scores = np.array([s.score for s in spans], dtype=float)

    weights = adapt_edge_weights(similarity, scores, mu=0.3, epochs=8)
    costs = edge_cost_matrix(similarity, weights)
    mst = prim_mst(costs)

    return spans, mst


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "In the quiet forest, the wind whispered through the leaves, "
        "carrying secrets of ancient trees and hidden paths."
    )
    spans, mst_edges = hybrid_process(sample)

    print("Extracted spans (showing first 5):")
    for s in spans[:5]:
        print(f"  [{s.start}:{s.end}] '{s.text}' score={s.score:.3f}")

    print("\nMinimum‑Cost Spanning Tree edges (node indices and cost):")
    for u, v, c in mst_edges:
        print(f"  {u} -- {v}  cost={c:.4f}")

    print("\nTotal tree cost:", sum(c for _, _, c in mst_edges))