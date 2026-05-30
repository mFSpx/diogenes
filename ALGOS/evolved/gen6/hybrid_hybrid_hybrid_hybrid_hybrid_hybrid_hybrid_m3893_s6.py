# DARWIN HAMMER — match 3893, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m2411_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1830_s0.py (gen4)
# born: 2026-05-29T23:52:28Z

"""Hybrid Fusion of RBF Surrogate and Regret‑Bandit‑Koopman Engines.

Parents:
- **Parent A**: `hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m2411_s1.py`
  Provides edge‑wise Gaussian RBF surrogate modelling and node‑prior extraction.
- **Parent B**: `hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1830_s0.py`
  Provides regret‑weighted probability distributions, Gini‑based similarity,
  bandit‑style confidence bounds and a Koopman‑operator forecast.

Mathematical Bridge:
The node‑prior scores `π(e)` from Parent A form a non‑negative vector that can be
interpreted as a probability distribution after normalisation.  Its Gini
coefficient `G(π)` (Parent B) quantifies the dispersion of the priors and is used
as a *similarity* factor that scales the RBF kernel width `ε` and the
exploration term of the bandit index.  The Koopman operator `K` (Parent B) is
applied to the raw edge feature vector `φ(e)` before RBF prediction, providing
an exploitation component.  The final fused score for an edge `e` is

    S(e) = π(e) · f_RBF(K·φ(e); ε·(1+G)) + η·‖K·φ(e)‖·G

where `f_RBF` is the Gaussian RBF surrogate and `η` is a tunable exploration
coefficient.  This single expression fuses the core topologies of both parents
into a unified decision rule."""


import math
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# Basic geometric utilities (Parent A)
# ----------------------------------------------------------------------


def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


# ----------------------------------------------------------------------
# Graph edge representation (Parent A)
# ----------------------------------------------------------------------


class Edge:
    def __init__(self, node1: str, node2: str):
        self.node1 = node1
        self.node2 = node2

    def __eq__(self, other):
        return (self.node1 == other.node1 and self.node2 == other.node2) or (
            self.node1 == other.node2 and self.node2 == other.node1
        )

    def __hash__(self):
        return hash((self.node1, self.node2))

    def __repr__(self):
        return f"Edge({self.node1!r}, {self.node2!r})"


def edge_feature(edge: Edge) -> List[float]:
    """Encode an edge as a numeric vector using ASCII codes of its node names."""
    return [ord(c) for c in edge.node1 + edge.node2]


# ----------------------------------------------------------------------
# RBF surrogate (Parent A)
# ----------------------------------------------------------------------


class RBFSurrogate:
    def __init__(self, centers: List[List[float]], weights: List[float], epsilon: float = 1.0):
        self.centers = centers
        self.weights = weights
        self.epsilon = epsilon

    def predict(self, x: List[float]) -> float:
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon) for w, c in zip(self.weights, self.centers)
        )


def fit_rbf(points: List[List[float]], values: List[float], epsilon: float = 1.0, ridge: float = 1e-9) -> RBFSurrogate:
    """Fit a Gaussian RBF interpolant to (points, values)."""
    A = np.array(
        [[gaussian(euclidean(p1, p2), epsilon) for p2 in points] for p1 in points],
        dtype=float,
    )
    A += ridge * np.eye(len(points))
    weights = np.linalg.solve(A, np.array(values, dtype=float))
    return RBFSurrogate(points, weights.tolist(), epsilon)


# ----------------------------------------------------------------------
# Node prior extraction (Parent A)
# ----------------------------------------------------------------------


def extract_node_priors(motifs: List[List[Edge]], epsilon: float = 1.0) -> Dict[Edge, float]:
    """Compute a prior score for each edge based on intra‑motif Gaussian similarity."""
    node_priors: Dict[Edge, float] = {}
    for motif in motifs:
        for edge in motif:
            score = 0.0
            for other_edge in motif:
                if edge != other_edge:
                    score += gaussian(
                        euclidean(
                            edge_feature(edge),
                            edge_feature(other_edge),
                        ),
                        epsilon,
                    )
            node_priors[edge] = score / len(motif) if motif else 0.0
    return node_priors


# ----------------------------------------------------------------------
# Regret‑weighted distribution & Gini coefficient (Parent B)
# ----------------------------------------------------------------------


def normalize_distribution(values: List[float]) -> List[float]:
    total = sum(values)
    return [v / total for v in values] if total > 0 else [0.0 for _ in values]


def gini_coefficient(probs: List[float]) -> float:
    """Compute Gini coefficient of a probability distribution."""
    if not probs:
        return 0.0
    sorted_p = sorted(probs)
    n = len(probs)
    cumulative = sum((i + 1) * p for i, p in enumerate(sorted_p))
    return 1.0 - 2.0 * cumulative / (n * sum(sorted_p))


def compute_regret_weighted_distribution(
    priors: Dict[Edge, float],
    costs: Dict[Edge, float],
) -> Dict[Edge, float]:
    """Regret weighting: higher cost reduces the probability mass."""
    raw = {e: max(p - costs.get(e, 0.0), 0.0) for e, p in priors.items()}
    total = sum(raw.values())
    if total == 0:
        # fallback to uniform distribution
        uniform = 1.0 / len(raw) if raw else 0.0
        return {e: uniform for e in raw}
    return {e: v / total for e, v in raw.items()}


# ----------------------------------------------------------------------
# Koopman forecast (Parent B)
# ----------------------------------------------------------------------


def koopman_forecast(feature: List[float], K: np.ndarray) -> np.ndarray:
    """Apply a linear Koopman operator to a feature vector."""
    vec = np.array(feature, dtype=float)
    return K @ vec


# ----------------------------------------------------------------------
# Hybrid fusion core (mathematical bridge)
# ----------------------------------------------------------------------


def hybrid_fuse(
    motifs: List[List[Edge]],
    epsilon: float = 1.0,
    eta: float = 0.5,
    K: np.ndarray = None,
) -> Dict[Edge, float]:
    """
    Produce a fused score for each edge by:
    1. Extracting node priors (Parent A).
    2. Building a regret‑weighted probability distribution (Parent B).
    3. Computing the Gini coefficient G of that distribution (bridge).
    4. Scaling the RBF kernel width ε←ε·(1+G) and using G in the exploration term.
    5. Forecasting edge features with a Koopman operator K (Parent B) before RBF.
    6. Combining prior, RBF prediction and Koopman‑driven exploration.
    """
    # 1. Node priors
    node_priors = extract_node_priors(motifs, epsilon)

    # 2. Mock costs (for demonstration we assign random costs)
    random.seed(0)
    costs = {e: random.random() for e in node_priors}

    # 3. Regret‑weighted distribution and Gini
    regret_dist = compute_regret_weighted_distribution(node_priors, costs)
    G = gini_coefficient(list(regret_dist.values()))

    # 4. Prepare data for RBF fitting
    edges = list(node_priors.keys())
    raw_features = [edge_feature(e) for e in edges]

    # 5. Koopman operator (use identity if not supplied)
    dim = len(raw_features[0])
    if K is None:
        K = np.eye(dim)
    koopman_features = [koopman_forecast(f, K).tolist() for f in raw_features]

    # 6. Fit RBF on Koopman‑transformed features using scaled epsilon
    scaled_eps = epsilon * (1.0 + G)
    rbf_model = fit_rbf(koopman_features, [node_priors[e] for e in edges], epsilon=scaled_eps)

    # 7. Compute final fused scores
    fused_scores: Dict[Edge, float] = {}
    for e, k_feat in zip(edges, koopman_features):
        rbf_pred = rbf_model.predict(k_feat)
        exploration = eta * np.linalg.norm(k_feat) * G
        fused_scores[e] = node_priors[e] * rbf_pred + exploration
    return fused_scores


# ----------------------------------------------------------------------
# Demonstration functions (require at least three)
# ----------------------------------------------------------------------


def demo_node_prior_and_gini():
    """Show extraction of priors and the resulting Gini coefficient."""
    # Simple motif: a triangle of three edges
    e1, e2, e3 = Edge("A", "B"), Edge("B", "C"), Edge("C", "A")
    motifs = [[e1, e2, e3]]
    priors = extract_node_priors(motifs, epsilon=0.8)
    costs = {e: 0.1 for e in priors}
    regret_dist = compute_regret_weighted_distribution(priors, costs)
    G = gini_coefficient(list(regret_dist.values()))
    print("Priors:", priors)
    print("Regret‑weighted distribution:", regret_dist)
    print("Gini coefficient:", G)


def demo_rbf_and_koopman():
    """Fit an RBF surrogate on Koopman‑transformed edge features and predict."""
    edges = [Edge("X", "Y"), Edge("Y", "Z"), Edge("Z", "X")]
    features = [edge_feature(e) for e in edges]
    # Identity Koopman for simplicity
    K = np.eye(len(features[0]))
    koop_feat = [koopman_forecast(f, K).tolist() for f in features]
    values = [random.random() for _ in edges]
    model = fit_rbf(koop_feat, values, epsilon=0.9)
    preds = [model.predict(f) for f in koop_feat]
    print("Original values:", values)
    print("RBF predictions after Koopman:", preds)


def demo_hybrid_fusion():
    """Run the full hybrid fusion pipeline on a synthetic motif set."""
    # Build two motifs with overlapping edges
    edges = [
        Edge("A", "B"),
        Edge("B", "C"),
        Edge("C", "D"),
        Edge("D", "A"),
        Edge("A", "C"),
    ]
    motif1 = edges[:3]  # A‑B, B‑C, C‑D
    motif2 = edges[2:]  # C‑D, D‑A, A‑C
    fused = hybrid_fuse([motif1, motif2], epsilon=1.0, eta=0.3)
    for e, score in fused.items():
        print(f"{e}: {score:.4f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    print("=== Demo: Node Prior & Gini ===")
    demo_node_prior_and_gini()
    print("\n=== Demo: RBF + Koopman ===")
    demo_rbf_and_koopman()
    print("\n=== Demo: Full Hybrid Fusion ===")
    demo_hybrid_fusion()