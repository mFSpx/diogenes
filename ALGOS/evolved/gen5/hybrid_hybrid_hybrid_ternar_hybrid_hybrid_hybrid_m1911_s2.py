# DARWIN HAMMER — match 1911, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_ternary_route_hybrid_endpoint_circ_m233_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_counterfactual_effec_m316_s1.py (gen4)
# born: 2026-05-29T23:39:47Z

import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np

# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Point = Tuple[float, float]          # 2‑D coordinates of a node
Edge = Tuple[str, str]               # connection between node identifiers
Morphology = Tuple[float, float, float]  # (length, width, height)

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int


@dataclass(frozen=True)
class CausalEffect:
    """Container for a causal effect estimate."""
    effect_id: str
    treatment: str
    outcome: str
    confounders: Tuple[str, ...]
    ate_estimate: float | None
    ate_confidence_interval: Tuple[float, float] | None
    refutation_passed: bool
    refutation_methods: Tuple[str, ...]
    heterogeneous_effects: Dict[str, float]


# ----------------------------------------------------------------------
# Core mathematical utilities
# ----------------------------------------------------------------------
def euclidean_length(a: Point, b: Point) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def reconstruction_risk_score(
    unique_quasi_identifiers: int,
    total_records: int,
    laplace_smoothing: float = 1.0,
) -> float:
    """
    A more principled reconstruction risk score based on the
    proportion of unique quasi‑identifiers with Laplace smoothing.
    Returns a probability in [0, 1].
    """
    if total_records <= 0:
        return 0.0
    # Laplace smoothing prevents a score of exactly 0 or 1
    numer = unique_quasi_identifiers + laplace_smoothing
    denom = total_records + 2 * laplace_smoothing
    return max(0.0, min(1.0, numer / denom))


def anonymize_for_indexing(
    record: Dict[str, Any],
    redact_keys: set[str] | None = None,
) -> Dict[str, Any]:
    """Redact obvious PII before indexing."""
    default_redact = {"email", "phone", "ssn", "secret", "token", "password"}
    redact = redact_keys or default_redact
    return {
        k: ("<redacted>" if k.lower() in redact else v)
        for k, v in record.items()
    }


def dp_aggregate(
    values: Iterable[float],
    epsilon: float = 1.0,
    sensitivity: float = 1.0,
) -> float:
    """
    Differential‑privacy aware aggregation.
    Returns the sum plus Laplace noise calibrated to (epsilon, sensitivity).
    """
    total = float(sum(values))
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale)
    return total + noise


# ----------------------------------------------------------------------
# Bayesian edge weight update
# ----------------------------------------------------------------------
def bayesian_edge_posterior(
    prior: float,
    likelihood: float,
    prior_weight: float = 1.0,
    likelihood_weight: float = 1.0,
) -> float:
    """
    Compute a simple Bayesian posterior for a scalar quantity.
    The posterior is a weighted average of prior and likelihood.
    """
    return (prior_weight * prior + likelihood_weight * likelihood) / (
        prior_weight + likelihood_weight
    )


def sample_ate_from_ci(
    ate_estimate: float,
    ci: Tuple[float, float],
    rng: np.random.Generator,
) -> float:
    """
    Approximate a normal distribution from a confidence interval.
    Assumes the CI is a 95 % interval.
    """
    low, high = ci
    sigma = (high - low) / (2 * 1.96)  # 95 % ≈ 1.96 σ each side
    return max(0.0, rng.normal(ate_estimate, sigma))


def compute_updated_edge_weights(
    nodes: Dict[str, Point],
    edges: List[Edge],
    risk_score: float,
    causal_effect: CausalEffect | None,
    rng: np.random.Generator,
) -> Dict[Edge, float]:
    """
    Produce Bayesian‑updated edge weights that blend three signals:
      1. Geometric length (baseline prior)
      2. Reconstruction risk (likelihood that the edge will cause exhaustion)
      3. Causal effect on risk (modulates the likelihood)
    """
    updated: Dict[Edge, float] = {}
    for edge in edges:
        # 1️⃣ Prior = geometric length
        length_ab = euclidean_length(nodes[edge[0]], nodes[edge[1]])

        # 2️⃣ Likelihood = risk_score (higher risk → larger cost)
        #    We map risk ∈[0,1] to a multiplicative factor ∈[1,2]
        likelihood = length_ab * (1.0 + risk_score)

        # 3️⃣ Causal adjustment (optional)
        if causal_effect and causal_effect.ate_estimate is not None:
            ate = sample_ate_from_ci(
                causal_effect.ate_estimate,
                causal_effect.ate_confidence_interval or (0.0, 0.0),
                rng,
            )
            # Treat ate as a proportional increase on the likelihood
            likelihood *= (1.0 + ate)

        # Bayesian blend – give the geometric prior a modest weight
        posterior = bayesian_edge_posterior(
            prior=length_ab,
            likelihood=likelihood,
            prior_weight=1.0,
            likelihood_weight=2.0,
        )
        updated[edge] = posterior
    return updated


# ----------------------------------------------------------------------
# Tree cost utilities
# ----------------------------------------------------------------------
def tree_cost(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    path_weight: float = 0.2,
    edge_weights: Dict[Edge, float] | None = None,
) -> float:
    """
    Compute the total cost of traversing the tree from `root`.
    If `edge_weights` is supplied, they replace the geometric length.
    The cost of a path is the sum of edge weights multiplied by `path_weight`.
    """
    # Build adjacency for BFS/DFS traversal
    adjacency: Dict[str, List[Tuple[str, Edge]]] = {n: [] for n in nodes}
    for e in edges:
        a, b = e
        adjacency[a].append((b, e))
        adjacency[b].append((a, e))

    visited = set()
    total = 0.0
    stack = [(root, 0.0)]  # (current_node, accumulated_cost)

    while stack:
        node, acc = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        total += acc
        for neighbor, edge in adjacency[node]:
            if neighbor in visited:
                continue
            weight = (
                edge_weights[edge]
                if edge_weights and edge in edge_weights
                else euclidean_length(nodes[edge[0]], nodes[edge[1]])
            )
            stack.append((neighbor, acc + weight * path_weight))
    return total


# ----------------------------------------------------------------------
# High‑level hybrid decision function
# ----------------------------------------------------------------------
def hybrid_decision(
    nodes: Dict[str, Point],
    edges: List[Edge],
    root: str,
    path_weight: float = 0.2,
    edge_weights: Dict[Edge, float] | None = None,
    reconstruction_risk: float = 0.5,
    causal_effect: CausalEffect | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Compute a decision metric that reflects:
      • Baseline material cost (tree_cost)
      • Updated cost after Bayesian fusion of risk & causal signals
    The function returns the posterior cost; a lower value indicates a
    healthier routing configuration.
    """
    rng = rng or np.random.default_rng()

    # Baseline cost (no risk or causal information)
    baseline = tree_cost(nodes, edges, root, path_weight, edge_weights)

    # Fuse risk & causal information into edge weights
    fused_weights = compute_updated_edge_weights(
        nodes,
        edges,
        reconstruction_risk,
        causal_effect,
        rng,
    )
    posterior = tree_cost(nodes, edges, root, path_weight, fused_weights)

    # Return a weighted blend of baseline and posterior to keep
    # the decision interpretable while still rewarding risk‑aware updates.
    return 0.6 * baseline + 0.4 * posterior


# ----------------------------------------------------------------------
# Example causal estimator (placeholder – replace with real estimator)
# ----------------------------------------------------------------------
def estimate_causal_effect(
    treatment: str,
    outcome: str,
    confounders: List[str],
    data: Dict[str, Any],
) -> CausalEffect:
    """
    Dummy estimator that returns a plausible effect structure.
    In production this would invoke a statistical model (e.g., TMLE,
    doubly‑robust estimator, or Bayesian hierarchical model).
    """
    effect_id = f"Effect_{treatment}_{outcome}"
    ate_estimate = 0.12  # modest increase in risk
    ate_ci = (0.05, 0.19)  # 95 % confidence interval
    refutation_passed = True
    refutation_methods = ("Placebo", "Subset")
    heterogeneous = {"high_income": 0.08, "low_income": 0.16}
    return CausalEffect(
        effect_id=effect_id,
        treatment=treatment,
        outcome=outcome,
        confounders=tuple(confounders),
        ate_estimate=ate_estimate,
        ate_confidence_interval=ate_ci,
        refutation_passed=refutation_passed,
        refutation_methods=tuple(refutation_methods),
        heterogeneous_effects=heterogeneous,
    )


# ----------------------------------------------------------------------
# Smoke test / demonstration
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny graph
    nodes = {
        "A": (0.0, 0.0),
        "B": (3.0, 4.0),
        "C": (6.0, 8.0),
    }
    edges = [("A", "B"), ("B", "C")]
    root = "A"

    # Compute a more realistic risk score
    risk = reconstruction_risk_score(unique_quasi_identifiers=12, total_records=150)

    # Obtain a causal effect estimate (placeholder)
    causal = estimate_causal_effect(
        treatment="k_anonymity",
        outcome="reconstruction_risk",
        confounders=["age", "zip"],
        data={},
    )

    # Run the hybrid decision
    decision_value = hybrid_decision(
        nodes=nodes,
        edges=edges,
        root=root,
        path_weight=0.2,
        reconstruction_risk=risk,
        causal_effect=causal,
    )
    print(f"Hybrid decision value: {decision_value:.4f}")
    sys.exit(0)