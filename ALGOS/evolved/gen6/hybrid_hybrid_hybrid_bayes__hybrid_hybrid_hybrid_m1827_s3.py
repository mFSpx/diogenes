# DARWIN HAMMER — match 1827, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_bayes_update__hybrid_possum_filter_m210_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m607_s1.py (gen5)
# born: 2026-05-29T23:39:01Z

"""Hybrid Bayesian‑Privacy / NLMS Adaptive Filter

Parents:
- hybrid_hybrid_bayes_update__hybrid_possum_filter_m210_s1.py
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m607_s1.py

Mathematical bridge:
The spatial‑aware privacy risk vector 𝑟∈ℝⁿ (parent A) is normalised to a prior
probability vector 𝑝 (∑pᵢ=1).  In the NLMS adaptive filter of parent B the
weight update is

    w←w+μ·(d−y)·x / (ε+‖x‖²)

where μ is the learning rate.  We replace μ by a *composite learning factor*
λᵢ=μ·pᵢ·hᵢ for each sample i, where hᵢ is a health‑score derived from a
TreeNode’s morphological size and the entity’s privacy risk.  Thus the
Bayesian‑derived spatial prior directly modulates the NLMS adaptation,
producing a unified update rule that respects both statistical (Bayes)
and signal‑processing (NLMS) dynamics.

The module implements:
1. spatial_aware_privacy_prior – builds the prior vector from entities.
2. health_score – combines a TreeNode’s size with the corresponding prior.
3. NLMSFilter – an adaptive filter whose learning rate is scaled per‑sample
   by λᵢ = μ·pᵢ·hᵢ.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (derived from parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def _signature(e: Entity) -> str:
    """Canonical signature used for quasi‑identifier matching."""
    return (e.address_signature or e.category).strip().lower()


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Base risk per entity (parent A)."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def spatial_aware_privacy_risk_vector(
    entities: List[Entity], delta_m: float
) -> np.ndarray:
    """
    Compute a risk value for each entity based on:
      * number of spatially close, signature‑matching peers,
      * base reconstruction risk.
    Returns a raw (non‑normalised) vector r∈ℝⁿ.
    """
    n = len(entities)
    risks = np.zeros(n, dtype=float)

    for i, e_i in enumerate(entities):
        # Count peers that share the same signature and lie within delta_m.
        similar = sum(
            1
            for j, e_j in enumerate(entities)
            if i != j
            and _signature(e_i) == _signature(e_j)
            and haversine_m((e_i.lat, e_i.lon), (e_j.lat, e_j.lon)) <= delta_m
        )
        # Unique quasi‑identifiers for this entity:
        uq = similar + 1  # include self
        risks[i] = reconstruction_risk_score(uq, n)
    return risks


def spatial_aware_privacy_prior(
    entities: List[Entity], delta_m: float = 500.0
) -> np.ndarray:
    """
    Normalise the raw risk vector to obtain a proper probability distribution
    (the Bayesian prior).  If all risks are zero, a uniform prior is returned.
    """
    raw = spatial_aware_privacy_risk_vector(entities, delta_m)
    total = raw.sum()
    if total == 0.0:
        return np.full_like(raw, 1.0 / len(raw))
    return raw / total


# ----------------------------------------------------------------------
# Structures from parent B (NLMS + health score)
# ----------------------------------------------------------------------
@dataclass
class TreeNode:
    """Morphology‑driven node used for health‑score computation."""
    name: str
    size: int  # e.g., number of sub‑components or memory footprint
    prior_probability: float = 0.0  # will be filled from the Bayesian prior


def health_score(node: TreeNode, prior: float) -> float:
    """
    Composite health score h ∈ (0,1] that blends:
      * node size (larger size ⇒ more robust, capped at 1),
      * Bayesian prior (higher privacy risk ⇒ lower health).
    The formula is:
        h = (1 - α·prior) * sigmoid(size)
    where α∈[0,1] controls the influence of privacy risk.
    """
    α = 0.7  # weighting of privacy risk
    # Sigmoid to map size ∈ ℕ to (0,1]
    size_factor = 1.0 / (1.0 + math.exp(-0.01 * (node.size - 50)))
    return max(0.0, min(1.0, (1.0 - α * prior) * size_factor))


# ----------------------------------------------------------------------
# Hybrid NLMS filter that consumes the Bayesian prior via health scores
# ----------------------------------------------------------------------
class NLMSFilter:
    """
    Normalised Least‑Mean‑Squares adaptive filter with per‑sample composite
    learning factor λᵢ = μ·pᵢ·hᵢ.
    """

    def __init__(self, order: int, mu: float = 0.1, eps: float = 1e-6):
        """
        Parameters
        ----------
        order : int
            Number of taps (filter length).
        mu : float
            Base learning rate (0<mu≤1).
        eps : float
            Regularisation term to avoid division by zero.
        """
        self.order = order
        self.mu = mu
        self.eps = eps
        self.w = np.zeros(order, dtype=float)  # filter weights

    def predict(self, x: np.ndarray) -> float:
        """Return current output y = wᵀx."""
        return float(np.dot(self.w, x))

    def update(
        self,
        x: np.ndarray,
        d: float,
        prior: float,
        health: float,
    ) -> None:
        """
        Perform a single NLMS weight update.

        Parameters
        ----------
        x : np.ndarray
            Input vector of length `order`.
        d : float
            Desired (target) response.
        prior : float
            Bayesian prior probability for the current sample.
        health : float
            Health score derived from the associated TreeNode.
        """
        y = self.predict(x)
        e = d - y
        # Composite learning factor λ = μ * prior * health
        lam = self.mu * prior * health
        norm = np.dot(x, x) + self.eps
        self.w += (lam * e / norm) * x

    def batch_train(
        self,
        X: np.ndarray,
        d: np.ndarray,
        priors: np.ndarray,
        healths: np.ndarray,
    ) -> None:
        """
        Train on a whole batch where each row of X corresponds to one sample.
        """
        for i in range(X.shape[0]):
            self.update(X[i], d[i], priors[i], healths[i])


# ----------------------------------------------------------------------
# High‑level hybrid operation (exposes three public functions)
# ----------------------------------------------------------------------
def compute_hybrid_priors(entities: List[Entity]) -> Tuple[np.ndarray, List[TreeNode]]:
    """
    Convert entities to a Bayesian prior and create matching TreeNode objects.
    Returns (prior_vector, list_of_TreeNode) where each node's
    `prior_probability` field is populated.
    """
    prior = spatial_aware_privacy_prior(entities)
    nodes = []
    for e, p in zip(entities, prior):
        # Derive a synthetic size from the entity's score (scaled) for demo.
        size = int(10 + 90 * e.score)  # size ∈ [10,100]
        node = TreeNode(name=e.id, size=size, prior_probability=p)
        nodes.append(node)
    return prior, nodes


def compute_health_vector(nodes: List[TreeNode]) -> np.ndarray:
    """
    Produce a health‑score vector h∈ℝⁿ aligned with the prior vector.
    """
    h = np.array([health_score(node, node.prior_probability) for node in nodes])
    return h


def hybrid_nlms_train(
    entities: List[Entity],
    inputs: np.ndarray,
    targets: np.ndarray,
    filter_order: int = 4,
) -> NLMSFilter:
    """
    End‑to‑end training routine:
      1. Compute Bayesian priors from entities.
      2. Build health scores.
      3. Initialise NLMS filter.
      4. Run batch training with composite learning factors.

    Parameters
    ----------
    entities : List[Entity]
        Source data (must have length equal to inputs.shape[0]).
    inputs : np.ndarray
        Input matrix X of shape (N, filter_order).
    targets : np.ndarray
        Desired response vector d of length N.
    filter_order : int
        Order of the NLMS filter.

    Returns
    -------
    NLMSFilter
        Trained filter (weights can be inspected via `.w`).
    """
    if inputs.shape[0] != len(entities) or targets.shape[0] != len(entities):
        raise ValueError("Dimension mismatch between entities and data matrices.")

    prior_vec, nodes = compute_hybrid_priors(entities)
    health_vec = compute_health_vector(nodes)

    nlms = NLMSFilter(order=filter_order, mu=0.2)
    nlms.batch_train(inputs, targets, prior_vec, health_vec)
    return nlms


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic dataset
    random.seed(0)
    np.random.seed(0)

    demo_entities = [
        Entity(
            id=f"E{i}",
            lat=37.0 + random.uniform(-0.01, 0.01),
            lon=-122.0 + random.uniform(-0.01, 0.01),
            category="A" if i % 2 == 0 else "B",
            score=random.random(),
        )
        for i in range(8)
    ]

    # Input matrix: each sample is a random vector of length 4
    X = np.random.randn(len(demo_entities), 4)
    # Desired response: a noisy linear combination of the inputs
    true_w = np.array([0.5, -0.3, 0.2, 0.1])
    d = X @ true_w + 0.05 * np.random.randn(len(demo_entities))

    # Run the hybrid training
    trained_filter = hybrid_nlms_train(demo_entities, X, d, filter_order=4)

    # Print results – should run without error
    print("Trained NLMS weights:", trained_filter.w)
    print("Prior probabilities:", spatial_aware_privacy_prior(demo_entities))
    print("Health scores:", compute_health_vector(
        [TreeNode(name=e.id,
                  size=int(10 + 90 * e.score),
                  prior_probability=0.0) for e in demo_entities]
    ))