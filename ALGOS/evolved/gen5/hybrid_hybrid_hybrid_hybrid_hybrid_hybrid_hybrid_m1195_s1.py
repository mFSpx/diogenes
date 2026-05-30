# DARWIN HAMMER — match 1195, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_bayes_claim_k_m166_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s3.py (gen3)
# born: 2026-05-29T23:33:37Z

"""Hybrid Algorithm: Distributed Leader Election ↔ Hoeffding Tree ↔ Bayesian Edge Reliability ↔ Tropical Max‑Plus Cost
   + Endpoint Morphology ↔ SSIM ↔ Recovery Priority

Parents
-------
* **Parent A** – `hybrid_hybrid_hybrid_distri_hybrid_bayes_claim_k_m166_s1.py`  
  Provides probabilistic acceptance (`p_accept`), Hoeffding bound decisions,
  Bayesian edge posteriors (`P(H|E)`) and tropical max‑plus algebra on a cost
  matrix.

* **Parent B** – `hybrid_hybrid_hybrid_endpoi_hybrid_ssim_hybrid_d_m1_s3.py`  
  Supplies morphological indices (`righting_time_index`, `recovery_priority`)
  and a structural similarity (SSIM) based endpoint circuit‑breaker.

Mathematical Bridge
-------------------
Both families expose a *confidence* scalar in the interval **[0, 1]**:

* `p_accept = exp(-ΔE / T)` – temperature‑scaled confidence for a tree split /
  leader election (Parent A).
* `posterior = P(H|E)` – Bayesian confidence of edge reliability (Parent A).
* `ssim_value` – image‑based similarity confidence (Parent B).
* `recovery_priority` – morphology‑derived confidence of node robustness
  (Parent B).

The hybrid algorithm multiplies these confidences to obtain a **composite weight**


w_ij = p_accept_ij * posterior_ij * ssim_ij * recovery_priority_i


and injects it into the tropical max‑plus evaluation of a routing / decision
tree:


Ĉ_ij = C_ij * w_ij                     # element‑wise scaling of raw edge length
cost   = max_path_sum_tropical(Ĉ)      # max‑plus path aggregation


The three core functions below demonstrate this fusion:
1. `acceptance_probability` – Hoeffding‑bound based split decision.
2. `bayesian_edge_update` – Beta‑conjugate posterior update for edge reliability.
3. `tropical_weighted_max_path` – builds the weighted cost matrix using
   SSIM, morphology priority and the above confidences, then computes the
   tropical max‑plus path cost.

The module can be used in distributed leader‑election loops, streaming
Hoeffding‑tree learners, or morphology‑aware routing optimisers."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, replace
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# 1. Probabilistic acceptance (Parent A)
# ----------------------------------------------------------------------
def acceptance_probability(delta_energy: float, temperature: float) -> float:
    """Metropolis‑style acceptance probability.

    Args:
        delta_energy: Energy difference ΔE (positive → worse state).
        temperature: Simulated‑annealing temperature T > 0.

    Returns:
        Probability in [0,1].
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return math.exp(-delta_energy / temperature)


def hoeffding_bound(num_samples: int, epsilon: float, delta: float = 0.05) -> bool:
    """Hoeffding bound decision: returns True if the observed mean is
    statistically significant enough to trigger a split/leadership change.

    Args:
        num_samples: Number of independent observations.
        epsilon: Desired error bound.
        delta: Failure probability (default 0.05).

    Returns:
        Boolean decision.
    """
    if num_samples <= 0:
        return False
    bound = math.sqrt(math.log(2 / delta) / (2 * num_samples))
    return bound < epsilon


# ----------------------------------------------------------------------
# 2. Bayesian edge reliability (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class EdgeBetaPrior:
    """Beta prior parameters for a Bernoulli edge reliability."""
    alpha: float = 1.0
    beta: float = 1.0


def bayesian_edge_update(prior: EdgeBetaPrior, successes: int, failures: int) -> Tuple[float, EdgeBetaPrior]:
    """Update edge reliability posterior with new evidence.

    Args:
        prior: Current Beta prior.
        successes: Number of observed successful transmissions.
        failures: Number of observed failures.

    Returns:
        posterior_mean: Expected reliability P(H|E) in [0,1].
        new_prior: Updated Beta parameters (conjugate posterior).
    """
    new_alpha = prior.alpha + successes
    new_beta = prior.beta + failures
    posterior_mean = new_alpha / (new_alpha + new_beta)
    return posterior_mean, EdgeBetaPrior(new_alpha, new_beta)


# ----------------------------------------------------------------------
# 3. Morphology & Recovery Priority (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1] based on righting time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# 4. SSIM endpoint circuit breaker (Parent B)
# ----------------------------------------------------------------------
def _mean_std(arr: np.ndarray) -> Tuple[float, float]:
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    return mu, sigma


def ssim(x: List[float], y: List[float], dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index Measure between two 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("signals must have the same length")
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)

    mu_x, sigma_x = _mean_std(x_arr)
    mu_y, sigma_y = _mean_std(y_arr)
    cov_xy = float(np.cov(x_arr, y_arr, ddof=1)[0, 1])

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)

    return float(numerator / denominator)


# ----------------------------------------------------------------------
# 5. Tropical max‑plus algebra utilities (Parent A)
# ----------------------------------------------------------------------
def t_add(a: float, b: float) -> float:
    """Tropical addition = max."""
    return max(a, b)


def t_mul(a: float, b: float) -> float:
    """Tropical multiplication = ordinary addition."""
    return a + b


def tropical_path_sum(matrix: np.ndarray, start: int, end: int) -> float:
    """Compute the tropical max‑plus path sum from start to end using
    a simple Floyd‑Warshall‑like relaxation (O(n³))."""
    n = matrix.shape[0]
    # Initialize tropical distance matrix: -inf for unreachable, 0 on diagonal
    dist = np.full((n, n), -np.inf)
    np.fill_diagonal(dist, 0.0)

    # Direct edges (tropical multiplication = addition)
    for i in range(n):
        for j in range(n):
            if not np.isinf(matrix[i, j]):  # assume np.inf marks no edge
                dist[i, j] = t_mul(0.0, matrix[i, j])  # = matrix[i,j]

    # Relaxation
    for k in range(n):
        for i in range(n):
            for j in range(n):
                via_k = t_mul(dist[i, k], dist[k, j])
                dist[i, j] = t_add(dist[i, j], via_k)

    return dist[start, end]


# ----------------------------------------------------------------------
# 6. Hybrid core: weighted tropical max‑plus evaluation
# ----------------------------------------------------------------------
def tropical_weighted_max_path(
    raw_cost: np.ndarray,
    edge_posteriors: np.ndarray,
    acceptance_probs: np.ndarray,
    ssim_matrix: np.ndarray,
    recovery_priorities: np.ndarray,
    start: int,
    end: int,
) -> float:
    """Compute tropical max‑plus cost on a weighted matrix that fuses all
    confidences.

    The composite weight for edge (i,j) is

        w_ij = acceptance_probs[i,j] *
               edge_posteriors[i,j] *
               ssim_matrix[i,j] *
               recovery_priorities[i]

    The weighted cost matrix is Ĉ_ij = raw_cost_ij * w_ij.
    Missing edges are represented by np.inf in `raw_cost`.

    Args:
        raw_cost: NxN matrix of physical edge lengths (np.inf → no edge).
        edge_posteriors: NxN matrix of Bayesian edge reliabilities.
        acceptance_probs: NxN matrix of Metropolis acceptance probabilities.
        ssim_matrix: NxN matrix of SSIM values between endpoint signals.
        recovery_priorities: length‑N vector of node recovery priorities.
        start: source node index.
        end: destination node index.

    Returns:
        Tropical max‑plus path cost (float). Returns -np.inf if no path exists.
    """
    if raw_cost.shape != edge_posteriors.shape or raw_cost.shape != acceptance_probs.shape:
        raise ValueError("Matrix dimensions must match")
    if raw_cost.shape[0] != ssim_matrix.shape[0] or raw_cost.shape[1] != ssim_matrix.shape[1]:
        raise ValueError("SSIM matrix dimensions must match")
    if raw_cost.shape[0] != recovery_priorities.shape[0]:
        raise ValueError("Recovery priority length must match node count")

    # Composite weight (broadcast recovery_priorities over rows)
    weight = (
        acceptance_probs
        * edge_posteriors
        * ssim_matrix
        * recovery_priorities[:, np.newaxis]
    )

    # Weighted cost matrix: element‑wise product, preserve np.inf for missing edges
    weighted_cost = np.where(np.isinf(raw_cost), np.inf, raw_cost * weight)

    # Tropical max‑plus path
    return tropical_path_sum(weighted_cost, start, end)


# ----------------------------------------------------------------------
# 7. Demonstration helpers
# ----------------------------------------------------------------------
def _random_signal(length: int) -> List[float]:
    return [random.randint(0, 255) for _ in range(length)]


def build_demo_graph(num_nodes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a small random demo graph and associated matrices.

    Returns:
        raw_cost, edge_posteriors, acceptance_probs, ssim_matrix, recovery_priorities
    """
    rng = np.random.default_rng(42)

    # Raw edge lengths: random positive values, np.inf for ~30% missing edges
    raw = rng.uniform(1.0, 10.0, size=(num_nodes, num_nodes))
    mask = rng.random(size=(num_nodes, num_nodes)) < 0.3
    raw[mask] = np.inf
    np.fill_diagonal(raw, 0.0)  # zero self‑cost

    # Edge posteriors: start from uniform Beta(1,1) and apply random evidence
    post = np.empty_like(raw)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if np.isinf(raw[i, j]):
                post[i, j] = 0.0
            else:
                prior = EdgeBetaPrior()
                successes = rng.integers(0, 5)
                failures = rng.integers(0, 5)
                post[i, j], _ = bayesian_edge_update(prior, successes, failures)

    # Acceptance probabilities: use a dummy ΔE proportional to raw cost
    temperature = 5.0
    accept = np.where(np.isinf(raw), 0.0, np.exp(-raw / temperature))

    # SSIM matrix: compare random signals per node
    signals = [_random_signal(64) for _ in range(num_nodes)]
    ssim_mat = np.empty((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            ssim_mat[i, j] = ssim(signals[i], signals[j]) if i != j else 1.0

    # Recovery priorities from random morphologies
    morphologies = [
        Morphology(
            length=rng.uniform(0.5, 2.0),
            width=rng.uniform(0.5, 2.0),
            height=rng.uniform(0.5, 2.0),
            mass=rng.uniform(0.5, 5.0),
        )
        for _ in range(num_nodes)
    ]
    rec_prio = np.array([recovery_priority(m) for m in morphologies])

    return raw, post, accept, ssim_mat, rec_prio


# ----------------------------------------------------------------------
# 8. Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a demo graph with 5 nodes
    raw_cost, post, accept, ssim_mat, rec_prio = build_demo_graph(5)

    # Choose arbitrary start / end nodes
    start_node = 0
    end_node = 4

    cost = tropical_weighted_max_path(
        raw_cost,
        post,
        accept,
        ssim_mat,
        rec_prio,
        start_node,
        end_node,
    )

    print(f"Tropical weighted max‑plus cost from node {start_node} to {end_node}: {cost:.4f}")

    # Simple Hoeffding bound demonstration
    decision = hoeffding_bound(num_samples=150, epsilon=0.1)
    print(f"Hoeffding decision (should split?): {decision}")

    # Acceptance probability example
    prob = acceptance_probability(delta_energy=2.3, temperature=5.0)
    print(f"Acceptance probability for ΔE=2.3, T=5.0: {prob:.4f}")