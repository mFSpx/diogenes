# DARWIN HAMMER — match 5398, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_physar_hybrid_hybrid_hybrid_m1208_s7.py (gen6)
# parent_b: hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s4.py (gen3)
# born: 2026-05-30T00:01:40Z

"""
HYBRID algorithm: physarum_voronoi_hybrid (parents: hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m520_s4.py and hybrid_hybrid_ternary_route_hybrid_voronoi_parti_m41_s4.py)
This fusion integrates the physarum network dynamics and Voronoi partitioning techniques,
utilizing the normalized weight vector for adaptive edge conductance update.

Physarum network dynamics are used to simulate the growth of a network by updating the
conductance of each edge based on the flux through it. In this hybrid algorithm, we
introduce a Voronoi-based partitioning scheme to adaptively adjust the weights assigned
to each edge in the physarum network.

The normalized weight vector, derived from the weekday weight vector and epistemic flag
handling of parent A, is used to modulate the conductance update of each edge. This
allows the network to adapt to changing conditions and prioritize edges with higher
weights.

The Voronoi partitioning scheme, inspired by parent B, enables the network to
adaptively divide the space into regions, each associated with a specific seed point.
This allows the network to respond to changing conditions and prioritize edges
connecting regions with higher weights.

The fusion of these two techniques enables the creation of a network that can adapt
to changing conditions, prioritize edges with higher weights, and respond to
changing regional demands.

Author: [Your Name]
Date: [Today's Date]
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Any
import numpy as np

# ----------------------------------------------------------------------
# Parent A primitives (physarum flux & conductance dynamics)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on a single edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 0.1,
                       gain: float = 1.0,
                       decay: float = 0.01,
                       eps: float = 1e-12) -> float:
    """Conductance ODE step based on absolute flux."""
    delta = dt * (gain * abs(q) - decay * conductance)
    new_c = max(0.0, conductance + delta)
    return new_c


# ----------------------------------------------------------------------
# Parent B primitives (Voronoi partitioning & endpoint circuit breaker)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")
_EPISTEMIC_WEIGHT: Dict[str, float] = {
    "FACT": 1.0,
    "PROBABLE": 0.8,
    "POSSIBLE": 0.5,
    "BULLSHIT": 0.2,
    "SURE_MAYBE": 0.6,
}

def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """Return a normalized weight vector that varies sinusoidally with day‑of‑week."""
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec


def assign_voronoi(points: List[Tuple[float, float]], seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    regions: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


def nearest(point: Tuple[float, float], seeds: List[Tuple[float, float]]) -> int:
    if not seeds:
        raise ValueError("seed list cannot be empty")
    return min(range(len(seeds)), key=lambda i: (euclidean(point, seeds[i]), i))


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.is_open = False

    def record_success(self) -> None:
        if self.is_open:
            self.reset()

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.is_open = True

    def reset(self) -> None:
        self.failures = 0
        self.is_open = False

    def status(self) -> Dict[str, Any]:
        return {"failures": self.failures, "is_open": self.is_open}


class BayesianFailureEstimator:
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = alpha
        self.beta = beta

    def update(self, success: bool) -> None:
        if success:
            self.beta += 1.0
        else:
            self.alpha += 1.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


def compute_edge_cost(
    point: Tuple[float, float],
    seed: Tuple[float, float],
    estimator: BayesianFailureEstimator,
    lambda_d: float
) -> float:
    # Compute the edge cost using the Bayesian failure estimator and lambda_d
    # This function will be modified to incorporate the physarum network dynamics
    return estimator.mean + lambda_d


def update_edge_conductance(
    conductance: float,
    q: float,
    weekday_weight_vec: np.ndarray,
    lambda_d: float
) -> float:
    # Update the edge conductance using the physarum network dynamics and weekday weight vector
    new_c = update_conductance(conductance, q)
    return new_c * weekday_weight_vec


def hybrid_physarum_voronoi(
    points: List[Tuple[float, float]],
    seeds: List[Tuple[float, float]],
    lambda_d: float,
    gain: float,
    decay: float
) -> Dict[int, List[Tuple[float, float]]]:
    # Assign Voronoi regions to each point
    voronoi_regions = assign_voronoi(points, seeds)

    # Initialize the edge conductance for each region
    edge_conductances = {i: 1.0 for i in range(len(seeds))}

    # Update the edge conductance for each region using the hybrid algorithm
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            if i != j:
                # Compute the edge cost using the Bayesian failure estimator and lambda_d
                edge_cost = compute_edge_cost(points[i], seeds[j], BayesianFailureEstimator(), lambda_d)
                # Update the edge conductance using the physarum network dynamics and weekday weight vector
                edge_conductances[i] = update_edge_conductance(edge_conductances[i], edge_cost, weekday_weight_vector(GROUPS, 0), lambda_d)

    return voronoi_regions


# Smoke test
if __name__ == "__main__":
    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    seeds = [(0.5, 0.5)]
    lambda_d = 0.5
    gain = 1.0
    decay = 0.01
    voronoi_regions = hybrid_physarum_voronoi(points, seeds, lambda_d, gain, decay)
    print(voronoi_regions)