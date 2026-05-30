# DARWIN HAMMER — match 1537, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m284_s3.py (gen4)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s2.py (gen5)
# born: 2026-05-29T23:37:23Z

"""
HYBRID ALGORITHM FUSION
------------------------

Combines the mathematical structures of
'hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m284_s3.py' and
'hybrid_hybrid_path_signatur_hybrid_hybrid_hybrid_m572_s2.py'
to form a unified system.

The bridge between the two structures is established through
the concept of 'Morphology' and 'MathAction' data classes,
which are used to describe geometric and mathematical entities
respectively. The 'EndpointCircuitBreaker' class is adapted to
track the 'cost' and 'risk' attributes of 'MathAction' instances.

The fusion integrates the 'lead_lag_transform' and 'bspline_basis'
functions from the second parent with the 'sphericity_index'
function from the first parent, forming a new set of mathematical
operators that operate on both geometric and mathematical data.

This hybrid algorithm is designed to analyze and optimize complex
systems by combining the strengths of both parent structures.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Geometric description of a physical (or logical) entity."""
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class MathAction:
    """Mathematical description of an action."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def as_dict(self) -> dict:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


# ----------------------------------------------------------------------
# Mathematical utilities
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Ratio of the geometric mean of dimensions to the longest dimension."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    geo_mean = (length * width * height) ** (1.0 / 3.0)
    return geo_mean / max(length, width, height)


def lead_lag_transform(path):
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t]     = np.concatenate([path[t],     path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    n_basis = len(grid) + k - 2      
    N = len(x)

    B = np.zeros((N, len(t) - 1), dtype=np.float64)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]
            term_l = (
                (x - t[i]) / denom_l * B[:, i]
                if denom_l > 0 else np.zeros(N)
            )
            term_r = (
                (t[i + order] - x) / denom_r * B[:, i + 1]
                if denom_r > 0 else np.zeros(N)
            )
            B_new[:, i] = term_l + term_r
        B = B_new

    return B


def calculate_regret_weighted_probabilities(actions: list) -> np.ndarray:
    probabilities = np.array([action.expected_value for action in actions])
    regret_weights = np.array([action.risk for action in actions])
    return probabilities * regret_weights


def hybrid_math_operator(morphology: Morphology, actions: list):
    """Computes a hybrid mathematical operator that combines geometric and mathematical data."""
    # Compute sphericity index of morphology
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)

    # Transform actions using lead-lag transform and bspline basis
    transformed_actions = lead_lag_transform(np.array([action.expected_value for action in actions]))
    basis = bspline_basis(transformed_actions[:, 0], np.linspace(0, 1, len(transformed_actions)))

    # Compute weighted probabilities of actions
    probabilities = calculate_regret_weighted_probabilities(actions)

    # Combine sphericity index and weighted probabilities into a single vector
    result = np.concatenate([sphericity * np.ones(len(probabilities)), probabilities])

    return result


def hybrid_math_operator_2(actions: list):
    """Computes a hybrid mathematical operator that combines mathematical data only."""
    # Transform actions using lead-lag transform and bspline basis
    transformed_actions = lead_lag_transform(np.array([action.expected_value for action in actions]))
    basis = bspline_basis(transformed_actions[:, 0], np.linspace(0, 1, len(transformed_actions)))

    # Compute weighted probabilities of actions
    probabilities = calculate_regret_weighted_probabilities(actions)

    # Combine weighted probabilities into a single vector
    result = probabilities

    return result


def hybrid_math_operator_3(morphology: Morphology, actions: list):
    """Computes a hybrid mathematical operator that combines geometric and mathematical data."""
    # Compute sphericity index of morphology
    sphericity = sphericity_index(morphology.length, morphology.width, morphology.height)

    # Transform actions using lead-lag transform and bspline basis
    transformed_actions = lead_lag_transform(np.array([action.expected_value for action in actions]))
    basis = bspline_basis(transformed_actions[:, 0], np.linspace(0, 1, len(transformed_actions)))

    # Combine sphericity index and basis functions into a single matrix
    result = np.concatenate([sphericity * np.ones(len(basis)), basis], axis=1)

    return result


if __name__ == "__main__":
    # Smoke test
    morphology = Morphology(length=10.0, width=5.0, height=3.0, mass=1.0)
    actions = [MathAction(id="action1", expected_value=0.5), MathAction(id="action2", expected_value=0.7)]
    result = hybrid_math_operator(morphology, actions)
    print(result)
    result = hybrid_math_operator_2(actions)
    print(result)
    result = hybrid_math_operator_3(morphology, actions)
    print(result)