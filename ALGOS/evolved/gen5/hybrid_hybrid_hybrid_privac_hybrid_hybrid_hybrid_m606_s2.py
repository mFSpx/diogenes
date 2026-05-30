# DARWIN HAMMER — match 606, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_privacy_model_hybrid_hybrid_fisher_m33_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m274_s0.py (gen4)
# born: 2026-05-29T23:29:56Z

"""Hybrid Privacy‑Fisher‑Circuit Model

This module fuses the core topologies of:

* **Parent A** – *hybrid_hybrid_privacy_model_hybrid_hybrid_fisher_m33_s1.py*  
  which builds a composite resource matrix **A** (RAM, privacy‑load) and weights the
  total load by a Gaussian‑based Fisher score.

* **Parent B** – *hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m274_s0.py*  
  which introduces a Fisher‑score‑driven *EndpointCircuitBreaker* and a
  morphology‑based priority for model selection.

**Mathematical bridge**

Both parents rely on the same Fisher information of a Gaussian beam:

\[
\mathcal{F}(θ; μ, σ)=\frac{(\partial_θ g(θ;μ,σ))^2}{g(θ;μ,σ)},
\qquad
g(θ;μ,σ)=\exp\!\Big(-\tfrac12\big(\tfrac{θ-μ}{σ}\big)^2\Big)
\]

In the hybrid algorithm the Fisher score is used **(i)** to weight the
privacy‑aware resource load and **(ii)** to modulate the state of the
circuit‑breaker that protects the system from excessive error.  The resulting
scalar `load_weight` multiplies the composite load vector, while the same
score (computed on the prediction error) decides whether the breaker opens.

The fusion yields a single unified decision pipeline that respects RAM,
privacy budget, model morphology and runtime reliability."""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared Fisher utilities (identical to both parents)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam g(θ;center,width)."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float = 0.0, width: float = 1.0,
                 eps: float = 1e-12) -> float:
    """Fisher information of the Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Domain specific helpers (privacy & morphology)
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Normalized reconstruction risk ∈[0,1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0,
                        unique_quasi_identifiers / total_records))


_TIER_FACTOR: Dict[str, float] = {"T1": 1.0, "T2": 2.0, "T3": 3.0}


def tier_factor(tier: str) -> float:
    """Map tier string to numeric sensitivity."""
    return _TIER_FACTOR.get(tier.upper(), 1.0)


@dataclass
class Model:
    """Compact description of a candidate model."""
    name: str
    ram_mb: float                     # RAM consumption in megabytes
    tier: str                         # Privacy tier, e.g. "T1"/"T2"/"T3"
    quasi_id_count: int               # Number of quasi‑identifiers
    total_records: int                # Population size for risk calculation
    morphology: Tuple[float, float]   # (length, width) in arbitrary units


def privacy_load(model: Model, alpha: float = 1.0) -> float:
    """
    Compute privacy‑load p(m) = α · tier_factor(tier) · mean(r),
    where r is the reconstruction risk of the model.
    """
    risk = reconstruction_risk_score(model.quasi_id_count,
                                     model.total_records)
    return alpha * tier_factor(model.tier) * risk


def morphology_priority(model: Model) -> float:
    """
    Derive a soft priority from morphology.
    Larger surface (length·width) gets higher priority.
    """
    length, width = model.morphology
    return length * width


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def build_resource_matrix(models: List[Model],
                          alpha: float = 1.0) -> np.ndarray:
    """
    Build composite resource matrix A ∈ ℝ^{n×2} where column 0 = RAM,
    column 1 = privacy‑load.
    """
    rows = []
    for m in models:
        rows.append([m.ram_mb, privacy_load(m, alpha)])
    return np.array(rows, dtype=float)


def weighted_load(selection: np.ndarray,
                  resource_matrix: np.ndarray,
                  theta: float,
                  center: float = 0.0,
                  width: float = 1.0) -> np.ndarray:
    """
    Compute L = (Aᵀ·x) * FisherScore(θ;center,width).

    *selection* is a binary vector x ∈ {0,1}ⁿ indicating which models are loaded.
    The function returns a 2‑element vector [ram_load, privacy_load] after
    Fisher weighting.
    """
    if selection.shape[0] != resource_matrix.shape[0]:
        raise ValueError("selection length must match number of models")
    # column‑wise total load
    total_load = resource_matrix.T @ selection
    weight = fisher_score(theta, center, width)
    return total_load * weight


class EndpointCircuitBreaker:
    """
    Circuit breaker whose state is driven by the Fisher‑weighted prediction error.
    If the weighted error exceeds a configurable threshold, the breaker opens.
    """
    def __init__(self, error_threshold: float = 0.5, failure_limit: int = 3):
        self.error_threshold = error_threshold
        self.failure_limit = failure_limit
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def evaluate_error(self, error: float,
                       theta: float,
                       center: float = 0.0,
                       width: float = 1.0) -> None:
        """
        Update breaker state based on Fisher‑weighted error.
        """
        weighted_error = error * fisher_score(theta, center, width)
        if weighted_error > self.error_threshold:
            self.failures += 1
            if self.failures >= self.failure_limit:
                self.open = True
        else:
            # successful observation resets the counter
            self.failures = 0
            self.open = False

    def allow(self) -> bool:
        """Return True if the circuit is closed (operations permitted)."""
        return not self.open


def select_models(models: List[Model],
                  ram_budget: float,
                  privacy_budget: float,
                  theta: float,
                  center: float = 0.0,
                  width: float = 1.0,
                  alpha: float = 1.0) -> List[Model]:
    """
    Greedy selector that respects both resource budgets while favouring
    morphology priority.  The Fisher‑weighted load is used to test feasibility.
    """
    A = build_resource_matrix(models, alpha)
    # Sort by morphology priority descending
    ordered = sorted(models,
                     key=lambda m: morphology_priority(m),
                     reverse=True)
    selection = np.zeros(len(models), dtype=int)
    for idx, model in enumerate(ordered):
        # Tentative selection vector
        trial = selection.copy()
        trial[models.index(model)] = 1
        load = weighted_load(trial, A, theta, center, width)
        if load[0] <= ram_budget and load[1] <= privacy_budget:
            selection[models.index(model)] = 1
    # Return the concrete model objects that were selected
    return [m for i, m in enumerate(models) if selection[i] == 1]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny catalogue of models
    catalogue = [
        Model(name="A", ram_mb=120.0, tier="T1",
              quasi_id_count=30, total_records=1000,
              morphology=(2.0, 1.5)),
        Model(name="B", ram_mb=200.0, tier="T2",
              quasi_id_count=80, total_records=1200,
              morphology=(1.0, 2.5)),
        Model(name="C", ram_mb=90.0, tier="T3",
              quasi_id_count=10, total_records=500,
              morphology=(3.0, 0.8)),
    ]

    # Budgets
    RAM_BUDGET = 300.0          # MB
    PRIVACY_BUDGET = 1.5        # abstract units

    # Fisher parameters (these could be learned in a real system)
    THETA = 0.2
    CENTER = 0.0
    WIDTH = 1.0

    # Perform selection
    selected = select_models(catalogue,
                             ram_budget=RAM_BUDGET,
                             privacy_budget=PRIVACY_BUDGET,
                             theta=THETA,
                             center=CENTER,
                             width=WIDTH,
                             alpha=0.8)

    print("Selected models:", [m.name for m in selected])

    # Simulate a prediction error stream and feed the circuit breaker
    breaker = EndpointCircuitBreaker(error_threshold=0.4, failure_limit=2)
    errors = [0.05, 0.12, 0.35, 0.48, 0.22]  # arbitrary error magnitudes
    for i, err in enumerate(errors, 1):
        breaker.evaluate_error(err, theta=THETA, center=CENTER, width=WIDTH)
        status = "OPEN" if not breaker.allow() else "CLOSED"
        print(f"Step {i}: error={err:.3f}, breaker={status}")

    # Demonstrate weighted load computation directly
    A = build_resource_matrix(catalogue, alpha=0.8)
    x = np.array([1, 0, 1], dtype=int)  # load models A and C
    load_vec = weighted_load(x, A, theta=THETA, center=CENTER, width=WIDTH)
    print(f"Weighted load (RAM, privacy): {load_vec[0]:.2f} MB, {load_vec[1]:.3f}")