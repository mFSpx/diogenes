# DARWIN HAMMER — match 2286, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_hybrid_hybrid_m606_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m480_s4.py (gen4)
# born: 2026-05-29T23:41:50Z

"""Hybrid Algorithm integrating privacy‑risk/fisher scoring (Parent A) with morphology‑driven recovery priority and RLCT estimation (Parent B).

Mathematical Bridge:
- Each candidate model *i* is described by a feature vector  
  **fᵢ(θ) = [ rᵢ , pᵢ·Fᵢ(θ) , 1‑πᵢ ]**  
  where  
    * rᵢ = RAM requirement,  
    * pᵢ = reconstruction‑risk score (Parent A),  
    * Fᵢ(θ) = Fisher score for a chosen θ (Parent A),  
    * πᵢ = recovery priority derived from morphology (Parent B).  
- The total system load for a binary selection vector **x** is the bilinear form  

  **L(θ, x) = xᵀ W(θ) x**,  

  with a diagonal weight matrix **W(θ)** whose i‑th diagonal entry is the scalar load  
  **ℓᵢ(θ) = rᵢ + pᵢ·Fᵢ(θ) · (1‑πᵢ)**.  
- Model selection proceeds by greedy minimisation of **L** under a hard budget *B* while an
  `EndpointCircuitBreaker` monitors successive overload failures.
- The `PheromoneRLCTSystem` supplies a dynamic pheromone weight *γ* based on the
  RLCT estimate from loss histories; this weight scales the budget *B*,
  providing a feedback loop between learning dynamics (Parent B) and
  privacy‑aware resource allocation (Parent A).

The code below implements this unified system with three demonstrative functions.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np

# ---------- Parent A components ----------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Normalized reconstruction risk in [0,1]."""
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

class EndpointCircuitBreaker:
    """Simple circuit breaker that opens after a configurable number of overload failures."""
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open = True

# ---------- Parent B components ----------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

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
    """Normalized priority in [0,1]; higher righting time → lower priority."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

class PheromoneRLCTSystem:
    """Tracks pheromone levels keyed by experiment identifiers."""
    def __init__(self):
        self.pheromone_signals: Dict[str, float] = {}

    @staticmethod
    def estimate_rlct_from_losses(train_losses_per_n: List[float], n_values: List[float]) -> float:
        """Linear regression in log‑log‑log space to estimate the RLCT."""
        losses = np.asarray(train_losses_per_n, dtype=np.float64)
        ns = np.asarray(n_values, dtype=np.float64)

        if np.any(ns <= np.e):
            raise ValueError("all n_values must be > e for log(log(n)) to be positive")
        if losses.shape != ns.shape:
            raise ValueError("train_losses_per_n and n_values must have the same length")

        y = np.log(np.maximum(losses, 1e-300))
        x = np.log(np.log(ns))

        x_c = x - x.mean()
        y_c = y - y.mean()
        var_x = (x_c ** 2).sum()
        if var_x < 1e-15:
            raise ValueError("n_values have no variance in log(log(n)) space")
        return float((x_c * y_c).sum() / var_x)

    def update_pheromone(self, key: str, rlct: float, decay: float = 0.9) -> None:
        """Exponential decay of previous signal then add new RLCT estimate."""
        prev = self.pheromone_signals.get(key, 0.0)
        self.pheromone_signals[key] = decay * prev + (1 - decay) * rlct

    def get_signal(self, key: str) -> float:
        return self.pheromone_signals.get(key, 0.0)

# ---------- Hybrid data structures ----------
@dataclass
class Model:
    """Unified representation of a candidate model."""
    name: str
    ram: float                                   # RAM demand (GB)
    unique_quasi_identifiers: int
    total_records: int
    fisher_center: float
    fisher_width: float
    morphology: Morphology

    def privacy_risk(self) -> float:
        return reconstruction_risk_score(self.unique_quasi_identifiers, self.total_records)

    def fisher(self, theta: float) -> float:
        return fisher_score(theta, self.fisher_center, self.fisher_width)

    def priority(self) -> float:
        return recovery_priority(self.morphology)

# ---------- Hybrid core functions ----------
def compute_individual_load(model: Model, theta: float) -> float:
    """
    Load contribution ℓᵢ(θ) = rᵢ + pᵢ·Fᵢ(θ)·(1‑πᵢ)
    where rᵢ is RAM, pᵢ is privacy risk, Fᵢ is Fisher score, πᵢ is recovery priority.
    """
    r = model.ram
    p = model.privacy_risk()
    f = model.fisher(theta)
    pi = model.priority()
    return r + p * f * (1.0 - pi)

def total_system_load(models: List[Model], selection: List[int], theta: float) -> float:
    """
    Bilinear form L(θ, x) = xᵀ W(θ) x with diagonal W(θ) containing individual loads.
    `selection` is a binary list of the same length as `models`.
    """
    if len(models) != len(selection):
        raise ValueError("models and selection must have the same length")
    loads = np.array([compute_individual_load(m, theta) for m in models], dtype=np.float64)
    x = np.asarray(selection, dtype=np.float64)
    return float((x * loads).sum())  # diagonal matrix reduces to element‑wise product

def greedy_model_selection(models: List[Model],
                           budget: float,
                           theta: float,
                           breaker: EndpointCircuitBreaker) -> List[int]:
    """
    Greedy selection of models that keeps total load ≤ budget.
    Returns a binary selection vector. The breaker opens if the budget is exceeded
    more than `failure_threshold` times.
    """
    # Sort models by increasing individual load (most efficient first)
    indexed = list(enumerate(models))
    indexed.sort(key=lambda pair: compute_individual_load(pair[1], theta))

    selection = [0] * len(models)
    current_load = 0.0

    for idx, model in indexed:
        candidate_load = compute_individual_load(model, theta)
        if current_load + candidate_load <= budget:
            selection[idx] = 1
            current_load += candidate_load
        else:
            # overload attempt – record failure
            breaker.record_failure()
            if breaker.open:
                # abort selection early if circuit is open
                break
    else:
        # No overloads occurred
        breaker.record_success()

    return selection

def adaptive_budget(base_budget: float,
                    pheromone_system: PheromoneRLCTSystem,
                    key: str,
                    scale: float = 0.5) -> float:
    """
    Adjusts the budget according to the pheromone signal (RLCT estimate).
    Higher RLCT → larger effective budget.
    """
    signal = pheromone_system.get_signal(key)
    return base_budget * (1.0 + scale * signal)

# ---------- Smoke test ----------
if __name__ == "__main__":
    # Create a few synthetic models
    models = [
        Model(
            name="A",
            ram=2.0,
            unique_quasi_identifiers=150,
            total_records=1000,
            fisher_center=0.5,
            fisher_width=0.1,
            morphology=Morphology(length=1.2, width=0.8, height=0.5, mass=0.3)
        ),
        Model(
            name="B",
            ram=3.5,
            unique_quasi_identifiers=80,
            total_records=500,
            fisher_center=0.6,
            fisher_width=0.15,
            morphology=Morphology(length=1.0, width=1.0, height=0.6, mass=0.5)
        ),
        Model(
            name="C",
            ram=1.0,
            unique_quasi_identifiers=20,
            total_records=200,
            fisher_center=0.4,
            fisher_width=0.08,
            morphology=Morphology(length=0.9, width=0.7, height=0.4, mass=0.2)
        ),
    ]

    theta = 0.55
    base_budget = 6.0
    breaker = EndpointCircuitBreaker(failure_threshold=2)

    # Simulate a learning episode to generate RLCT pheromone
    rlct_system = PheromoneRLCTSystem()
    losses = [0.9, 0.7, 0.55, 0.45, 0.38]
    ns = [100, 200, 400, 800, 1600]
    rlct_est = rlct_system.estimate_rlct_from_losses(losses, ns)
    rlct_system.update_pheromone(key="demo", rlct=rlct_est)

    # Adaptive budget based on pheromone
    effective_budget = adaptive_budget(base_budget, rlct_system, key="demo")

    selection = greedy_model_selection(models, effective_budget, theta, breaker)

    selected_names = [m.name for sel, m in zip(selection, models) if sel]
    print(f"Effective budget: {effective_budget:.3f}")
    print(f"Circuit breaker open: {breaker.open}")
    print(f"Selected models: {selected_names}")
    print(f"Total load: {total_system_load(models, selection, theta):.3f}")