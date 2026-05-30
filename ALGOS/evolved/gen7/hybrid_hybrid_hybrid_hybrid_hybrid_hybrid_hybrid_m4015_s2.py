# DARWIN HAMMER — match 4015, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2286_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_cockpit_metri_m1831_s4.py (gen4)
# born: 2026-05-29T23:53:13Z

import math
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np


# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Model:
    """Parameters describing a candidate model."""
    r: float   # base RAM requirement (scalar load)
    p: float   # reconstruction‑risk coefficient
    F: float   # Fisher information score
    pi: float  # recovery priority (0 ≤ pi ≤ 1)


# ----------------------------------------------------------------------
# Utility functions – mathematical building blocks
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int,
                              total_records: int) -> float:
    """Normalized reconstruction risk in [0, 1]."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian kernel centred at *center* with standard deviation *width*."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float,
                 center: float,
                 width: float,
                 eps: float = 1e-12) -> float:
    """
    Approximate Fisher information for a Gaussian beam.
    Uses a small epsilon to avoid division by zero.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def anti_slop_ratio(claims_with_evidence: int,
                    total_claims_emitted: int) -> float:
    """Evidence ratio, clipped to [0, 1]."""
    if total_claims_emitted <= 0:
        return 1.0
    return min(1.0, claims_with_evidence / total_claims_emitted)


def audit_quality_factor(anti_slop: float,
                         cockpit_honesty: float,
                         audit_debt: float) -> float:
    """
    Compute the audit quality factor Q.
    The denominator is protected against zero to keep Q bounded.
    """
    return anti_slop * cockpit_honesty / (1.0 + max(audit_debt, 0.0))


# ----------------------------------------------------------------------
# System components
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """
    Simple circuit breaker that opens after *max_failures* consecutive
    overload attempts.  It can be reset manually.
    """
    def __init__(self, max_failures: int):
        if max_failures < 1:
            raise ValueError("max_failures must be >= 1")
        self.max_failures = max_failures
        self.failures = 0

    def record_failure(self) -> None:
        self.failures += 1

    def reset(self) -> None:
        self.failures = 0

    @property
    def is_open(self) -> bool:
        return self.failures >= self.max_failures


class PheromoneRLCTSystem:
    """
    Maintains a scalar pheromone that modulates the exploration
    parameter *theta*.  The update rule is deliberately simple:
    pheromone ← (1‑α)·pheromone + α·reward,  where α∈(0,1) is a learning rate.
    """
    def __init__(self, base_pheromone: float, learning_rate: float = 0.2):
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1]")
        self.base_pheromone = base_pheromone
        self.pheromone = base_pheromone
        self.alpha = learning_rate

    def update(self, reward: float) -> None:
        """Blend the observed *reward* into the pheromone level."""
        self.pheromone = (1.0 - self.alpha) * self.pheromone + self.alpha * reward

    def theta(self) -> float:
        """
        Map the current pheromone to a usable *theta* in [0, 1].
        A sigmoid squashes the raw pheromone.
        """
        return 1.0 / (1.0 + math.exp(-self.pheromone))


# ----------------------------------------------------------------------
# Hybrid optimisation core
# ----------------------------------------------------------------------
class HybridModelSelector:
    """
    Integrates model‑selection (minimise bilinear load L) with audit‑quality
    weighting.  The algorithm proceeds greedily but respects the circuit
    breaker and adapts *theta* via the pheromone system.
    """
    def __init__(self,
                 budget: float,
                 circuit_breaker: EndpointCircuitBreaker,
                 pheromone_system: PheromoneRLCTSystem,
                 audit_params: Tuple[float, float, float]):
        """
        *budget* – maximum allowable total load.
        *circuit_breaker* – monitors overload failures.
        *pheromone_system* – supplies the adaptive theta.
        *audit_params* – (anti_slop, cockpit_honesty, audit_debt) for Q.
        """
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.budget = budget
        self.cb = circuit_breaker
        self.ph = pheromone_system
        self.anti_slop, self.cockpit_honesty, self.audit_debt = audit_params

    @staticmethod
    def _model_load_vector(models: List[Model],
                           theta: float) -> np.ndarray:
        """
        Return the diagonal of W(θ) as a 1‑D array.
        ℓᵢ(θ) = rᵢ + pᵢ·Fᵢ(θ)·(1‑πᵢ)
        """
        loads = np.empty(len(models), dtype=float)
        for i, m in enumerate(models):
            Fi_theta = fisher_score(theta, center=m.F, width=0.1)  # width is a tunable hyper‑parameter
            loads[i] = m.r + m.p * Fi_theta * (1.0 - m.pi)
        return loads

    def _audit_quality(self) -> float:
        """Compute the audit quality factor Q."""
        return audit_quality_factor(self.anti_slop,
                                    self.cockpit_honesty,
                                    self.audit_debt)

    def select(self, models: List[Model]) -> List[Model]:
        """
        Greedy selection respecting the bilinear load L and the audit quality.
        The effective load is scaled by 1/Q so that higher audit quality
        relaxes the budget constraint.
        """
        if self.cb.is_open:
            raise RuntimeError("Circuit breaker is open – selection aborted")

        theta = self.ph.theta()
        loads = self._model_load_vector(models, theta)
        Q = self._audit_quality()
        if Q <= 0:
            raise RuntimeError("Invalid audit quality factor Q ≤ 0")

        # Effective load incorporates audit quality
        effective_load = loads / Q

        # Greedy ordering by smallest effective load
        order = np.argsort(effective_load)
        selected: List[Model] = []
        total_load = 0.0

        for idx in order:
            candidate_load = loads[idx]          # raw load for budget accounting
            if total_load + candidate_load <= self.budget:
                selected.append(models[idx])
                total_load += candidate_load
            else:
                # Record a failure – too heavy for the remaining budget
                self.cb.record_failure()
                if self.cb.is_open:
                    break

        # Provide a reward signal to the pheromone system:
        # reward = (total_load / budget) * Q  (higher is better)
        reward = (total_load / self.budget) * Q
        self.ph.update(reward)

        return selected


# ----------------------------------------------------------------------
# Policy stub (kept minimal – focus is on the hybrid selection)
# ----------------------------------------------------------------------
def update_policy(updates: List[Dict[str, Any]]) -> None:
    """
    Placeholder for a bandit‑style policy update.
    In a full implementation this would adjust action probabilities
    based on observed rewards.
    """
    pass


# ----------------------------------------------------------------------
# Demonstration entry‑point
# ----------------------------------------------------------------------
def main() -> None:
    # Sample models – in practice these would be generated or loaded.
    models = [
        Model(r=1.0, p=0.5, F=0.8, pi=0.2),
        Model(r=2.0, p=0.3, F=0.9, pi=0.1),
        Model(r=3.0, p=0.7, F=0.6, pi=0.3),
        Model(r=0.8, p=0.4, F=0.7, pi=0.15),
        Model(r=1.5, p=0.6, F=0.85, pi=0.05),
    ]

    # Initialise system components
    circuit_breaker = EndpointCircuitBreaker(max_failures=3)
    pheromone_system = PheromoneRLCTSystem(base_pheromone=0.0, learning_rate=0.15)

    # Example audit parameters (could be derived from an external audit module)
    anti_slop = anti_slop_ratio(claims_with_evidence=42, total_claims_emitted=50)
    cockpit_honesty = 0.9          # domain‑specific trust metric
    audit_debt = 0.2               # accumulated audit lag

    selector = HybridModelSelector(
        budget=10.0,
        circuit_breaker=circuit_breaker,
        pheromone_system=pheromone_system,
        audit_params=(anti_slop, cockpit_honesty, audit_debt)
    )

    try:
        selected = selector.select(models)
    except RuntimeError as exc:
        print(f"Selection aborted: {exc}", file=sys.stderr)
        return

    print("Selected models (r, p, F, pi):")
    for m in selected:
        print(f"{m.r:.2f}, {m.p:.2f}, {m.F:.2f}, {m.pi:.2f}")

    print(f"\nFinal pheromone level: {pheromone_system.pheromone:.4f}")
    print(f"Derived theta: {pheromone_system.theta():.4f}")
    print(f"Circuit breaker failures: {circuit_breaker.failures}")


if __name__ == "__main__":
    main()