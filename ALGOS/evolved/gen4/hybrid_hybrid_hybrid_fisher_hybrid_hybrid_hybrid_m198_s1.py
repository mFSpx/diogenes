# DARWIN HAMMER — match 198, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s0.py (gen3)
# born: 2026-05-29T23:27:30Z

"""Hybrid Fisher–Risk VRAM Scheduler
Parents:
- hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s2.py (Gaussian beam, Fisher information)
- hybrid_hybrid_hybrid_privac_hybrid_endpoint_circ_m28_s0.py (Reconstruction risk, VRAM budgeting, circuit breaker)

Mathematical bridge:
Both parents expose a *probabilistic score* that can be interpreted as a weight:
    • Fisher information I(θ) = (∂ℓ/∂θ)² / ℓ, where ℓ is a Gaussian beam intensity.
    • Reconstruction risk R = unique_quasi_identifiers / total_records.
We fuse them by forming a composite cost C = I(θ) · (R·RAM), i.e. the Fisher score modulated by the
expected memory load (RAM) scaled by the privacy‑risk probability.  This unified metric drives a
minimum‑cost model selection under a VRAM budget while respecting a circuit‑breaker that guards
against repeated failures.
"""

import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A components – Gaussian beam & Fisher information
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity centred at *center* with standard deviation *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a single‑parameter Gaussian model.
    I(θ) = (∂ℓ/∂θ)² / ℓ, where ℓ = Gaussian intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def gaussian_filter(data: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a 1‑D Gaussian smoothing kernel (sigma) to *data*."""
    kernel = np.array([gaussian_beam(x, 0.0, sigma) for x in data])
    kernel /= kernel.sum()
    return np.convolve(data, kernel, mode="same")


# ----------------------------------------------------------------------
# Parent B components – privacy risk, VRAM cost, circuit breaker
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """
    Simple privacy‑risk estimator.
    R = unique / total, clipped to [0, 1].
    """
    if total_records <= 0:
        return 0.0
    risk = unique_quasi_identifiers / total_records
    return max(0.0, min(1.0, risk))


class EndpointCircuitBreaker:
    """Failure counter that opens after *failure_threshold* consecutive failures."""

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


# ----------------------------------------------------------------------
# Hybrid core – unified cost metric and scheduler
# ----------------------------------------------------------------------
def combined_metric(
    theta: float,
    center: float,
    width: float,
    model: ModelTier,
    uid: int,
    total: int,
) -> float:
    """
    Unified cost C = I(θ) * (R * RAM).

    - I(θ)   : Fisher information from the Gaussian beam.
    - R      : Reconstruction risk (probability of re‑identification).
    - RAM    : Model memory consumption in MB.
    """
    fisher = fisher_score(theta, center, width)
    risk = reconstruction_risk_score(uid, total)
    return fisher * (risk * model.ram_mb)


def schedule_models(
    models: List[ModelTier],
    theta: float,
    center: float,
    width: float,
    uid: int,
    total: int,
    vram_budget_mb: int,
    breaker: EndpointCircuitBreaker,
) -> List[Tuple[ModelTier, float]]:
    """
    Greedy selector that picks models with the lowest combined_metric while
    respecting the VRAM budget.  Models that cause a failure are recorded in
    *breaker*; once the breaker opens they are excluded from further consideration.
    Returns a list of (model, metric) pairs for the selected models.
    """
    # Compute metric for each model
    scored = [
        (model, combined_metric(theta, center, width, model, uid, total))
        for model in models
    ]
    # Sort ascending – lower metric = more desirable
    scored.sort(key=lambda pair: pair[1])

    selected: List[Tuple[ModelTier, float]] = []
    used_vram = 0

    for model, metric in scored:
        if not breaker.allow():
            # Circuit open – stop scheduling further models
            break
        # Simulate a stochastic access that may fail (10 % chance)
        if random.random() < 0.10:
            breaker.record_failure()
            continue
        else:
            breaker.record_success()

        if used_vram + model.ram_mb > vram_budget_mb:
            continue  # cannot fit, skip
        selected.append((model, metric))
        used_vram += model.ram_mb

    return selected


def smooth_risk_over_time(
    timestamps: np.ndarray,
    raw_risks: np.ndarray,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Treat *raw_risks* as a noisy time‑series of reconstruction risk and
    apply Gaussian smoothing (the same kernel used in the Fisher branch).
    Returns the smoothed risk values.
    """
    if timestamps.shape != raw_risks.shape:
        raise ValueError("timestamps and raw_risks must have the same shape")
    # Normalise timestamps to a uniform grid for convolution
    sorted_idx = np.argsort(timestamps)
    sorted_risks = raw_risks[sorted_idx]
    return gaussian_filter(sorted_risks, sigma)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a synthetic pool of models
    pool = [
        ModelTier(name="tiny", ram_mb=256, tier="A"),
        ModelTier(name="small", ram_mb=512, tier="A"),
        ModelTier(name="medium", ram_mb=1024, tier="B"),
        ModelTier(name="large", ram_mb=2048, tier="B"),
        ModelTier(name="xl", ram_mb=4096, tier="C"),
    ]

    # Parameters for the Fisher component
    theta_val = 0.75
    center_val = 0.5
    width_val = 0.2

    # Privacy parameters
    unique_qi = 123
    total_records = 10000

    # VRAM budget (in MB)
    budget = 4096

    # Initialise circuit breaker
    cb = EndpointCircuitBreaker(failure_threshold=2)

    selected = schedule_models(
        models=pool,
        theta=theta_val,
        center=center_val,
        width=width_val,
        uid=unique_qi,
        total=total_records,
        vram_budget_mb=budget,
        breaker=cb,
    )

    print("Selected models under budget:")
    for mdl, metric in selected:
        print(f" - {mdl.name:<5} (RAM={mdl.ram_mb} MB) metric={metric:.6g}")

    # Demonstrate risk smoothing over a dummy timeline
    times = np.arange(0, 100, 1, dtype=float)
    raw_risk = np.clip(np.random.normal(loc=0.02, scale=0.015, size=times.shape), 0, 1)
    smoothed = smooth_risk_over_time(times, raw_risk, sigma=3.0)
    print("\nRisk smoothing demo (first 10 values):")
    for t, r_raw, r_smooth in zip(times[:10], raw_risk[:10], smoothed[:10]):
        print(f"t={int(t):02d} raw={r_raw:.4f} smooth={r_smooth:.4f}")

    sys.exit(0)