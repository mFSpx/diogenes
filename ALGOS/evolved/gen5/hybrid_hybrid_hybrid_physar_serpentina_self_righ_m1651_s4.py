# DARWIN HAMMER — match 1651, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_physarum_netw_hybrid_gliner_zero_s_m66_s2.py (gen4)
# parent_b: serpentina_self_righting.py (gen0)
# born: 2026-05-29T23:38:08Z

"""HybridFluxMorphology
Parent A: hybrid_hybrid_physarum_netw_hybrid_gliner_zero_s_m66_s2.py
Parent B: serpentina_self_righting.py

Mathematical bridge:
- Parent A evolves a conductance C via the ODE  dC/dt = gain·|q| – decay·C,
  where q = flux = C/ℓ·(p₁–p₂).
- Parent B provides a morphology‑dependent righting‑time index R(m) that
  depends on a flatness index f = (L+W)/(2·H).

We map the pressure difference (p₁–p₂) to the flatness index f of a morphology.
Thus the flux becomes q = C/ℓ·f, and the conductance update is directly driven
by the morphology. The updated conductance is then fed back as a scaling factor
for the recovery priority ρ = min(1, R(m)/max_index).  The hybrid priority
combines the textual span score s with the morphology‑driven recovery
priority, yielding a unified metric that couples the network‑flow dynamics
with the physical righting dynamics.
"""

import math
import random
import sys
import pathlib
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# ----------------------------------------------------------------------
# Core data structures (from Parent A)
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_diff: float, eps: float = 1e-12) -> float:
    """Flux based on conductance, edge length and pressure difference."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * pressure_diff


def update_conductance(conductance: float, q: float, dt: float = 1.0,
                       gain: float = 1.0, decay: float = 0.05) -> float:
    """Euler step of dC/dt = gain·|q| – decay·C, clipped at zero."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


# ----------------------------------------------------------------------
# Core data structures (from Parent B)
# ----------------------------------------------------------------------
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


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """0..1 priority for scheduler rescue/rollback assistance."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


# ----------------------------------------------------------------------
# Hybrid utilities
# ----------------------------------------------------------------------
def span_to_morphology(span: Span,
                       base_length: float = 1.0,
                       base_width: float = 0.5,
                       base_height: float = 0.2) -> Morphology:
    """
    Convert a textual Span into a synthetic Morphology.
    The mapping is heuristic:
      - length ∝ span length
      - width ∝ span score (scaled)
      - height ∝ (end‑start) normalized
      - mass ∝ score * length
    """
    span_len = max(1, span.end - span.start)
    length = base_length * span_len
    width = base_width * (0.5 + span.score)  # score in [0,1] → width in [0.5,1.0]·base_width
    height = base_height * (0.5 + 0.5 * span_len / 100.0)  # cap at reasonable values
    mass = max(0.1, span.score * length)  # avoid zero mass
    return Morphology(length=length, width=width, height=height, mass=mass)


def hybrid_flux_conductance(span: Span,
                            conductance: float,
                            edge_length: float,
                            dt: float = 1.0,
                            gain: float = 1.0,
                            decay: float = 0.05) -> Tuple[float, float]:
    """
    Compute flux using the morphology‑derived pressure difference,
    then update conductance.
    Returns (new_conductance, flux_value).
    """
    # Map span → morphology and obtain flatness as pressure difference surrogate
    morph = span_to_morphology(span)
    pressure_diff = flatness_index(morph.length, morph.width, morph.height)
    q = flux(conductance, edge_length, pressure_diff)
    new_c = update_conductance(conductance, q, dt=dt, gain=gain, decay=decay)
    return new_c, q


def hybrid_priority(span: Span,
                    conductance: float,
                    edge_length: float,
                    dt: float = 1.0,
                    gain: float = 1.0,
                    decay: float = 0.05,
                    max_index: float = 10.0) -> float:
    """
    Combine textual span score with morphology‑driven recovery priority.
    The conductance update influences the recovery priority via a scaling factor.
    """
    # Update conductance using morphology‑aware flux
    new_c, q = hybrid_flux_conductance(span, conductance, edge_length,
                                       dt=dt, gain=gain, decay=decay)

    # Derive morphology again (could reuse but kept separate for clarity)
    morph = span_to_morphology(span)

    # Base recovery priority from morphology
    base_priority = recovery_priority(morph, max_index=max_index)

    # Modulate priority by the normalized conductance (0‑1)
    norm_c = math.tanh(new_c)  # smooth bounded map
    modulated_priority = base_priority * (0.5 + 0.5 * norm_c)

    # Fuse with span's intrinsic score (weighted average)
    final_priority = 0.6 * modulated_priority + 0.4 * span.score
    return final_priority


# ----------------------------------------------------------------------
# Demonstration class (optional but illustrative)
# ----------------------------------------------------------------------
class HybridFluxMorphologySystem:
    """
    System that maintains a conductance reservoir and evaluates spans
    using the hybrid priority metric.
    """

    def __init__(self,
                 initial_conductance: float = 1.0,
                 edge_length: float = 1.0,
                 dt: float = 1.0,
                 gain: float = 1.0,
                 decay: float = 0.05):
        self.conductance = initial_conductance
        self.edge_length = edge_length
        self.dt = dt
        self.gain = gain
        self.decay = decay

    def evaluate_span(self, span: Span) -> float:
        """Return hybrid priority and internally update conductance."""
        priority = hybrid_priority(
            span,
            conductance=self.conductance,
            edge_length=self.edge_length,
            dt=self.dt,
            gain=self.gain,
            decay=self.decay
        )
        # Update internal conductance for next call
        self.conductance, _ = hybrid_flux_conductance(
            span,
            self.conductance,
            self.edge_length,
            dt=self.dt,
            gain=self.gain,
            decay=self.decay
        )
        return priority

    def reset(self, conductance: float = 1.0) -> None:
        self.conductance = conductance


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a few synthetic spans
    spans = [
        Span(start=0, end=10, text="alpha", label="A", score=0.2),
        Span(start=15, end=45, text="beta", label="B", score=0.7),
        Span(start=50, end=120, text="gamma", label="C", score=0.5)
    ]

    system = HybridFluxMorphologySystem(initial_conductance=0.8, edge_length=2.0,
                                        dt=0.5, gain=0.9, decay=0.1)

    for i, sp in enumerate(spans, 1):
        prio = system.evaluate_span(sp)
        print(f"Span {i} ({sp.text}) – hybrid priority: {prio:.4f}, conductance: {system.conductance:.4f}")

    # Ensure no exceptions and reasonable values
    assert 0.0 <= system.conductance <= 10.0
    print("Smoke test completed successfully.")