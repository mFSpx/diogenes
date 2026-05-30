# DARWIN HAMMER — match 2731, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_nlms_omni_cha_m142_s4.py (gen3)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m278_s1.py (gen4)
# born: 2026-05-29T23:43:52Z

import math
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """
    Return a cyclic day index in [0, 6] where 0 = Monday, 6 = Sunday.
    The extra ``+1`` mirrors the original implementation but keeps the
    result in the same range.
    """
    return (date(year, month, day).weekday() + 1) % 7


# ----------------------------------------------------------------------
# Core components
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """
    Simple circuit‑breaker that tracks consecutive failures.
    The failure rate is exposed as a *privacy‑load* that can be fed
    into downstream probabilistic models.
    """

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        """Reset the failure counter and close the breaker."""
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        """Increment the failure counter and open the breaker if needed."""
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        """Return ``True`` if the circuit is closed (i.e. work may proceed)."""
        return not self.open

    def failure_rate(self) -> float:
        """
        Normalised failure rate in ``[0, 1]``.
        This value is later interpreted as a *privacy‑load*.
        """
        return min(self.failures / self.failure_threshold, 1.0)


class Morphology:
    """
    Geometric description of an endpoint.
    Provides derived shape descriptors used to adapt Gaussian parameters.
    """

    def __init__(self, length: float, width: float, height: float, mass: float):
        if min(length, width, height, mass) <= 0:
            raise ValueError("All dimensions must be positive")
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

    @property
    def sphericity(self) -> float:
        """
        Ratio of geometric mean to maximal dimension, ∈ (0, 1].
        Used to shrink/expand Gaussian beams proportionally to shape.
        """
        gm = (self.length * self.width * self.height) ** (1.0 / 3.0)
        return gm / max(self.length, self.width, self.height)


# ----------------------------------------------------------------------
# Probabilistic primitives
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """
    Standard Gaussian kernel evaluated at ``theta``.
    ``width`` is the standard deviation; must be > 0.
    """
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_information(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian with mean ``center`` and std. dev. ``width``.
    ``eps`` guards against division by zero.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Domain‑specific helpers
# ----------------------------------------------------------------------
def _tokenise(text: str) -> List[str]:
    """Very light tokeniser – split on whitespace and strip punctuation."""
    return [t.strip(".,;:!?()[]{}\"'").lower() for t in text.split() if t]


def compute_cognitive_risk(text: Union[str, float, int]) -> float:
    """
    Compute a toy *cognitive‑risk* score from free‑form text.
    If the input is not a string we fall back to ``str()`` so the function
    never raises on type errors.
    The score is the count of risk‑related keywords.
    """
    if not isinstance(text, str):
        text = str(text)

    tokens = _tokenise(text)
    risk_keywords = {"evidence", "planning", "threat", "vulnerability", "risk"}
    return float(sum(1 for t in tokens if any(t.startswith(k) for k in risk_keywords)))


# ----------------------------------------------------------------------
# Integrated hybrid model
# ----------------------------------------------------------------------
@dataclass
class Entity:
    """
    Represents a logical entity that participates in the hybrid decision model.
    ``spatial_load`` is free‑form text describing the entity's context.
    ``cognitive_risk`` is an optional pre‑computed risk score; if omitted it
    will be derived from ``spatial_load`` on demand.
    """
    spatial_load: Union[str, float, int]
    cognitive_risk: float = None

    def risk(self) -> float:
        """Lazily compute the risk if it was not supplied at construction."""
        if self.cognitive_risk is None:
            self.cognitive_risk = compute_cognitive_risk(self.spatial_load)
        return self.cognitive_risk


class HybridModel:
    """
    Deep integration of:
      * Endpoint circuit‑breaker (privacy‑load)
      * Morphology‑driven Gaussian beam shaping
      * Fisher information‑based localisation
      * Text‑derived cognitive risk

    The public ``evaluate`` method returns a scalar that can be used for
    downstream decision making.
    """

    def __init__(
        self,
        breaker: EndpointCircuitBreaker,
        morphology: Morphology,
        entity: Entity,
        *,
        beam_center: float = 0.0,
        beam_width: float = 1.0,
    ):
        self.breaker = breaker
        self.morphology = morphology
        self.entity = entity
        self.beam_center = beam_center
        self.beam_width = beam_width

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _adjusted_beam_width(self) -> float:
        """
        Adjust the Gaussian beam width by the endpoint's sphericity.
        More spherical objects produce narrower beams (higher localisation).
        """
        # Guard against pathological sphericity values (should be (0,1])
        s = max(min(self.morphology.sphericity, 1.0), 1e-6)
        return self.beam_width * s

    def _privacy_load(self) -> float:
        """
        Derive a *privacy‑load* from the circuit‑breaker failure rate.
        The load is blended with the entity's cognitive risk to increase
        sensitivity when the system is unstable.
        """
        base = self.breaker.failure_rate()
        risk = self.entity.risk()
        # Simple convex combination; weights can be tuned later.
        return 0.6 * base + 0.4 * (risk / max(risk, 1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate(self, theta: float) -> float:
        """
        Compute the hybrid score for a given observation angle ``theta``.
        The computation chain is:

        1. Adjust Gaussian beam width using morphology.
        2. Evaluate Fisher information at ``theta``.
        3. Modulate by the privacy‑load (circuit‑breaker + risk).
        4. Return the product as the final hybrid score.
        """
        if not self.breaker.allow():
            # Circuit is open → no contribution.
            return 0.0

        width_adj = self._adjusted_beam_width()
        fisher = fisher_information(theta, self.beam_center, width_adj)
        load = self._privacy_load()
        return fisher * load

    # ------------------------------------------------------------------
    # Convenience wrappers mirroring the original script
    # ------------------------------------------------------------------
    def morphology_based_gaussian(self, theta: float) -> float:
        """Gaussian beam shaped by endpoint morphology."""
        width_adj = self._adjusted_beam_width()
        return gaussian_beam(theta, self.beam_center, width_adj)

    def entity_based_fisher(self, theta: float) -> float:
        """Fisher information scaled by the entity's cognitive risk."""
        risk = self.entity.risk()
        width = self.beam_width * risk if risk > 0 else self.beam_width
        return fisher_information(theta, self.beam_center, width)


# ----------------------------------------------------------------------
# Demonstration / simple sanity check
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise components
    breaker = EndpointCircuitBreaker(failure_threshold=3)
    morphology = Morphology(length=1.0, width=2.0, height=3.0, mass=5.0)
    entity = Entity(spatial_load="evidence planning threat", cognitive_risk=None)

    # Simulate a few failures to illustrate privacy‑load effect
    for _ in range(2):
        breaker.record_failure()

    model = HybridModel(
        breaker=breaker,
        morphology=morphology,
        entity=entity,
        beam_center=0.0,
        beam_width=1.0,
    )

    theta_vals = np.linspace(-2.0, 2.0, 9)
    for th in theta_vals:
        print(f"Theta={_pct(th): .2f} → HybridScore={_pct(model.evaluate(th))}")

    # Show the auxiliary functions for comparison with the original script
    print("Morphology‑based Gaussian:", _pct(model.morphology_based_gaussian(0.5)))
    print("Entity‑based Fisher:", _pct(model.entity_based_fisher(0.5)))