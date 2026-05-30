# DARWIN HAMMER — match 5252, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1444_s0.py (gen6)
# parent_b: hybrid_hybrid_percyphon_hyb_pheromone_m337_s0.py (gen3)
# born: 2026-05-30T00:00:59Z

"""Hybrid algorithm merging:
- Parent A: SSIM-based similarity, multivector encoding, hybrid routing utilities.
- Parent B: Morphology sphericity index, pheromone‑style adaptive signal, endpoint circuit breaker.

Mathematical bridge:
1. Morphology dimensions are treated as a signal vector.  SSIM (from Parent A) quantifies the similarity
   between two morphology vectors.
2. The sphericity index (from Parent B) is multiplied by the SSIM value to obtain an adaptive pheromone
   strength that respects the system’s shape.
3. The resulting scalar is encoded into a multivector (Parent A) which is then combined with a
   packet‑based hybrid score.  The final decision is gated by an endpoint circuit breaker
   (Parent B)."""

import sys
import random
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any

# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------
PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)


def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index (SSIM) for two equal‑length signals."""
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)


class Multivector:
    """Simple multivector with dictionary‑based basis representation."""

    def __init__(self, components: Dict[Tuple[int, ...], float], n: int):
        self.n = n
        # store keys as frozenset for order‑independent lookup
        self.components = {frozenset(k): float(v) for k, v in components.items()}

    def __add__(self, other: "Multivector") -> "Multivector":
        if self.n != other.n:
            raise ValueError("Multivectors must have the same dimension")
        result: Dict[frozenset, float] = {}
        for k in set(self.components) | set(other.components):
            result[k] = self.components.get(k, 0.0) + other.components.get(k, 0.0)
        return Multivector({tuple(k): v for k, v in result.items()}, self.n)

    def scale(self, factor: float) -> "Multivector":
        return Multivector({tuple(k): v * factor for k, v in self.components.items()}, self.n)

    def __repr__(self) -> str:
        comps = {tuple(k): v for k, v in self.components.items()}
        return f"Multivector(n={self.n}, components={comps})"


def hybrid_score(packet: Dict[str, Any]) -> float:
    """Score a packet by comparing its payload to a prototype via SSIM."""
    payload = packet.get("payload")
    if not isinstance(payload, (list, tuple)):
        return 0.0
    try:
        payload_vec = np.asarray(payload, dtype=np.float64)
        if payload_vec.size != PROTOTYPE_VECTOR.size:
            # truncate or pad with zeros to match size
            if payload_vec.size > PROTOTYPE_VECTOR.size:
                payload_vec = payload_vec[: PROTOTYPE_VECTOR.size]
            else:
                pad = np.zeros(PROTOTYPE_VECTOR.size - payload_vec.size)
                payload_vec = np.concatenate([payload_vec, pad])
        return compute_ssim(payload_vec.tolist(), PROTOTYPE_VECTOR.tolist())
    except Exception:
        return 0.0


# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(length: float, width: float, height: float) -> float:
    """Classic sphericity for a rectangular prism."""
    if min(length, width, height) <= 0:
        raise ValueError("Dimensions must be positive")
    volume = length * width * height
    surface = 2 * (length * width + length * height + width * height)
    # Sphericity = (π^{1/3} * (6V)^{2/3}) / A
    return (math.pi ** (1.0 / 3.0) * (6 * volume) ** (2.0 / 3.0)) / surface


class EndpointCircuitBreaker:
    """Simple failure‑count circuit breaker."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        return not self.open


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def morphology_vector(morph: Morphology) -> List[float]:
    """Convert Morphology to a numeric vector for SSIM."""
    return [morph.length, morph.width, morph.height, morph.mass]


def pheromone_strength(morph: Morphology, reference: Morphology) -> float:
    """
    Adaptive pheromone signal = sphericity * SSIM(morph, reference).

    The reference morphology can be a design target or the previous state.
    """
    vec_a = morphology_vector(morph)
    vec_b = morphology_vector(reference)
    similarity = compute_ssim(vec_a, vec_b)
    sph = sphericity_index(morph.length, morph.width, morph.height)
    return sph * similarity


def encode_pheromone_to_multivector(strength: float) -> Multivector:
    """
    Encode a scalar pheromone strength into a 3‑dimensional multivector.
    Basis vectors are arbitrarily chosen as (0,), (1,), (2,).
    """
    components = {
        (0,): strength,
        (1,): strength * 0.5,
        (2,): strength * 0.25,
    }
    return Multivector(components, n=3)


def hybrid_decision(
    packet: Dict[str, Any],
    morph: Morphology,
    reference: Morphology,
    breaker: EndpointCircuitBreaker,
) -> Tuple[bool, float]:
    """
    Compute a decision for a packet.

    Returns (allowed, final_score):
    - allowed: False if circuit breaker is open.
    - final_score: weighted sum of SSIM‑based packet score and pheromone‑derived multivector norm.
    """
    if not breaker.allow():
        return False, 0.0

    pkt_score = hybrid_score(packet)  # from Parent A
    pheromone = pheromone_strength(morph, reference)  # bridge
    mv = encode_pheromone_to_multivector(pheromone)

    # Simple multivector norm (sqrt of sum of squares of components)
    norm = math.sqrt(sum(v ** 2 for v in mv.components.values()))

    final_score = 0.6 * pkt_score + 0.4 * (norm / (norm + 1))  # keep in [0,1]

    # Randomly decide based on final_score threshold
    allowed = random.random() < final_score
    return allowed, final_score


def update_breaker_from_outcome(breaker: EndpointCircuitBreaker, success: bool) -> None:
    """Update circuit breaker based on the outcome of a decision."""
    if success:
        breaker.record_success()
    else:
        breaker.record_failure()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a dummy packet
    packet_example = {"payload": [0.21, 0.49, 0.31, 0.68, 0.12]}

    # Current and reference morphologies
    current_morph = Morphology(length=2.0, width=1.5, height=1.0, mass=3.2)
    reference_morph = Morphology(length=2.1, width=1.4, height=1.05, mass=3.0)

    # Circuit breaker
    breaker = EndpointCircuitBreaker(failure_threshold=2)

    # Run several decisions to see breaker behavior
    for i in range(5):
        allowed, score = hybrid_decision(packet_example, current_morph, reference_morph, breaker)
        print(f"Iteration {i+1}: allowed={allowed}, score={score:.4f}, breaker_open={not breaker.allow()}")
        update_breaker_from_outcome(breaker, allowed)

    sys.exit(0)