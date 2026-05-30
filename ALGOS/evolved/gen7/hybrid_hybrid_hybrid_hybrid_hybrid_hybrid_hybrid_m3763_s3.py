# DARWIN HAMMER — match 3763, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_path_s_geometric_product_m161_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m1537_s1.py (gen6)
# born: 2026-05-29T23:51:31Z

"""Hybrid Fusion of Path Signature & Morphology

Parents:
- hybrid_hybrid_hybrid_path_s_geometric_product_m161_s0.py (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_path_s_m1537_s1.py (Algorithm B)

Mathematical Bridge:
Algorithm A represents a path signature as a Clifford multivector, enabling
geometric products between signature components. Algorithm B describes
geometric entities via the `Morphology` dataclass and tracks action metrics
through `MathAction` and `EndpointCircuitBreaker`.

The fusion maps the physical attributes of a `Morphology` instance onto a
multivector (scalar ↔ mass, vector ↔ length/width/height). The path signature
is also encoded as a multivector. Their geometric product yields a *hybrid
signature* that simultaneously captures temporal path information and static
geometric characteristics. This hybrid signature drives cost/risk updates
for `MathAction` objects, while a sphericity‑derived scaling factor modulates
the lead‑lag transformed path for downstream processing.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Fallback Clifford algebra implementation (used if geometric_product is absent)
# ----------------------------------------------------------------------
try:
    from geometric_product import Multivector, _multiply_blades  # type: ignore
except Exception:  # pragma: no cover
    class Multivector:
        """Very small Clifford multivector placeholder.

        Components are stored in a dict mapping blade identifiers (tuples of
        basis indices) to scalar coefficients. The empty tuple ``()`` denotes
        the scalar part.
        """
        def __init__(self, components=None):
            self.components = components or {}

        def __add__(self, other):
            result = self.components.copy()
            for k, v in other.components.items():
                result[k] = result.get(k, 0.0) + v
            return Multivector(result)

        def __mul__(self, other):
            """Geometric product (simplified).

            For demonstration purposes we only implement scalar‑scalar
            multiplication and scalar‑vector multiplication. Full blade
            algebra is omitted.
            """
            if isinstance(other, Multivector):
                a = self.components.get((), 0.0)
                b = other.components.get((), 0.0)
                # scalar part multiplies scalars; other blades are ignored
                return Multivector({(): a * b})
            else:  # scalar multiplication
                return Multivector({k: v * other for k, v in self.components.items()})

        __rmul__ = __mul__

        def __repr__(self):
            return f"Multivector({self.components})"

    def _multiply_blades(a, b):  # pragma: no cover
        return a * b

# ----------------------------------------------------------------------
# Core data structures from Parent B
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

# ----------------------------------------------------------------------
# Functions from Parent A (adapted)
# ----------------------------------------------------------------------
def lead_lag_transform(path):
    """Create the lead‑lag transformed version of a discrete path."""
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out

def path_to_multivector(path):
    """Encode a discrete path as a simple multivector.

    - Scalar part: sum of all coordinates.
    - Bivector parts: summed pairwise products of coordinate dimensions.
    """
    path = np.asarray(path, dtype=float)
    scalar = float(np.sum(path))
    comps = {(): scalar}
    d = path.shape[1]
    for i in range(d):
        for j in range(i + 1, d):
            biv = float(np.sum(path[:, i] * path[:, j]))
            comps[(i, j)] = biv
    return Multivector(comps)

def morphology_to_multivector(morph):
    """Map Morphology attributes onto a multivector.

    - Scalar part: mass.
    - Vector part (grade‑1): length, width, height stored on basis 0,1,2.
    """
    comps = {(): morph.mass,
             (0,): morph.length,
             (1,): morph.width,
             (2,): morph.height}
    return Multivector(comps)

# ----------------------------------------------------------------------
# Hybrid operations (new)
# ----------------------------------------------------------------------
def hybrid_signature(morph: Morphology, path) -> Multivector:
    """Geometric product of the path‑signature multivector with the morphology multivector."""
    mv_path = path_to_multivector(path)
    mv_morph = morphology_to_multivector(morph)
    return mv_path * mv_morph

def sphericity_index(morph: Morphology) -> float:
    """Approximate sphericity of a rectangular solid.

    Uses the classic formula:
        Ψ = π^{1/3} (6V)^{2/3} / A
    where V is volume and A is surface area.
    """
    vol = morph.length * morph.width * morph.height
    area = 2 * (morph.length * morph.width +
                morph.length * morph.height +
                morph.width * morph.height)
    if area == 0:
        return 0.0
    return (math.pi ** (1.0 / 3.0)) * (6.0 * vol) ** (2.0 / 3.0) / area

def hybrid_operator(path, morph: Morphology):
    """Apply lead‑lag transform and scale by the sphericity index."""
    transformed = lead_lag_transform(path)
    scale = sphericity_index(morph)
    return transformed * scale

def evaluate_action(morph: Morphology, action: MathAction,
                    path, breaker: EndpointCircuitBreaker):
    """Update a MathAction's cost/risk using the magnitude of the hybrid signature.

    The magnitude is defined as the sum of absolute multivector coefficients.
    """
    hv = hybrid_signature(morph, path)
    magnitude = sum(abs(v) for v in hv.components.values())

    new_cost = action.cost + 0.1 * magnitude
    new_risk = action.risk + 0.05 * magnitude

    updated_action = MathAction(id=action.id,
                                expected_value=action.expected_value,
                                cost=new_cost,
                                risk=new_risk)

    # Simple risk‑threshold policy for the circuit breaker
    risk_threshold = 10.0
    if updated_action.risk > risk_threshold:
        breaker.record_failure()
    else:
        breaker.record_success()

    return updated_action, breaker

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example discrete path (3‑dimensional)
    path = np.array([[0.0, 1.0, 2.0],
                     [3.0, 4.0, 5.0],
                     [6.0, 7.0, 8.0]])

    # Sample morphology
    morph = Morphology(length=2.0, width=1.5, height=3.0, mass=5.0)

    # Baseline action
    action = MathAction(id="A1", expected_value=100.0)

    # Circuit breaker instance
    breaker = EndpointCircuitBreaker(failure_threshold=2)

    # Hybrid signature demonstration
    hv = hybrid_signature(morph, path)
    print("Hybrid signature:", hv)

    # Hybrid operator demonstration
    op = hybrid_operator(path, morph)
    print("Hybrid operator shape:", op.shape)
    print("First row of scaled lead‑lag matrix:", op[0])

    # Action evaluation demonstration
    updated_action, breaker = evaluate_action(morph, action, path, breaker)
    print("Updated action:", updated_action)
    print("Circuit breaker state -> open:", breaker.open,
          "failures:", breaker.failures,
          "last_event:", breaker.last_event_at)