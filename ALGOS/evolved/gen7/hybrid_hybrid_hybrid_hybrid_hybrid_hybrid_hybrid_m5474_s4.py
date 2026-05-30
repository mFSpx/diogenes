# DARWIN HAMMER — match 5474, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2244_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2344_s0.py (gen6)
# born: 2026-05-30T00:02:08Z

"""Hybrid algorithm merging Parent A and Parent B.

Parent A contributes a geometric‑algebra layer: the `Multivector` class and the
geometric product that encodes morphology (length, width, height, mass) of
model tiers.  
Parent B contributes statistical‑learning layers: Fisher information,
Hoeffding bound, Gini impurity and a pheromone‑style circuit‑breaker that
modulates model reliability.

**Mathematical bridge** – The scalar Fisher information computed from a
Gaussian beam (`θ`, `center`, `width`) is injected as a weight on the
coefficients of a `Multivector`.  The resulting weighted multivector is then
used together with a Hoeffding confidence radius and a Gini impurity term to
produce a single “priority” score for a model tier.  The circuit‑breaker
state multiplies this score, effectively acting as a dynamic scaling factor
that can shut down unreliable tiers.

The three core functions below demonstrate this fusion:
1. `geometric_product` – pure algebraic combination of two multivectors.
2. `fisher_weighted_multivector` – embeds Fisher information into a multivector.
3. `hybrid_priority` – merges geometry, statistics and circuit‑breaker state
   into a unified priority metric.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Dict, Tuple, FrozenSet, Iterable, List

# ----------------------------------------------------------------------
# Parent A – Morphology & Multivector
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

class Multivector:
    """A simple multivector where each blade is a frozenset of basis indices
    mapped to a scalar coefficient."""
    def __init__(self, blades: Dict[FrozenSet[int], float] = None):
        self.blades: Dict[FrozenSet[int], float] = blades or {}

    def __repr__(self) -> str:
        return f"Multivector({self.blades})"

    def copy(self) -> 'Multivector':
        return Multivector(self.blades.copy())

    def norm(self) -> float:
        """Euclidean norm of the coefficient vector."""
        return math.sqrt(sum(v * v for v in self.blades.values()))

# ----------------------------------------------------------------------
# Helper for blade multiplication (geometric product)
# ----------------------------------------------------------------------
def _blade_sign(indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)
        elif lst[i] == lst[i + 1]:
            # cancel equal indices (e_i * e_i = 1)
            lst.pop(i)
            lst.pop(i)
            n -= 2
            sign = sign  # unchanged
            i = max(i - 1, 0)
        else:
            i += 1
    return tuple(lst), sign

def _multiply_blades(blade_a: FrozenSet[int], blade_b: FrozenSet[int]) -> Tuple[FrozenSet[int], int]:
    combined = tuple(sorted(blade_a | blade_b))
    # The geometric product is associative; we need to count sign changes
    # from re‑ordering the concatenated list.
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """Geometric product of two multivectors."""
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in a.blades.items():
        for blade_b, coeff_b in b.blades.items():
            new_blade, sign = _multiply_blades(blade_a, blade_b)
            result[new_blade] = result.get(new_blade, 0.0) + sign * coeff_a * coeff_b
    return Multivector(result)

# ----------------------------------------------------------------------
# Parent B – Statistical primitives
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    """Hoeffding bound ε = sqrt( (R² * ln(1/δ)) / (2n) )."""
    if range_ <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("Invalid arguments for Hoeffding bound")
    return math.sqrt((range_ ** 2 * math.log(1 / delta)) / (2 * n))

def gini_impurity(class_counts: List[int]) -> float:
    """Standard Gini impurity given raw class counts."""
    total = sum(class_counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in class_counts]
    return 1.0 - sum(p * p for p in probs)

# ----------------------------------------------------------------------
# Circuit‑breaker (Parent A)
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Simple failure‑count circuit breaker."""
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = max(0, self.failures - 1)
        if self.failures < self.failure_threshold:
            self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.open = True

# ----------------------------------------------------------------------
# Hybrid core functions (the mathematical bridge)
# ----------------------------------------------------------------------
def fisher_weighted_multivector(theta: float, center: float, width: float,
                               base_mv: Multivector) -> Multivector:
    """
    Embed Fisher information as a scalar weight on every blade of ``base_mv``.
    The resulting multivector carries statistical confidence alongside geometric
    structure.
    """
    weight = fisher_score(theta, center, width)
    weighted_blades = {blade: coeff * weight for blade, coeff in base_mv.blades.items()}
    return Multivector(weighted_blades)

def hybrid_priority(morph: Morphology,
                    mv_a: Multivector,
                    mv_b: Multivector,
                    cb: EndpointCircuitBreaker,
                    theta: float,
                    center: float,
                    width: float,
                    class_counts: List[int],
                    range_: float,
                    delta: float,
                    n_observations: int) -> float:
    """
    Compute a unified priority score for a model tier.

    Steps:
    1. Geometric product of the two multivectors → captures relational geometry.
    2. Norm of the product → scalar distance measure.
    3. Fisher‑weighted version of the product → statistical confidence.
    4. Combine with Gini impurity (lower impurity → higher priority).
    5. Apply Hoeffding bound as a confidence radius denominator.
    6. Modulate by circuit‑breaker state (open → penalise).

    Returns a single float; larger values indicate higher priority.
    """
    # 1‑2. Geometric relation
    prod_mv = geometric_product(mv_a, mv_b)
    geom_dist = prod_mv.norm() + 1e-12  # avoid zero division

    # 3. Fisher weighting
    fisher_mv = fisher_weighted_multivector(theta, center, width, prod_mv)
    fisher_norm = fisher_mv.norm() + 1e-12

    # 4. Gini impurity term (invert because lower impurity = better)
    gini = gini_impurity(class_counts)
    impurity_factor = 1.0 - gini  # in [0,1]

    # 5. Hoeffding confidence radius
    hoeff = hoeffding_bound(range_, delta, n_observations) + 1e-12

    # 6. Circuit‑breaker scaling
    cb_factor = 0.5 if cb.open else 1.0

    # Morphology scaling – simple linear combination of physical attributes
    morph_factor = (morph.length + morph.width + morph.height + morph.mass) / 4.0

    # Final hybrid priority
    priority = (geom_dist * fisher_norm * impurity_factor * morph_factor) / (hoeff) * cb_factor
    return priority

def update_circuit_breaker(cb: EndpointCircuitBreaker, success: bool) -> None:
    """Utility wrapper to record an outcome."""
    if success:
        cb.record_success()
    else:
        cb.record_failure()

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple multivectors: e1 and e2 basis vectors
    mv1 = Multivector({frozenset({1}): 1.0})
    mv2 = Multivector({frozenset({2}): 2.0, frozenset(): 0.5})  # scalar + e2

    morph = Morphology(length=1.2, width=0.8, height=0.5, mass=2.3)

    cb = EndpointCircuitBreaker(failure_threshold=2)

    # Simulate a few outcomes to toggle the breaker
    update_circuit_breaker(cb, success=False)
    update_circuit_breaker(cb, success=False)  # should open now
    update_circuit_breaker(cb, success=True)   # partial recovery

    theta, center, width = 0.3, 0.0, 1.0
    class_counts = [30, 70]  # binary classification
    range_ = 2.0
    delta = 0.05
    n = 150

    priority = hybrid_priority(
        morph,
        mv1,
        mv2,
        cb,
        theta,
        center,
        width,
        class_counts,
        range_,
        delta,
        n,
    )
    print(f"Hybrid priority score: {priority:.6f}")
    # Ensure no exception was raised
    sys.exit(0)