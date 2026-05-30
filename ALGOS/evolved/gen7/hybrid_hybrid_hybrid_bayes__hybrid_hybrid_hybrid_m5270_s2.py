# DARWIN HAMMER — match 5270, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_bayes_update__hybrid_hybrid_hybrid_m1675_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1257_s2.py (gen6)
# born: 2026-05-30T00:01:00Z

"""Hybrid Fusion of Bayesian‑Geometric‑RBF Updates with Epistemic Certainty Metadata.

Parents:
- hybrid_hybrid_bayes_update__hybrid_hybrid_hybrid_m1675_s3.py (Bayesian updates,
  geometric algebra, Voronoi, radial basis functions)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1257_s2.py (Epistemic certainty
  flags, morphology, engine endpoint description)

Mathematical bridge:
The Bayesian prior is expressed as a multivector (geometric algebra blade
coefficients).  Each blade’s coefficient is scaled by an *epistemic certainty
weight* derived from a `CertaintyFlag` (confidence expressed in basis points).
Morphological attributes of an `EngineEndpoint` (mass, volume) are mapped to a
resource scalar that modulates the RBF kernel bandwidth σ:
    σ = σ₀ / (1 + α·resource)
Thus the posterior for an observation x is obtained by a weighted Gaussian
kernel evaluated on the geometric distance between the observation’s multivector
representation and each prior blade, where the weight combines Bayesian likelihood,
the epistemic certainty factor, and the morphology‑driven kernel width.

The three public functions demonstrate this fused pipeline:
1. `hybrid_state_update` – Bayesian‑RBF update of a prior multivector using an
   observation and endpoint metadata.
2. `voronoi_partition_epistemic` – Voronoi region assignment of points in
   multivector space, with each region labelled by the most confident epistemic
   flag among candidate endpoints.
3. `marginal_probability_mv` – marginal probability of a set of multivectors
   after integrating epistemic‑weighted RBF contributions.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Geometric algebra utilities (from Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                del lst[j : j + 2]
                n -= 2
                # duplicate indices cancel (e_i ^ e_i = 0)
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(
    blade_a: FrozenSet[int], blade_b: FrozenSet[int]
) -> Tuple[FrozenSet[int], int]:
    """Geometric product of two basis blades.

    Returns (resulting_blade, sign) where sign = ±1 accounts for anticommutation.
    """
    # Concatenate the index lists, then sort while tracking swaps.
    combined = list(blade_a) + list(blade_b)
    sorted_blade, sign = _blade_sign(combined)
    return frozenset(sorted_blade), sign


def multivector_product(a: Dict[FrozenSet[int], float],
                        b: Dict[FrozenSet[int], float]) -> Dict[FrozenSet[int], float]:
    """Geometric product of two multivectors represented as dicts {blade: coeff}."""
    result: Dict[FrozenSet[int], float] = {}
    for blade_a, coeff_a in a.items():
        for blade_b, coeff_b in b.items():
            blade_res, sign = _multiply_blades(blade_a, blade_b)
            result[blade_res] = result.get(blade_res, 0.0) + sign * coeff_a * coeff_b
    return result


# ----------------------------------------------------------------------
# Epistemic certainty and endpoint metadata (from Parent B)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def resource_scalar(self) -> float:
        """A simple scalar combining mass and volume, used to modulate kernel width."""
        return self.mass * self.volume


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # basis points, 0..10000
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def weight(self) -> float:
        """Convert basis points to a probability‑like weight in [0,1]."""
        return self.confidence_bps / 10000.0


@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"
    certainty: CertaintyFlag = field(default_factory=lambda: CertaintyFlag(
        label="PROBABLE",
        confidence_bps=5000,
        authority_class="default",
        rationale="auto-generated",
    ))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def _gaussian_rbf(dist: float, sigma: float) -> float:
    """Radial basis function value for Euclidean distance `dist` and bandwidth `sigma`."""
    if sigma <= 0:
        sigma = 1e-9
    return math.exp(- (dist ** 2) / (2 * sigma ** 2))


def _multivector_norm(mv: Dict[FrozenSet[int], float]) -> float:
    """Euclidean norm of the coefficient vector of a multivector."""
    return math.sqrt(sum(c * c for c in mv.values()))


def hybrid_state_update(
    prior_mv: Dict[FrozenSet[int], float],
    observation_mv: Dict[FrozenSet[int], float],
    endpoint: EngineEndpoint,
    sigma_base: float = 1.0,
    alpha: float = 0.5,
) -> Dict[FrozenSet[int], float]:
    """
    Perform a Bayesian‑RBF update of a prior multivector using an observation.

    The update rule is:
        posterior(blade) ∝ prior(blade) *
            Σ_k weight_k * RBF(‖observation − blade‖; σ_k)

    where
        weight_k = certainty.weight
        σ_k = sigma_base / (1 + alpha * resource_scalar)

    Returns a new multivector (dict) representing the posterior.
    """
    # Compute morphology‑driven bandwidth
    resource = endpoint.morphology.resource_scalar
    sigma = sigma_base / (1.0 + alpha * resource)

    # Epistemic weight
    w = endpoint.certainty.weight

    # Distance between observation and each prior blade (using norm of geometric product)
    posterior: Dict[FrozenSet[int], float] = {}
    for blade, coeff in prior_mv.items():
        # Geometric product between observation and blade gives a multivector;
        # we use its norm as a proxy for distance in geometric algebra space.
        prod = multivector_product({blade: 1.0}, observation_mv)
        dist = _multivector_norm(prod)

        rbf_val = _gaussian_rbf(dist, sigma)
        updated_coeff = coeff * (w * rbf_val + (1 - w))  # blend with prior when certainty low
        posterior[blade] = updated_coeff

    # Normalise posterior to unit L2 norm for numerical stability
    norm = _multivector_norm(posterior)
    if norm > 0:
        for b in posterior:
            posterior[b] /= norm
    return posterior


def voronoi_partition_epistemic(
    points_mv: List[Dict[FrozenSet[int], float]],
    endpoints: List[EngineEndpoint],
) -> List[Tuple[int, str]]:
    """
    Assign each multivector point to the nearest endpoint in geometric‑algebra space.
    The nearest endpoint is the one whose *reference multivector* (the endpoint's
    resource‑scaled identity blade) minimizes the RBF‑weighted distance.
    The returned list contains tuples (point_index, assigned_epistemic_label).
    """
    # Build a simple reference multivector for each endpoint:
    #   ref_mv = {e0: resource_scalar}
    # where e0 is the scalar blade (empty frozenset).
    refs = [
        ({frozenset(): endpoint.morphology.resource_scalar}, endpoint.certainty)
        for endpoint in endpoints
    ]

    assignments: List[Tuple[int, str]] = []
    for idx, pt in enumerate(points_mv):
        best_score = -math.inf
        best_label = "UNKNOWN"
        for ref_mv, certainty in refs:
            # Distance via geometric product norm
            prod = multivector_product(ref_mv, pt)
            dist = _multivector_norm(prod)
            # Convert distance to similarity via RBF (no bandwidth scaling here)
            sim = _gaussian_rbf(dist, sigma=1.0)
            score = sim * certainty.weight
            if score > best_score:
                best_score = score
                best_label = certainty.label
        assignments.append((idx, best_label))
    return assignments


def marginal_probability_mv(
    mv_set: List[Dict[FrozenSet[int], float]],
    endpoints: List[EngineEndpoint],
    sigma_base: float = 1.0,
    alpha: float = 0.5,
) -> float:
    """
    Compute a scalar marginal probability for a collection of multivectors.
    The probability is the product over points of a weighted sum of RBF kernels
    centred at each endpoint's reference multivector, where the weight is the
    epistemic certainty.
    """
    # Pre‑compute endpoint reference multivectors and bandwidths
    refs = []
    for ep in endpoints:
        resource = ep.morphology.resource_scalar
        sigma = sigma_base / (1.0 + alpha * resource)
        ref_mv = {frozenset(): resource}
        refs.append((ref_mv, sigma, ep.certainty.weight))

    log_prob = 0.0
    for mv in mv_set:
        point_contrib = 0.0
        for ref_mv, sigma, w in refs:
            prod = multivector_product(ref_mv, mv)
            dist = _multivector_norm(prod)
            point_contrib += w * _gaussian_rbf(dist, sigma)
        # Avoid log(0)
        point_contrib = max(point_contrib, 1e-12)
        log_prob += math.log(point_contrib)
    return math.exp(log_prob)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple 2‑blade prior: scalar (empty set) and e1 (frozenset({1}))
    prior = {
        frozenset(): 0.6,
        frozenset({1}): 0.4,
    }

    # Observation: a rotated version (for demo we reuse same structure)
    observation = {
        frozenset(): 0.5,
        frozenset({1}): 0.5,
    }

    # Create an endpoint with modest morphology and high certainty
    morph = Morphology(length=2.0, width=1.0, height=0.5, mass=10.0)
    certainty = CertaintyFlag(
        label="FACT",
        confidence_bps=9000,
        authority_class="engineer",
        rationale="test case",
    )
    endpoint = EngineEndpoint(
        engine_id="E1",
        channel="alpha",
        residency="local",
        runtime="python3.11",
        resource_class="standard",
        always_on=True,
        endpoint="http://example.com/api",
        capabilities=["compute", "store"],
        morphology=morph,
        certainty=certainty,
    )

    posterior = hybrid_state_update(prior, observation, endpoint)
    print("Posterior multivector coefficients:")
    for blade, coeff in posterior.items():
        print(f"  Blade {sorted(blade) if blade else 'scalar'}: {coeff:.4f}")

    # Voronoi test with three random points
    random_points = []
    for _ in range(3):
        # random multivector with scalar and e1 components
        random_points.append({
            frozenset(): random.random(),
            frozenset({1}): random.random(),
        })

    assignments = voronoi_partition_epistemic(random_points, [endpoint])
    print("\nVoronoi assignments (point index -> epistemic label):")
    for idx, label in assignments:
        print(f"  Point {idx}: {label}")

    # Marginal probability over the random points
    marg = marginal_probability_mv(random_points, [endpoint])
    print(f"\nMarginal probability of the point set: {marg:.6e}")

    sys.exit(0)