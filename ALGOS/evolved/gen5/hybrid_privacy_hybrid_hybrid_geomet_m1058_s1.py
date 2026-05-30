# DARWIN HAMMER — match 1058, survivor 1
# gen: 5
# parent_a: privacy.py (gen0)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_ternar_m35_s1.py (gen4)
# born: 2026-05-29T23:32:41Z

import math
import numpy as np
import random
from typing import Any, Dict, Iterable, List, Tuple, Set, FrozenSet, Union

# ----------------------------------------------------------------------
# Privacy / Anonymization helpers
# ----------------------------------------------------------------------
def reconstruction_risk_score(
    unique_quasi_identifiers: int,
    total_records: int,
    *,
    epsilon: float = 1.0,
    delta: float = 1e-5,
) -> float:
    """
    Differential‑privacy inspired reconstruction risk.

    The classic risk estimate is ``U / N`` (unique quasi‑identifiers / total
    records).  To make the score robust we clamp it to ``[0, 1]`` and apply a
    Laplace smoothing controlled by ``epsilon`` – a higher epsilon yields a
    score closer to the raw ratio, while a lower epsilon adds more uncertainty.

    Parameters
    ----------
    unique_quasi_identifiers: int
        Number of records that are unique with respect to the quasi‑identifiers.
    total_records: int
        Total number of records in the dataset.
    epsilon: float, optional
        Privacy budget used for Laplace smoothing (default 1.0).
    delta: float, optional
        Small constant to avoid division‑by‑zero (default 1e‑5).

    Returns
    -------
    float
        A value in ``[0, 1]`` representing the reconstruction risk.
    """
    if total_records <= 0:
        return 0.0
    raw = unique_quasi_identifiers / max(total_records, delta)
    # Laplace smoothing: add Laplace(0, 1/epsilon) noise and clamp.
    noise = np.random.laplace(0.0, 1.0 / max(epsilon, delta))
    return float(np.clip(raw + noise, 0.0, 1.0))


def anonymize_for_indexing(
    record: Dict[str, Any],
    redact_keys: Union[Set[str], None] = None,
) -> Dict[str, Any]:
    """
    Redact sensitive fields before the record is used as a key in any data
    structure (e.g., a hash table for Voronoi seeds).

    Parameters
    ----------
    record: dict
        Original record.
    redact_keys: set | None
        Keys that should be replaced by ``'<redacted>'``.  If ``None`` a sensible
        default set is used.

    Returns
    -------
    dict
        A copy of ``record`` with the selected keys redacted.
    """
    default_redact = {
        "email",
        "phone",
        "ssn",
        "secret",
        "token",
        "password",
        "address",
        "name",
    }
    redact = {k.lower() for k in (redact_keys or default_redact)}
    return {
        k: ("<redacted>" if k.lower() in redact else v) for k, v in record.items()
    }


def dp_aggregate(
    values: Iterable[float],
    *,
    epsilon: float = 1.0,
    sensitivity: float = 1.0,
) -> float:
    """
    Differential‑privacy sum with Laplace noise.

    Parameters
    ----------
    values: iterable of float
        The values to aggregate.
    epsilon: float
        Privacy budget.
    sensitivity: float
        Global sensitivity of the sum (default 1.0).

    Returns
    -------
    float
        Noisy sum.
    """
    true_sum = float(sum(values))
    scale = sensitivity / max(epsilon, 1e-9)
    noise = np.random.laplace(0.0, scale)
    return true_sum + noise


# ----------------------------------------------------------------------
# Geometric algebra helpers (very small subset)
# ----------------------------------------------------------------------
def _blade_sign(indices: Iterable[int]) -> Tuple[List[int], int]:
    """
    Return a sorted list of basis indices together with the sign (+1 / -1)
    resulting from the required swaps to achieve the sorted order.
    Repeated indices cancel each other (Grassmann algebra rule).
    """
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign = -sign
            i = max(i - 1, 0)  # re‑check previous pair after swap
        elif lst[i] == lst[i + 1]:
            # duplicate basis vectors annihilate
            lst.pop(i)
            lst.pop(i)
            n -= 2
            sign = sign  # no sign change for cancellation
            i = max(i - 1, 0)
        else:
            i += 1
    return lst, sign


def _multiply_blades(
    blade_a: FrozenSet[int], blade_b: FrozenSet[int]
) -> Tuple[FrozenSet[int], int]:
    """
    Geometric product of two blades (represented as frozensets of basis indices).
    Returns the resulting blade and the sign factor.
    """
    combined = list(blade_a) + list(blade_b)
    sorted_basis, sign = _blade_sign(combined)
    return frozenset(sorted_basis), sign


class Multivector:
    """
    Very lightweight multivector implementation supporting only addition
    and geometric product of basis blades.
    """

    def __init__(self, components: Dict[FrozenSet[int], float] | None = None):
        # ``components`` maps a blade (frozenset of basis indices) to its scalar.
        self.components: Dict[FrozenSet[int], float] = components or {}

    def __add__(self, other: "Multivector") -> "Multivector":
        result = self.components.copy()
        for blade, coeff in other.components.items():
            result[blade] = result.get(blade, 0.0) + coeff
        return Multivector(result)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product."""
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                result[blade_res] = result.get(blade_res, 0.0) + sign * coeff_a * coeff_b
        return Multivector(result)

    def __repr__(self) -> str:
        terms = [
            f"{coeff:g}{''.join(str(i) for i in sorted(blade)) or '1'}"
            for blade, coeff in self.components.items()
            if abs(coeff) > 1e-12
        ]
        return " + ".join(terms) if terms else "0"


def point_to_multivector(point: Iterable[float]) -> Multivector:
    """
    Encode a Euclidean point as a multivector where each coordinate multiplies
    a distinct basis vector.  For a 2‑D point ``(x, y)`` we obtain ``x e1 + y e2``.
    """
    comps = {frozenset({i + 1}): float(coord) for i, coord in enumerate(point)}
    return Multivector(comps)


# ----------------------------------------------------------------------
# Hybrid geometric / privacy integration
# ----------------------------------------------------------------------
def _risk_weighted_distance(
    point: np.ndarray,
    seed: np.ndarray,
    risk: float,
) -> float:
    """
    Core distance metric: Euclidean distance scaled by a monotone function of
    reconstruction risk.  The function ``1 + risk`` preserves ordering while
    giving higher‑risk points a larger effective distance.
    """
    euclid = np.linalg.norm(point - seed)
    return euclid * (1.0 + risk)


def hybrid_voronoi_partition(
    points: List[Iterable[float]],
    seeds: List[Iterable[float]],
    *,
    unique_quasi_identifiers: int,
    total_records: int,
    epsilon: float = 1.0,
) -> Dict[Tuple[float, ...], Tuple[float, ...]]:
    """
    Assign each point to the nearest seed using a distance that is weighted by
    a privacy‑aware reconstruction risk score.

    The risk score is computed **once** for the whole dataset (it depends only
    on the global quasi‑identifier statistics) and then applied uniformly to
    all distance calculations, ensuring deterministic behaviour while still
    reflecting the privacy posture of the data.

    Parameters
    ----------
    points, seeds : list of iterable of floats
        Coordinates in Euclidean space.
    unique_quasi_identifiers, total_records : int
        Global statistics required for the risk score.
    epsilon : float, optional
        Privacy budget used when computing the risk score.

    Returns
    -------
    dict
        Mapping ``point -> nearest_seed`` where both keys and values are
        immutable ``tuple`` objects.
    """
    if not points or not seeds:
        raise ValueError("Both points and seeds must contain at least one element.")

    risk = reconstruction_risk_score(
        unique_quasi_identifiers, total_records, epsilon=epsilon
    )

    seed_arr = [np.asarray(s, dtype=float) for s in seeds]
    assignments: Dict[Tuple[float, ...], Tuple[float, ...]] = {}

    for pt in points:
        pt_arr = np.asarray(pt, dtype=float)
        best_seed = None
        best_dist = math.inf
        for s_arr in seed_arr:
            d = _risk_weighted_distance(pt_arr, s_arr, risk)
            if d < best_dist:
                best_dist = d
                best_seed = s_arr
        assignments[tuple(pt_arr)] = tuple(best_seed)  # type: ignore[arg-type]
    return assignments


def hybrid_ternary_route(
    points: List[Iterable[float]],
    seeds: List[Iterable[float]],
    *,
    unique_quasi_identifiers: int,
    total_records: int,
    epsilon: float = 1.0,
) -> Dict[Tuple[float, ...], Tuple[float, ...]]:
    """
    Build a simple routing graph where each point is connected to the seed
    selected by ``hybrid_voronoi_partition``.  The graph is represented as a
    dictionary ``point -> seed``.  This function showcases how the privacy‑aware
    Voronoi assignment can be reused as a building block for higher‑level
    algorithms (e.g., path planning, clustering).

    Parameters
    ----------
    Same as :func:`hybrid_voronoi_partition`.

    Returns
    -------
    dict
        Mapping ``point -> assigned_seed``.
    """
    return hybrid_voronoi_partition(
        points,
        seeds,
        unique_quasi_identifiers=unique_quasi_identifiers,
        total_records=total_records,
        epsilon=epsilon,
    )


# ----------------------------------------------------------------------
# Demonstration / simple sanity check (executed only when run as script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample geometric data
    points = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    seeds = [[0.0, 0.0], [10.0, 10.0]]

    # Fake privacy statistics
    uq = 2   # two records are unique w.r.t. quasi‑identifiers
    N = 10   # total records

    assignments = hybrid_voronoi_partition(
        points,
        seeds,
        unique_quasi_identifiers=uq,
        total_records=N,
        epsilon=0.8,
    )
    print("Voronoi assignments (point → seed):")
    for p, s in assignments.items():
        print(f"  {p} → {s}")

    route = hybrid_ternary_route(
        points,
        seeds,
        unique_quasi_identifiers=uq,
        total_records=N,
        epsilon=0.8,
    )
    print("\nTernary route (identical to assignments in this simple version):")
    for p, s in route.items():
        print(f"  {p} → {s}")

    # Privacy‑aware aggregation example
    values = [random.random() for _ in range(5)]
    noisy_sum = dp_aggregate(values, epsilon=0.5)
    print(f"\nNoisy DP sum of {values[:3]}… : {noisy_sum:.4f}")

    # Anonymization demo
    record = {
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
        "ssn": "123-45-6789",
    }
    print("\nOriginal record:", record)
    print("Anonymized for indexing:", anonymize_for_indexing(record))

    # Multivector demo
    mv_a = point_to_multivector([1, 2])
    mv_b = point_to_multivector([3, 4])
    print("\nMultivector A:", mv_a)
    print("Multivector B:", mv_b)
    print("Geometric product A * B:", mv_a * mv_b)