# DARWIN HAMMER — match 932, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_geomet_m165_s1.py (gen3)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m192_s3.py (gen4)
# born: 2026-05-29T23:31:47Z

import numpy as np
import math
from datetime import date as dt, datetime, timezone, timedelta
from typing import Dict, List, Tuple, Union, Iterable, Optional


# ----------------------------------------------------------------------
# Multivector utilities (geometric algebra core)
# ----------------------------------------------------------------------
class Multivector:
    """
    Simple multivector for a Euclidean Clifford algebra 𝔾(n).

    * ``components`` maps a frozenset of basis indices to a scalar coefficient.
    * The empty frozenset represents the scalar (grade‑0) part.
    """

    def __init__(self, components: Dict[frozenset[int], float], n: int):
        self.n = int(n)
        # discard near‑zero entries to keep the representation sparse
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }

    # ------------------------------------------------------------------
    # Basic algebraic operations
    # ------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        """Return the grade‑k part of the multivector."""
        return Multivector(
            {
                blade: coef
                for blade, coef in self.components.items()
                if len(blade) == k
            },
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) coefficient."""
        return self.components.get(frozenset(), 0.0)

    # ------------------------------------------------------------------
    # Python magic methods
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            label = (
                "1"
                if not blade
                else "e" + "".join(str(i) for i in sorted(blade))
            )
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __mul__(self, other: Union[float, int]) -> "Multivector":
        """Scalar multiplication."""
        if not isinstance(other, (int, float)):
            raise TypeError("Multivector can only be multiplied by a scalar.")
        return Multivector(
            {blade: coef * other for blade, coef in self.components.items()},
            self.n,
        )

    __rmul__ = __mul__


# ----------------------------------------------------------------------
# Date / weekday utilities (doomsday calendar core)
# ----------------------------------------------------------------------
def _to_numpy_datetime64(
    years: np.ndarray, months: np.ndarray, days: np.ndarray
) -> np.ndarray:
    """
    Convert three integer arrays to a ``datetime64[D]`` array.
    The function works element‑wise and is fully vectorised.
    """
    # Build an ISO‑8601 string array: YYYY‑MM‑DD
    iso_strings = np.char.add(
        np.char.add(years.astype(str), "-"),
        np.char.add(months.astype(str).rjust(2, "0"), "-"),
    )
    iso_strings = np.char.add(iso_strings, days.astype(str).rjust(2, "0"))
    return iso_strings.astype("datetime64[D]")


def weekday_numbers(
    years: np.ndarray, months: np.ndarray, days: np.ndarray
) -> np.ndarray:
    """
    Return weekday numbers for the supplied dates.

    The output follows the convention used in the original code:
        Sun → 0, Mon → 1, …, Sat → 6
    """
    dates = _to_numpy_datetime64(years, months, days)

    # ``astype('datetime64[D]')`` → number of days since the Unix epoch.
    # The epoch (1970‑01‑01) was a Thursday (weekday 4).  Adding 4 aligns
    # the count with the ISO weekday (Mon=0 … Sun=6).  Finally we shift
    # to the required Sun=0 … Sat=6 ordering.
    days_since_epoch = dates.astype("int64")
    iso_weekday = (days_since_epoch + 4) % 7          # Mon=0 … Sun=6
    return (iso_weekday + 1) % 7                     # Sun=0 … Sat=6


def weekday_counts(
    dates: Iterable[Union[dt, Tuple[int, int, int]]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split an iterable of ``datetime.date`` objects or ``(y,m,d)`` tuples
    into three NumPy integer arrays: years, months, days.
    """
    years, months, days = [], [], []
    for d in dates:
        if isinstance(d, dt):
            years.append(d.year)
            months.append(d.month)
            days.append(d.day)
        elif isinstance(d, tuple) and len(d) == 3:
            y, m, day = d
            years.append(int(y))
            months.append(int(m))
            days.append(int(day))
        else:
            raise TypeError(
                "Each element must be a datetime.date or a (year, month, day) tuple."
            )
    return (
        np.asarray(years, dtype=np.int32),
        np.asarray(months, dtype=np.int32),
        np.asarray(days, dtype=np.int32),
    )


# ----------------------------------------------------------------------
# Health / decision hygiene core
# ----------------------------------------------------------------------
def health_score(
    reconstruction_risk_score: float,
    failures: float,
    failure_threshold: float,
    recovery_priority: float,
) -> float:
    """
    Compute a bounded health metric.

    The original formula omitted the explicit computation of ``failure_rate``.
    Here we calculate it safely and clamp the final result to the interval [0, 1].

    Parameters
    ----------
    reconstruction_risk_score : float
        A risk factor in the range [0, 1].
    failures : float
        Observed failure count (non‑negative).
    failure_threshold : float
        Threshold at which failures become critical (must be > 0).
    recovery_priority : float
        Priority of recovery actions, also in [0, 1].

    Returns
    -------
    float
        Health score in [0, 1].
    """
    if failure_threshold <= 0:
        raise ValueError("failure_threshold must be positive.")
    failure_rate = min(max(failures / failure_threshold, 0.0), 1.0)

    # Original expression with added safety checks
    raw = (1.0 - (reconstruction_risk_score * failure_rate)) * (1.0 - recovery_priority)
    return min(max(raw, 0.0), 1.0)


def embed_weekday_distribution(
    weekday_numbers: np.ndarray, n: int = 7
) -> Multivector:
    """
    Encode the weekday histogram into a multivector.

    Each weekday *i* (0 = Sun, …, 6 = Sat) is mapped to a basis vector
    ``e{i+1}``.  The coefficient is the relative frequency of that weekday.
    The scalar part is set to 1.0 (baseline) so that the health factor
    can modulate the whole multivector later.

    Parameters
    ----------
    weekday_numbers : np.ndarray
        Array of weekday indices (0‑6).
    n : int, optional
        Dimension of the underlying Clifford algebra (default 7).

    Returns
    -------
    Multivector
        A multivector whose grade‑1 part reflects the weekday distribution.
    """
    total = weekday_numbers.size
    if total == 0:
        # Degenerate case – return a pure scalar.
        return Multivector({frozenset(): 1.0}, n)

    # Compute histogram (counts for each weekday)
    counts = np.bincount(weekday_numbers, minlength=7)
    frequencies = counts / total

    components: Dict[frozenset[int], float] = {frozenset(): 1.0}
    for i, freq in enumerate(frequencies):
        if freq > 0:
            # Basis index i+1 (to keep 0 reserved for the scalar)
            components[frozenset({i + 1})] = float(freq)
    return Multivector(components, n)


def decision_hygiene_score(multivector: Multivector, health: float) -> float:
    """
    Combine the multivector with the health metric.

    The scalar part is multiplied by ``health`` (as in the original design),
    while the grade‑1 part is also scaled to reflect that health influences
    the weighting of weekday‑specific decisions.

    Returns a single float that can be interpreted as a “score”.
    """
    # Scale the scalar part
    scalar_contrib = multivector.scalar_part() * health

    # Scale the grade‑1 (weekday) part by the same health factor and sum its coefficients
    grade1 = multivector.grade(1)
    vector_contrib = sum(grade1.components.values()) * health

    return scalar_contrib + vector_contrib


# ----------------------------------------------------------------------
# Public API – the hybrid operation
# ----------------------------------------------------------------------
def hybrid_operation(
    dates: Iterable[Union[dt, Tuple[int, int, int]]],
    reconstruction_risk_score: float,
    failures: float,
    failure_threshold: float,
    recovery_priority: float,
    algebra_dim: int = 7,
) -> float:
    """
    Perform the fused computation:

    1. Convert dates → weekday numbers.
    2. Embed the weekday distribution into a multivector.
    3. Compute a bounded health score.
    4. Combine the two into a decision‑hygiene score.

    The function now uses the *actual* weekday distribution rather than a
    placeholder scalar multivector, making the integration between the two
    parent modules mathematically deeper.
    """
    # 1. Split dates and compute weekdays
    years, months, days = weekday_counts(dates)
    weekdays = weekday_numbers(years, months, days)

    # 2. Build a multivector that reflects the weekday histogram
    mv = embed_weekday_distribution(weekdays, n=algebra_dim)

    # 3. Compute health (bounded)
    health = health_score(
        reconstruction_risk_score,
        failures,
        failure_threshold,
        recovery_priority,
    )

    # 4. Final decision hygiene score
    return decision_hygiene_score(mv, health)


# ----------------------------------------------------------------------
# Simple demo when run as a script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_dates = [
        dt(2022, 1, 1),
        dt(2022, 1, 2),
        dt(2022, 1, 3),
        (2022, 1, 4),
        (2022, 1, 5),
    ]

    result = hybrid_operation(
        dates=demo_dates,
        reconstruction_risk_score=0.5,
        failures=2,
        failure_threshold=10,
        recovery_priority=0.1,
    )
    print("Hybrid decision‑hygiene score:", result)