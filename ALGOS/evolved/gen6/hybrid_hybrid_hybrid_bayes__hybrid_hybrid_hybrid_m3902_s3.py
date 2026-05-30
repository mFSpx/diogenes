# DARWIN HAMMER — match 3902, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_bayes_update__hybrid_hybrid_hybrid_m1675_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1085_s0.py (gen5)
# born: 2026-05-29T23:52:30Z

import sys
import math
import random
import pathlib
import re
from datetime import datetime
from typing import List, Tuple, Dict, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Geometric algebra core
# ----------------------------------------------------------------------
def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
    """Sort indices and compute the sign of the permutation.
    Identical indices cancel (Grassmann algebra)."""
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
                # cancel pair
                del lst[j : j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(
    blade_a: frozenset[int], blade_b: frozenset[int]
) -> Tuple[frozenset[int], int]:
    """Geometric product of two blades represented by index sets."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def geometric_product(
    vec_a: Tuple[float, float], vec_b: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Return (scalar, bivector) where
        scalar = a·b (dot product)
        bivector = |a∧b| (magnitude of the wedge product)
    The wedge magnitude is computed via blade multiplication.
    """
    # basis: 0 -> x, 1 -> y
    blade_a = frozenset(i for i, v in enumerate(vec_a) if abs(v) > 1e-12)
    blade_b = frozenset(i for i, v in enumerate(vec_b) if abs(v) > 1e-12)

    # scalar part (dot product)
    scalar = sum(ai * bi for ai, bi in zip(vec_a, vec_b))

    # bivector magnitude via wedge (antisymmetric part)
    # a∧b = a*b - b*a  => magnitude = |det([a,b])|
    # Using blade multiplication to respect sign cancellation
    _, sign = _multiply_blades(blade_a, blade_b)
    bivector = sign * (vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0])
    return scalar, abs(bivector)


# ----------------------------------------------------------------------
# Bayesian update core
# ----------------------------------------------------------------------
def bayes_update(prior: float, likelihood: float) -> float:
    """Standard Bayesian update for binary hypothesis."""
    numerator = prior * likelihood
    denominator = numerator + (1 - prior) * (1 - likelihood)
    return numerator / denominator if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Radial basis function core
# ----------------------------------------------------------------------
def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))


# ----------------------------------------------------------------------
# Doomsday weekday frequencies
# ----------------------------------------------------------------------
def weekday_frequencies(dates: Iterable[datetime]) -> Dict[int, float]:
    """Return normalized frequency of each weekday (0=Mon … 6=Sun)."""
    counts = np.zeros(7, dtype=int)
    for d in dates:
        counts[d.weekday()] += 1
    total = counts.sum()
    return {i: counts[i] / total for i in range(7)} if total > 0 else {i: 0.0 for i in range(7)}


# ----------------------------------------------------------------------
# Gini coefficient
# ----------------------------------------------------------------------
def gini_coefficient(values: List[float]) -> float:
    """Compute Gini coefficient of a non‑negative list."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    if np.any(arr < 0):
        raise ValueError("Gini is defined for non‑negative values")
    sorted_arr = np.sort(arr)
    n = len(arr)
    cumulative = np.cumsum(sorted_arr)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return gini


# ----------------------------------------------------------------------
# Regex feature detection
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|"
    r"screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority)\b", re.I
)


def regex_feature_vector(text: str) -> List[int]:
    """Return a binary feature vector [evidence_present, planning_present]."""
    return [
        int(bool(EVIDENCE_RE.search(text))),
        int(bool(PLANNING_RE.search(text))),
    ]


# ----------------------------------------------------------------------
# Bandit action selection
# ----------------------------------------------------------------------
def epsilon_greedy_select(
    q_values: List[float], epsilon: float = 0.1
) -> int:
    """Return index of selected action using epsilon‑greedy policy."""
    if random.random() < epsilon:
        return random.randrange(len(q_values))
    return int(np.argmax(q_values))


# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_reward(
    point: Tuple[float, float],
    prototype: Tuple[float, float],
    prior: float,
    text: str,
    weekday_weights: Dict[int, float],
) -> float:
    """
    Compute a reward for a single (point, text) pair.

    Steps:
    1. Geometric product with a prototype → (dot, bivector).
    2. Use dot as the numeric distribution for Gini (single value → Gini=0).
       Instead we feed the absolute bivector magnitude to a Gaussian RBF
       to obtain a likelihood.
    3. Bayesian update of the prior using the likelihood.
    4. Multiply by a weekday contextual factor (weekday of today).
    5. Multiply by a regex feature score (sum of binary features).
    """
    # 1. geometric product
    dot, biv = geometric_product(point, prototype)

    # 2. Gaussian likelihood from bivector magnitude (distance‑like)
    likelihood = gaussian_rbf(biv, epsilon=0.5)

    # 3. Bayesian posterior
    posterior = bayes_update(prior, likelihood)

    # 4. Weekday contextual factor (today's weekday)
    today_wd = datetime.now().weekday()
    weekday_factor = weekday_weights.get(today_wd, 0.0)

    # 5. Feature score
    feature_score = sum(regex_feature_vector(text))

    # Combine factors
    reward = posterior * weekday_factor * feature_score
    return reward


# ----------------------------------------------------------------------
# Usage example
# ----------------------------------------------------------------------
def main():
    point = (1.0, 2.0)
    prototype = (2.0, 3.0)
    prior = 0.5
    text = "This is a test with evidence and a plan."
    weekday_weights = {i: 1.0 / 7 for i in range(7)}

    reward = hybrid_reward(point, prototype, prior, text, weekday_weights)
    print(reward)


if __name__ == "__main__":
    main()