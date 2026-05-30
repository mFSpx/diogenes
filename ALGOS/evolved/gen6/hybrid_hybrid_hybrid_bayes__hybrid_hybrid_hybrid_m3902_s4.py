# DARWIN HAMMER — match 3902, survivor 4
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

def _blade_sign(indices: List[int]) -> Tuple[List[int], int]:
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
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(
    blade_a: frozenset[int], blade_b: frozenset[int]
) -> Tuple[frozenset[int], int]:
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


def geometric_product(
    vec_a: Tuple[float, float], vec_b: Tuple[float, float]
) -> Tuple[float, float]:
    blade_a = frozenset(i for i, v in enumerate(vec_a) if abs(v) > 1e-12)
    blade_b = frozenset(i for i, v in enumerate(vec_b) if abs(v) > 1e-12)

    scalar = sum(ai * bi for ai, bi in zip(vec_a, vec_b))

    _, sign = _multiply_blades(blade_a, blade_b)
    bivector = sign * (vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0])
    return scalar, abs(bivector)


def bayes_update(prior: float, likelihood: float) -> float:
    numerator = prior * likelihood
    denominator = numerator + (1 - prior) * (1 - likelihood)
    return numerator / denominator if denominator != 0 else 0.0


def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    return math.exp(-((epsilon * r) ** 2))


def compute_phash(values: List[float]) -> int:
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def weekday_frequencies(dates: Iterable[datetime]) -> Dict[int, float]:
    counts = np.zeros(7, dtype=int)
    for d in dates:
        counts[d.weekday()] += 1
    total = counts.sum()
    return {i: counts[i] / total for i in range(7)} if total > 0 else {i: 0.0 for i in range(7)}


def gini_coefficient(values: List[float]) -> float:
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


EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|"
    r"screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority)\b", re.I
)


def regex_feature_vector(text: str) -> List[int]:
    return [
        int(bool(EVIDENCE_RE.search(text))),
        int(bool(PLANNING_RE.search(text))),
    ]


def epsilon_greedy_select(
    q_values: List[float], epsilon: float = 0.1
) -> int:
    if random.random() < epsilon:
        return random.randrange(len(q_values))
    return int(np.argmax(q_values))


def hybrid_reward(
    point: Tuple[float, float],
    prototype: Tuple[float, float],
    prior: float,
    text: str,
    weekday_weights: Dict[int, float],
) -> float:
    dot, biv = geometric_product(point, prototype)
    likelihood = gaussian_rbf(biv, epsilon=0.5)
    posterior = bayes_update(prior, likelihood)
    today_wd = datetime.now().weekday()
    weekday_factor = weekday_weights.get(today_wd, 0.0)
    regex_features = regex_feature_vector(text)
    regex_score = sum(regex_features)
    return posterior * weekday_factor * regex_score


def improved_hybrid_reward(
    point: Tuple[float, float],
    prototype: Tuple[float, float],
    prior: float,
    text: str,
    weekday_weights: Dict[int, float],
    gini_values: List[float],
) -> float:
    dot, biv = geometric_product(point, prototype)
    likelihood = gaussian_rbf(biv, epsilon=0.5)
    posterior = bayes_update(prior, likelihood)
    today_wd = datetime.now().weekday()
    weekday_factor = weekday_weights.get(today_wd, 0.0)
    regex_features = regex_feature_vector(text)
    regex_score = sum(regex_features)
    gini_factor = 1 - gini_coefficient(gini_values)
    return posterior * weekday_factor * regex_score * gini_factor


def main():
    point = (1.0, 2.0)
    prototype = (3.0, 4.0)
    prior = 0.5
    text = "This is a test text with evidence and planning."
    weekday_weights = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5, 5: 0.6, 6: 0.7}
    gini_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    reward = improved_hybrid_reward(point, prototype, prior, text, weekday_weights, gini_values)
    print(reward)


if __name__ == "__main__":
    main()