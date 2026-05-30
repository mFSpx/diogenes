# DARWIN HAMMER — match 871, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m84_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_hybrid_bandit_m232_s1.py (gen3)
# born: 2026-05-29T23:31:23Z

"""
Hybrid Fusion Module
====================

This module fuses the core topologies of the two parent algorithms:

* **Parent A** – geometric‑algebra based *Multivector* operations whose geometric
  product is modulated by a pheromone signal.
* **Parent B** – a liquid‑time‑constant (LTC) network that adapts work‑share
  allocation together with a Count‑Min sketch that approximates empirical
  log‑likelihoods for a bandit router.

**Mathematical Bridge**

The bridge is built on the observation that both parents treat a *scalar
signal* as a modulating factor:

* In Parent A the pheromone signal scales the coefficients of the geometric
  product.
* In Parent B the LTC network outputs a scalar that scales the allocation of
  resources, while the Count‑Min sketch supplies a scalar log‑likelihood estimate.

The fused algorithm therefore:

1. Uses the LTC network to produce an adaptive *pheromone* value from the
   current day‑of‑week and external features.
2. Applies that pheromone to the geometric product of multivectors.
3. Feeds the resulting scalar into a Count‑Min sketch to obtain a fast
   log‑likelihood estimate that drives bandit‑style action selection.

The three public functions below illustrate this integrated workflow.
"""

import math
import random
import sys
from datetime import date
from pathlib import Path
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Set, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared utilities
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


# ----------------------------------------------------------------------
# Geometric Algebra core (Parent A)
# ----------------------------------------------------------------------
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list.
    Identical indices cancel (Grassmann algebra property)."""
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
                # cancel duplicate index
                lst.pop(j)
                lst.pop(j)  # the next element shifts left
                n -= 2
                i = -1  # restart outer loop because list changed
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: Dict[frozenset, float], n: int):
        # store only non‑zero coefficients
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    # ------------------------------------------------------------------
    # Basic grade / scalar helpers
    # ------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        """Return a new Multivector keeping only grade‑k blades."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items()
             if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) coefficient."""
        return self.components.get(frozenset(), 0.0)

    # ------------------------------------------------------------------
    # Algebraic operations
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
            if abs(result[blade]) < 1e-12:
                del result[blade]
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) - coef
            if abs(result[blade]) < 1e-12:
                del result[blade]
        return Multivector(result, self.n)

    def scale(self, scalar: float) -> "Multivector":
        """Uniformly scale all coefficients."""
        if scalar == 0:
            return Multivector({}, self.n)
        return Multivector({b: c * scalar for b, c in self.components.items()}, self.n)

    def geometric_product(self, other: "Multivector", pheromone: float = 1.0) -> "Multivector":
        """Geometric product with pheromone‑modulated coefficients."""
        result: Dict[frozenset, float] = defaultdict(float)
        for blade_a, coef_a in self.components.items():
            for blade_b, coef_b in other.components.items():
                blade_res, sign = _multiply_blades(blade_a, blade_b)
                result[blade_res] += sign * coef_a * coef_b * pheromone
        # prune near‑zero entries
        result = {b: c for b, c in result.items() if abs(c) > 1e-12}
        return Multivector(result, self.n)

    def __repr__(self):
        terms = [f"{c:.3g}*e{sorted(list(b)) if b else '0'}" for b, c in self.components.items()]
        return " + ".join(terms) if terms else "0"


# ----------------------------------------------------------------------
# Liquid‑Time‑Constant Network (Parent B)
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def ltc_f(x: np.ndarray, I: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Liquid‑time‑constant (LTC) activation.
    Concatenates external input ``x`` with internal state ``I``,
    applies a linear map and a sigmoid non‑linearity.
    """
    concat = np.concatenate([x, I], axis=0)
    return sigmoid(W @ concat + b)


# ----------------------------------------------------------------------
# Count‑Min Sketch (Parent B)
# ----------------------------------------------------------------------
class CountMinSketch:
    """
    Simple Count‑Min sketch with pairwise‑independent hash functions.
    Stores integer counts; ``estimate`` returns the minimum over hash rows.
    """

    def __init__(self, depth: int = 5, width: int = 2 ** 10, seed: int = 0):
        self.depth = depth
        self.width = width
        rng = np.random.default_rng(seed)
        # random a, b for each hash: (a * x + b) % prime % width
        self.prime = 2 ** 31 - 1
        self.a = rng.integers(1, self.prime, size=depth, dtype=np.int64)
        self.b = rng.integers(0, self.prime, size=depth, dtype=np.int64)
        self.table = np.zeros((depth, width), dtype=np.int64)

    def _hash(self, item: int) -> List[int]:
        return [((self.a[i] * item + self.b[i]) % self.prime) % self.width for i in range(self.depth)]

    def add(self, item: int, count: int = 1):
        for row, col in enumerate(self._hash(item)):
            self.table[row, col] += count

    def estimate(self, item: int) -> int:
        return min(self.table[row, col] for row, col in enumerate(self._hash(item)))


# ----------------------------------------------------------------------
# Hybrid Functions (the required three)
# ----------------------------------------------------------------------
def allocate_adaptive_workshare(
    *,
    total_units: float,
    deterministic_target_pct: float = 90.0,
    pheromone_signal: float = None,
) -> Dict[str, float]:
    """
    Allocate work units among the four model groups.

    1. A deterministic baseline based on the day of the week.
    2. An LTC network that consumes the baseline percentage and an optional
       external pheromone signal, producing a scaling factor ``alpha``.
    3. The final allocation is ``alpha`` times the baseline, renormalised to
       sum to ``total_units``.
    """
    # ---- deterministic baseline -------------------------------------------------
    weekday = date.today().weekday()  # 0 = Monday
    baseline_pct = deterministic_target_pct / 100.0
    baseline = {g: 0.0 for g in GROUPS}
    fav = GROUPS[weekday % len(GROUPS)]
    baseline[fav] = baseline_pct
    # distribute the remainder uniformly
    remainder = (1.0 - baseline_pct) / (len(GROUPS) - 1)
    for g in GROUPS:
        if g != fav:
            baseline[g] = remainder

    # ---- LTC‑driven pheromone (if not supplied) ---------------------------------
    if pheromone_signal is None:
        # Simple feature vector: [baseline for fav group, weekday normalized]
        x = np.array([baseline[fav], weekday / 6.0], dtype=np.float64)
        I = np.zeros(2, dtype=np.float64)  # internal state placeholder
        # Small random weights (fixed seed for reproducibility)
        rng = np.random.default_rng(42)
        W = rng.normal(scale=0.5, size=(2, x.size + I.size))
        b = rng.normal(scale=0.1, size=2)
        pheromone_signal = float(ltc_f(x, I, W, b).mean())
        pheromone_signal = max(0.0, min(1.0, pheromone_signal))  # clamp to [0,1]

    # ---- scale baseline by pheromone -------------------------------------------
    scaled = {g: v * pheromone_signal for g, v in baseline.items()}
    # renormalise to total_units
    total_frac = sum(scaled.values())
    if total_frac == 0:
        # fallback to equal split
        equal = total_units / len(GROUPS)
        return {g: equal for g in GROUPS}
    factor = total_units / total_frac
    allocation = {g: v * factor for g, v in scaled.items()}
    # round percentages for readability
    allocation = {g: _pct(v) for g, v in allocation.items()}
    return allocation


def hybrid_select_action(
    mv: Multivector,
    pheromone_signal: float,
    sketch: CountMinSketch,
) -> Tuple[str, float]:
    """
    Select an action (one of the GROUPS) using:

    * The pheromone‑modulated geometric product of ``mv`` with itself.
    * The scalar part of the product as a key for the Count‑Min sketch.
    * The sketch provides an estimated count → log‑likelihood.
    * The group with the highest estimated log‑likelihood is returned.
    """
    # geometric self‑product, pheromone scales the coefficients
    product = mv.geometric_product(mv, pheromone=pheromone_signal)
    scalar = product.scalar_part()
    # map the (possibly negative) scalar to a non‑negative integer key
    key = int(abs(scalar) * 1e6)  # preserve some precision
    # query sketch for each group (different seeds per group)
    scores = {}
    for idx, group in enumerate(GROUPS):
        # perturb key with group index to obtain distinct estimates
        estimate = sketch.estimate(key + idx)
        # log‑likelihood (add 1 to avoid log(0))
        scores[group] = math.log(estimate + 1)
    # pick max
    chosen = max(scores, key=scores.get)
    return chosen, scores[chosen]


def hybrid_rlct_estimate(sketch: CountMinSketch, steps: int = 1000) -> float:
    """
    Compute a rough RLCT (real log‑canonical‑threshold) estimate from the sketch.

    The empirical log‑likelihood sum is approximated by summing the log of
    sketch estimates over a uniform sample of integer keys.
    The RLCT estimate is the average log‑likelihood per step.
    """
    total_log = 0.0
    rng = np.random.default_rng(123)
    for _ in range(steps):
        key = rng.integers(0, 10 ** 6)
        est = sketch.estimate(int(key))
        total_log += math.log(est + 1)
    return total_log / steps


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Allocate workshare
    alloc = allocate_adaptive_workshare(total_units=1000.0)
    print("Adaptive allocation:", alloc)

    # 2. Build a simple multivector (Cl(3,0))
    #   e1 + 2 e2 + 3 e1∧e2
    mv = Multivector(
        {
            frozenset(): 0.0,                # scalar (ignored)
            frozenset({1}): 1.0,
            frozenset({2}): 2.0,
            frozenset({1, 2}): 3.0,
        },
        n=3,
    )
    print("Original multivector:", mv)

    # 3. Create a Count‑Min sketch and seed it with some fake observations
    cms = CountMinSketch(depth=4, width=2048, seed=7)
    # simulate counts for keys 0‑9
    for k in range(10):
        cms.add(k, count=random.randint(1, 50))

    # 4. Hybrid action selection
    pheromone = 0.73  # could be obtained from allocate_adaptive_workshare
    action, score = hybrid_select_action(mv, pheromone, cms)
    print(f"Hybrid selected action: {action} (score={_pct(score)})")

    # 5. RLCT estimate
    rlct = hybrid_rlct_estimate(cms, steps=500)
    print("RLCT estimate:", _pct(rlct))