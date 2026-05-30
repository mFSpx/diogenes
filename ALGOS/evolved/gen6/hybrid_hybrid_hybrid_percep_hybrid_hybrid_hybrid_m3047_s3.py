# DARWIN HAMMER — match 3047, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_perceptual_de_hybrid_hybrid_rlct_g_m578_s2.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_rlct_grokking_m749_s0.py (gen5)
# born: 2026-05-29T23:47:30Z

"""Hybrid Perceptual‑RLCT‑Multivector Algorithm
================================================

Parents
-------
* **Algorithm A** – *hybrid_hybrid_perceptual_de_hybrid_hybrid_rlct_g_m578_s2.py*  
  Provides a radial‑basis surrogate model, a Real Log‑Canonical‑Threshold (RLCT)
  estimator and a Normalised Least‑Mean‑Squares (NLMS) adaptive filter.

* **Algorithm B** – *hybrid_hybrid_hybrid_hybrid_hybrid_rlct_grokking_m749_s0.py*  
  Supplies a geometric‑algebra `Multivector` representation together with a
  decision‑hygiene scoring routine that uses the multivector’s Clifford product.

Mathematical Bridge
-------------------
The bridge is built on two observations:

1. The NLMS weight vector can be promoted to a *grade‑1 multivector* where each
   basis blade `e_i` stores the weight for feature `i`.  The geometric product
   of this weight multivector with an input‑vector multivector yields a scalar
   that is exactly the dot‑product used by NLMS, while higher‑grade parts can
   encode richer geometric relationships.

2. The RLCT, which measures model complexity, can be turned into an *adaptive
   step‑size* `μ` for NLMS.  By coupling `μ` with the multivector‑based weight
   update we obtain a unified adaptation law that respects both statistical
   regularisation (RLCT) and geometric structure (Clifford algebra).

The resulting hybrid algorithm therefore:

* encodes NLMS weights as a `Multivector`,
* computes an RLCT‑derived step size,
* updates the weight multivector using the geometric product with the input
  multivector,
* evaluates similarity between perceptual‑hash vectors with a Gaussian RBF,
* feeds the RBF scores into a decision‑hygiene entropy that is computed from
  the multivector’s component magnitudes.

The three public functions below illustrate the complete pipeline.
"""

import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

Vector = Sequence[float]
Blade = Tuple[int, ...]  # e.g. (0,) for e0, (0,1) for e0∧e1


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _multiply_blades(a: Blade, b: Blade) -> Tuple[Blade, int]:
    """
    Geometric product of two basis blades in a Euclidean Clifford algebra
    Cl(n,0).  Returns (result_blade, sign) where sign ∈ {+1, -1}.
    The implementation is limited to the grade‑1 case (vectors) but works
    for any blades by applying the anticommutation rule and the rule
    e_i*e_i = 1.
    """
    # concatenate indices and count swaps needed to sort
    result = list(a) + list(b)
    sign = 1
    # bubble‑sort while counting swaps (each swap flips sign)
    for i in range(len(result)):
        for j in range(len(result) - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
                sign *= -1
    # cancel pairs of equal indices (e_i*e_i = 1)
    i = 0
    while i < len(result) - 1:
        if result[i] == result[i + 1]:
            # remove the pair
            del result[i : i + 2]
            # no sign change because e_i*e_i = +1
            i = max(i - 1, 0)
        else:
            i += 1
    return tuple(result), sign


class Multivector:
    """Simple Euclidean multivector (Cl(n,0))."""

    def __init__(self, components: Dict[Blade, float] = None, n: int = 0):
        self.n = int(n)
        self.components: Dict[Blade, float] = {}
        if components:
            for k, v in components.items():
                if abs(v) > 1e-12:
                    self.components[tuple(k)] = float(v)

    def __add__(self, other: "Multivector") -> "Multivector":
        res = Multivector(self.components.copy(), self.n)
        for b, v in other.components.items():
            res.components[b] = res.components.get(b, 0.0) + v
            if abs(res.components[b]) < 1e-12:
                del res.components[b]
        return res

    __radd__ = __add__

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-1.0) * other

    def __rmul__(self, scalar: float) -> "Multivector":
        """Scalar multiplication."""
        if not isinstance(scalar, (int, float)):
            raise TypeError("Only scalar multiplication is supported")
        return Multivector({b: scalar * v for b, v in self.components.items()}, self.n)

    __mul__ = __rmul__  # scalar * multivector

    def geometric_product(self, other: "Multivector") -> "Multivector":
        """Full geometric (Clifford) product."""
        result = Multivector({}, self.n)
        for ba, va in self.components.items():
            for bb, vb in other.components.items():
                blade, sign = _multiply_blades(ba, bb)
                result.components[blade] = result.components.get(blade, 0.0) + sign * va * vb
        # prune near‑zero entries
        result.components = {b: v for b, v in result.components.items() if abs(v) > 1e-12}
        return result

    def scalar_part(self) -> float:
        """Return the grade‑0 (scalar) component, or 0 if absent."""
        return self.components.get((), 0.0)

    def grade(self, k: int) -> "Multivector":
        """Extract only blades of grade *k*."""
        return Multivector({b: v for b, v in self.components.items() if len(b) == k}, self.n)

    def magnitude_squared(self) -> float:
        """Sum of squares of all components (Euclidean norm in coefficient space)."""
        return sum(v * v for v in self.components.values())

    def probability_distribution(self) -> List[float]:
        """Normalize squared magnitudes to a probability vector (used for entropy)."""
        mags = np.array([v * v for v in self.components.values()], dtype=float)
        total = mags.sum()
        if total == 0.0:
            return [0.0] * len(mags)
        return (mags / total).tolist()

    def __repr__(self) -> str:
        return f"Multivector({self.components}, n={self.n})"


def vector_to_multivector(x: Vector) -> Multivector:
    """Encode a real vector as a grade‑1 multivector (each component → e_i)."""
    comps = { (i,): float(val) for i, val in enumerate(x) }
    return Multivector(comps, n=len(x))


def multivector_dot(w: Multivector, x: Multivector) -> float:
    """
    Dot product between two grade‑1 multivectors via the geometric product.
    The scalar part of w ⊙ x equals the usual Euclidean dot product.
    """
    prod = w.geometric_product(x)
    return prod.scalar_part()


def compute_rlct(X: np.ndarray) -> float:
    """
    Approximate the Real Log‑Canonical Threshold for a data matrix X
    (samples × features).  The RLCT of a linear model with Gaussian prior is
    proportional to the sum of log eigenvalues of the empirical covariance.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2‑D array")
    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov + 1e-12 * np.eye(cov.shape[0]))
    # Avoid log(0) by clipping
    eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)
    rlct = np.sum(np.log(eigvals))
    return float(rlct)


def hybrid_nlms_update(
    weight_mv: Multivector,
    x_vec: Vector,
    target: float,
    rlct: float,
    mu_base: float = 0.5,
    eps: float = 1e-9,
) -> Multivector:
    """
    NLMS‑style adaptive update where the step size μ is modulated by the RLCT.
    The weight vector is stored as a grade‑1 multivector.

    Parameters
    ----------
    weight_mv : Multivector
        Current weight (grade‑1) multivector.
    x_vec : Vector
        Input feature vector.
    target : float
        Desired scalar output.
    rlct : float
        Real Log‑Canonical Threshold (complexity measure).
    mu_base : float, optional
        Base NLMS learning rate (default 0.5).  Must satisfy 0 < μ_base < 2.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    Multivector
        Updated weight multivector.
    """
    if not (0.0 < mu_base < 2.0):
        raise ValueError("mu_base must be in (0, 2)")

    # Convert input to a grade‑1 multivector
    x_mv = vector_to_multivector(x_vec)

    # Predicted output using geometric product (scalar part)
    y = multivector_dot(weight_mv, x_mv)

    error = target - y

    # RLCT‑scaled learning rate: larger RLCT → smaller step (regularisation)
    mu = mu_base / (1.0 + abs(rlct))

    norm_sq = np.dot(x_vec, x_vec) + eps
    step = (mu * error) / norm_sq

    # Weight update: w_{new} = w + step * x  (geometric product reduces to scalar‑scaled vector)
    updated = weight_mv + step * x_mv
    return updated


def hybrid_decision_hygiene_score(
    sample: Vector,
    weight_mv: Multivector,
    dataset: List[Vector],
    epsilon: float = 1.0,
) -> float:
    """
    Compute a decision‑hygiene score for *sample*.

    The score is a weighted average of RBF similarities between *sample* and each
    element of *dataset*.  Weights are derived from the Shannon entropy of the
    multivector’s component magnitude distribution, encouraging epistemic
    certainty.

    Returns
    -------
    float
        Higher values indicate a more “hygienic” (low‑entropy, high‑similarity)
        decision.
    """
    # RBF similarities
    sims = np.array([gaussian(euclidean(sample, d), epsilon) for d in dataset], dtype=float)
    if sims.sum() == 0.0:
        sims = np.ones_like(sims) / len(sims)

    # Entropy term from multivector component distribution
    probs = np.array(weight_mv.probability_distribution(), dtype=float)
    if len(probs) == 0:
        entropy = 0.0
    else:
        # avoid log(0)
        probs = np.clip(probs, 1e-12, 1.0)
        entropy = -np.sum(probs * np.log(probs))

    # Hygiene score: similarity weighted by (1‑entropy) (entropy normalised)
    max_entropy = math.log(len(probs)) if len(probs) > 1 else 0.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    hygiene_factor = 1.0 - norm_entropy  # higher when entropy low

    score = hygiene_factor * np.mean(sims)
    return float(score)


def build_hybrid_epistemic_tree(
    data: List[Vector],
    weight_mv: Multivector,
    max_depth: int = 3,
    min_samples_split: int = 2,
) -> Dict[str, Any]:
    """
    Very lightweight recursive tree builder that uses the hybrid decision‑hygiene
    score as the splitting criterion.  The tree structure is a nested dict:

        {
            "type": "leaf" or "node",
            "depth": int,
            "score": float,                 # hygiene score of the node
            "indices": List[int],           # data indices reaching the node
            "left":  <sub‑tree> or None,
            "right": <sub‑tree> or None,
            "axis": int,                    # split dimension (if node)
            "threshold": float,             # split threshold (if node)
        }

    This implementation is intentionally simple and serves only as a proof of
    concept for the hybrid pipeline.
    """
    def recurse(indices: List[int], depth: int) -> Dict[str, Any]:
        subset = [data[i] for i in indices]
        score = hybrid_decision_hygiene_score(
            sample=random.choice(subset),
            weight_mv=weight_mv,
            dataset=subset,
        )
        node: Dict[str, Any] = {
            "type": "leaf",
            "depth": depth,
            "score": score,
            "indices": indices,
            "left": None,
            "right": None,
            "axis": None,
            "threshold": None,
        }

        if depth >= max_depth or len(indices) < min_samples_split:
            return node

        # Choose split axis with largest variance
        arr = np.array(subset)