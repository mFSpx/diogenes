# DARWIN HAMMER — match 1334, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_percyp_hybrid_caputo_fracti_m610_s1.py (gen4)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_hybrid_hybrid_m105_s0.py (gen4)
# born: 2026-05-29T23:35:37Z

import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility: Timestamp
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Z format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Circuit breaker (unchanged, but with typing)
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Simple circuit‑breaker that blocks further processing after N failures."""

    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = now_z()

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = now_z()

    def allow(self) -> bool:
        return not self.open


# ----------------------------------------------------------------------
# Gamma function – Lanczos approximation with memoisation
# ----------------------------------------------------------------------
_gamma_cache: Dict[float, float] = {}


def _gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function with simple caching."""
    if z in _gamma_cache:
        return _gamma_cache[z]

    # Coefficients for g=7, n=9 (Abramowitz & Stegun)
    g = 7
    p = np.array(
        [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ]
    )
    if z < 0.5:
        # Reflection formula
        result = math.pi / (math.sin(math.pi * z) * _gamma_lanczos(1 - z))
    else:
        z_minus_one = z - 1
        a = p[0]
        for i in range(1, len(p)):
            a += p[i] / (z_minus_one + i)
        t = z_minus_one + g + 0.5
        result = (
            math.sqrt(2 * math.pi)
            * t ** (z_minus_one + 0.5)
            * math.exp(-t)
            * a
        )
    _gamma_cache[z] = result
    return result


# ----------------------------------------------------------------------
# Fractional derivative – Grünwald‑Letnikov (GL) scheme
# ----------------------------------------------------------------------
def gl_caputo_derivative(alpha: float, series: List[float]) -> float:
    """
    Approximate the Caputo derivative of order ``alpha`` for a discrete series
    using the Grünwald‑Letnikov definition.

    D^α f(t_n) ≈ (1 / h^α) * Σ_{k=0}^{n} (-1)^k * C(α, k) * f(t_{n‑k})

    where C(α, k) = Γ(k‑α) / (Γ(-α) Γ(k+1))

    The step size ``h`` is assumed to be 1 (unit sampling).  The function returns
    the derivative at the last time index (n = len(series)-1).
    """
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be in (0,1) for the Caputo derivative.")
    n = len(series) - 1
    if n < 0:
        return 0.0

    # Pre‑compute binomial‑type coefficients using recursion for stability
    coeff = 1.0
    derivative = coeff * series[n]  # k = 0 term
    for k in range(1, n + 1):
        coeff *= (alpha - k + 1) / k  # C(α, k) = C(α, k‑1) * (α‑k+1)/k
        term = ((-1) ** k) * coeff * series[n - k]
        derivative += term

    return derivative  # h = 1 ⇒ no scaling needed


# ----------------------------------------------------------------------
# NLMS core (unchanged)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Dot‑product prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    One NLMS adaptation step.
    Returns the updated weight vector and the instantaneous error.
    """
    y = nlms_predict(weights, x)
    e = target - y
    norm_x_sq = max(np.dot(x, x), eps)
    step = (mu / (norm_x_sq + eps)) * e * x
    new_weights = weights + step
    return new_weights, e


# ----------------------------------------------------------------------
# Bayesian marginal with principled handling of epistemic certainty
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, certainty: float, fp_scale: float = 0.1) -> float:
    """
    Compute the posterior error probability given a prior and an epistemic
    certainty factor ``certainty`` ∈ [0,1].

    The likelihood of *no* error is taken as the certainty itself; the likelihood
    of an error is ``1‑certainty``.  A small false‑positive term ``fp`` prevents
    division by zero and can be tuned via ``fp_scale``.
    """
    likelihood_error = 1.0 - certainty
    fp = certainty * fp_scale
    numerator = prior * likelihood_error
    denominator = numerator + (1.0 - prior) * (1.0 - likelihood_error) + fp
    return 0.0 if denominator == 0.0 else numerator / denominator


# ----------------------------------------------------------------------
# Prior from NLMS scores – softmax across the two nodes
# ----------------------------------------------------------------------
def softmax_prior(s_i: float, s_j: float, eps: float = 1e-12) -> float:
    """
    Produce a normalized prior probability for an edge from two NLMS scores.
    The softmax ensures the prior lies in (0,1) even when scores are negative
    or have large magnitude differences.
    """
    max_s = max(s_i, s_j, 0.0)  # shift to avoid overflow
    exp_i = math.exp(s_i - max_s)
    exp_j = math.exp(s_j - max_s)
    total = exp_i + exp_j + eps
    return exp_i / total  # probability that node i is “preferred’’


# ----------------------------------------------------------------------
# Deeply integrated hybrid edge weight
# ----------------------------------------------------------------------
def hybrid_edge_weight(
    i: int,
    j: int,
    nodes: Dict[int, Tuple[float, float]],
    nlms_weights: np.ndarray,
    x_i: np.ndarray,
    x_j: np.ndarray,
    certainty: float,
    alpha: float,
    error_series: List[float],
    eps: float = 1e-12,
    fp_scale: float = 0.1,
    frac_factor_gain: float = 1.0,
) -> float:
    """
    Compute a mathematically tighter hybrid weight for edge (i, j).

    The integration follows three tightly coupled steps:

    1. **NLMS‑driven prior** – a softmax of the two prediction scores yields a
       proper probability in (0,1).

    2. **Epistemic‑certainty Bayesian update** – the prior is combined with the
       epistemic certainty to obtain a posterior error probability.

    3. **Fractional‑error dynamics** – a Grünwald‑Letnikov Caputo derivative of the
       NLMS error series modulates the weight, rewarding edges whose error
       evolution is smooth (small derivative) and penalising erratic edges.

    The Euclidean distance provides a geometric baseline; all three
    contributions are combined multiplicatively to preserve the MST ordering
    while respecting each mathematical domain.
    """
    # --- 1. NLMS decision scores -------------------------------------------------
    s_i = nlms_predict(nlms_weights, x_i)
    s_j = nlms_predict(nlms_weights, x_j)

    # --- 2. Prior via softmax ----------------------------------------------------
    prior = softmax_prior(s_i, s_j, eps)

    # --- 3. Bayesian posterior error probability ---------------------------------
    marginal = bayes_marginal(prior, certainty, fp_scale)

    # --- 4. Euclidean distance ---------------------------------------------------
    xi, yi = nodes[i]
    xj, yj = nodes[j]
    d = math.hypot(xi - xj, yi - yj) + eps  # avoid zero distance

    # --- 5. Fractional dynamics factor -------------------------------------------
    # Use the GL derivative on the *error* series; the series is expected to be
    # ordered from oldest to newest, with the most recent error at the end.
    frac_der = gl_caputo_derivative(alpha, error_series)
    # Scale the absolute derivative; the gain allows the user to control its
    # influence relative to the geometric term.
    fractional_factor = 1.0 + frac_factor_gain * abs(frac_der)

    # --- 6. Final hybrid weight --------------------------------------------------
    # (1‑marginal) is the probability of *no* error; we want larger weights for
    # more reliable edges, hence the multiplication.
    weight = d * (1.0 - marginal) * fractional_factor + eps
    return weight


# ----------------------------------------------------------------------
# Kruskal MST with integrated circuit‑breaker
# ----------------------------------------------------------------------
class DisjointSet:
    """Union‑Find structure with path compression."""

    def __init__(self, elements: List[int]):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return False
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        elif self.rank[xroot] > self.rank[yroot]:
            self.parent[yroot] = xroot
        else:
            self.parent[yroot] = xroot
            self.rank[xroot] += 1
        return True


def kruskal_mst(
    nodes: Dict[int, Tuple[float, float]],
    edges: List[Tuple[int, int]],
    edge_weights: Dict[Tuple[int, int], float],
    breaker: EndpointCircuitBreaker,
) -> List[Tuple[int, int]]:
    """
    Build a Minimum Spanning Tree using Kruskal's algorithm.
    The ``breaker`` can abort insertion after repeated failures (e.g. due to
    missing weight entries).  The function returns the list of edges that form
    the MST.
    """
    # Sort edges by weight (ascending)
    sorted_edges = sorted(edges, key=lambda e: edge_weights.get(e, math.inf))

    ds = DisjointSet(list(nodes.keys()))
    mst: List[Tuple[int, int]] = []

    for e in sorted_edges:
        if not breaker.allow():
            break
        try:
            w = edge_weights[e]
        except KeyError:
            breaker.record_failure()
            continue

        if ds.union(e[0], e[1]):
            mst.append(e)
            breaker.record_success()
        else:
            # Edge would create a cycle – not a failure for the breaker
            pass

    return mst


# ----------------------------------------------------------------------
# Example driver (illustrative, not executed on import)
# ----------------------------------------------------------------------
def _example_usage():
    # Dummy data for illustration
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0)}
    edges = [(0, 1), (1, 2), (0, 2)]

    # Random NLMS weights and feature vectors
    rng = np.random.default_rng(42)
    nlms_weights = rng.normal(size=5)
    x_i = rng.normal(size=5)
    x_j = rng.normal(size=5)

    # Simulated error series (e.g., last 20 NLMS errors)
    error_series = list(rng.normal(scale=0.1, size=20))

    # Compute edge weights
    edge_weights = {}
    for i, j in edges:
        w = hybrid_edge_weight(
            i,
            j,
            nodes,
            nlms_weights,
            x_i,
            x_j,
            certainty=0.85,
            alpha=0.6,
            error_series=error_series,
        )
        edge_weights[(i, j)] = w

    breaker = EndpointCircuitBreaker(failure_threshold=2)
    mst = kruskal_mst(nodes, edges, edge_weights, breaker)
    print("MST edges:", mst)


if __name__ == "__main__":
    _example_usage()