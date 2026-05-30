# DARWIN HAMMER — match 5215, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1729_s0.py (gen5)
# parent_b: hybrid_hybrid_gliner_zero_s_hybrid_hybrid_hybrid_m2123_s2.py (gen5)
# born: 2026-05-30T00:00:42Z

"""Hybrid Algorithm Fusion of:
- hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1729_s0.py (Sheaf + burst admission)
- hybrid_hybrid_gliner_zero_s_hybrid_hybrid_hybrid_m2123_s2.py (Span health scoring & endpoint update)

Mathematical Bridge:
The node sections of a Sheaf are interpreted as *feature vectors* that can be
treated as pseudo‑spans.  The original burst‑admission score (based on a
perceptual hash of the section) is extended by the health score derived from
the GLiNER‑style span confidence (score·(1‑score)).  Both quantities are
combined linearly after a normalization step that uses a simple normal‑
distribution quantile (≈ norm.ppf) implemented with a rational approximation.
Thus the hybrid score fuses the Hamming‑distance‑driven admission model with the
probabilistic health model, and the resulting scalar drives endpoint updates
in the same way the original algorithm updated service‑endpoint metrics."""

import sys
import random
import math
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core structures from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class StrikeState:
    velocity: float
    distance: float
    peak_velocity: float


def compute_phash(values: List[float]) -> int:
    """Return a 64‑bit perceptual hash of the numeric list."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class Sheaf:
    """Simple sheaf over a finite graph.

    Each node stores a vector (section).  Directed edges carry linear restriction
    maps stored as NumPy matrices.
    """

    def __init__(self, node_dims: Dict[str, int], edge_list: List[Tuple[str, str]]):
        self.node_dims = dict(node_dims)
        self.edges = [tuple(e) for e in edge_list]          # undirected
        self._sections: Dict[str, np.ndarray] = {}
        # (src, dst) -> (restriction matrix, its transpose)
        self._restrictions: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

    def add_section(self, node: str, values: List[float]) -> None:
        dim = self.node_dims.get(node)
        if dim is None:
            raise ValueError(f"Node {node!r} not defined in node_dims.")
        arr = np.array(values, dtype=float)
        if arr.shape[0] != dim:
            raise ValueError(f"Section dimension mismatch for node {node}.")
        self._sections[node] = arr

    def add_restriction(self, src: str, dst: str, matrix: np.ndarray) -> None:
        """Add a linear map from src‑space to dst‑space."""
        if matrix.shape[0] != self.node_dims[src] or matrix.shape[1] != self.node_dims[dst]:
            raise ValueError("Restriction matrix dimensions do not match node dimensions.")
        self._restrictions[(src, dst)] = (matrix, matrix.T)

    def get_section(self, node: str) -> np.ndarray:
        return self._sections[node]

    # ------------------------------------------------------------------
    # Parent‑A specific burst admission model
    # ------------------------------------------------------------------
    def burst_admission_score(
        self,
        node: str,
        work_value: float,
        cost: float,
        urgency: float,
        reference_phash: int | None = None,
    ) -> float:
        """Score ∈ [0,1] evaluating how worthy a node section is.

        The score combines:
        - Hamming distance between the node's perceptual hash and a reference.
        - A logistic‑style term that rewards high work and penalises cost/urgency.
        """
        section = self.get_section(node)
        phash = compute_phash(section.tolist())
        if reference_phash is None:
            reference_phash = phash  # self‑reference yields distance 0
        ham = hamming_distance(phash, reference_phash) / 64.0  # normalised
        # logistic blend
        effort = work_value / (1.0 + cost + urgency)
        raw = (1.0 - ham) * math.tanh(effort)
        return max(0.0, min(1.0, raw))


# ----------------------------------------------------------------------
# Core structures from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str


@dataclass
class Endpoint:
    health_score: float
    failure_rate: float
    recovery_priority: float


def hybrid_compute_health_scores(spans: List[Span]) -> List[float]:
    """Health score per span: s·(1‑s), maximal at s=0.5."""
    return [span.score * (1.0 - span.score) for span in spans]


# Simple normal‑quantile approximation (Abramowitz & Stegun 26.2.23)
def _norm_ppf(p: float) -> float:
    """Return an approximation of the inverse CDF of N(0,1).  Valid for 0<p<1."""
    # Coefficients
    a = [ -3.969683028665376e+01,  2.209460984245205e+02,
          -2.759285104469687e+02,  1.383577518672690e+02,
          -3.066479806614716e+01,  2.506628277459239e+00 ]

    b = [ -5.447609879822406e+01,  1.615858368580409e+02,
          -1.556989798598866e+02,  6.680131188771972e+01,
          -1.328068155288572e+01 ]

    c = [ -7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
           4.374664141464968e+00,  2.938163982698783e+00 ]

    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00 ]

    # Define break‑points
    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    # Central region
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def confidence_interval(r: float, delta: float, n: int) -> float:
    """Wilson‑score‑like interval width for proportion r."""
    if n == 0:
        return 0.0
    z = _norm_ppf(1 - delta / 2)
    return z * math.sqrt(r * (1 - r) / n)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_node_score(
    sheaf: Sheaf,
    node: str,
    work: float,
    cost: float,
    urgency: float,
    reference_phash: int | None = None,
) -> float:
    """
    Combine Parent‑A burst admission with Parent‑B health scoring.

    Steps
    -----
    1. Compute the original burst admission (0‑1).
    2. Treat the node's vector as a pseudo‑Span with score = mean(section).
    3. Derive a health term via s·(1‑s).
    4. Normalise both terms with a common z‑score (using the normal‑ppf
       approximation) and blend them linearly.
    """
    # 1. Burst admission
    burst = sheaf.burst_admission_score(node, work, cost, urgency, reference_phash)

    # 2. Pseudo‑span from the node vector
    section = sheaf.get_section(node)
    pseudo_score = float(np.mean(section))
    pseudo_span = Span(0, len(section), "", "node", pseudo_score, "hybrid")
    health = hybrid_compute_health_scores([pseudo_span])[0]

    # 3. Normalise (simple min‑max across the two terms)
    min_val = min(burst, health)
    max_val = max(burst, health) if max(burst, health) > 0 else 1.0
    norm_burst = (burst - min_val) / (max_val - min_val)
    norm_health = (health - min_val) / (max_val - min_val)

    # 4. Blend (weight can be tuned; here equal weight)
    blended = 0.5 * norm_burst + 0.5 * norm_health
    return blended


def hybrid_update_endpoints(
    endpoints: List[Endpoint],
    spans: List[Span],
    delta: float = 0.05,
) -> List[Endpoint]:
    """
    Update each endpoint's health_score using the confidence interval derived
    from the span health scores.  Failure_rate and recovery_priority are
    adjusted proportionally to the interval width.
    """
    health_scores = hybrid_compute_health_scores(spans)
    n = len(spans)
    updated = []
    for ep, h in zip(endpoints, health_scores):
        ci_width = confidence_interval(h, delta, n)
        new_health = max(0.0, min(1.0, h - ci_width))
        new_failure = ep.failure_rate * (1.0 + ci_width)
        new_recovery = ep.recovery_priority * (1.0 - ci_width)
        updated.append(Endpoint(new_health, new_failure, new_recovery))
    return updated


def hybrid_matrix_propagation(sheaf: Sheaf, src: str, dst: str) -> np.ndarray:
    """
    Propagate a section from src to dst using the stored restriction matrix,
    then re‑evaluate the burst admission on the destination node.
    The returned array is the transformed section; side‑effects on the sheaf
    are not performed (pure function).
    """
    if (src, dst) not in sheaf._restrictions:
        raise KeyError(f"No restriction from {src} to {dst}.")
    matrix, _ = sheaf._restrictions[(src, dst)]
    src_section = sheaf.get_section(src)
    transformed = matrix @ src_section
    # optional: store the transformed section (commented to keep purity)
    # sheaf._sections[dst] = transformed
    return transformed


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny sheaf with two nodes
    node_dims = {"A": 3, "B": 3}
    edges = [("A", "B")]
    sheaf = Sheaf(node_dims, edges)

    # Random sections
    random.seed(42)
    sheaf.add_section("A", [random.random() for _ in range(3)])
    sheaf.add_section("B", [random.random() for _ in range(3)])

    # Add a simple restriction (identity)
    identity = np.eye(3)
    sheaf.add_restriction("A", "B", identity)

    # Compute hybrid node score for node A
    score_A = hybrid_node_score(sheaf, "A", work_value=2.0, cost=0.5, urgency=0.3)
    print(f"Hybrid score for node A: {score_A:.4f}")

    # Generate pseudo‑spans from node vectors (as if they were extracted)
    spans = [
        Span(0, 3, "", "node", float(np.mean(sheaf.get_section("A"))), "hybrid"),
        Span(0, 3, "", "node", float(np.mean(sheaf.get_section("B"))), "hybrid"),
    ]

    # Create dummy endpoints
    endpoints = [
        Endpoint(health_score=0.8, failure_rate=0.1, recovery_priority=0.5),
        Endpoint(health_score=0.6, failure_rate=0.2, recovery_priority=0.7),
    ]

    # Update endpoints using hybrid health information
    updated_eps = hybrid_update_endpoints(endpoints, spans)
    for i, ep in enumerate(updated_eps, 1):
        print(f"Endpoint {i}: {asdict(ep)}")

    # Demonstrate matrix propagation
    transformed = hybrid_matrix_propagation(sheaf, "A", "B")
    print(f"Transformed section A→B: {transformed}")