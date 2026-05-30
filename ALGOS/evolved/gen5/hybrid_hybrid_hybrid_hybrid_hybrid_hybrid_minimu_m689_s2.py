# DARWIN HAMMER — match 689, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_hybrid_m58_s0.py (gen4)
# parent_b: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s3.py (gen2)
# born: 2026-05-29T23:30:26Z

"""Hybrid algorithm merging Fisher‑score weighted similarity (Parent A) with epistemic certainty propagation (Parent B).

Mathematical bridge:
- Parent A provides a Fisher information scalar `I(θ)= (∂_θ G(θ))² / G(θ)` where `G` is a Gaussian beam.
- Parent B supplies epistemic certainty flags whose numeric confidence (`confidence_bps/10000`) can be interpreted as probabilities.
- We construct a sheaf‑graph `S` (nodes ↔ feature subspaces) and assign each edge a Fisher weight `w_{uv}=I(θ_{uv})`.
- Certainty values `c_u` at nodes are propagated through the graph by the linear operation  
  `c' = D⁻¹ W c`, where `W` is the Fisher‑weighted adjacency matrix and `D` the diagonal degree matrix.
- The resulting propagated confidence vector `c'` is then used to weight similarity metrics
  (e.g., SSIM) computed by Parent A, yielding a unified decision metric that respects both
  information geometry and epistemic confidence.

The module implements this fusion with three core functions:
1. `fisher_weight_matrix` – builds the Fisher‑weighted adjacency matrix for a sheaf.
2. `propagate_certainty` – performs the normalized graph‑propagation of certainty flags.
3. `certainty_weighted_ssim` – computes SSIM between two signals and scales it by the
   propagated certainty.

All components are pure NumPy/Python and runnable as a self‑contained script.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Fisher / Gaussian / SSIM utilities
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity G(θ)."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information I(θ) = (∂_θ G)² / G."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural similarity index (SSIM) for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


# ----------------------------------------------------------------------
# Parent B – Epistemic certainty helpers
# ----------------------------------------------------------------------


EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
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
            object.__setattr__(self, "generated_at",
                               datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def certainty(label: str, *, confidence_bps: int, authority_class: str,
              rationale: str, evidence_refs: Iterable[str] = ()) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


def filesystem_observation(*, sha256: str, path: str,
                           mtime_utc: str | None = None) -> CertaintyFlag:
    refs = [f"sha256:{sha256}", f"path:{path}"]
    if mtime_utc:
        refs.append(f"mtime:{mtime_utc}")
    return certainty(
        "FACT",
        confidence_bps=10000,
        authority_class="filesystem_observation",
        rationale="Local file bytes were hashed and copied into CAS; this proves byte custody, not semantic truth.",
        evidence_refs=refs,
    )


def parser_extraction(*, sha256: str, extract_method: str,
                      injection_detected: bool = False) -> CertaintyFlag:
    if injection_detected:
        return certainty(
            "BULLSHIT",
            confidence_bps=0,
            authority_class="parser_extraction",
            rationale="Injection detected – result is untrustworthy.",
            evidence_refs=(f"sha256:{sha256}", f"method:{extract_method}"),
        )
    return certainty(
        "FACT",
        confidence_bps=9000,
        authority_class="parser_extraction",
        rationale="Parser successfully extracted data without injection.",
        evidence_refs=(f"sha256:{sha256}", f"method:{extract_method}"),
    )


# ----------------------------------------------------------------------
# Sheaf representation (from Parent A) – lightweight for this fusion
# ----------------------------------------------------------------------


class Sheaf:
    """Simple sheaf storing node dimensions and undirected edges."""

    def __init__(self, node_dims: Dict[str, int], edges: List[Tuple[str, str]]):
        """
        Args:
            node_dims: Mapping node_name → dimension (int, unused in current math but kept for compatibility).
            edges: List of (node_u, node_v) tuples defining adjacency.
        """
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        self._nodes = list(self.node_dims.keys())
        self._node_index = {n: i for i, n in enumerate(self._nodes)}

    @property
    def order(self) -> int:
        """Number of nodes."""
        return len(self._nodes)

    def adjacency_matrix(self) -> np.ndarray:
        """Unweighted adjacency matrix A (size N×N)."""
        n = self.order
        A = np.zeros((n, n), dtype=float)
        for u, v in self.edges:
            i, j = self._node_index[u], self._node_index[v]
            A[i, j] = A[j, i] = 1.0
        return A

    def node_names(self) -> List[str]:
        return list(self._nodes)


# ----------------------------------------------------------------------
# Fusion core
# ----------------------------------------------------------------------


def fisher_weight_matrix(sheaf: Sheaf,
                         theta_map: Dict[Tuple[str, str], float],
                         center: float = 0.0,
                         width: float = 1.0) -> np.ndarray:
    """
    Build a Fisher‑information weighted adjacency matrix W.

    For each edge (u, v) we compute w_uv = I(θ_uv) using the Fisher score.
    Missing θ values default to the centre (yielding the maximal Gaussian intensity).

    Returns:
        W – symmetric matrix of shape (N, N) where N = sheaf.order.
    """
    n = sheaf.order
    W = np.zeros((n, n), dtype=float)
    for u, v in sheaf.edges:
        i, j = sheaf._node_index[u], sheaf._node_index[v]
        theta = theta_map.get((u, v), theta_map.get((v, u), center))
        w = fisher_score(theta, center, width)
        W[i, j] = W[j, i] = w
    return W


def propagate_certainty(sheaf: Sheaf,
                        cert_flags: Dict[str, CertaintyFlag],
                        weight_matrix: np.ndarray) -> Dict[str, float]:
    """
    Perform one step of normalized graph propagation of certainty.

    Let c be the vector of raw confidences (confidence_bps / 10000) aligned with sheaf nodes.
    Propagation rule: c' = D⁻¹ W c, where D = diag(W·1) is the degree matrix.
    The result is a vector of propagated confidences in [0, 1].

    Returns:
        Mapping node_name → propagated confidence (float).
    """
    n = sheaf.order
    # Build raw confidence vector
    c = np.zeros(n, dtype=float)
    for name, flag in cert_flags.items():
        if name not in sheaf._node_index:
            raise KeyError(f"Certainty flag for unknown node '{name}'")
        idx = sheaf._node_index[name]
        c[idx] = flag.confidence_bps / 10000.0

    # Weighted adjacency
    W = weight_matrix.copy()
    # Degree (avoid division by zero)
    deg = W.sum(axis=1)
    deg[deg == 0] = 1.0
    D_inv = np.diag(1.0 / deg)

    c_prime = D_inv @ (W @ c)

    # Clip to [0,1] for safety
    c_prime = np.clip(c_prime, 0.0, 1.0)

    return {sheaf._nodes[i]: float(c_prime[i]) for i in range(n)}


def certainty_weighted_ssim(x: np.ndarray,
                            y: np.ndarray,
                            sheaf: Sheaf,
                            propagated_confidence: Dict[str, float],
                            dynamic_range: float = 255.0) -> float:
    """
    Compute SSIM between two signals and weight it by the average propagated confidence
    across all sheaf nodes. This couples the perceptual similarity (Parent A) with the
    epistemic trust (Parent B).

    Returns:
        Weighted SSIM score in [0, 1].
    """
    base_ssim = ssim(x, y, dynamic_range=dynamic_range)
    avg_conf = np.mean(list(propagated_confidence.values()))
    return float(base_ssim * avg_conf)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Construct a minimal sheaf with three nodes and a triangle topology
    node_dims = {"A": 2, "B": 2, "C": 2}
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    sheaf = Sheaf(node_dims, edges)

    # Assign synthetic theta values to edges (radians)
    theta_map = {
        ("A", "B"): 0.2,
        ("B", "C"): -0.1,
        ("C", "A"): 0.0,
    }

    # Build Fisher‑weighted matrix
    W = fisher_weight_matrix(sheaf, theta_map, center=0.0, width=1.0)

    # Create epistemic certainty flags for each node
    cert_flags = {
        "A": filesystem_observation(sha256="a"*64, path="/tmp/a.txt"),
        "B": parser_extraction(sha256="b"*64, extract_method="regex"),
        "C": certainty(
            "PROBABLE",
            confidence_bps=7500,
            authority_class="heuristic",
            rationale="Statistical model suggests plausibility."
        ),
    }

    # Propagate certainty through the Fisher‑weighted graph
    propagated = propagate_certainty(sheaf, cert_flags, W)

    # Dummy signals for SSIM (simple grayscale ramps)
    x = np.linspace(0, 255, 512, dtype=float)
    y = np.linspace(0, 255, 512, dtype=float) + np.random.normal(0, 5, 512)

    # Compute the hybrid metric
    score = certainty_weighted_ssim(x, y, sheaf, propagated)

    print("Fisher‑weighted adjacency matrix W:\n", np.round(W, 4))
    print("\nPropagated confidences per node:")
    for n, val in propagated.items():
        print(f"  {n}: {val:.4f}")
    print(f"\nWeighted SSIM score: {score:.6f}")

    # Ensure the script exits cleanly
    sys.exit(0)