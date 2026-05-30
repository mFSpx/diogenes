# DARWIN HAMMER — match 2975, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1660_s1.py (gen6)
# parent_b: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_minimu_m2629_s1.py (gen3)
# born: 2026-05-29T23:46:59Z

"""Hybrid Algorithm integrating:
- Parent A: morphological recovery priority with Caputo fractional diffusion.
- Parent B: Hoeffding bound, Gini coefficient, and epistemic certainty flags.

Mathematical bridge:
The recovery priorities computed from morphological data (Parent A) are diffused across a
graph whose edge weights are derived from cosine similarity of feature vectors.
The diffusion operator is a discrete Caputo fractional power (order α) of the
graph Laplacian (I‑W).  The statistical uncertainty of each diffused priority is
quantified with the Hoeffding bound (Parent B).  The bound magnitude is then
translated into an epistemic certainty flag, while the Gini coefficient of the
priority distribution provides an inequality measure that can modulate confidence.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Iterable

import numpy as np

# ---------- Parent A components ----------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1] derived from righting time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if den == 0 else float(np.dot(a, b) / den)

# ---------- Parent B components ----------
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
            object.__setattr__(self, "generated_at", "2026-05-29T23:25:17Z")

    def as_dict(self) -> dict:
        return asdict(self)

def certainty(label: str,
              *,
              confidence_bps: int,
              authority_class: str,
              rationale: str,
              evidence_refs: Iterable[str] = ()) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    """Hoeffding bound for a bounded random variable in [0,1]."""
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    if n <= 0:
        raise ValueError("sample size n must be positive")
    return math.sqrt(math.log(2.0 / delta) / (2.0 * n))

def gini_coefficient(x: np.ndarray) -> float:
    """Compute Gini coefficient of a 1‑D array."""
    if x.ndim != 1:
        raise ValueError("x must be 1‑dimensional")
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

# ---------- Hybrid core ----------
def feature_vector(m: Morphology) -> np.ndarray:
    """Create a raw feature vector from morphology dimensions."""
    return np.array([m.length, m.width, m.height, m.mass], dtype=float)

def edge_weight_matrix(morphologies: List[Morphology]) -> np.ndarray:
    """W_{ij} = 1 - cosine_similarity(feature_i, feature_j). Diagonal = 0."""
    n = len(morphologies)
    W = np.zeros((n, n), dtype=float)
    feats = [feature_vector(m) for m in morphologies]
    for i in range(n):
        for j in range(i + 1, n):
            w = 1.0 - _cos(feats[i], feats[j])
            W[i, j] = W[j, i] = w
    return W

def binom_frac(alpha: float, k: int) -> float:
    """Generalized binomial coefficient for real alpha."""
    return math.gamma(alpha + 1) / (math.gamma(k + 1) * math.gamma(alpha - k + 1))

def caputo_fractional_diffusion(priorities: np.ndarray,
                                W: np.ndarray,
                                alpha: float = 0.5,
                                terms: int = 10) -> np.ndarray:
    """
    Apply a discrete Caputo fractional operator of order alpha to the priority vector.
    Implements (I - W)^α * priorities via a truncated binomial series.
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0,1]")
    I = np.eye(W.shape[0])
    M = I - W
    result = np.zeros_like(priorities, dtype=float)
    for k in range(terms + 1):
        coeff = binom_frac(alpha, k) * ((-1) ** k)
        Mk = np.linalg.matrix_power(M, k)
        result += coeff * Mk @ priorities
    return result

def evaluate_certainty(diffused: np.ndarray,
                       n_samples: int = 30,
                       delta: float = 0.05) -> List[CertaintyFlag]:
    """
    For each diffused priority compute a Hoeffding bound and translate it
    into an epistemic certainty flag.
    """
    flags: List[CertaintyFlag] = []
    for i, val in enumerate(diffused):
        bound = hoeffding_bound(val, delta, n_samples)
        # Map bound size to label
        if bound < 0.01:
            label = "SURE_MAYBE"
        elif bound < 0.03:
            label = "FACT"
        elif bound < 0.07:
            label = "PROBABLE"
        else:
            label = "POSSIBLE"
        confidence = int(max(0, min(10000, (1.0 - bound) * 10000)))
        flags.append(
            certainty(
                label,
                confidence_bps=confidence,
                authority_class="HybridSystem",
                rationale=f"Hoeffding bound={bound:.4f} after diffusion",
                evidence_refs=(),
            )
        )
    return flags

def hybrid_process(morphologies: List[Morphology],
                   alpha: float = 0.5,
                   diffusion_terms: int = 12,
                   n_samples: int = 30) -> Tuple[np.ndarray, List[CertaintyFlag]]:
    """
    Full pipeline:
    1. Compute recovery priorities from morphologies.
    2. Build edge‑weight graph.
    3. Apply Caputo fractional diffusion.
    4. Assess statistical uncertainty with Hoeffding bound.
    5. Return diffused priorities and epistemic certainty flags.
    """
    # 1. Priorities
    pri = np.array([recovery_priority(m) for m in morphologies], dtype=float)

    # 2. Graph
    W = edge_weight_matrix(morphologies)

    # 3. Diffusion
    diffused = caputo_fractional_diffusion(pri, W, alpha=alpha, terms=diffusion_terms)

    # 4. Certainty evaluation
    flags = evaluate_certainty(diffused, n_samples=n_samples)

    return diffused, flags

# ---------- Smoke test ----------
if __name__ == "__main__":
    # Create a small population of morphologies
    random.seed(42)
    morphs = [
        Morphology(length=random.uniform(0.5, 2.0),
                   width=random.uniform(0.5, 2.0),
                   height=random.uniform(0.3, 1.0),
                   mass=random.uniform(0.2, 5.0))
        for _ in range(5)
    ]

    diff, cert = hybrid_process(morphs, alpha=0.6, diffusion_terms=15, n_samples=50)

    print("Diffused priorities:")
    for i, v in enumerate(diff):
        print(f"  Node {i}: {v:.4f}")

    print("\nCertainty flags:")
    for i, f in enumerate(cert):
        print(f"  Node {i}: {f.label} (confidence {f.confidence_bps/100:.2f}%)")
    # Show Gini of the diffused distribution
    print("\nGini coefficient of diffused priorities:",
          f"{gini_coefficient(diff):.4f}")