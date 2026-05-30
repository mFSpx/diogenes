# DARWIN HAMMER — match 5798, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_fracti_m2251_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_hybrid_m1197_s3.py (gen5)
# born: 2026-05-30T00:04:55Z

import numpy as np
import math
from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Dict, List

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
            object.__setattr__(self, "generated_at", "2024-01-01T00:00:00Z")

    def as_dict(self) -> Dict[str, Any]:
        return dataclass.asdict(self)


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 1‑D arrays."""
    return np.linalg.norm(a - b)


def rbf_similarity_matrix(
    features: List[np.ndarray],
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Vectorised computation of the Gaussian RBF similarity matrix.

    Returns
    -------
    S : ndarray, shape (n, n)
        S[i, j] = exp(-epsilon^2 * ||x_i - x_j||^2)
    """
    X = np.asarray(features, dtype=float)          # (n, d)
    # Using (a-b)^2 = a^2 + b^2 - 2ab
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)          # (n, 1)
    dists_sq = sq_norms + sq_norms.T - 2.0 * X @ X.T           # (n, n)
    np.clip(dists_sq, 0, None, out=dists_sq)                  # numerical safety
    return np.exp(-epsilon ** 2 * dists_sq)


def tropical_relu(matrix: np.ndarray) -> np.ndarray:
    """
    Tropical “max‑plus” projection using the ReLU (max(x,0)) element‑wise.
    """
    return np.maximum(matrix, 0.0)


def gini_coefficient(values: Iterable[float]) -> float:
    """
    Gini coefficient for a non‑negative 1‑D iterable.
    """
    xs = np.asarray(list(values), dtype=float)
    if xs.size == 0 or xs.sum() == 0:
        return 0.0
    if np.any(xs < 0):
        raise ValueError("values must be non‑negative")
    xs_sorted = np.sort(xs)
    n = xs_sorted.size
    cum = np.cumsum(xs_sorted)
    # Gini formula using the Lorenz curve
    gini = (n + 1 - 2 * np.sum(cum) / xs_sorted.sum()) / n
    return float(gini)


def hybrid_operation(features: List[np.ndarray], 
                     labels: List[str], 
                     confidence_bps: List[int], 
                     authority_class: List[str], 
                     rationale: List[str]) -> List[CertaintyFlag]:
    """
    This function performs the hybrid operation by first computing the RBF similarity matrix 
    of the input features, then using the minhash operation to generate a compact representation 
    of the text data, and finally using the fractional power binding operation to model the 
    strength of the causal relationships between the text data and the epistemic certainty flags.
    """
    similarity_matrix = rbf_similarity_matrix(features)
    minhash_values = np.min(similarity_matrix, axis=1)
    certainty_flags = []
    for i in range(len(labels)):
        fractional_binding = minhash_values[i] ** 0.5
        certainty_flags.append(certainty(
            label=labels[i],
            confidence_bps=int(confidence_bps[i] * fractional_binding),
            authority_class=authority_class[i],
            rationale=rationale[i],
            evidence_refs=(str(minhash_values[i]),)
        ))
    return certainty_flags


def compute_gini_coefficient_of_certainty_flags(certainty_flags: List[CertaintyFlag]) -> float:
    """
    This function computes the Gini coefficient of the confidence bps values of the certainty flags.
    """
    confidence_bps_values = [flag.confidence_bps for flag in certainty_flags]
    return gini_coefficient(confidence_bps_values)


def main():
    features = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    labels = ["FACT", "PROBABLE", "POSSIBLE"]
    confidence_bps = [1000, 2000, 3000]
    authority_class = ["high", "medium", "low"]
    rationale = ["good", "fair", "poor"]

    certainty_flags = hybrid_operation(features, labels, confidence_bps, authority_class, rationale)
    gini_coefficient_value = compute_gini_coefficient_of_certainty_flags(certainty_flags)

    print("Certainty Flags:")
    for flag in certainty_flags:
        print(flag.as_dict())
    print("Gini Coefficient:", gini_coefficient_value)


if __name__ == "__main__":
    main()