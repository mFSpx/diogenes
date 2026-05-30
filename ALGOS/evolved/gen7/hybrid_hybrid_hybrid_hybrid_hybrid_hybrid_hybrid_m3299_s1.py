# DARWIN HAMMER — match 3299, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2380_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_ternar_m132_s0.py (gen3)
# born: 2026-05-29T23:49:06Z

"""Hybrid Algorithm: Fusion of HLTXR Tropical Max‑Plus Propagation (Parent A) with 
Decision‑Hygiene Feature Extraction & Ternary Routing (Parent B).

Mathematical Bridge
-------------------
Parent A provides a tropical (max‑plus) matrix multiplication `tropical_matmul` that
drives the broadcast strength vector of the HLTXR algorithm.  
Parent B extracts a sparse, integer‑valued feature vector from free‑form text via
regular‑expression based decision‑hygiene analysis.

The fusion treats the decision‑hygiene feature vector as a *dynamic weighting mask* that
modulates the tropical algebra:

* Each row of the left‑hand matrix `A` is scaled by a normalized feature weight before
  max‑plus multiplication, effectively biasing the propagation in proportion to the
  semantic evidence present in the input text.
* The same normalized weight also perturbs the HLTXR gradient/hessian computation
  (`adjusted_grad_hess`) and is further combined with a morphology‑derived scaling
  factor (sphericity/flatness) that plays the role of a discrete curvature term.

Finally, the resulting tropical product is fed to a ternary routing operator that
splits the output space into three disjoint regions (low, medium, high) – the
“ternary lens” from Parent B – providing a unified decision signal that can be used
by downstream XGBoost‑style learners or graph‑curvature analyses.

The module therefore integrates the core matrix algebra of Parent A with the
semantic‑driven weighting and routing logic of Parent B into a single coherent
pipeline.
"""

import math
import random
import re
import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# ----------------------------------------------------------------------
# Parent B – Decision‑hygiene feature extraction
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no c)\b", re.I)


def extract_features(text: str) -> np.ndarray:
    """
    Extract a 5‑dimensional integer feature vector from `text` using the
    decision‑hygiene regexes.
    """
    feats = [
        len(EVIDENCE_RE.findall(text)),
        len(PLANNING_RE.findall(text)),
        len(DELAY_RE.findall(text)),
        len(SUPPORT_RE.findall(text)),
        len(BOUNDARY_RE.findall(text)),
    ]
    return np.array(feats, dtype=float)


# ----------------------------------------------------------------------
# Parent A – Tropical (max‑plus) matrix multiplication
# ----------------------------------------------------------------------
def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Max‑plus matrix multiplication:
        (A ⊗ B)[i, j] = max_k (A[i, k] + B[k, j])
    """
    n, m = A.shape
    p = B.shape[1]
    C = np.empty((n, p), dtype=float)
    for i in range(n):
        for j in range(p):
            C[i, j] = max(A[i, k] + B[k, j] for k in range(m))
    return C


# ----------------------------------------------------------------------
# Morphology utilities (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def sphericity_index(m: Morphology) -> float:
    """Dimension‑less sphericity index."""
    if min(m.length, m.width, m.height) <= 0:
        raise ValueError("dimensions must be positive")
    return (m.length * m.width * m.height) ** (1.0 / 3.0) / m.length


def flatness_index(m: Morphology) -> float:
    """Dimension‑less flatness index."""
    if min(m.length, m.width, m.height) <= 0:
        raise ValueError("dimensions must be positive")
    return m.height / max(m.length, m.width)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def weighted_tropical(A: np.ndarray, B: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Apply row‑wise scaling to `A` using `weights` (length = A.shape[0]) before
    max‑plus multiplication.  The weights are assumed to be non‑negative and
    normalized to sum to 1.
    """
    if weights.shape[0] != A.shape[0]:
        raise ValueError("Weight vector length must match number of rows of A")
    A_scaled = A * weights[:, np.newaxis]
    return tropical_matmul(A_scaled, B)


def adjusted_grad_hess(
    logistic_loss: float,
    alpha: float,
    s: float,
    H: float,
    feature_weight: float,
    morph_factor: float,
) -> Tuple[float, float]:
    """
    HLTXR gradient/hessian adjusted by:
      * `feature_weight` – mean of the decision‑hygiene feature vector,
      * `morph_factor`   – product of sphericity and flatness (a proxy for
        discrete curvature).
    """
    base_grad = logistic_loss * (1 - logistic_loss)
    base_hess = logistic_loss * (1 - logistic_loss) * (1 - 2 * logistic_loss)

    grad = base_grad + alpha * s * H * feature_weight * morph_factor
    hess = base_hess + alpha * s * H * feature_weight * morph_factor
    return grad, hess


def ternary_route(matrix: np.ndarray) -> np.ndarray:
    """
    Partition the rows of `matrix` into three routing zones:
        0 – low‑value zone (first third of rows),
        1 – medium‑value zone (second third),
        2 – high‑value zone (last third).
    Returns an integer array of shape (matrix.shape[0],) with the assigned route.
    """
    n = matrix.shape[0]
    thirds = n // 3
    routes = np.empty(n, dtype=int)
    routes[:thirds] = 0
    routes[thirds : 2 * thirds] = 1
    routes[2 * thirds :] = 2
    # Edge case when n < 3: assign all to route 1
    if n < 3:
        routes[:] = 1
    return routes


def hybrid_process(
    A: np.ndarray,
    B: np.ndarray,
    text: str,
    morph: Morphology,
    alpha: float = 0.5,
    s: float = 1.0,
    H: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    End‑to‑end hybrid pipeline:

    1. Extract decision‑hygiene features from `text`.
    2. Normalise them to obtain row‑weights for the tropical product.
    3. Compute a weighted tropical multiplication `C = weighted_tropical(A, B, w)`.
    4. Derive a combined morphology factor `m_factor = sphericity * flatness`.
    5. For each element of `C`, compute a gradient/hessian pair using
       `adjusted_grad_hess`.
    6. Route the rows of `C` through the ternary lens.

    Returns a tuple `(C, grad_hess_array, routes)` where:
        * `C` is the weighted tropical result,
        * `grad_hess_array` is an `(n, 2)` array of (grad, hess) per row,
        * `routes` is the ternary routing vector.
    """
    # 1‑2. Feature extraction and weight normalisation
    feats = extract_features(text)
    # Pad or truncate to match rows of A
    if feats.shape[0] < A.shape[0]:
        feats = np.pad(feats, (0, A.shape[0] - feats.shape[0]), constant_values=0)
    elif feats.shape[0] > A.shape[0]:
        feats = feats[: A.shape[0]]
    weight_sum = feats.sum() + 1e-12
    w = feats / weight_sum

    # 3. Weighted tropical multiplication
    C = weighted_tropical(A, B, w)

    # 4. Morphology factor
    sph = sphericity_index(morph)
    flt = flatness_index(morph)
    m_factor = sph * flt

    # 5. Gradient/Hessian per row (using the row‑wise mean of C as logistic loss proxy)
    grads = []
    for i in range(C.shape[0]):
        logistic_proxy = 1.0 / (1.0 + math.exp(-C[i].mean()))  # sigmoid of row mean
        grad, hess = adjusted_grad_hess(
            logistic_proxy, alpha, s, H, feature_weight=w.mean(), morph_factor=m_factor
        )
        grads.append([grad, hess])
    grad_hess_array = np.array(grads, dtype=float)

    # 6. Ternary routing
    routes = ternary_route(C)

    return C, grad_hess_array, routes


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Construct small random matrices (5 rows to match feature length)
    A = np.random.uniform(-2, 2, size=(5, 4))
    B = np.random.uniform(-2, 2, size=(4, 3))

    sample_text = (
        "The evidence was verified and the plan was approved. "
        "We will wait until tomorrow before proceeding, but the team is ready to help."
    )

    morph = Morphology(length=3.0, width=2.0, height=1.5, mass=4.2)

    C, grad_hess, routes = hybrid_process(A, B, sample_text, morph)

    print("Weighted tropical product (C):")
    print(C)
    print("\nGradient/Hessian per row:")
    print(grad_hess)
    print("\nTernary routes:")
    print(routes)