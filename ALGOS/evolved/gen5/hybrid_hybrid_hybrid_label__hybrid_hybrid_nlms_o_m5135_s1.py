# DARWIN HAMMER — match 5135, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_label_foundry_path_signature_m231_s0.py (gen3)
# parent_b: hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s1.py (gen4)
# born: 2026-05-30T00:00:01Z

"""
HybridLabelSignatureNLMS

Parents:
- hybrid_hybrid_label_foundry_path_signature_m231_s0.py (Label aggregation + level‑2 path signature scaling)
- hybrid_hybrid_nlms_omni_cha_hybrid_rectified_flo_m835_s1.py (Normalized LMS predictor + straight‑line interpolant)

Mathematical Bridge
-------------------
The level‑2 signature tensor S₂ of a trajectory X(t) captures geometric
interactions between coordinates.  We flatten the element‑wise exponential
scaled tensor

    S₂ˢᶜ = exp(ρ·S₂)

into a feature vector φ ∈ ℝ^{d²}.  This vector feeds a Normalized LMS (NLMS)
predictor w·φ that estimates a *confidence correction* for the label
aggregation performed by the parent‑A logic.  The NLMS error, defined as the
difference between the NLMS prediction and the vote‑based confidence, drives
the NLMS weight update.  The straight‑line interpolant of the Rectified Flow
provides additional synthetic points that enrich the signature without
altering the original trajectory.

The resulting hybrid system:
1. Computes a vote‑based confidence ĉ from labeling‑function results.
2. Extracts φ from the (possibly enriched) path using the scaled signature.
3. Predicts a correction 𝛿 = w·φ with NLMS.
4. Returns a fused confidence c = clip(ĉ + 𝛿, 0, 1) and updates w online.

All operations rely only on NumPy and the Python standard library.
"""

from __future__ import annotations
import sys
import pathlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  # binary 0/1


@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  # in [0,1]


def labeling_function(name: str | None = None):
    """Decorator that annotates a labeling function with a name."""
    def deco(fn: Callable[[dict], int]) -> Callable[[dict], int]:
        fn.lf_name = name or fn.__name__
        return fn
    return deco


# ----------------------------------------------------------------------
# Parent A core – majority vote aggregation
# ----------------------------------------------------------------------
def aggregate_votes(batches: List[List[LabelingFunctionResult]]) -> Dict[str, float]:
    """
    Compute the vote‑based confidence for each document.
    Returns a mapping doc_id → confidence (mean of binary votes).
    """
    votes: Dict[str, List[int]] = defaultdict(list)
    for batch in batches:
        for r in batch:
            if r.label in (0, 1):
                votes[r.doc_id].append(r.label)
    return {doc_id: float(np.mean(lbls)) for doc_id, lbls in votes.items()}


# ----------------------------------------------------------------------
# Parent B core – NLMS predictor
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
    Single NLMS weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Learning rate ∈ (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    new_weights, error
    """
    error = target - nlms_predict(weights, x)
    norm_sq = np.linalg.norm(x) ** 2
    weights = weights + mu * error * x / (norm_sq + eps)
    return weights, error


def interpolant(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Straight‑line interpolant Z_t = t * x1 + (1‑t) * x0.
    Supports batch broadcasting: x0, x1 shape (B, d); t shape (B,).
    """
    t = np.reshape(t, (-1, 1))
    return t * x1 + (1 - t) * x0


# ----------------------------------------------------------------------
# Signature utilities (from Parent A, enriched with interpolant)
# ----------------------------------------------------------------------
def level2_signature(path: np.ndarray) -> np.ndarray:
    """
    Compute the level‑2 iterated‑integral tensor S₂ for a path.
    path shape: (T, d). Returns (d, d) tensor.

    S₂[i, j] = Σ_{t=1}^{T‑1} (X_{t}ⁱ − X₀ⁱ) * (ΔX_{t})ʲ
    where ΔX_{t} = X_{t+1} − X_{t}.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time, dim)")
    base = path[:-1] - path[0]          # shape (T‑1, d)
    inc = np.diff(path, axis=0)         # shape (T‑1, d)
    return np.einsum("ti,tj->ij", base, inc)


def scaled_signature_features(path: np.ndarray, rho: float) -> np.ndarray:
    """
    Compute exp(ρ·S₂) and flatten to a feature vector φ.
    """
    S2 = level2_signature(path)
    S2_scaled = np.exp(rho * S2)        # element‑wise exponential
    return S2_scaled.ravel()           # shape (d*d,)


def enrich_path_with_interpolant(path: np.ndarray, n_extra: int = 5) -> np.ndarray:
    """
    Insert n_extra equally‑spaced interpolated points between each consecutive
    pair of original points using the straight‑line interpolant.
    """
    if n_extra <= 0:
        return path
    T, d = path.shape
    enriched = [path[0]]
    for i in range(T - 1):
        x0, x1 = path[i], path[i + 1]
        ts = np.linspace(0, 1, n_extra + 2)[1:-1]   # interior points
        mids = interpolant(np.broadcast_to(x0, (len(ts), d)),
                           np.broadcast_to(x1, (len(ts), d)),
                           ts)
        enriched.extend(mids)
        enriched.append(x1)
    return np.stack(enriched, axis=0)


# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_aggregate(
    batches: List[List[LabelingFunctionResult]],
    path: np.ndarray,
    rho: float,
    weights: np.ndarray,
    mu: float = 0.5,
) -> Tuple[List[ProbabilisticLabel], np.ndarray]:
    """
    Perform a hybrid aggregation:

    1. Vote‑based confidence ĉ from labeling‑function results.
    2. Enrich the path, extract φ = flattened exp(ρ·S₂).
    3. Predict correction δ = w·φ with NLMS.
    4. Fuse confidences: c = clip(ĉ + δ, 0, 1).
    5. Update NLMS weights using target = ĉ (the pure vote confidence).

    Returns the list of ProbabilisticLabel objects and the updated weights.
    """
    # 1. vote confidence per document
    vote_conf = aggregate_votes(batches)                     # doc_id → float

    # 2. feature extraction
    enriched = enrich_path_with_interpolant(path, n_extra=3)
    phi = scaled_signature_features(enriched, rho)          # (d*d,)

    # 3. NLMS prediction (single scalar correction applied to all docs)
    delta = nlms_predict(weights, phi)

    # 4. fuse confidences
    out: List[ProbabilisticLabel] = []
    for doc_id, c_hat in vote_conf.items():
        c = float(np.clip(c_hat + delta, 0.0, 1.0))
        out.append(ProbabilisticLabel(doc_id, 0, c))

    # 5. weight update (use average vote confidence as target for stability)
    target = float(np.mean(list(vote_conf.values())) if vote_conf else 0.0)
    new_weights, _ = nlms_update(weights, phi, target, mu=mu)

    return out, new_weights


def initialize_nlms_weights(feature_dim: int) -> np.ndarray:
    """Initialize NLMS weight vector with small random values."""
    rng = np.random.default_rng()
    return rng.normal(loc=0.0, scale=0.01, size=feature_dim)


def demo_hybrid():
    """Simple smoke‑test demonstrating the hybrid pipeline."""
    # Synthetic labeling‑function results for two documents
    batches = [
        [
            LabelingFunctionResult("lf_a", "doc1", 1),
            LabelingFunctionResult("lf_b", "doc1", 0),
            LabelingFunctionResult("lf_c", "doc1", 1),
        ],
        [
            LabelingFunctionResult("lf_a", "doc2", 0),
            LabelingFunctionResult("lf_b", "doc2", 0),
            LabelingFunctionResult("lf_c", "doc2", 1),
        ],
    ]

    # Synthetic 2‑D path (e.g., a simple curve)
    t = np.linspace(0, 2 * math.pi, 10)
    path = np.column_stack([np.cos(t), np.sin(t)])  # shape (10, 2)

    rho = 0.3
    phi_dim = (path.shape[1]) ** 2          # d*d after flattening
    w = initialize_nlms_weights(phi_dim)

    labels, w_new = hybrid_aggregate(batches, path, rho, w, mu=0.6)

    print("Hybrid Probabilistic Labels:")
    for pl in labels:
        print(f"  doc_id={pl.doc_id}, label={pl.label}, confidence={pl.confidence:.4f}")

    print("\nNLMS weights change norm:",
          np.linalg.norm(w_new - w))


if __name__ == "__main__":
    demo_hybrid()