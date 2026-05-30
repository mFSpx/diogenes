# DARWIN HAMMER — match 2716, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m160_s2.py (gen4)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s3.py (gen3)
# born: 2026-05-29T23:43:43Z

"""Hybrid Certainty‑Geometric Fisher‑SSIM (HCGF) algorithm.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m160_s2.py (Hybrid Sheaf‑Certainty Cohomology + GA rotor)
- hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s3.py (Fisher‑SSIM routing + decision‑hygiene pruning)

Mathematical bridge:
The certainty flag supplies a scalar weight w_c ∈ [0,1] that scales both the
geometric product (a 2‑D GA rotor) and the statistical weights derived from
Fisher information and Shannon entropy.  The rotor R(w_c)=exp(‑½ w_c π e₁e₂) is
implemented as a 2‑D rotation matrix; it rotates the input feature vector.
Simultaneously w_c modulates the normalized Fisher weight w_f and the normalized
entropy weight w_h that appear in the unified decision metric

    M(t) = p(t) · [ w_f·SSIM(x,y) + w_h·H·Σ_i w_i·f_i ] · (1 + w_c),

where p(t)=1/(1+t) is a decreasing‑pruning probability, H is the Shannon entropy
of the extracted feature multiset, and w_i are raw feature counts.  The resulting
algorithm fuses topological certainty handling, geometric transformation, and
information‑theoretic routing into a single coherent system.
"""

import os
import sys
import math
import random
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Tuple, Iterable, List, Dict, Any
import numpy as np
from collections import Counter

# ----------------------------------------------------------------------
# Constants & utilities
# ----------------------------------------------------------------------
DEFAULT_BUDGET_MB = int(os.environ.get("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.environ.get("LUCIDOTA_VRAM_RESERVE_MB", "768"))

def now_z() -> str:
    """Current UTC time in ISO‑8601 Zulu format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ----------------------------------------------------------------------
# Parent A – Epistemic certainty helpers (adapted)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    """Immutable container for epistemic certainty."""
    label: str
    confidence_bps: int  # basis points, 0..10000
    authority_class: str = ""
    rationale: str = ""
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = field(default_factory=now_z)

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")

def certainty_weight(flag: CertaintyFlag) -> float:
    """Normalize confidence to [0,1]."""
    return flag.confidence_bps / 10000.0

# ----------------------------------------------------------------------
# Parent B – Fisher‑SSIM helpers (adapted)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def normalized_fisher_weight(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Normalized Fisher weight w_f = I/(I+ε)."""
    I = fisher_score(theta, center, width, eps)
    return I / (I + eps)

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index (SSIM) for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)

def decreasing_pruning(t: int) -> float:
    """Simple decreasing‑pruning probability p(t)=1/(1+t)."""
    return 1.0 / (1.0 + max(t, 0))

def shannon_entropy(tokens: Iterable[str]) -> float:
    """Shannon entropy of a token multiset."""
    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts.values()], dtype=float)
    return -float(np.sum(probs * np.log(probs + 1e-12)))

# ----------------------------------------------------------------------
# Geometric Algebra rotor utilities (2‑D implementation)
# ----------------------------------------------------------------------
def rotor_matrix(angle: float) -> np.ndarray:
    """2‑D rotation matrix representing a GA rotor for angle radians."""
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

def rotate_vector(v: np.ndarray, rotor: np.ndarray) -> np.ndarray:
    """Apply rotor to vector v."""
    if v.shape != (2,) or rotor.shape != (2, 2):
        raise ValueError("v must be shape (2,) and rotor must be (2,2)")
    return rotor @ v

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_rotate(v: np.ndarray, flag: CertaintyFlag) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate a 2‑D feature vector using a rotor whose angle is scaled by certainty.
    Returns (rotated_vector, rotor_matrix).
    """
    w_c = certainty_weight(flag)                # ∈[0,1]
    angle = w_c * (math.pi / 2)                 # up to 90°
    R = rotor_matrix(angle)
    v_rot = rotate_vector(v, R)
    return v_rot, R

def hybrid_decision_metric(x: np.ndarray,
                           y: np.ndarray,
                           feature_tokens: List[str],
                           flag: CertaintyFlag,
                           t: int,
                           theta: float,
                           center: float,
                           width: float) -> float:
    """
    Compute the unified decision metric M(t) that blends:
    - certainty‑scaled Fisher weight,
    - SSIM similarity,
    - entropy‑weighted feature sum,
    - decreasing pruning probability,
    - an overall boost (1 + w_c) from the certainty flag.
    """
    # Certainty scaling
    w_c = certainty_weight(flag)

    # Statistical weights
    w_f = normalized_fisher_weight(theta, center, width)          # Fisher
    H = shannon_entropy(feature_tokens)                           # Entropy
    w_h = H / (H + 1e-12)                                          # Normalized entropy

    # Similarity
    s = ssim(x, y)

    # Feature importance term Σ_i w_i·f_i
    token_counts = Counter(feature_tokens)
    feature_sum = float(sum(token_counts.values()))               # simple count sum

    # Pruning probability
    p = decreasing_pruning(t)

    # Core metric
    M = p * (w_f * s + w_h * H * feature_sum)

    # Certainty boost
    M *= (1.0 + w_c)

    return M

def hybrid_step(v: np.ndarray,
                x: np.ndarray,
                y: np.ndarray,
                feature_tokens: List[str],
                flag: CertaintyFlag,
                t: int,
                theta: float,
                center: float,
                width: float) -> Dict[str, Any]:
    """
    Perform one hybrid iteration:
    1. Rotate the feature vector.
    2. Compute the decision metric.
    3. Return a bundle of intermediate results for inspection.
    """
    v_rot, rotor = hybrid_rotate(v, flag)
    metric = hybrid_decision_metric(x, y, feature_tokens, flag, t, theta, center, width)

    return {
        "original_vector": v,
        "rotated_vector": v_rot,
        "rotor": rotor,
        "decision_metric": metric,
        "certainty_weight": certainty_weight(flag),
        "time_step": t
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a deterministic certainty flag
    flag = CertaintyFlag(label="PROBABLE", confidence_bps=7300, authority_class="test")

    # Simple 2‑D vector
    v = np.array([1.0, 0.0])

    # Synthetic 1‑D signals for SSIM
    rng = np.random.default_rng(42)
    x = rng.integers(0, 256, size=100).astype(float)
    y = x + rng.normal(0, 5, size=100)  # noisy copy

    # Feature tokens (pretend they were extracted by regexes)
    features = ["alpha", "beta", "alpha", "gamma", "beta", "beta"]

    # Parameters for Fisher/gaussian beam
    theta = 0.3
    center = 0.0
    width = 1.0

    # Perform a hybrid step at time t=5
    result = hybrid_step(v, x, y, features, flag, t=5, theta=theta, center=center, width=width)

    # Print a concise summary
    print("Hybrid step results:")
    print(f"  Certainty weight      : {result['certainty_weight']:.3f}")
    print(f"  Original vector       : {result['original_vector']}")
    print(f"  Rotated vector        : {result['rotated_vector']}")
    print(f"  Decision metric M(t) : {result['decision_metric']:.6f}")
    print(f"  Time step             : {result['time_step']}")
    print("  Rotor matrix:")
    print(result["rotor"])
    sys.exit(0)