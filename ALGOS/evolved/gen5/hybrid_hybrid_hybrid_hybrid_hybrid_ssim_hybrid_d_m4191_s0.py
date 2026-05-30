# DARWIN HAMMER — match 4191, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_path_s_caputo_fractional_m736_s1.py (gen4)
# parent_b: hybrid_ssim_hybrid_decision_hygi_m9_s0.py (gen2)
# born: 2026-05-29T23:54:15Z

"""Hybrid Algorithm: Caputo Fractional Path & Decision Hygiene SSIM Fusion

This module fuses two distinct parent algorithms:

* **Parent A** – provides a lead‑lag transform for path‑wise causality encoding,
  B‑spline basis evaluation and a Lanczos Gamma approximation (fractional calculus
  tools).

* **Parent B** – extracts a nine‑dimensional decision‑hygiene feature vector from
  text, computes a Structural Similarity Index (SSIM) between feature vectors,
  and scores the decision using positive/negative weight vectors.

**Mathematical Bridge**

The bridge is built on the observation that both parents operate on *vectors*:
the lead‑lag transform maps a temporal path to a higher‑dimensional interleaved
space, while SSIM measures similarity between two vectors.  
We therefore:

1. Convert a raw text into a *feature path* by repeating the extracted feature
   vector over a small temporal horizon.
2. Apply the **lead‑lag transform** to obtain an interleaved representation.
3. Approximate the transformed path with **B‑splines** (using the basis from
   Parent A). The spline coefficients act as a smooth surrogate of the original
   path.
4. Compute **SSIM** between the original feature vector and the spline‑averaged
   vector; this similarity becomes a *weight* for the decision‑hygiene score.
5. Combine the weighted decision‑hygiene score with the Lanczos‑Gamma‑based
   fractional scaling (optional) to produce the final hybrid metric.

The resulting function `hybrid_decision_score` returns a single scalar that
encapsulates path‑smoothness, similarity to reference texts, and the original
decision‑hygiene evaluation.
"""

import numpy as np
import math
import re
import random
import sys
from pathlib import Path
from collections import Counter
from typing import List, Tuple

# ----------------------------------------------------------------------
# Parent A components
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform: interleave (lead, lag) channels for causality encoding.
    Input `path` shape: (T, d)
    Output shape: (2*T-1, 2*d)
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array (time, dim)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def bspline_basis(x: np.ndarray, grid: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Evaluate B‑spline basis functions of order `k` at positions `x`.
    Returns a matrix B of shape (len(x), len(grid)+k-2).
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    # Knot vector with clamped ends
    t = np.concatenate([
        np.full(k - 1, grid[0]),
        grid,
        np.full(k - 1, grid[-1]),
    ])

    N = len(x)
    B = np.zeros((N, len(t) - 1), dtype=np.float64)

    # Zeroth order (piecewise constant)
    for i in range(len(t) - 1):
        B[:, i] = np.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
    B[:, -1] = np.where(x == t[-1], 1.0, B[:, -1])

    # Recursion for higher orders
    for order in range(2, k + 1):
        B_new = np.zeros((N, len(t) - order), dtype=np.float64)
        for i in range(len(t) - order):
            denom_l = t[i + order - 1] - t[i]
            denom_r = t[i + order] - t[i + 1]

            term_l = ((x - t[i]) / denom_l) * B[:, i] if denom_l > 1e-12 else 0.0
            term_r = ((t[i + order] - x) / denom_r) * B[:, i + 1] if denom_r > 1e-12 else 0.0
            B_new[:, i] = term_l + term_r
        B = B_new
    return B


def gamma_lanczos(z: float) -> float:
    """
    Lanczos approximation of Gamma(z) for real z > 0.
    Uses g=7 coefficients (Numerical Recipes).
    """
    if z < 0.5:
        # Reflection formula
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))

    # Shift to the right of the pole
    z -= 1
    _g = 7
    _c = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ], dtype=np.float64)

    a = _c[0]
    for i in range(1, len(_c)):
        a += _c[i] / (z + i)

    t = z + _g + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * a


# ----------------------------------------------------------------------
# Parent B components
# ----------------------------------------------------------------------
_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

# Regular expressions for each feature
_REGEX_MAP = {
    "evidence": re.compile(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
        re.I,
    ),
    "planning": re.compile(
        r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
        re.I,
    ),
    "delay": re.compile(
        r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
        re.I,
    ),
    "support": re.compile(
        r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
        re.I,
    ),
    "boundary": re.compile(
        r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
        re.I,
    ),
    "outcome": re.compile(
        r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
        re.I,
    ),
    "impulsive": re.compile(
        r"\b(?:impulse|rash|spontaneous|quick|snap|hasty|without thinking|instinct|gut|react)\b",
        re.I,
    ),
    "scarcity": re.compile(
        r"\b(?:scarce|limited|rare|shortage|few|only|cannot get|unavailable|hard to find)\b",
        re.I,
    ),
    "risk": re.compile(
        r"\b(?:risk|danger|threat|hazard|peril|exposure|vulnerable|unsafe|dangerous)\b",
        re.I,
    ),
}


def _count_feature(text: str, pattern: re.Pattern) -> int:
    """Return the number of non‑overlapping matches of `pattern` in `text`."""
    return len(pattern.findall(text))


def extract_feature_vector(text: str) -> np.ndarray:
    """
    Produce a 9‑dimensional decision‑hygiene feature vector from `text`.
    Order follows `_FEATURE_ORDER`.
    """
    counts = [_count_feature(text, _REGEX_MAP[feat]) for feat in _FEATURE_ORDER]
    return np.array(counts, dtype=np.float64)


def decision_hygiene_score(features: np.ndarray) -> float:
    """
    Compute a raw decision‑hygiene score:
        score = (pos_weights·features) - (neg_weights·features)
    """
    pos = np.dot(_POSITIVE_WEIGHTS, features)
    neg = np.dot(_NEGATIVE_WEIGHTS, features)
    return float(pos - neg)


def ssim_index(x: np.ndarray, y: np.ndarray, C1: float = 1e-4, C2: float = 9e-4) -> float:
    """
    Simplified 1‑D Structural Similarity Index (SSIM) between vectors `x` and `y`.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("SSIM requires vectors of equal shape")

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = ((x - mu_x) ** 2).mean()
    sigma_y = ((y - mu_y) ** 2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def path_from_features(features: np.ndarray, steps: int = 5) -> np.ndarray:
    """
    Create a synthetic temporal path by linearly interpolating `features`
    over `steps` time points.
    Returns an array of shape (steps, len(features)).
    """
    if steps < 2:
        raise ValueError("steps must be >= 2")
    # Simple linear interpolation between zero vector and the feature vector
    start = np.zeros_like(features)
    end = features
    t = np.linspace(0, 1, steps)[:, None]
    path = start + t * (end - start)
    return path


def spline_smooth_path(transformed: np.ndarray, grid_points: int = 20) -> np.ndarray:
    """
    Approximate `transformed` (shape (T, D)) with a B‑spline.
    Returns the spline‑averaged path of shape (T, D).
    """
    T, D = transformed.shape
    # Parameterize by normalized time
    t_vals = np.linspace(0, 1, T)
    grid = np.linspace(0, 1, grid_points)

    # Build basis matrix (T x (grid+order-2))
    B = bspline_basis(t_vals, grid, k=3)  # cubic B‑spline
    # Least‑squares solve for coefficients per dimension
    coeffs = np.linalg.lstsq(B, transformed, rcond=None)[0]  # shape (basis, D)
    smooth = B @ coeffs
    return smooth


def hybrid_decision_score(
    text: str,
    reference_texts: List[str],
    steps: int = 5,
    grid_points: int = 20,
) -> float:
    """
    Compute the hybrid decision score for `text` against a list of `reference_texts`.

    Procedure:
    1. Extract feature vector `f` from `text`.
    2. Build a synthetic path from `f` and apply the lead‑lag transform.
    3. Smooth the transformed path with a cubic B‑spline.
    4. Collapse the smoothed path back to a single vector (mean over time).
    5. For each reference, compute SSIM between the collapsed vector and the
       reference's feature vector; keep the maximum similarity `s_max`.
    6. Compute raw decision‑hygiene score `dh`.
    7. Final hybrid metric = s_max * dh * gamma_lanczos(1 + dh/1e4) (fractional scaling).

    Returns a scalar hybrid metric.
    """
    # 1. Feature extraction
    f = extract_feature_vector(text)

    # 2. Path creation & lead‑lag
    path = path_from_features(f, steps=steps)
    transformed = lead_lag_transform(path)

    # 3. Spline smoothing
    smooth = spline_smooth_path(transformed, grid_points=grid_points)

    # 4. Collapse back to a representative vector (mean across time, then halve dim)
    # The lead‑lag doubles the dimension; we average lead and lag halves.
    D2 = smooth.shape[1]
    lead = smooth[:, : D2 // 2]
    lag = smooth[:, D2 // 2 :]
    collapsed = (lead + lag) / 2.0
    rep_vector = collapsed.mean(axis=0)

    # 5. SSIM against references
    sims = []
    for ref in reference_texts:
        ref_vec = extract_feature_vector(ref)
        # Align dimensions: reference is 9‑dim, rep_vector may be larger (depends on steps)
        # Reduce rep_vector to first 9 components (or pad reference)
        min_len = min(len(rep_vector), len(ref_vec))
        sim = ssim_index(rep_vector[:min_len], ref_vec[:min_len])
        sims.append(sim)
    s_max = max(sims) if sims else 0.0

    # 6. Decision hygiene raw score
    dh = decision_hygiene_score(f)

    # 7. Fractional scaling via Lanczos Gamma (acts as a smooth non‑linear factor)
    scaling = gamma_lanczos(1.0 + dh / 1e4)

    return s_max * dh * scaling


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "We have verified the source and recorded the logs. "
        "The plan includes a timeline and budget. "
        "We should wait until tomorrow before proceeding."
    )
    references = [
        "Evidence was confirmed and documented. The roadmap is set.",
        "Delay the release; there is a risk of scarcity.",
        "Support from the team is available. Outcome is successful.",
    ]

    score = hybrid_decision_score(sample_text, references)
    print(f"Hybrid decision score: {score:.4f}")

    # Simple sanity checks (