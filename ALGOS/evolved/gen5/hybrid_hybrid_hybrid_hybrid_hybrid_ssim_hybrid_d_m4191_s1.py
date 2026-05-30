# DARWIN HAMMER — match 4191, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_path_s_caputo_fractional_m736_s1.py (gen4)
# parent_b: hybrid_ssim_hybrid_decision_hygi_m9_s0.py (gen2)
# born: 2026-05-29T23:54:15Z

import numpy as np
import math
import re
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
    sigma_x = np.sqrt((x - mu_x) ** 2).mean()
    sigma_y = np.sqrt((y - mu_y) ** 2).mean()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    c1 = (C1 ** 2)
    c2 = (C2 ** 2)

    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))

    return ssim


# ----------------------------------------------------------------------
# Hybrid Algorithm
# ----------------------------------------------------------------------
def create_feature_path(text: str, T: int) -> np.ndarray:
    """
    Convert a raw text into a *feature path* by repeating the extracted feature
    vector over a small temporal horizon `T`.
    """
    feature_vector = extract_feature_vector(text)
    return np.tile(feature_vector, (T, 1))


def hybrid_decision_score(text: str, reference_text: str, T: int = 5) -> float:
    """
    Compute a hybrid decision score by fusing path smoothness, similarity to
    reference texts, and the original decision-hygiene evaluation.

    Args:
    - text (str): Input text to evaluate.
    - reference_text (str): Reference text for SSIM comparison.
    - T (int): Temporal horizon for feature path creation.

    Returns:
    - A scalar hybrid decision score.
    """
    # Create feature path
    feature_path = create_feature_path(text, T)

    # Apply lead-lag transform
    interleaved_path = lead_lag_transform(feature_path)

    # Approximate with B-splines
    grid = np.linspace(0, 1, interleaved_path.shape[0])
    spline_basis = bspline_basis(grid, grid, k=3)
    spline_coefficients = np.dot(interleaved_path, spline_basis.T)

    # Compute SSIM with reference text
    reference_features = extract_feature_vector(reference_text)
    ssim_vector = np.zeros(T)
    for t in range(T):
        ssim_vector[t] = ssim_index(reference_features, spline_coefficients[t])

    # Compute weighted decision-hygiene score
    features = extract_feature_vector(text)
    decision_score = decision_hygiene_score(features)
    weighted_score = np.dot(ssim_vector, features) / T

    # Optional fractional scaling
    fractional_scale = gamma_lanczos(0.5)  # Use a fixed fractional order
    scaled_score = weighted_score * fractional_scale

    return scaled_score