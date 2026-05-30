# DARWIN HAMMER — match 1234, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_ternary_lens__m71_s0.py (gen3)
# parent_b: hybrid_hybrid_gliner_zero_s_hybrid_doomsday_cale_m49_s2.py (gen2)
# born: 2026-05-29T23:35:59Z

"""Hybrid Fisher‑Ternary‑Gini Router
Parent A: hybrid_fisher_locali_hybrid_ternary_lens (Gaussian beam → Fisher score,
SSIM between 1‑D signals).
Parent B: hybrid_gliner_zero_s_hybrid_doomsday_cale (date weekday utilities,
Gini coefficient, label handling).

Mathematical bridge
------------------
1. A continuous confidence `F = FisherScore(θ)` (Parent A) is used as a scalar
   probability‑weight for each entry of a ternary evidence vector `v ∈ {‑1,0,+1}`
   (Parent A & B).  
   `w_i = F * |v_i|` yields a non‑negative weighted histogram.

2. The normalized histogram `p_i = w_i / Σ w_i` is fed simultaneously to
   *Shannon entropy* `H = - Σ p_i log₂ p_i` (Parent A) **and** to the
   *Gini coefficient* `G = 1 - Σ p_i²` (a simplified Gini for probability
   vectors, derived from Parent B’s inequality measure).

3. A structural similarity index `S = SSIM(x, y)` between two 1‑D signals
   (Parent A) is computed.

4. The final routing decision fuses the three information‑theoretic quantities:
   `Decision = α·H + β·G + γ·S`, where `α,β,γ` are tunable scalars.

The module implements this fused pipeline and demonstrates its use with
three public functions.

"""

import math
import random
import sys
from pathlib import Path
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A building blocks
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


def _mean_std(arr: np.ndarray) -> tuple[float, float]:
    """Return mean and standard deviation of a 1‑D array."""
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    return mean, std


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Simplified SSIM for 1‑D signals.
    Returns a value in [0, 1].
    """
    if x.shape != y.shape:
        raise ValueError("inputs must have the same shape")
    mu_x, sigma_x = _mean_std(x)
    mu_y, sigma_y = _mean_std(y)
    sigma_xy = float(np.mean((x - mu_x) * (y - mu_y)))

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return numerator / denominator if denominator != 0 else 0.0


# ----------------------------------------------------------------------
# Parent‑B building blocks
# ----------------------------------------------------------------------
DEFAULT_LABELS = [
    "Operator", "Rainmaker", "Paladin / God-Mode", "Psyche / State-Collapse",
    "Forensic Shield", "Infinite Sink", "Anchor Weight", "Server Wipe",
    "API Rate Limiting", "Environment Migration", "Cruelty Protocols",
    "Master’s Eye", "Chrono-Ledger", "KRAMPUSCHEWING", "KORPUS",
    "DIOGENES", "FairyFuse", "Job Fair Allocator", "Darwinian Surfaces",
    "Command Envelope Protocol",
]


def parse_labels(raw: str | None) -> list[str]:
    """Parse a comma‑separated label string or fall back to defaults."""
    if raw is None:
        return DEFAULT_LABELS
    return [lbl.strip() for lbl in raw.split(",") if lbl.strip()]


def weekday_sakamoto(years: np.ndarray, months: np.ndarray, days: np.ndarray) -> np.ndarray:
    """
    Vectorised weekday calculation using Tomohiko Sakamoto's algorithm.
    Returns 0 = Sunday … 6 = Saturday.
    """
    y = years.astype(np.int64)
    m = months.astype(np.int64)
    d = days.astype(np.int64)

    m_adj = np.where(m < 3, m + 12, m)
    y_adj = np.where(m < 3, y - 1, y)

    t = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)
    w = (y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400 + t[m_adj - 1] + d) % 7
    return w.astype(np.int8)


def gini_coefficient(probs: np.ndarray) -> float:
    """
    Gini coefficient for a probability vector (Σ p_i = 1, p_i ≥ 0).
    For a probability distribution this reduces to 1 - Σ p_i².
    """
    if probs.ndim != 1:
        raise ValueError("probs must be 1‑D")
    if np.any(probs < 0):
        raise ValueError("probabilities must be non‑negative")
    return 1.0 - float(np.sum(probs ** 2))


def sha256_text(text: str) -> str:
    """SHA‑256 hash of the supplied Unicode text."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def ternary_vector_from_text(text: str, patterns: dict[str, re.Pattern]) -> np.ndarray:
    """
    Produce a ternary vector from `text` using a dict of regex patterns.
    For each pattern:
        - match → +1
        - negative look‑ahead match → -1
        - no match → 0
    The order of keys determines vector ordering.
    """
    vec = []
    for label, pat in patterns.items():
        if pat.search(text):
            vec.append(1)
        elif pat.search(f"not {text}"):  # placeholder for a “negative” pattern
            vec.append(-1)
        else:
            vec.append(0)
    return np.array(vec, dtype=np.int8)


def weighted_distribution(theta: float, center: float, width: float,
                         ternary_vec: np.ndarray) -> np.ndarray:
    """
    Compute a normalized weighted probability distribution:
        w_i = FisherScore * |v_i|
        p_i = w_i / Σ w_i
    If the ternary vector is all zeros, return a uniform distribution.
    """
    if ternary_vec.ndim != 1:
        raise ValueError("ternary_vec must be 1‑D")
    f = fisher_score(theta, center, width)
    weights = f * np.abs(ternary_vec.astype(np.float64))
    total = weights.sum()
    if total == 0:
        # uniform fallback
        return np.full_like(weights, 1.0 / weights.size, dtype=np.float64)
    return weights / total


def hybrid_metric(theta: float, center: float, width: float,
                  ternary_vec: np.ndarray,
                  sig_x: np.ndarray, sig_y: np.ndarray,
                  alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3) -> float:
    """
    Compute the fused routing metric:
        H = Shannon entropy of weighted distribution
        G = Gini coefficient of the same distribution
        S = SSIM between two signals
        Decision = α·H + β·G + γ·S
    """
    p = weighted_distribution(theta, center, width, ternary_vec)

    # Shannon entropy (base 2)
    eps = np.finfo(float).eps
    H = -float(np.sum(p * np.log2(p + eps)))

    # Gini from the same probability vector
    G = gini_coefficient(p)

    # Structural similarity of the two signals
    S = ssim(sig_x, sig_y)

    return alpha * H + beta * G + gamma * S


# ----------------------------------------------------------------------
# Example high‑level usage
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


def route_text(text: str, reference_signal: np.ndarray,
               theta: float, center: float, width: float,
               patterns: dict[str, re.Pattern],
               alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3) -> Span:
    """
    Produce a `Span` describing the routing decision for `text`.

    Steps:
        1. Build a ternary evidence vector from regex `patterns`.
        2. Generate a synthetic signal from the text (character codes) for SSIM.
        3. Compute the hybrid metric.
        4. Choose a label based on the highest absolute ternary entry.
        5. Return a Span with the metric as its score.
    """
    # 1. ternary vector
    tern_vec = ternary_vector_from_text(text, patterns)

    # 2. synthetic signal: normalized Unicode code points
    codes = np.fromiter((ord(ch) for ch in text), dtype=np.int32, count=len(text))
    if codes.size == 0:
        codes = np.array([0], dtype=np.int32)
    sig_x = (codes - codes.mean()) / (codes.std() + 1e-12)

    # Ensure reference has same length (pad or truncate)
    if reference_signal.shape != sig_x.shape:
        min_len = min(reference_signal.size, sig_x.size)
        sig_x = sig_x[:min_len]
        ref_sig = reference_signal[:min_len]
    else:
        ref_sig = reference_signal

    # 3. hybrid metric
    decision_score = hybrid_metric(theta, center, width, tern_vec,
                                   sig_x, ref_sig, alpha, beta, gamma)

    # 4. pick label: the pattern with the largest absolute weight
    if tern_vec.size == 0:
        chosen_label = "Unlabeled"
    else:
        idx = int(np.argmax(np.abs(tern_vec)))
        chosen_label = list(patterns.keys())[idx]

    # 5. construct Span (using dummy indices)
    return Span(start=0, end=len(text), text=text,
                label=chosen_label, score=decision_score)


def demo_hybrid():
    """Run a quick demonstration of the hybrid routing pipeline."""
    # Example regex patterns (very simple for demo purposes)
    patterns = {
        "ALPHA": re.compile(r"[A-M]"),
        "NUMERIC": re.compile(r"\d"),
        "SPECIAL": re.compile(r"[!@#\$%]"),
    }

    sample_text = "Hello World! 123"
    # Reference signal: a sine wave of same length
    t = np.linspace(0, 2 * math.pi, num=len(sample_text))
    reference = np.sin(t)

    # Fisher parameters
    theta, center, width = 0.5, 0.0, 1.0

    span = route_text(sample_text, reference, theta, center, width, patterns)
    print(f"Routing Span -> label: {span.label}, score: {span.score:.4f}, hash: {sha256_text(span.text)}")


if __name__ == "__main__":
    demo_hybrid()