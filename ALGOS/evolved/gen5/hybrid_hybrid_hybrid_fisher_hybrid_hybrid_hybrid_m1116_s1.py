# DARWIN HAMMER — match 1116, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_fisher_locali_hybrid_ternary_lens__m71_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m264_s1.py (gen4)
# born: 2026-05-29T23:32:52Z

"""Hybrid Fisher‑Ternary‑Regex‑RBF Router
This module fuses the core mathematics of two parent algorithms:

* **Parent A** – Fisher‑localization & SSIM routing.
  - Gaussian beam → Fisher information score.
  - Ternary evidence vector → weighted Shannon entropy.

* **Parent B** – Regex‑driven feature extraction & Radial‑Basis‑Function (RBF) surrogate.
  - Regex patterns produce categorical counts.
  - An RBF model maps a feature vector to a scalar prediction.

**Mathematical bridge**  
The Fisher information score, computed from a Gaussian beam parameterised by `theta`,
is used as a *confidence weight* for each regex‑derived categorical count.  
These confidence‑weighted counts are turned into a ternary vector (‑1, 0, +1) whose
absolute values are multiplied by the Fisher score to obtain a weighted histogram.
The histogram yields a Shannon entropy `H`.  

Simultaneously, the same weighted counts form the input to an RBF surrogate
model producing a prediction `R`.  The final routing metric combines the entropy,
the RBF prediction and the structural similarity index `S` between the packet
text and a reference sample:


Decision = α·H + β·R + γ·S


The three functions below demonstrate this hybrid pipeline.

"""

import math
import random
import sys
from pathlib import Path
import re
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


def ssim_1d(x: np.ndarray, y: np.ndarray,
            dynamic_range: float = 255.0,
            k1: float = 0.01, k2: float = 0.03) -> float:
    """Simplified SSIM for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError("inputs must have the same shape")
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return numerator / denominator


# ----------------------------------------------------------------------
# Parent‑B building blocks
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
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety)\b",
    re.I,
)

REGEX_CATEGORIES = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
}


def extract_regex_features(text: str) -> dict:
    """Count matches for each regex category."""
    counts = {}
    for name, pattern in REGEX_CATEGORIES.items():
        counts[name] = len(pattern.findall(text))
    return counts


def ternary_vector_from_counts(counts: dict) -> np.ndarray:
    """Map counts to ternary values: +1 if count>0, 0 otherwise."""
    vec = np.array([1 if c > 0 else 0 for c in counts.values()], dtype=float)
    # Convert to -1/0/+1 by centering around zero (here only 0/1 needed)
    return vec * 2 - 1  # 0→-1, 1→+1


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def fisher_weighted_entropy(theta: float, center: float, width: float,
                            counts: dict) -> float:
    """
    Compute Shannon entropy of a Fisher‑weighted ternary histogram.

    Steps
    -----
    1. Compute Fisher score `F`.
    2. Build ternary vector `t` from regex counts.
    3. Weight: w_i = F * |t_i|
    4. Normalise to a probability distribution `p`.
    5. Return `- Σ p_i log2 p_i`.
    """
    F = fisher_score(theta, center, width)
    t = ternary_vector_from_counts(counts)
    w = F * np.abs(t)  # absolute ternary values (0 or 1) scaled by F
    if w.sum() == 0:
        # avoid division by zero – entropy of a deterministic distribution is 0
        return 0.0
    p = w / w.sum()
    entropy = -np.sum(p * np.log2(p + 1e-12))
    return float(entropy)


def rbf_predict(feature_vec: np.ndarray,
                centers: np.ndarray,
                widths: np.ndarray,
                weights: np.ndarray) -> float:
    """
    Radial‑Basis Function surrogate prediction.

    φ_i(x) = exp( -||x - c_i||² / (2 * σ_i²) )
    y = Σ w_i * φ_i(x)

    Parameters
    ----------
    feature_vec : (D,) input feature vector.
    centers     : (N, D) RBF centres.
    widths      : (N,) σ_i for each centre (must be >0).
    weights     : (N,) linear weights.

    Returns
    -------
    Scalar prediction.
    """
    diff = centers - feature_vec  # (N, D)
    sq_norm = np.einsum('ij,ij->i', diff, diff)  # (N,)
    phi = np.exp(-sq_norm / (2.0 * (widths ** 2) + 1e-12))
    return float(np.dot(weights, phi))


def hybrid_decision(packet_text: str,
                    reference_text: str,
                    theta: float,
                    center: float,
                    width: float,
                    rbf_centers: np.ndarray,
                    rbf_widths: np.ndarray,
                    rbf_weights: np.ndarray,
                    alpha: float = 0.4,
                    beta: float = 0.4,
                    gamma: float = 0.2) -> float:
    """
    Unified routing metric combining:
      * Fisher‑weighted entropy `H`
      * RBF surrogate output `R`
      * SSIM between packet and reference `S`

    Decision = α·H + β·R + γ·S
    """
    # 1️⃣ Regex feature extraction
    counts = extract_regex_features(packet_text)

    # 2️⃣ Fisher‑weighted entropy
    H = fisher_weighted_entropy(theta, center, width, counts)

    # 3️⃣ Build feature vector for RBF (use raw counts, normalized)
    raw_vec = np.array(list(counts.values()), dtype=float)
    if raw_vec.max() > 0:
        feature_vec = raw_vec / raw_vec.max()
    else:
        feature_vec = raw_vec

    # 4️⃣ RBF prediction
    R = rbf_predict(feature_vec, rbf_centers, rbf_widths, rbf_weights)

    # 5️⃣ SSIM (convert strings to uint8 arrays via ord)
    pkt_arr = np.frombuffer(packet_text.encode('utf-8'), dtype=np.uint8)
    ref_arr = np.frombuffer(reference_text.encode('utf-8'), dtype=np.uint8)
    min_len = min(len(pkt_arr), len(ref_arr))
    if min_len == 0:
        S = 0.0
    else:
        S = ssim_1d(pkt_arr[:min_len].astype(float),
                    ref_arr[:min_len].astype(float))

    # 6️⃣ Combine
    decision = alpha * H + beta * R + gamma * S
    return decision


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Example packet and reference texts
    packet = ("The evidence shows that the plan was verified. "
              "Please check the log and the screenshot for proof.")
    reference = ("Verified evidence and logs confirm the plan. "
                 "Check the screenshot for additional proof.")

    # Fisher beam parameters
    theta_val = 0.3
    center_val = 0.0
    width_val = 1.0

    # RBF surrogate configuration (5‑dimensional because we have 5 regex categories)
    N = 8  # number of RBF centres
    D = len(REGEX_CATEGORIES)
    rbf_centers = np.random.uniform(-1, 1, size=(N, D))
    rbf_widths = np.random.uniform(0.5, 1.5, size=N)
    rbf_weights = np.random.uniform(-1, 1, size=N)

    score = hybrid_decision(packet,
                            reference,
                            theta=theta_val,
                            center=center_val,
                            width=width_val,
                            rbf_centers=rbf_centers,
                            rbf_widths=rbf_widths,
                            rbf_weights=rbf_weights)

    print(f"Hybrid routing decision score: {score:.4f}")