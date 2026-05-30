# DARWIN HAMMER — match 4125, survivor 5
# gen: 6
# parent_a: hybrid_decision_hygiene_hybrid_hybrid_hybrid_m1886_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_hybrid_krampu_m11_s2.py (gen4)
# born: 2026-05-29T23:53:40Z

"""
Hybrid Algorithm: Decision Hygiene Entropy + Fisher‑SSIM + Ollivier‑Ricci Regularization

Parents:
- hybrid_decision_hygiene_hybrid_hybrid_hybrid_m1886_s1.py (Decision‑hygiene feature extraction,
  Shannon entropy of category counts, social‑interaction modulation)
- hybrid_hybrid_hybrid_fisher_hybrid_hybrid_krampu_m11_s2.py (Fisher information for a Gaussian
  beam, SSIM similarity, Ollivier‑Ricci curvature regularization, unified metric)

Mathematical Bridge:
Both parents employ Shannon entropy (H) as a core scalar that quantifies uncertainty.
Parent B combines H with a Fisher‑information weight w_f = I/(I+ε) to balance SSIM and a
feature‑importance term Σ_i w_i·f_i.  Parent A supplies the feature counts w_i and binary
flags f_i via regex‑based decision‑hygiene analysis, and its entropy H is already
computed from those counts.  The fusion therefore uses the same H for both the
entropy‑weight w_h = H/(H+ε) and the decision‑hygiene term, while the Fisher‑information
derived from a Gaussian beam supplies w_f.  The resulting unified decision metric is

    M = w_f·SSIM(x, y) + w_h·H·Σ_i w_i·f_i + λ·Ω(W)

where Ω(W) is an Ollivier‑Ricci curvature‑like regularizer on a weight matrix W.
The code below implements this hybrid system with three public functions demonstrating
the combined operation.
"""

import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# Decision‑hygiene feature extraction (Parent A)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|"
    r"before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|"
    r"advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|"
    r"protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|"
    r"filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|"
    r"tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|"
    r"rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|"
    r"crisis|collapse)\b",
    re.I,
)

CATEGORY_REGEXES = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
    "scarcity": SCARCITY_RE,
    "risk": RISK_RE,
}


def extract_feature_counts(text: str) -> dict:
    """
    Scan *text* with the compiled regexes and return a dictionary mapping each
    category to the number of matches found.
    """
    counts = {}
    for name, pattern in CATEGORY_REGEXES.items():
        matches = pattern.findall(text)
        counts[name] = len(matches)
    return counts


def shannon_entropy(counts: dict) -> float:
    """
    Compute the Shannon entropy of the categorical count distribution.
    Zero‑count categories are ignored.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * math.log(p, 2) for p in probs)


# ----------------------------------------------------------------------
# Fisher‑SSIM core (Parent B)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam.
    I(θ) = (∂_θ I(θ))² / I(θ)
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """
    One‑dimensional Structural Similarity Index (SSIM).
    """
    if x.shape != y.shape:
        raise ValueError("samples must have equal shape")
    if x.size == 0:
        raise ValueError("samples must not be empty")
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    sigma_xy = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mx * my + c1) * (2 * sigma_xy + c2)
    denominator = (mx ** 2 + my ** 2 + c1) * (vx + vy + c2)
    return numerator / denominator


# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature regularizer (simplified)
# ----------------------------------------------------------------------
def curvature_regularization(W: np.ndarray) -> float:
    """
    Approximate a curvature‑like regularizer for a weight matrix W.
    The implementation uses the Frobenius norm of the graph Laplacian
    L = D - W, where D is the degree matrix.  This mimics the tendency of
    Ollivier‑Ricci curvature to penalise highly irregular connections.
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")
    degrees = np.sum(W, axis=1)
    D = np.diag(degrees)
    L = D - W
    return np.linalg.norm(L, ord="fro")


# ----------------------------------------------------------------------
# Unified hybrid operation
# ----------------------------------------------------------------------
def unified_metric(text_a: str,
                   text_b: str,
                   theta: float,
                   center: float,
                   width: float,
                   W: np.ndarray,
                   lambda_reg: float = 0.1,
                   eps: float = 1e-12) -> float:
    """
    Compute the hybrid decision metric M for two textual inputs.

    Steps:
    1. Extract decision‑hygiene counts from *text_a* (the “source” document).
    2. Compute Shannon entropy H of those counts.
    3. Build binary flags f_i (1 if count > 0 else 0) and raw counts w_i.
    4. Compute Fisher information I(θ) from the Gaussian beam parameters.
    5. Compute normalized weights:
           w_f = I / (I + ε)
           w_h = H / (H + ε)
    6. Convert each text to a numeric 1‑D signal (ASCII codes) and evaluate SSIM.
    7. Compute curvature regularization Ω(W).
    8. Assemble M = w_f·SSIM + w_h·H·Σ_i w_i·f_i + λ·Ω(W).

    The function returns the scalar metric M.
    """
    # 1‑3. Decision‑hygiene features
    counts = extract_feature_counts(text_a)
    H = shannon_entropy(counts)
    flags = {k: (1 if v > 0 else 0) for k, v in counts.items()}
    weighted_sum = sum(counts[k] * flags[k] for k in counts)

    # 4‑5. Fisher information and normalized weights
    I = fisher_score(theta, center, width, eps=eps)
    w_f = I / (I + eps)
    w_h = H / (H + eps)

    # 6. Text → numeric signal (ASCII values) and SSIM
    # Truncate or pad to the same length
    min_len = min(len(text_a), len(text_b))
    if min_len == 0:
        raise ValueError("input texts must not be empty")
    sig_a = np.frombuffer(text_a[:min_len].encode("utf-8"), dtype=np.uint8).astype(float)
    sig_b = np.frombuffer(text_b[:min_len].encode("utf-8"), dtype=np.uint8).astype(float)
    ssim_score = ssim(sig_a, sig_b)

    # 7. Curvature regularization
    curvature = curvature_regularization(W)

    # 8. Assemble metric
    M = w_f * ssim_score + w_h * H * weighted_sum + lambda_reg * curvature
    return M


# ----------------------------------------------------------------------
# Additional helper demonstrating hybrid sub‑components
# ----------------------------------------------------------------------
def decision_hygiene_vector(text: str) -> np.ndarray:
    """
    Return a feature vector (ordered by CATEGORY_REGEXES) of binary flags
    indicating presence (1) or absence (0) of each category in *text*.
    """
    counts = extract_feature_counts(text)
    return np.array([1 if counts[name] > 0 else 0 for name in CATEGORY_REGEXES], dtype=float)


def fisher_weighted_ssim(text_a: str,
                         text_b: str,
                         theta: float,
                         center: float,
                         width: float) -> float:
    """
    Compute w_f·SSIM where w_f is the Fisher‑information weight.
    """
    I = fisher_score(theta, center, width)
    w_f = I / (I + 1e-12)
    min_len = min(len(text_a), len(text_b))
    if min_len == 0:
        raise ValueError("texts must not be empty")
    sig_a = np.frombuffer(text_a[:min_len].encode("utf-8"), dtype=np.uint8).astype(float)
    sig_b = np.frombuffer(text_b[:min_len].encode("utf-8"), dtype=np.uint8).astype(float)
    return w_f * ssim(sig_a, sig_b)


def entropy_weighted_feature_score(text: str) -> float:
    """
    Compute w_h·H·Σ_i w_i·f_i for a single text.
    """
    counts = extract_feature_counts(text)
    H = shannon_entropy(counts)
    w_h = H / (H + 1e-12)
    flags = {k: (1 if v > 0 else 0) for k, v in counts.items()}
    weighted_sum = sum(counts[k] * flags[k] for k in counts)
    return w_h * H * weighted_sum


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example texts
    txt1 = (
        "We need to verify the source, plan the next steps, and ensure the "
        "boundary is respected. The outcome must be verified and documented."
    )
    txt2 = (
        "First, check the evidence, then create a checklist. Avoid any impulsive "
        "actions. The final result should be shipped and verified."
    )

    # Gaussian beam parameters (arbitrary but plausible)
    theta_val = 0.7
    center_val = 0.5
    width_val = 0.2

    # Random weight matrix for curvature regularizer (5×5)
    rng = np.random.default_rng(42)
    W_mat = rng.random((5, 5))
    # Make it symmetric to resemble an undirected graph adjacency matrix
    W_mat = (W_mat + W_mat.T) / 2

    metric = unified_metric(
        text_a=txt1,
        text_b=txt2,
        theta=theta_val,
        center=center_val,
        width=width_val,
        W=W_mat,
        lambda_reg=0.05,
    )
    print(f"Unified hybrid metric M = {metric:.6f}")

    # Demonstrate the auxiliary helpers
    vec = decision_hygiene_vector(txt1)
    print("Decision‑hygiene binary vector:", vec)

    fw_ssim = fisher_weighted_ssim(txt1, txt2, theta_val, center_val, width_val)
    print(f"Fisher‑weighted SSIM = {fw_ssim:.6f}")

    eh_score = entropy_weighted_feature_score(txt1)
    print(f"Entropy‑weighted feature score = {eh_score:.6f}")