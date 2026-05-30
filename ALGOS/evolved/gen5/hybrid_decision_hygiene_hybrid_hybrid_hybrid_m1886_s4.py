# DARWIN HAMMER — match 1886, survivor 4
# gen: 5
# parent_a: decision_hygiene.py (gen0)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_hybrid_m207_s1.py (gen4)
# born: 2026-05-29T23:39:28Z

"""Hybrid Decision Hygiene & Capybara Optimization.

Parents:
- decision_hygiene.py (Algorithm A) – provides deterministic regex‑based feature
  counts and a linear scoring scheme.
- hybrid_hybrid_hybrid_decisi_hybrid_hybrid_m207_s1.py (Algorithm B) – uses
  Shannon entropy of the same feature counts to modulate a radial‑basis‑function
  (RBF) surrogate model that drives a Capybara‑style optimization loop.

Mathematical bridge:
The feature count vector **c** ∈ ℝ⁹ (evidence, planning, …, risk) produced by
Algorithm A is first normalised to a probability distribution **p** = c / Σc.
Its Shannon entropy H(p) = − Σ pᵢ log₂ pᵢ quantifies uncertainty in the decision
hygiene profile.  This scalar H(p) is then employed as a multiplicative
modulation factor for the linear score Sₗᵢₙₑₐᵣ derived from A and as an additive bias
for the RBF surrogate output S_RBF:

    S_hybrid = (1 + H(p)) · S_linear + S_RBF .

Thus the entropy acts as a “bridge” linking the deterministic hygiene scoring
with the probabilistic optimisation dynamics of the Capybara algorithm.  The
RBF surrogate uses centres **C** ∈ ℝⁿˣ⁹, weights **w** ∈ ℝⁿ and a kernel width
γ > 0:

    K(x, c_j) = exp(−γ ‖x − c_j‖²)
    S_RBF   = Σ_j w_j K(x, c_j)

The resulting module supplies a self‑contained hybrid scoring system.
"""

import re
import math
import random
import sys
import pathlib
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Regex patterns – identical to those in decision_hygiene.py
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|"
    r"criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|"
    r"first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|"
    r"support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|"
    r"safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|"
    r"closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|"
    r"destroy|revenge|scorched|burn it|quit|give up)\b",
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


# ----------------------------------------------------------------------
# 1️⃣ Feature counting (Algorithm A core)
# ----------------------------------------------------------------------
def counts(text: str) -> Dict[str, int]:
    """Return a dictionary of regex‑based token counts."""
    return {
        "evidence_count": len(EVIDENCE_RE.findall(text or "")),
        "planning_count": len(PLANNING_RE.findall(text or "")),
        "delay_count": len(DELAY_RE.findall(text or "")),
        "support_count": len(SUPPORT_RE.findall(text or "")),
        "boundary_count": len(BOUNDARY_RE.findall(text or "")),
        "outcome_count": len(OUTCOME_RE.findall(text or "")),
        "impulsive_count": len(IMPULSIVE_RE.findall(text or "")),
        "scarcity_count": len(SCARCITY_RE.findall(text or "")),
        "risk_count": len(RISK_RE.findall(text or "")),
    }


def linear_score(c: Dict[str, int]) -> float:
    """
    Deterministic linear score mirroring the spirit of the original
    `score_features` from decision_hygiene.py.
    Positive categories are weighted heavily; negative/impulsive categories are
    penalised.
    """
    # Positive weights (empirical, similar magnitude to the original)
    pos = (
        c["evidence_count"] * 1600
        + c["planning_count"] * 800
        + c["support_count"] * 500
        + c["boundary_count"] * 1200
        + c["outcome_count"] * 1000
    )
    # Negative/risk weights
    neg = (
        c["impulsive_count"] * 2000
        + c["scarcity_count"] * 1500
        + c["risk_count"] * 2500
        + c["delay_count"] * 300
    )
    return pos - neg


# ----------------------------------------------------------------------
# 2️⃣ Shannon entropy bridge (Algorithm B core)
# ----------------------------------------------------------------------
def shannon_entropy(count_vec: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute base‑2 Shannon entropy of a non‑negative vector.
    The vector is first normalised to a probability distribution.
    """
    total = count_vec.sum()
    if total == 0:
        return 0.0
    p = count_vec / total
    # Guard against log(0) by adding eps
    return -np.sum(p * np.log2(p + eps))


# ----------------------------------------------------------------------
# 3️⃣ Radial‑Basis‑Function surrogate model
# ----------------------------------------------------------------------
def rbf_kernel(x: np.ndarray, c: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gaussian RBF kernel between a single vector x and an array of centres c.
    Returns a 1‑D array of kernel values.
    """
    diff = c - x  # shape (n_centers, n_features)
    sq_norm = np.einsum("ij,ij->i", diff, diff)
    return np.exp(-gamma * sq_norm)


def rbf_predict(
    x: np.ndarray,
    centres: np.ndarray,
    weights: np.ndarray,
    gamma: float,
) -> float:
    """
    Predict a scalar output for input vector x using a weighted sum of RBF kernels.
    centres.shape == (n_centers, n_features)
    weights.shape == (n_centers,)
    """
    kernels = rbf_kernel(x, centres, gamma)  # (n_centers,)
    return float(np.dot(weights, kernels))


# ----------------------------------------------------------------------
# 4️⃣ Hybrid scoring function – the mathematical fusion
# ----------------------------------------------------------------------
def hybrid_score(
    text: str,
    centres: np.ndarray,
    weights: np.ndarray,
    gamma: float = 0.1,
) -> float:
    """
    Compute the fused decision‑hygiene / Capybara score.

    Steps:
    1. Extract raw counts (Algorithm A).
    2. Form a numeric vector `v` (order matches the regex dictionary).
    3. Compute linear score S_linear.
    4. Compute Shannon entropy H of `v`.
    5. Compute RBF surrogate output S_rbf.
    6. Fuse via   S_hybrid = (1 + H) * S_linear + S_rbf
    """
    c = counts(text)

    # Preserve ordering consistent with the regex definitions
    order = [
        "evidence_count",
        "planning_count",
        "delay_count",
        "support_count",
        "boundary_count",
        "outcome_count",
        "impulsive_count",
        "scarcity_count",
        "risk_count",
    ]
    vec = np.array([c[k] for k in order], dtype=float)

    s_linear = linear_score(c)
    entropy = shannon_entropy(vec)

    s_rbf = rbf_predict(vec, centres, weights, gamma)

    return (1.0 + entropy) * s_linear + s_rbf


# ----------------------------------------------------------------------
# 5️⃣ Utility: generate a random surrogate model (for demo / testing)
# ----------------------------------------------------------------------
def random_surrogate(
    n_centers: int = 5,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Produce random centres, weights and a gamma value suitable for the hybrid
    model.  The random state is optional for reproducibility.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, 5, size=(n_centers, 9))  # 9 features
    weights = rng.uniform(-1, 1, size=n_centers)
    gamma = rng.uniform(0.05, 0.2)
    return centres, weights, gamma


# ----------------------------------------------------------------------
# 6️⃣ Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "I have evidence that the plan was approved, but I feel impulsive and scared. "
        "We need support from a friend and a clear outcome. "
        "There is a risk of delay because of scarce resources."
    )

    centres, weights, gamma = random_surrogate(n_centers=7, seed=42)

    score = hybrid_score(sample_text, centres, weights, gamma)

    print(f"Hybrid score for sample text: {score:.3f}")

    # Verify that entropy is in a sensible range (0‑~3.2 for 9 categories)
    c = counts(sample_text)
    vec = np.array(
        [
            c["evidence_count"],
            c["planning_count"],
            c["delay_count"],
            c["support_count"],
            c["boundary_count"],
            c["outcome_count"],
            c["impulsive_count"],
            c["scarcity_count"],
            c["risk_count"],
        ],
        dtype=float,
    )
    print(f"Feature vector: {vec}")
    print(f"Shannon entropy: {shannon_entropy(vec):.4f}")
    print(f"Linear score: {linear_score(c):.3f}")
    print(f"RBF contribution: {rbf_predict(vec, centres, weights, gamma):.3f}")