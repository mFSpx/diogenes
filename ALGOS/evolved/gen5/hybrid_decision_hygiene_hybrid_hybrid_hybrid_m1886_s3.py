# DARWIN HAMMER — match 1886, survivor 3
# gen: 5
# parent_a: decision_hygiene.py (gen0)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_hybrid_m207_s1.py (gen4)
# born: 2026-05-29T23:39:28Z

"""Hybrid Decision Hygiene + Capybara Optimization (Hybrid Algorithm)

Parents:
- decision_hygiene.py (Algorithm A): regex‑based feature counting and deterministic scoring.
- hybrid_hybrid_hybrid_decisi_hybrid_hybrid_m207_s1.py (Algorithm B): uses Shannon entropy of the same feature counts as a signal for a radial‑basis‑function (RBF) surrogate that drives a Capybara‑style optimization step.

Mathematical bridge:
Both parents expose the *same* feature‑count vector `c = (c₁,…,c₉)`.  
Algorithm A maps `c → sₐ = w·c` (a weighted linear score).  
Algorithm B maps `c → H(c) = -∑ pᵢ log₂ pᵢ` where `pᵢ = cᵢ / Σc`.  
The hybrid treats the entropy `H(c)` as a *modulation* factor for an RBF surrogate
`f(x) = Σ αⱼ·exp(-γ‖x-µⱼ‖²)` with `x = [sₐ, H(c)]`.  
Thus the final decision score is  

    S_hybrid = sₐ + H(c)·f([sₐ, H(c)])

which fuses deterministic hygiene scoring with uncertainty‑driven optimization.

The implementation below provides:
1. `counts(text)` – identical to Algorithm A.
2. `linear_score(c)` – weighted sum (Algorithm A core).
3. `shannon_entropy(c)` – entropy of the count distribution.
4. `rbf_predict(x, centers, alphas, gamma)` – RBF surrogate (Algorithm B core).
5. `hybrid_decision_score(text, centers, alphas, gamma)` – full fused computation.

All code is pure Python 3 with only the allowed standard‑library modules and NumPy."""

import re
import math
import random
import sys
import pathlib
import numpy as np
from typing import Dict, Tuple, List

# ----------------------------------------------------------------------
# Regex patterns – shared with both parent algorithms
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)
SUPPORT_RE = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)
OUTCOME_RE = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)
IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I)
SCARCITY_RE = re.compile(r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b", re.I)
RISK_RE = re.compile(r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b", re.I)

# ----------------------------------------------------------------------
# 1. Feature counting (identical to Algorithm A)
# ----------------------------------------------------------------------
def counts(text: str) -> Dict[str, int]:
    """Return a dictionary of keyword‑category counts extracted from *text*."""
    return {
        "evidence_count": len(EVIDENCE_RE.findall(text or "")),
        "planning_count": len(PLANNING_RE.findall(text or "")),
        "delay_count":    len(DELAY_RE.findall(text or "")),
        "support_count":  len(SUPPORT_RE.findall(text or "")),
        "boundary_count": len(BOUNDARY_RE.findall(text or "")),
        "outcome_count":  len(OUTCOME_RE.findall(text or "")),
        "impulsive_count":len(IMPULSIVE_RE.findall(text or "")),
        "scarcity_count": len(SCARCITY_RE.findall(text or "")),
        "risk_count":     len(RISK_RE.findall(text or "")),
    }

# ----------------------------------------------------------------------
# 2. Linear weighted score (core of Algorithm A)
# ----------------------------------------------------------------------
# Weights are chosen to reflect the relative importance of each category.
_WEIGHTS = {
    "evidence_count":  1600,
    "planning_count":  1200,
    "delay_count":    -800,
    "support_count":   500,
    "boundary_count": 1000,
    "outcome_count":  1500,
    "impulsive_count":-2000,
    "scarcity_count": -1500,
    "risk_count":     -3000,
}

def linear_score(c: Dict[str, int]) -> int:
    """Compute the deterministic hygiene score `sₐ = w·c`."""
    return sum(_WEIGHTS[k] * v for k, v in c.items())

# ----------------------------------------------------------------------
# 3. Shannon entropy of the count distribution (Algorithm B)
# ----------------------------------------------------------------------
def shannon_entropy(c: Dict[str, int]) -> float:
    """Return H(c) = -∑ pᵢ log₂ pᵢ where pᵢ = cᵢ / Σc."""
    total = sum(c.values())
    if total == 0:
        return 0.0
    probs = [v / total for v in c.values() if v > 0]
    return -sum(p * math.log2(p) for p in probs)

# ----------------------------------------------------------------------
# 4. Radial‑basis‑function surrogate (Algorithm B core)
# ----------------------------------------------------------------------
def rbf_kernel(x: np.ndarray, mu: np.ndarray, gamma: float) -> float:
    """Gaussian RBF kernel exp(-γ‖x‑μ‖²)."""
    diff = x - mu
    return math.exp(-gamma * np.dot(diff, diff))

def rbf_predict(
    x: np.ndarray,
    centers: np.ndarray,
    alphas: np.ndarray,
    gamma: float,
) -> float:
    """
    Predict f(x) = Σ_j α_j · exp(-γ‖x‑μ_j‖²).

    *x* – 2‑D feature vector `[sₐ, H]`.
    *centers* – shape (m, 2) array of RBF centers μ_j.
    *alphas* – shape (m,) array of learned amplitudes.
    *gamma* – width parameter (scalar > 0).
    """
    kernels = np.array([rbf_kernel(x, mu, gamma) for mu in centers])
    return float(np.dot(alphas, kernels))

# ----------------------------------------------------------------------
# 5. Full hybrid decision score
# ----------------------------------------------------------------------
def hybrid_decision_score(
    text: str,
    centers: np.ndarray,
    alphas: np.ndarray,
    gamma: float = 1e-5,
) -> Tuple[float, Dict[str, int]]:
    """
    Compute the fused score `S_hybrid = sₐ + H·f([sₐ, H])`.

    Returns a tuple (final_score, raw_counts_dict).
    """
    c = counts(text)
    s_a = linear_score(c)
    H = shannon_entropy(c)

    # Feature vector for the RBF model
    x = np.array([float(s_a), H])

    # RBF surrogate prediction
    f_x = rbf_predict(x, centers, alphas, gamma)

    final = s_a + H * f_x
    return final, c

# ----------------------------------------------------------------------
# Helper to generate a tiny synthetic RBF model (for demonstration)
# ----------------------------------------------------------------------
def _generate_dummy_rbf_model(num_centers: int = 5, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Produce deterministic dummy RBF parameters:
      - centers: uniformly sampled in a plausible range for [sₐ, H].
      - alphas: small random amplitudes.
      - gamma: a modest width.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Rough bounds for sₐ (based on typical weights) and H (0‑~3 bits)
    s_min, s_max = -20000, 20000
    H_min, H_max = 0.0, 4.0

    centers = np.column_stack((
        np.random.uniform(s_min, s_max, size=num_centers),
        np.random.uniform(H_min, H_max, size=num_centers),
    ))
    alphas = np.random.uniform(-500, 500, size=num_centers)
    gamma = 1e-5  # keeps kernels broad enough for the wide sₐ range
    return centers, alphas, gamma

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "I have evidence that the plan was approved, but I need more support. "
        "We should delay the rollout until we have a proper checklist. "
        "If risk arises we must set boundaries and keep a record."
    )

    centers, alphas, gamma = _generate_dummy_rbf_model()
    score, cnts = hybrid_decision_score(sample_text, centers, alphas, gamma)

    print("Feature counts:")
    for k, v in cnts.items():
        print(f"  {k}: {v}")
    print(f"\nLinear hygiene score (sₐ): {linear_score(cnts)}")
    print(f"Shannon entropy (H): {shannon_entropy(cnts):.4f}")
    print(f"RBF surrogate output (f): {rbf_predict(np.array([linear_score(cnts), shannon_entropy(cnts)]), centers, alphas, gamma):.4f}")
    print(f"\nHybrid decision score (S_hybrid): {score:.2f}")