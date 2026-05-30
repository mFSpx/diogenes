# DARWIN HAMMER — match 2344, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hoeffd_m881_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m14_s4.py (gen3)
# born: 2026-05-29T23:41:57Z

"""Hybrid algorithm combining:
- Parent A (hybrid_hybrid_hybrid_hoeffd_m881_s0): Gaussian beam, Fisher information,
  Hoeffding bound, pheromone‑guided pruning.
- Parent B (hybrid_hybrid_hybrid_decisi_m14_s4): Regex‑based feature extraction
  and epistemic certainty flags.

Mathematical bridge:
Feature frequencies extracted from text are treated as observations θ.
Each feature i receives a Gaussian‑shaped likelihood centred at a learned
parameter μ_i (here the mean count) with width σ_i.  Fisher information
I_i(θ) quantifies the sensitivity of that likelihood.  The Hoeffding bound
ε = sqrt(R²·ln(1/δ)/(2·n)) supplies a confidence radius for the aggregated
information, while epistemic certainty weights w_f (from FLAG_CERTAINTY)
modulate the contribution of each feature.  A pheromone table stores
cumulative scores, acting as a reinforcement signal that biases future
decisions toward high‑certainty, high‑information features.
"""

import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent A core utilities
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def hoeffding_bound(range_: float, delta: float, n: int) -> float:
    """Hoeffding bound ε = sqrt( (R² * ln(1/δ)) / (2n) )."""
    if range_ <= 0 or not (0 < delta < 1) or n <= 0:
        raise ValueError("invalid arguments")
    return math.sqrt((range_ * range_ * math.log(1.0 / delta)) / (2.0 * n))


def gini_impurity(counts: List[int]) -> float:
    """Simple Gini impurity for a list of integer counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts]
    return 1.0 - sum(p * p for p in probs)


# ----------------------------------------------------------------------
# Parent B feature extraction & epistemic certainty
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
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now)\b", re.I
)
QUALITY_RE = re.compile(r"\b(?:quality|high|low|grade|rating)\b", re.I)
SECURITY_RE = re.compile(r"\b(?:security|secure|vulnerability|exploit)\b", re.I)
PERFORMANCE_RE = re.compile(r"\b(?:performance|fast|slow|latency)\b", re.I)
COMPLIANCE_RE = re.compile(r"\b(?:compliance|regulation|standard)\b", re.I)
COST_RE = re.compile(r"\b(?:cost|price|budget|expense)\b", re.I)

FEATURE_REGEXES: List[Tuple[str, re.Pattern]] = [
    ("evidence", EVIDENCE_RE),
    ("planning", PLANNING_RE),
    ("delay", DELAY_RE),
    ("quality", QUALITY_RE),
    ("security", SECURITY_RE),
    ("performance", PERFORMANCE_RE),
    ("compliance", COMPLIANCE_RE),
    ("cost", COST_RE),
    ("generic", re.compile(r"\b\w{7,}\b", re.I)),
]

EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "SURE_MAYBE", "BULLSHIT")
FLAG_CERTAINTY: Dict[str, float] = {
    "FACT": 0.95,
    "PROBABLE": 0.75,
    "POSSIBLE": 0.50,
    "SURE_MAYBE": 0.30,
    "BULLSHIT": 0.0,
}


def extract_features(text: str) -> Dict[str, int]:
    """Count occurrences of each feature regex in *text*."""
    counts: Dict[str, int] = {}
    for name, pattern in FEATURE_REGEXES:
        matches = pattern.findall(text)
        counts[name] = len(matches)
    return counts


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_fisher_per_feature(
    feature_counts: Dict[str, int],
    centers: Dict[str, float],
    widths: Dict[str, float],
) -> Dict[str, float]:
    """Compute Fisher information for each feature using its count as θ."""
    fisher_vals: Dict[str, float] = {}
    for feat, cnt in feature_counts.items():
        mu = centers.get(feat, 1.0)  # default centre
        sigma = widths.get(feat, 1.0)  # default width
        fisher_vals[feat] = fisher_score(float(cnt), mu, sigma)
    return fisher_vals


def pheromone_update(
    pheromones: Dict[str, float],
    feature_scores: Dict[str, float],
    decay: float = 0.9,
) -> None:
    """Reinforce pheromones with new scores and apply exponential decay."""
    for feat, score in feature_scores.items():
        old = pheromones.get(feat, 0.0)
        pheromones[feat] = decay * old + (1.0 - decay) * score


def hybrid_decision(
    text: str,
    delta: float = 0.05,
    range_: float = 1.0,
    epistemic_flag: str = "PROBABLE",
) -> Tuple[float, Dict[str, float]]:
    """
    End‑to‑end hybrid decision:
    1. Extract feature frequencies.
    2. Estimate per‑feature Fisher information.
    3. Combine with epistemic certainty weight.
    4. Apply Hoeffding bound to obtain a confidence‑adjusted score.
    5. Update pheromone table (global) as side‑effect.
    Returns (overall_score, per_feature_score_dict).
    """
    # 1. Feature extraction
    feats = extract_features(text)

    # 2. Parameter estimation for Gaussian beams (simple heuristics)
    centers = {k: max(v, 1) for k, v in feats.items()}          # μ_i ≈ count (avoid zero)
    widths = {k: math.sqrt(max(v, 1)) for k, v in feats.items()}  # σ_i ≈ sqrt(count)

    # 3. Fisher information per feature
    fisher_vals = compute_fisher_per_feature(feats, centers, widths)

    # 4. Epistemic certainty weight
    weight = FLAG_CERTAINTY.get(epistemic_flag, 0.5)

    # 5. Raw feature scores = weight * fisher
    raw_scores = {k: weight * v for k, v in fisher_vals.items()}

    # 6. Hoeffding bound for the whole observation set
    n_obs = max(sum(feats.values()), 1)
    epsilon = hoeffding_bound(range_, delta, n_obs)

    # 7. Adjust scores by confidence radius (higher epsilon → more conservative)
    adjusted_scores = {k: max(0.0, v - epsilon) for k, v in raw_scores.items()}

    # 8. Aggregate to a single decision metric (e.g., weighted mean)
    total_fisher = sum(fisher_vals.values()) or 1.0
    overall_score = sum(adjusted_scores.values()) / total_fisher

    # 9. Pheromone reinforcement (global mutable state)
    pheromone_update(_PHEROMONES, adjusted_scores)

    return overall_score, adjusted_scores


# Global pheromone store (acts as the "pheromone matrix" from Parent A)
_PHEROMONES: Dict[str, float] = {}


def get_pheromone_levels() -> Dict[str, float]:
    """Snapshot of current pheromone intensities."""
    return dict(_PHEROMONES)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = """
    The audit confirmed the evidence of a security breach. 
    Planning for remediation includes a detailed checklist and timeline. 
    Performance metrics will be monitored, and cost estimates are being refined.
    """
    score, per_feat = hybrid_decision(sample_text, epistemic_flag="FACT")
    print(f"Overall hybrid decision score: {score:.4f}")
    print("Per‑feature adjusted scores:")
    for f, s in per_feat.items():
        print(f"  {f}: {s:.6f}")
    print("Current pheromone levels:")
    for f, p in get_pheromone_levels().items():
        print(f"  {f}: {p:.6f}")