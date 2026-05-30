# DARWIN HAMMER — match 5492, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_nlms_hybrid_h_hybrid_endpoint_circ_m715_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2724_s0.py (gen5)
# born: 2026-05-30T00:02:17Z

"""Hybrid Algorithm integrating LTC adaptive filtering with Bandit‑Cued risk assessment.

Parents:
- hybrid_hybrid_nlms_hybrid_h_hybrid_endpoint_circ_m715_s0.py (LTC update + Morphology scaling)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2724_s0.py (Bandit action handling, regex feature extraction, weighted cue analysis)

Mathematical bridge:
The Bandit‑produced propensity scalar modulates the learning‑rate (μ) and time‑constant (τ) of the LTC update,
while the LTC‑derived failure‑threshold (1/(μ·τ)) rescales the geometric Morphology (including mass).
The cue‑risk score, computed from regex‑extracted binary features and signed weight vectors,
is injected as an additive bias into the LTC prediction, thus fusing decision‑risk quantification with
continuous adaptive filtering in a single unified step.
"""

import math
import random
import sys
import pathlib
import re
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Dataclasses (borrowed from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # scalar in (0, 1] indicating confidence
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

@dataclass
class Morphology:
    length: float
    width: float
    height: float
    mass: float                # additional geometric property

# ----------------------------------------------------------------------
# Feature extraction (Parent B)
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
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 2000, 2500, 3000], dtype=np.int64)

_REGEX_MAP: Dict[str, re.Pattern] = {
    "evidence": re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I),
    "planning": re.compile(r"\b(?:plan|checklist|steps?|sequence|roadmap|schedule|timeline)\b", re.I),
    "delay":   re.compile(r"\b(?:delay|latency|wait|postpone|slow)\b", re.I),
    "support": re.compile(r"\b(?:support|help|assist|aid|backup)\b", re.I),
    "boundary":re.compile(r"\b(?:boundary|limit|threshold|cap|border)\b", re.I),
    "outcome": re.compile(r"\b(?:outcome|result|consequence|effect)\b", re.I),
    "impulsive":re.compile(r"\b(?:impulsive|rash|hasty|quick|spontaneous)\b", re.I),
    "scarcity":re.compile(r"\b(?:scarcity|rare|limited|shortage)\b", re.I),
    "risk":    re.compile(r"\b(?:risk|danger|threat|hazard|exposure)\b", re.I),
}

def extract_features(text: str) -> np.ndarray:
    """Return a binary feature vector (len=_FEATURE_ORDER) indicating presence of each cue."""
    text = text.lower()
    feats = np.zeros(len(_FEATURE_ORDER), dtype=np.int64)
    for idx, key in enumerate(_FEATURE_ORDER):
        if _REGEX_MAP[key].search(text):
            feats[idx] = 1
    return feats

def cue_risk_score(features: np.ndarray) -> float:
    """Signed risk score: positive cues add, negative cues subtract."""
    pos = np.dot(_POSITIVE_WEIGHTS, features)
    neg = np.dot(_NEGATIVE_WEIGHTS, features)
    return (pos - neg) / 1e4   # scale to a modest magnitude

# ----------------------------------------------------------------------
# LTC core (Parent A)
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    return float(np.dot(weights, x))

def ltc_update(weights: np.ndarray,
               x: np.ndarray,
               target: float,
               mu: float = 0.5,
               eps: float = 1e-9,
               tau: float = 1.0,
               beta: float = 1.0) -> Tuple[np.ndarray, float, np.ndarray]:
    """Liquid‑Time‑Constant (LTC) weight update."""
    y = predict(weights, x)
    error = target - y
    power = np.dot(x, x) + eps
    next_weights = weights + mu * error * x / power
    g_t = np.clip(y + random.random() + beta * random.random(), 0.0, 1.0)
    dxdt = -(1.0 / tau + g_t) * x + g_t * np.random.uniform(0.0, 1.0, len(x))
    return next_weights, error, dxdt

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_predict(weights: np.ndarray,
                   x: np.ndarray,
                   morphology: Morphology,
                   context_text: str) -> float:
    """
    Predict using LTC weights, then bias the output with a cue‑risk score.
    The morphology volume (length·width·height) acts as a scaling factor.
    """
    base = predict(weights, x)
    features = extract_features(context_text)
    risk_bias = cue_risk_score(features)
    volume = morphology.length * morphology.width * morphology.height
    return base + risk_bias * math.log1p(volume)

def hybrid_update(weights: np.ndarray,
                  x: np.ndarray,
                  target: float,
                  morphology: Morphology,
                  bandit_action: BanditAction,
                  mu_base: float = 0.5,
                  eps: float = 1e-9,
                  tau_base: float = 1.0,
                  beta: float = 1.0) -> Tuple[np.ndarray, float, Morphology]:
    """
    LTC update where the Bandit propensity modulates μ and τ.
    Afterwards the failure‑threshold rescales the morphology (including mass).
    """
    # Modulate learning parameters with bandit confidence
    prop = max(min(bandit_action.propensity, 1.0), 1e-3)   # keep in (0,1]
    mu = mu_base * prop
    tau = tau_base / prop

    next_weights, error, _ = ltc_update(weights, x, target, mu, eps, tau, beta)

    # Failure‑threshold derived from LTC parameters
    failure_threshold = 1.0 / (mu * tau + eps)

    # Rescale geometry
    morphology.length *= failure_threshold
    morphology.width  *= failure_threshold
    morphology.height *= failure_threshold
    morphology.mass   *= failure_threshold  # mass follows the same scaling

    return next_weights, error, morphology

def select_bandit_action(context_id: str,
                         actions: List[BanditAction]) -> BanditAction:
    """
    Upper‑Confidence‑Bound (UCB) selection: expected_reward + confidence_bound * sqrt(log(N)/n)
    For simplicity, we treat propensity as the visit count proxy.
    """
    # In a real setting N would be total pulls; we approximate with sum of propensities.
    total = sum(a.propensity for a in actions) + 1e-9
    best = None
    best_score = -float('inf')
    for a in actions:
        ucb = a.expected_reward + a.confidence_bound * math.sqrt(math.log(total) / (a.propensity + 1e-9))
        if ucb > best_score:
            best_score = ucb
            best = a
    return best

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise dummy data
    dim = 5
    w = np.zeros(dim)
    x = np.random.rand(dim)
    target = 0.7

    morph = Morphology(length=1.0, width=0.5, height=0.3, mass=2.0)

    actions = [
        BanditAction(action_id="A", propensity=0.8, expected_reward=0.5, confidence_bound=0.2, algorithm="bandit_v1"),
        BanditAction(action_id="B", propensity=0.4, expected_reward=0.6, confidence_bound=0.1, algorithm="bandit_v1"),
        BanditAction(action_id="C", propensity=0.6, expected_reward=0.4, confidence_bound=0.3, algorithm="bandit_v1"),
    ]

    # Choose action based on context (dummy text)
    context = "The evidence confirms the risk and scarcity of resources."
    chosen = select_bandit_action("ctx_001", actions)

    # Hybrid prediction before update
    pred_before = hybrid_predict(w, x, morph, context)
    print(f"Prediction before update: {pred_before:.4f}")

    # Hybrid update step
    w, err, morph = hybrid_update(w, x, target, morph, chosen)
    print(f"Update error: {err:.4f}")
    print(f"Rescaled morphology: {asdict(morph)}")

    # Prediction after update
    pred_after = hybrid_predict(w, x, morph, context)
    print(f"Prediction after update: {pred_after:.4f}")

    sys.exit(0)