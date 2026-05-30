# DARWIN HAMMER — match 2557, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m480_s0.py (gen4)
# born: 2026-05-29T23:42:51Z

"""Hybrid Decision‑Hygiene & Morphology Uncertainty Fusion

Parents:
- hybrid_hybrid_hybrid_decisi_hybrid_hard_truth_ma_m167_s2.py (Algorithm A)
- hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m480_s0.py (Algorithm B)

Mathematical bridge:
Algorithm A produces a discrete feature‑count vector **x** ∈ ℝⁿ from
regex‑based hygiene and stylometry cues.  
Algorithm B defines a continuous state‑space model whose prior covariance
Σ₀ is shaped by physical morphology (sphericity, flatness, righting‑time)
and whose observation noise σ² is driven by epistemic certainty flags.

The hybrid treats **x** as an observation of the latent state **θ**.
A single Kalman‑like update yields the posterior mean  

    θ̂ = K x , K = Σ₀ (Σ₀ + σ²I)⁻¹  

and posterior covariance  

    Σ̂ = (I – K) Σ₀ .

The posterior mean is then combined with morphology‑derived scalars to
produce a unified hybrid score and cost.  This fuses the discrete
topology of regex‑extracted features with the continuous uncertainty
quantification of the RLCT‑based system."""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Algorithm A – regex feature extraction (simplified)
# ----------------------------------------------------------------------
EVIDENCE_RE = __import__("re").compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    __import__("re").I,
)
PLANNING_RE = __import__("re").compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|"
    r"triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    __import__("re").I,
)

# Minimal stylometry categories (full set omitted for brevity)
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set("about above across after against among around at before behind below beneath beside between beyond by".split()),
}


def _tokenize(text: str) -> List[str]:
    """Very simple whitespace tokenizer."""
    return text.lower().split()


def extract_feature_vector(text: str) -> np.ndarray:
    """
    Returns a feature vector whose entries are:
    0 – count of evidence‑type tokens,
    1 – count of planning‑type tokens,
    2… – counts of stylometry function‑category tokens.
    """
    tokens = _tokenize(text)
    ev_cnt = sum(1 for t in tokens if EVIDENCE_RE.fullmatch(t))
    pl_cnt = sum(1 for t in tokens if PLANNING_RE.fullmatch(t))

    cat_counts = []
    for cat, vocab in FUNCTION_CATS.items():
        cat_counts.append(sum(1 for t in tokens if t in vocab))

    vec = np.array([ev_cnt, pl_cnt] + cat_counts, dtype=float)
    return vec


# ----------------------------------------------------------------------
# Algorithm B – morphology and epistemic uncertainty
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = (
    "FACT",
    "PROBABLE",
    "POSSIBLE",
    "BULLSHIT",
    "SURE_MAYBE",
)


class Morphology:
    """Simple container for 3‑D object dimensions and mass."""

    __slots__ = ("length", "width", "height", "mass")

    def __init__(self, length: float, width: float, height: float, mass: float):
        if min(length, width, height, mass) <= 0:
            raise ValueError("All morphology parameters must be positive")
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric sphericity (unitless, ≤ 1)."""
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)


def flatness_index(length: float, width: float, height: float) -> float:
    """Flatness ratio, larger → flatter object."""
    return (length + width) / (2.0 * height)


def righting_time_index(morph: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    """A heuristic time scale derived from morphology."""
    fi = flatness_index(morph.length, morph.width, morph.height)
    return (morph.mass ** b) * math.exp(k * fi) / neck_lever


def epistemic_noise_variance(flag: str) -> float:
    """
    Maps an epistemic certainty flag to an observation noise variance.
    Lower variance → higher confidence.
    """
    mapping = {
        "FACT": 0.5,
        "PROBABLE": 1.0,
        "POSSIBLE": 2.0,
        "SURE_MAYBE": 4.0,
        "BULLSHIT": 8.0,
    }
    return mapping.get(flag.upper(), 2.0)


# ----------------------------------------------------------------------
# Hybrid core – Kalman‑style fusion of discrete features and continuous morphology
# ----------------------------------------------------------------------
def prior_covariance(morph: Morphology, dim: int) -> np.ndarray:
    """
    Constructs a diagonal prior covariance Σ₀ whose scale is driven by
    morphology sphericity and flatness.  The diagonal entries are identical,
    reflecting an isotropic prior on the feature space.
    """
    sph = sphericity_index(morph.length, morph.width, morph.height)
    flat = flatness_index(morph.length, morph.width, morph.height)
    scale = 1.0 + (1.0 - sph) * 2.0 + flat * 0.5  # heuristic scaling
    return np.eye(dim) * scale


def hybrid_posterior(text: str, morph: Morphology, flag: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs a single‑step Kalman‑like update.
    Returns posterior mean vector θ̂ and posterior covariance Σ̂.
    """
    x = extract_feature_vector(text)               # observation vector
    dim = x.shape[0]

    Σ0 = prior_covariance(morph, dim)               # prior covariance
    σ2 = epistemic_noise_variance(flag)            # observation noise variance
    R = np.eye(dim) * σ2                            # observation noise matrix

    # Kalman gain K = Σ0 (Σ0 + R)⁻¹
    K = Σ0 @ np.linalg.inv(Σ0 + R)

    # Posterior mean and covariance
    theta_hat = K @ x
    Σ_hat = (np.eye(dim) - K) @ Σ0

    return theta_hat, Σ_hat


def hybrid_score(text: str, morph: Morphology, flag: str) -> float:
    """
    Computes a scalar hybrid score:
        score = ||θ̂||₂ × sphericity × exp(‑righting_time/τ)
    where τ is a tunable time constant.
    """
    theta_hat, _ = hybrid_posterior(text, morph, flag)
    norm = np.linalg.norm(theta_hat)
    sph = sphericity_index(morph.length, morph.width, morph.height)
    rt = righting_time_index(morph)
    τ = 10.0  # time‑scale constant
    return norm * sph * math.exp(-rt / τ)


def hybrid_cost(text: str, morph: Morphology, flag: str) -> float:
    """
    Hybrid free‑energy‑like cost:
        cost = 0.5 * (log det Σ̂ + trace(Σ̂⁻¹ Σ0) + ||θ̂||² / σ²)
    Lower cost indicates better agreement between discrete evidence and the
    continuous morphological prior under the given epistemic certainty.
    """
    theta_hat, Σ_hat = hybrid_posterior(text, morph, flag)
    Σ0 = prior_covariance(morph, theta_hat.shape[0])
    σ2 = epistemic_noise_variance(flag)

    # Numerical safety: add tiny epsilon to diagonal before log‑det
    eps = 1e-12
    logdet = math.log(np.linalg.det(Σ_hat + np.eye(Σ_hat.shape[0]) * eps) + eps)

    inv_Σ_hat = np.linalg.inv(Σ_hat + np.eye(Σ_hat.shape[0]) * eps)
    trace_term = np.trace(inv_Σ_hat @ Σ0)

    quad_term = (theta_hat @ theta_hat) / σ2

    return 0.5 * (logdet + trace_term + quad_term)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The audit confirmed the evidence and source. "
        "We plan the next steps and schedule the timeline. "
        "I think we should verify the hash and screenshot."
    )
    sample_morph = Morphology(length=2.5, width=1.8, height=0.9, mass=3.2)
    for flag in ("FACT", "PROBABLE", "BULLSHIT"):
        vec = extract_feature_vector(sample_text)
        print(f"Feature vector ({flag}):", vec)

        theta, cov = hybrid_posterior(sample_text, sample_morph, flag)
        print(f"Posterior mean ({flag}):", theta)
        print(f"Posterior covariance diagonal ({flag}):", np.diag(cov))

        sc = hybrid_score(sample_text, sample_morph, flag)
        ct = hybrid_cost(sample_text, sample_morph, flag)
        print(f"Hybrid score ({flag}): {sc:.4f}")
        print(f"Hybrid cost ({flag}): {ct:.4f}")
        print("-" * 40)

    sys.exit(0)