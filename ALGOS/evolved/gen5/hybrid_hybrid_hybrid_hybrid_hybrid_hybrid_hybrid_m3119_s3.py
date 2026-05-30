# DARWIN HAMMER — match 3119, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_nlms_o_fisher_localization_m1155_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_regret_engine_m652_s0.py (gen4)
# born: 2026-05-29T23:48:02Z

"""
Hybrid Algorithm integrating:
- Parent A: Hybrid NLMS‑LTC Diffusion Fusion & Fisher localization (signature,
  Gaussian beam, Fisher information scoring, NLMS adaptation).
- Parent B: Morphology‑driven regret engine (SSM morphology indices,
  weighted Shannon entropy, Gini‑based weekday distribution, regret‑weighted
  action ranking).

Mathematical Bridge
-------------------
The bridge is built on a *state vector* that concatenates:
1. MinHash signature features derived from token sets (Parent A).
2. Morphology‑derived geometric indices (sphericity, flatness, righting time)
   (Parent B).

Weight adaptation for this state vector follows a Normalised Least‑Mean‑Squares
(NLMS) rule whose learning‑rate μ is *scaled* by the Fisher information score
computed from a Gaussian‑beam model (Parent A).  The adapted weights are then
used to modulate a *regret‑weighted* action valuation (Parent B).  The regret
modifier incorporates a Gini coefficient computed over a diffusion schedule,
thus coupling the diffusion schedule optimisation of Parent A with the
regret‑engine of Parent B.

The resulting system produces a single unified decision metric per action,
allowing a coherent prediction that respects both information‑theoretic
(Fisher) and morphology‑driven (regret) considerations.
"""

import math
import random
import sys
from pathlib import Path
import hashlib
import numpy as np

MAX64 = (1 << 64) - 1
EPS = 1e-12

# ----------------------------------------------------------------------
# Parent A building blocks
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on a seed and a token."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: list[str], k: int = 128) -> list[int]:
    """MinHash signature of a token set with k hash functions."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = EPS) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


# ----------------------------------------------------------------------
# Parent B building blocks
# ----------------------------------------------------------------------
class Morphology:
    __slots__ = ("length", "width", "height", "mass")
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass * (fi ** b) * k) / neck_lever


def gini_coefficient(values: np.ndarray) -> float:
    """Standard Gini coefficient for a 1‑D array of non‑negative numbers."""
    if values.ndim != 1:
        raise ValueError("values must be 1‑D")
    if np.any(values < 0):
        raise ValueError("values must be non‑negative")
    sorted_vals = np.sort(values)
    n = len(values)
    cumulative = np.cumsum(sorted_vals)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


# ----------------------------------------------------------------------
# Hybrid core
# ----------------------------------------------------------------------
def hybrid_state_vector(tokens: list[str], morphology: Morphology, k: int = 64) -> np.ndarray:
    """
    Build a joint state vector by concatenating:
    - Normalised MinHash signature (k elements)
    - Normalised morphology indices (sphericity, flatness, righting time)
    """
    sig = np.array(signature(tokens, k), dtype=np.float64)
    # Normalise signature to [0,1]
    sig = (sig - sig.min()) / (sig.max() - sig.min() + EPS)

    sph = sphericity_index(morphology.length, morphology.width, morphology.height)
    flt = flatness_index(morphology.length, morphology.width, morphology.height)
    rgt = righting_time_index(morphology)

    morph_vec = np.array([sph, flt, rgt], dtype=np.float64)
    # Normalise morphology vector (zero‑mean, unit‑var)
    if morph_vec.std() > 0:
        morph_vec = (morph_vec - morph_vec.mean()) / morph_vec.std()
    else:
        morph_vec = morph_vec - morph_vec.mean()

    return np.concatenate([sig, morph_vec])


def nlms_update(weights: np.ndarray,
                x: np.ndarray,
                d: float,
                mu: float,
                fisher_center: float,
                fisher_width: float,
                theta: float) -> np.ndarray:
    """
    NLMS weight update where the effective step size is scaled by a Fisher score.
    """
    if x.ndim != 1 or weights.shape != x.shape:
        raise ValueError("x and weights must be 1‑D vectors of the same length")
    norm_x_sq = np.dot(x, x) + EPS
    error = d - np.dot(weights, x)
    # Fisher scaling based on the current angle theta
    scale = fisher_score(theta, fisher_center, fisher_width)
    step = mu * scale / norm_x_sq
    return weights + step * x * error


def regret_weighted_score(actions: list[dict],
                          schedule: np.ndarray,
                          gini_weight: float = 0.5) -> list[float]:
    """
    Compute a regret‑adjusted score for each action.

    The base score is `expected_value - cost - risk`.  It is then multiplied by
    a schedule factor that is the normalized diffusion schedule entry.  Finally,
    the whole vector is shifted by a term proportional to the Gini coefficient
    of the schedule (higher inequality → higher regret penalty).
    """
    if schedule.ndim != 1:
        raise ValueError("schedule must be 1‑D")
    if not actions:
        return []

    # Normalise schedule to [0,1]
    sch_norm = (schedule - schedule.min()) / (schedule.max() - schedule.min() + EPS)

    base = np.array([a["expected_value"] - a.get("cost", 0.0) - a.get("risk", 0.0)
                     for a in actions], dtype=np.float64)

    # Apply schedule weighting
    weighted = base * sch_norm[: len(base)]

    # Gini‑based regret term
    gini = gini_coefficient(schedule)
    regret_term = gini_weight * gini * np.mean(base)

    return (weighted - regret_term).tolist()


def hybrid_predict(tokens: list[str],
                   morphology: Morphology,
                   actions: list[dict],
                   schedule: np.ndarray,
                   mu: float = 0.1,
                   fisher_center: float = 0.0,
                   fisher_width: float = 1.0) -> dict:
    """
    End‑to‑end hybrid prediction:
    1. Build state vector.
    2. Initialise NLMS weights (zeros).
    3. Perform a single NLMS adaptation using a synthetic desired signal
       (the mean of action base scores).
    4. Compute regret‑weighted scores.
    5. Return the action with the highest hybrid score.
    """
    state = hybrid_state_vector(tokens, morphology, k=64)

    # Initialise weights
    w = np.zeros_like(state)

    # Desired signal: average of raw action values
    d = np.mean([a["expected_value"] for a in actions])

    # Use the angle between state and a reference axis as theta
    theta = math.atan2(state[-1], state[0])  # simple proxy

    # NLMS adaptation
    w = nlms_update(w, state, d, mu, fisher_center, fisher_width, theta)

    # Project state onto adapted weights to obtain a scalar “confidence”
    confidence = np.dot(w, state)

    # Regret scores
    regret_scores = regret_weighted_score(actions, schedule)

    # Fuse confidence with regret scores (simple linear blend)
    blended = np.array(regret_scores) + confidence

    best_idx = int(np.argmax(blended))
    return actions[best_idx]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy token set
    tokens = ["alpha", "beta", "gamma", "delta", "epsilon"]

    # Example morphology
    morph = Morphology(length=2.5, width=1.2, height=0.8, mass=3.0)

    # Dummy actions
    actions = [
        {"id": "A", "expected_value": 10.0, "cost": 2.0, "risk": 1.0},
        {"id": "B", "expected_value": 8.0,  "cost": 1.0, "risk": 0.5},
        {"id": "C", "expected_value": 12.0, "cost": 3.0, "risk": 2.0},
    ]

    # Diffusion schedule (e.g., time‑varying weights)
    schedule = np.linspace(0.1, 1.0, 10)

    best_action = hybrid_predict(
        tokens=tokens,
        morphology=morph,
        actions=actions,
        schedule=schedule,
        mu=0.05,
        fisher_center=0.0,
        fisher_width=1.0,
    )
    print("Selected action:", best_action)