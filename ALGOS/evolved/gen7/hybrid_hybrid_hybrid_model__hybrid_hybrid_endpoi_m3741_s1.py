# DARWIN HAMMER — match 3741, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_model_vram_sc_hybrid_hybrid_hybrid_m1453_s3.py (gen6)
# parent_b: hybrid_hybrid_endpoint_circ_hybrid_hybrid_decisi_m189_s1.py (gen3)
# born: 2026-05-29T23:51:26Z

"""
Hybrid VRAM‑Bandit / Pheromone Scheduler with Morphology‑Aware Decision Engine

Parents:
- hybrid_hybrid_model_vram_sc_hybrid_hybrid_m1453_s3.py (Hybrid VRAM‑Bandit Scheduler)
- hybrid_hybrid_endpoint_circ_hybrid_hybrid_decisi_m189_s1.py (Hybrid Endpoint Decision Hygiene)

Mathematical Bridge:
The VRAM‑Bandit scheduler provides a *store equation* that evolves a scalar
resource store S(t) from inflow I and outflow O:

    S_{t+dt} = max(0, S_t + (α·I − β·O)·dt)

The pheromone subsystem supplies a probability vector ϕ(t) that decays
exponentially:

    ϕ_{t+dt} = ϕ_t · exp(−λ·dt)

The decision module yields a raw feature vector v∈ℝ⁹ from textual counts and
produces a score vector s = W⁺·v − W⁻·v, where W⁺,W⁻∈ℝ⁹ˣ⁹ are fixed
weight matrices.

We fuse these structures by:
1. Computing the KL‑divergence D_KL(ϕ_t‖ϕ_{t+dt}) which quantifies the
   informational loss of pheromone decay.
2. Embedding this divergence into the store dynamics as a multiplicative
   factor: 𝕀 = S_{t+dt}·(1 + D_KL).
3. Scaling the decision scores by the *recovery priority* p (derived from
   morphology) and by the normalized information metric 𝕀̂ = 𝕀 / (𝕀+ε).

The resulting hybrid score vector

    ŝ = p·𝕀̂·s

inherits temporal relevance (through S and ϕ), resource‑allocation confidence
(through the bandit inflow/outflow), and morphology‑aware decision weighting.
"""

import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Any

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2] if pathlib.Path(__file__).exists() else pathlib.Path(".")

@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # interpreted as inflow rate
    expected_reward: float
    confidence_bound: float    # interpreted as outflow rate
    algorithm: str


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


# ----------------------------------------------------------------------
# Parent‑A core: store dynamics & pheromone decay
# ----------------------------------------------------------------------
def store_equation(inflow: float, outflow: float, store: float,
                   alpha: float, beta: float, dt: float) -> float:
    """VRAM‑Bandit store update."""
    delta = alpha * inflow - beta * outflow
    return max(0.0, store + delta * dt)


def pheromone_decay(pheromone: np.ndarray, decay_rate: float, dt: float) -> np.ndarray:
    """Exponential decay of a pheromone probability vector."""
    if decay_rate < 0:
        raise ValueError("decay_rate must be non‑negative")
    factor = np.exp(-decay_rate * dt)
    decayed = pheromone * factor
    # Renormalize to keep a proper probability distribution
    total = decayed.sum()
    return decayed / total if total > 0 else decayed


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL‑divergence D_KL(p‖q) for discrete distributions."""
    eps = 1e-12
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


# ----------------------------------------------------------------------
# Parent‑B core: morphology priority & decision scoring
# ----------------------------------------------------------------------
def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology,
                        b: float = 1.0 / 3.0,
                        k: float = 0.35,
                        neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Maps righting‑time index to a unit interval [0,1]."""
    rti = righting_time_index(m)
    return max(0.0, min(1.0, rti / max_index))


def extract_feature_vector(text: str) -> np.ndarray:
    """
    Very lightweight regex‑based feature extraction.
    Returns a fixed‑size vector v∈ℝ⁹ where each component counts occurrences
    of a simple pattern.
    """
    patterns = [
        r"\bthe\b", r"\band\b", r"\bof\b",
        r"\bto\b", r"\b[a-z]{5,}\b", r"\d+",
        r"[A-Z][a-z]+", r"\b\w{2}\b", r"\b\w{3,}\b"
    ]
    counts = []
    lower = text.lower()
    for pat in patterns:
        counts.append(len([m for m in __import__("re").finditer(pat, lower)]))
    return np.array(counts, dtype=float)


# Fixed weight matrices (deterministic for reproducibility)
np.random.seed(0)
W_POS = np.random.uniform(0.0, 1.0, (9, 9))
W_NEG = np.random.uniform(0.0, 1.0, (9, 9))


def score_vector_from_features(v: np.ndarray) -> np.ndarray:
    """Linear transformation s = W⁺·v − W⁻·v."""
    return W_POS @ v - W_NEG @ v


def shannon_entropy(vec: np.ndarray) -> float:
    """Entropy of the normalized absolute values of a vector."""
    probs = np.abs(vec)
    total = probs.sum()
    if total == 0:
        return 0.0
    probs = probs / total
    eps = 1e-12
    return -float(np.sum(probs * np.log(probs + eps)))


# ----------------------------------------------------------------------
# Hybrid operations (bridge between the two parents)
# ----------------------------------------------------------------------
def hybrid_information_metric(pheromone: np.ndarray,
                              decay_rate: float,
                              dt: float,
                              inflow: float,
                              outflow: float,
                              store: float,
                              alpha: float = 1.0,
                              beta: float = 1.0) -> float:
    """
    Combines pheromone KL‑divergence with the VRAM‑Bandit store update.
    Returns a scalar information metric I.
    """
    decayed = pheromone_decay(pheromone, decay_rate, dt)
    D = kl_divergence(pheromone, decayed)          # informational loss
    S_next = store_equation(inflow, outflow, store, alpha, beta, dt)
    I = S_next * (1.0 + D)                         # embed divergence multiplicatively
    return I


def hybrid_decision_score(text: str,
                          morphology: Morphology,
                          bandit: BanditAction,
                          pheromone: np.ndarray,
                          decay_rate: float,
                          dt: float,
                          store: float) -> Dict[str, Any]:
    """
    End‑to‑end hybrid scoring:
    1. Extract raw features v from text.
    2. Transform to score vector s.
    3. Compute recovery priority p from morphology.
    4. Compute information metric I from pheromone & bandit dynamics.
    5. Scale s by p·Î where Î = I / (I + ε).
    6. Return scaled scores, entropy, and auxiliary diagnostics.
    """
    # 1. Feature extraction
    v = extract_feature_vector(text)

    # 2. Linear scoring
    s = score_vector_from_features(v)

    # 3. Morphology‑derived priority
    p = recovery_priority(morphology)

    # 4. Information metric (using bandit propensity/outflow as inflow/outflow)
    I = hybrid_information_metric(
        pheromone=pheromone,
        decay_rate=decay_rate,
        dt=dt,
        inflow=bandit.propensity,
        outflow=bandit.confidence_bound,
        store=store,
        alpha=1.0,
        beta=1.0
    )

    # 5. Normalized scaling factor
    eps = 1e-9
    I_hat = I / (I + eps)
    scale = p * I_hat
    s_hat = scale * s

    # 6. Entropy of the scaled scores
    entropy = shannon_entropy(s_hat)

    return {
        "raw_features": v.tolist(),
        "raw_scores": s.tolist(),
        "scaled_scores": s_hat.tolist(),
        "recovery_priority": p,
        "information_metric": I,
        "scale_factor": scale,
        "entropy": entropy
    }


def update_system_state(pheromone: np.ndarray,
                        decay_rate: float,
                        dt: float,
                        store: float,
                        bandit: BanditAction) -> Dict[str, Any]:
    """
    Performs a single simulation step:
    - Decays pheromone distribution.
    - Updates VRAM store using bandit inflow/outflow.
    Returns the new pheromone vector and store value.
    """
    new_pheromone = pheromone_decay(pheromone, decay_rate, dt)
    new_store = store_equation(
        inflow=bandit.propensity,
        outflow=bandit.confidence_bound,
        store=store,
        alpha=1.0,
        beta=1.0,
        dt=dt
    )
    return {
        "pheromone": new_pheromone,
        "store": new_store
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic initial state
    init_pheromone = np.full(5, 1.0 / 5)          # uniform distribution over 5 slots
    decay = 0.2
    timestep = 0.1
    init_store = 1024.0                           # MB

    # Example bandit action
    bandit = BanditAction(
        action_id="act-001",
        propensity=0.8,          # inflow rate
        expected_reward=1.2,
        confidence_bound=0.3,    # outflow rate
        algorithm="epsilon-greedy"
    )

    # Example morphology
    morph = Morphology(length=0.45, width=0.30, height=0.20, mass=12.0)

    # Example textual input
    sample_text = (
        "The quick brown fox jumps over the lazy dog and runs to the river. "
        "Numbers like 123 and 456 appear, as do capitalized Words."
    )

    # Run hybrid decision scoring
    result = hybrid_decision_score(
        text=sample_text,
        morphology=morph,
        bandit=bandit,
        pheromone=init_pheromone,
        decay_rate=decay,
        dt=timestep,
        store=init_store
    )

    print("Hybrid Decision Result:")
    for k, v in result.items():
        print(f"{k}: {v}")

    # Perform a system update step
    state = update_system_state(
        pheromone=init_pheromone,
        decay_rate=decay,
        dt=timestep,
        store=init_store,
        bandit=bandit
    )
    print("\nUpdated System State:")
    print(f"Pheromone distribution: {state['pheromone']}")
    print(f"Store (MB): {state['store']:.2f}")