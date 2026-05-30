# DARWIN HAMMER — match 4310, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m1627_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_endpoint_circ_m1225_s2.py (gen6)
# born: 2026-05-29T23:56:10Z

"""Hybrid Algorithm integrating Regret‑Weighted Decision Geometry (Parent A) 
and Morphological Sphericity‑Modulated Liquid‑Time‑Constant dynamics (Parent B).

Mathematical Bridge:
- Actions are encoded as a multivector **v = [E, C, R]** (expected value, cost, risk) 
  (Parent A).  
- The three components of **v** are interpreted as the morphological dimensions 
  (length, width, height) of a virtual object (Parent B).  
- From these dimensions the **sphericity index** 𝜎 = (L·W·H)^{1/3} / max(L,W,H) 
  is computed.  
- The sphericity 𝜎 modulates the time‑constant τ of a Liquid‑Time‑Constant (LTC) 
  update: τ = τ₀ / 𝜎.  This couples geometry to the dynamics of the regret‑weighted 
  strategy vector, yielding a hybrid state update.
- Finally, an entropy‑based hygiene score blends the Shannon entropy of the 
  regret‑weighted probabilities with the sphericity, providing a unified decision 
  metric.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple, Iterable, Callable, Dict

import numpy as np

# ----------------------------------------------------------------------
# Data structures (shared)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Atomic decision element."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class Morphology:
    """Geometric representation derived from a multivector."""
    length: float
    width: float
    height: float
    mass: float = 1.0  # default mass, can be tuned later

# ----------------------------------------------------------------------
# Parent A core utilities
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

def entropy(probabilities: List[float]) -> float:
    """Shannon entropy (base‑2)."""
    return -sum(p * math.log(p, 2) for p in probabilities if p > 0)

def regret_weighted_strategy(actions: List[MathAction]) -> np.ndarray:
    """Return normalized expected‑value weights."""
    ev = np.array([a.expected_value for a in actions], dtype=float)
    total = ev.sum()
    if total == 0:
        # avoid division by zero – uniform distribution
        return np.full_like(ev, 1.0 / len(ev))
    return ev / total

def multivector_representation(actions: List[MathAction]) -> np.ndarray:
    """Stack [expected, cost, risk] rows into a matrix."""
    return np.stack(
        [np.array([a.expected_value, a.cost, a.risk]) for a in actions],
        axis=0,
    )  # shape (n_actions, 3)

# ----------------------------------------------------------------------
# Parent B core utilities
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    """Dimensionless sphericity (0,1]."""
    if min(length, width, height) <= 0:
        raise ValueError("All dimensions must be positive.")
    geometric_mean = (length * width * height) ** (1.0 / 3.0)
    return geometric_mean / max(length, width, height)

def gaussian_beam(theta: float, center: float, width: float, sphericity: float) -> float:
    """Gaussian intensity modulated by sphericity."""
    if width <= 0:
        raise ValueError("width must be positive")
    # standard Gaussian term
    gauss = math.exp(-0.5 * ((theta - center) / width) ** 2)
    # sphericity acts as an amplitude scaler (higher sphericity → stronger beam)
    return gauss * sphericity

# ----------------------------------------------------------------------
# Hybrid bridging functions
# ----------------------------------------------------------------------
def multivector_to_morphology(multivector: np.ndarray) -> Morphology:
    """
    Convert a (n,3) multivector matrix into a single Morphology instance.
    The conversion aggregates each column by its mean and scales to a
    physically plausible size range.
    """
    if multivector.ndim != 2 or multivector.shape[1] != 3:
        raise ValueError("multivector must be of shape (n,3)")
    # Mean of each component across actions
    means = multivector.mean(axis=0)
    # Simple linear scaling to keep dimensions in a reasonable range
    scale = 1.0  # can be tuned
    length, width, height = (abs(v) * scale + 0.1 for v in means)  # avoid zero
    return Morphology(length=length, width=width, height=height)

def hybrid_ltc_update(state: np.ndarray,
                      input_vec: np.ndarray,
                      dt: float,
                      sphericity: float,
                      base_tau: float = 0.5) -> np.ndarray:
    """
    Liquid‑Time‑Constant (LTC) update where the time‑constant τ is
    inversely proportional to sphericity.

    new_state = state * exp(-dt/τ) + input_vec * (1 - exp(-dt/τ))
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if sphericity <= 0:
        raise ValueError("sphericity must be positive")
    tau = base_tau / sphericity  # higher sphericity → faster response
    decay = math.exp(-dt / tau)
    return state * decay + input_vec * (1.0 - decay)

def hybrid_decision_hygiene(actions: List[MathAction],
                            dt: float = 0.1) -> float:
    """
    Unified hygiene score:
        H = (1 - entropy(weights)) * sphericity
    where weights are regret‑weighted probabilities and sphericity is derived
    from the multivector‑to‑morphology conversion.
    """
    # 1️⃣ Regret‑weighted probabilities
    weights = regret_weighted_strategy(actions)
    # 2️⃣ Entropy of the distribution
    ent = entropy(weights.tolist())
    # 3️⃣ Multivector → Morphology → sphericity
    mv = multivector_representation(actions)
    morph = multivector_to_morphology(mv)
    sph = sphericity_index(morph.length, morph.width, morph.height)
    # 4️⃣ Combine (entropy is in [0, log2(n)], we normalise by max possible)
    max_ent = math.log(len(actions), 2) if len(actions) > 1 else 1.0
    norm_ent = ent / max_ent  # in [0,1]
    hygiene = (1.0 - norm_ent) * sph
    return hygiene

def hybrid_process_pool(actions_pool: List[List[MathAction]],
                        dt: float = 0.1) -> List[float]:
    """
    Process a pool of action sets. For each set we:
        • compute the regret‑weighted vector,
        • map it to morphology,
        • update an internal state via LTC,
        • return the resulting hygiene score.
    """
    # Initialise a dummy state vector (same size as number of actions in first set)
    if not actions_pool:
        return []
    n = len(actions_pool[0])
    state = np.zeros(n)
    scores = []
    for actions in actions_pool:
        # Regret‑weighted input vector
        inp = regret_weighted_strategy(actions)
        # Morphology & sphericity for this batch
        mv = multivector_representation(actions)
        morph = multivector_to_morphology(mv)
        sph = sphericity_index(morph.length, morph.width, morph.height)
        # LTC state update
        state = hybrid_ltc_update(state, inp, dt, sph)
        # Hygiene based on updated state (we reuse actions for entropy)
        scores.append(hybrid_decision_hygiene(actions, dt))
    return scores

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a small set of actions
    actions_set1 = [
        MathAction(id="A", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="B", expected_value=5.0, cost=1.5, risk=0.5),
        MathAction(id="C", expected_value=8.0, cost=3.0, risk=2.0),
    ]
    actions_set2 = [
        MathAction(id="D", expected_value=7.0, cost=2.5, risk=1.5),
        MathAction(id="E", expected_value=6.0, cost=2.0, risk=1.0),
        MathAction(id="F", expected_value=9.0, cost=3.5, risk=2.5),
    ]

    # Compute hybrid hygiene scores individually
    h1 = hybrid_decision_hygiene(actions_set1)
    h2 = hybrid_decision_hygiene(actions_set2)
    print(f"Hygiene score set 1: {h1:.4f}")
    print(f"Hygiene score set 2: {h2:.4f}")

    # Process a pool
    pool_scores = hybrid_process_pool([actions_set1, actions_set2])
    print("Pool hygiene scores:", [f"{s:.4f}" for s in pool_scores])

    # Demonstrate LTC update alone
    mv = multivector_representation(actions_set1)
    morph = multivector_to_morphology(mv)
    sph = sphericity_index(morph.length, morph.width, morph.height)
    state = np.zeros(len(actions_set1))
    inp = regret_weighted_strategy(actions_set1)
    new_state = hybrid_ltc_update(state, inp, dt=0.2, sphericity=sph)
    print("LTC updated state:", new_state)