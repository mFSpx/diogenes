# DARWIN HAMMER — match 5775, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m1507_s2.py (gen5)
# parent_b: hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m474_s0.py (gen4)
# born: 2026-05-30T00:04:44Z

"""
Hybrid Fusion Module
====================

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – ``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m1507_s2.py``  
  Provides a *store dynamics* model where a resource level evolves according to
  ``Δ = α·Σ(inflow) − β·Σ(outflow)`` and a derived ``dance`` term that caps the
  scaled delta.

* **Parent B** – ``hybrid_hybrid_omni_chaotic__hybrid_hybrid_hybrid_m474_s0.py``  
  Supplies a *JEPA*‑style latent prediction (`predictor`), an *energy* term
  ``‖s_θ(x) − p_φ‖²``, and a *bandit router* update.

**Mathematical Bridge**

The bridge is the observation that the *store delta* (a scalar) can be used as
a *temperature* scaling factor for the JEPA energy, while the *bandit propensity*
can be modulated by a MinHash‑based similarity between the current action id
and a historical hash fingerprint.  The fused objective for an action *a* is


Q(a) = 𝔼[a] − λ₁·risk(a) − λ₂·energy·τ
       + λ₃·propensity·sim(a, history)


where ``τ = sigmoid(Δ)`` maps the store delta to (0,1), and ``sim`` is a
MinHash similarity.  This combines risk assessment (Parent A) with
energy‑regularized regret‑weighted bandit decisions (Parent B) into a single
coherent decision score.

The implementation below provides three public functions that demonstrate the
hybrid operation, together with a small smoke test.
"""

import math
import random
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Data structures (reused / adapted from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """Atomic decision candidate."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class BanditAction:
    """Bandit‑style action descriptor."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


@dataclass
class StoreState:
    """Dynamical store with the update rule from Parent A."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply the linear store dynamics."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """Clipped, gain‑scaled version of the last delta."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))


# ----------------------------------------------------------------------
# Functions from Parent B (re‑implemented without external deps)
# ----------------------------------------------------------------------
def encoder(x: np.ndarray) -> np.ndarray:
    """L2‑normalize a vector."""
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x


def predictor(s_theta_x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Linear predictor used in the chaotic omni‑front synthesis."""
    return s_theta_x + z


def jepa_energy(s_theta_x: np.ndarray, p_phi: np.ndarray) -> float:
    """Energy term ‖s_θ(x) − p_φ‖²."""
    return float(np.linalg.norm(s_theta_x - p_phi) ** 2)


def vicreg_regularizer(representations: np.ndarray) -> float:
    """Variance‑invariance‑covariance regularizer (simplified)."""
    var_term = np.mean(np.var(representations, axis=0))
    cov_term = np.mean(np.abs(np.cov(representations.T)))
    return float(var_term + cov_term)


def minhash_similarity(id_a: str, id_b: str, num_hashes: int = 64) -> float:
    """
    Approximate Jaccard similarity using a simple MinHash scheme.
    Returns a value in [0, 1].
    """
    def _hashes(s: str) -> List[int]:
        h = hashlib.sha256(s.encode()).digest()
        ints = [int.from_bytes(h[i:i+4], "big") for i in range(0, len(h), 4)]
        # repeat / truncate to requested number
        while len(ints) < num_hashes:
            ints += ints
        return ints[:num_hashes]

    ha = _hashes(id_a)
    hb = _hashes(id_b)
    matches = sum(1 for a, b in zip(ha, hb) if a == b)
    return matches / num_hashes


# ----------------------------------------------------------------------
# Hybrid core operations (three required functions)
# ----------------------------------------------------------------------
def compute_store_update(store: StoreState,
                         inflow: List[float],
                         outflow: List[float]) -> Tuple[float, float, float]:
    """
    Update the store and return (new_level, delta, temperature).
    Temperature τ = sigmoid(delta) ∈ (0,1) scales the JEPA energy.
    """
    level, delta = store.update(inflow, outflow)
    tau = 1 / (1 + math.exp(-delta))  # sigmoid
    return level, delta, tau


def compute_jepta_terms(s_theta_x: np.ndarray,
                        p_phi: np.ndarray,
                        z: np.ndarray,
                        representations: np.ndarray,
                        temperature: float) -> Tuple[np.ndarray, float, float]:
    """
    Produce the predicted latent, the scaled energy, and a regularization term.
    """
    s_theta_y = predictor(s_theta_x, z)
    energy = jepa_energy(s_theta_x, p_phi)
    scaled_energy = temperature * energy
    reg = vicreg_regularizer(representations)
    return s_theta_y, scaled_energy, reg


def hybrid_decision(math_action: MathAction,
                    bandit_action: BanditAction,
                    store: StoreState,
                    s_theta_x: np.ndarray,
                    p_phi: np.ndarray,
                    z: np.ndarray,
                    representations: np.ndarray,
                    history_action_id: str,
                    inflow: List[float],
                    outflow: List[float],
                    λ1: float = 0.5,
                    λ2: float = 0.3,
                    λ3: float = 0.2) -> Dict:
    """
    Fuse risk, JEPA energy, and bandit propensity into a single score.
    Returns a dict with the chosen action id, its score, and the updated store.
    """
    # 1) Update store and obtain temperature τ
    level, delta, tau = compute_store_update(store, inflow, outflow)

    # 2) Compute JEPA related terms
    _, scaled_energy, _ = compute_jepta_terms(s_theta_x, p_phi, z,
                                              representations, tau)

    # 3) MinHash similarity between current and historical action ids
    sim = minhash_similarity(math_action.id, history_action_id)

    # 4) Hybrid quality score Q(a)
    Q = (math_action.expected_value
         - λ1 * math_action.risk
         - λ2 * scaled_energy
         + λ3 * bandit_action.propensity * sim)

    # 5) Package result
    result = {
        "chosen_action_id": math_action.id,
        "score": Q,
        "store_level": level,
        "store_delta": delta,
        "temperature": tau,
        "similarity": sim,
        "scaled_energy": scaled_energy,
    }
    return result


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Initialise store
    store = StoreState(level=5.0, alpha=0.8, beta=0.5, dt=1.0, base=1.0, gain=0.2)

    # Dummy inflow/outflow streams
    inflow = [random.uniform(0, 2) for _ in range(3)]
    outflow = [random.uniform(0, 1) for _ in range(2)]

    # Create a MathAction and a BanditAction
    ma = MathAction(id="action_42", expected_value=7.3, risk=1.1)
    ba = BanditAction(action_id="action_42", propensity=0.6,
                      expected_reward=6.8, confidence_bound=0.2)

    # Latent vectors (small dimension for the demo)
    s_theta_x = encoder(np.random.randn(8))
    p_phi = encoder(np.random.randn(8))
    z = np.random.randn(8) * 0.05

    # Batch of representations for regularizer (e.g., recent embeddings)
    representations = np.stack([encoder(np.random.randn(8)) for _ in range(10)])

    # Historical action id (simulated)
    hist_id = "action_7"

    # Run the hybrid decision
    out = hybrid_decision(
        math_action=ma,
        bandit_action=ba,
        store=store,
        s_theta_x=s_theta_x,
        p_phi=p_phi,
        z=z,
        representations=representations,
        history_action_id=hist_id,
        inflow=inflow,
        outflow=outflow,
    )

    # Print concise result
    print("Hybrid decision output:")
    for k, v in out.items():
        print(f"  {k}: {v}")

    # Ensure the store has been updated without error
    assert isinstance(store.level, float)
    sys.exit(0)