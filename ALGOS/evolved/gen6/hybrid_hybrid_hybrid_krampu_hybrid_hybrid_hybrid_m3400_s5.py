# DARWIN HAMMER — match 3400, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_krampus_brain_hybrid_hybrid_hybrid_m524_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_pheromone_inf_m1240_s5.py (gen4)
# born: 2026-05-29T23:49:50Z

"""Hybrid Krampus‑RBF‑Pheromone Algorithm
================================================

This module fuses **Parent Algorithm A** (Krampus brain‑map → context vector → hybrid
RBF surrogate with a coboundary operator) and **Parent Algorithm B**
(weekday‑driven stochastic weighting and exponential pheromone decay).

Mathematical bridge
-------------------
Both parents operate on real‑valued vector spaces:

* A supplies a *coboundary operator* `Δ = W_rbf @ c` where `W_rbf` are RBF
  kernel weights and `c` is the Krampus brain‑map context vector.
* B supplies a *row‑stochastic weekday weight vector* `ω(d)` for a set of
  groups and an *exponential decay* `φ(t) = ½^{t / τ}` for pheromone signals.

The fusion treats `Δ` as a feature vector that is first modulated by the
weekday‑dependent stochastic matrix `Ω(d) = diag(ω(d))` and then scaled by the
pheromone decay factor `φ(t)`.  The final hybrid score is


s = φ(t) · Ω(d) · (W_rbf @ c)


where all operations are element‑wise or matrix‑vector products.  This single
expression unifies the core topologies of both parents while preserving their
interpretability.

The implementation below provides three demonstrative functions:

1. `rbf_coboundary` – builds the coboundary operator from RBF weights and context.
2. `weekday_weight_vector` – reproduces B’s sinusoidal stochastic weighting.
3. `hybrid_score` – combines the coboundary, weekday weighting and pheromone
   decay into a unified scalar or vector output suitable for downstream action
   selection.

"""

import math
import random
import sys
from pathlib import Path
import numpy as np
import datetime

# ----------------------------------------------------------------------
# Utility functions (shared)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel used by the RBF surrogate."""
    return math.exp(-((epsilon * r) ** 2))

def krampus_brainmap_context_vector(krampus_brainmap: np.ndarray) -> np.ndarray:
    """Identity mapping – the brain‑map itself is the context vector."""
    return krampus_brainmap

# ----------------------------------------------------------------------
# Parent A – RBF surrogate + coboundary
# ----------------------------------------------------------------------
def rbf_kernel(distances: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """Apply Gaussian kernel element‑wise to a distance vector."""
    return np.exp(-((epsilon * distances) ** 2))

def rbf_coboundary(
    rbf_weights: np.ndarray,
    centers: np.ndarray,
    context: np.ndarray,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Compute the coboundary operator Δ = W_rbf @ φ(||c – μ||)
    where:
        - W_rbf : (m, n) weight matrix,
        - μ      : (n, d) RBF centers,
        - c      : (d,) context vector,
        - φ      : Gaussian kernel applied to Euclidean distances.
    Returns a vector of shape (m,).
    """
    # distances: (n,) – Euclidean distance from context to each centre
    diff = centers - context  # broadcast (n, d) - (d,) -> (n, d)
    distances = np.linalg.norm(diff, axis=1)  # (n,)
    kernel_vals = rbf_kernel(distances, epsilon)  # (n,)
    # weighted sum across centres
    return rbf_weights @ kernel_vals  # (m,)

# ----------------------------------------------------------------------
# Parent B – weekday stochastic weighting and pheromone decay
# ----------------------------------------------------------------------
GROUPS: tuple[str, ...] = ("codex", "groq", "cohere", "local_models")

def weekday_weight_vector(groups: list[str], dow: int) -> np.ndarray:
    """
    Normalised row‑stochastic weight vector for *groups* based on weekday ``dow``.
    ``dow`` follows the convention 0 = Sunday … 6 = Saturday.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow / 7)
    amplitude = 0.9
    weights = 0.5 + 0.5 * np.sin(base_angles + phase) * amplitude
    return weights / np.sum(weights)

class PheromoneSystem:
    """
    Minimal pheromone manager: stores a scalar signal per ``surface_key`` together
    with a half‑life.  The decay follows an exponential law:
        φ(t) = signal * 0.5^{t / τ}
    """
    def __init__(self):
        self._store: dict[str, dict] = {}

    def set_signal(
        self,
        surface_key: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> None:
        self._store[surface_key] = {
            "signal_value": float(signal_value),
            "half_life_seconds": float(half_life_seconds),
            "created_time": datetime.datetime.now(datetime.timezone.utc),
        }

    def decay_factor(self, surface_key: str, at_time: datetime.datetime | None = None) -> float:
        """Return the multiplicative decay factor φ(t) for the given key."""
        if surface_key not in self._store:
            raise KeyError(f"Pheromone key '{surface_key}' not found")
        entry = self._store[surface_key]
        now = at_time or datetime.datetime.now(datetime.timezone.utc)
        elapsed = (now - entry["created_time"]).total_seconds()
        half_life = entry["half_life_seconds"]
        if half_life <= 0:
            return 0.0
        return 0.5 ** (elapsed / half_life)

    def current_signal(self, surface_key: str, at_time: datetime.datetime | None = None) -> float:
        """Signal after exponential decay."""
        entry = self._store.get(surface_key)
        if entry is None:
            raise KeyError(f"Pheromone key '{surface_key}' not found")
        factor = self.decay_factor(surface_key, at_time)
        return entry["signal_value"] * factor

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_score(
    rbf_weights: np.ndarray,
    centers: np.ndarray,
    krampus_brainmap: np.ndarray,
    groups: list[str],
    pheromone_system: PheromoneSystem,
    surface_key: str,
    epsilon: float = 1.0,
) -> np.ndarray:
    """
    Unified hybrid score vector.

    Steps:
    1. Build context vector c from the Krampus brain‑map.
    2. Compute coboundary Δ = W_rbf @ φ(||c‑μ||).
    3. Obtain weekday stochastic weights ω based on the current weekday.
    4. Form a diagonal matrix Ω = diag(ω) sized to match Δ (broadcast if needed).
    5. Retrieve pheromone decay factor φ_p.
    6. Return s = φ_p * Ω * Δ   (element‑wise multiplication after broadcasting).

    The result has the same shape as Δ (typically (m,)).
    """
    # 1. Context
    c = krampus_brainmap_context_vector(krampus_brainmap)  # (d,)

    # 2. Coboundary
    delta = rbf_coboundary(rbf_weights, centers, c, epsilon)  # (m,)

    # 3. Weekday weight vector
    dow = datetime.datetime.now(datetime.timezone.utc).weekday()  # 0=Mon … 6=Sun
    # Convert to Sunday‑first convention used in parent B
    dow_sunday_first = (dow + 1) % 7
    omega = weekday_weight_vector(groups, dow_sunday_first)  # (g,)

    # 4. Align dimensions:
    #    If Δ has more entries than groups, repeat ω cyclically.
    if delta.shape[0] != len(omega):
        repeats = math.ceil(delta.shape[0] / len(omega))
        omega_expanded = np.tile(omega, repeats)[: delta.shape[0]]
    else:
        omega_expanded = omega

    # 5. Pheromone decay factor (scalar)
    phi_p = pheromone_system.decay_factor(surface_key)

    # 6. Hybrid combination
    return phi_p * (omega_expanded * delta)

def select_action(
    scores: np.ndarray,
    actions: list[str],
    temperature: float = 1.0,
) -> str:
    """
    Simple softmax action selector using the hybrid scores.
    Returns the action identifier with probability proportional to exp(score/τ).
    """
    if len(scores) != len(actions):
        raise ValueError("scores and actions must have the same length")
    logits = scores / max(temperature, 1e-8)
    max_logit = np.max(logits)
    exp_vals = np.exp(logits - max_logit)  # for numerical stability
    probs = exp_vals / np.sum(exp_vals)
    return random.choices(actions, weights=probs, k=1)[0]

def update_pheromone(
    pheromone_system: PheromoneSystem,
    surface_key: str,
    reward: float,
    half_life_seconds: float = 300.0,
) -> None:
    """
    Update the pheromone signal for a given surface based on received reward.
    The new signal is additive (clamped to non‑negative) and the half‑life is
    refreshed to the provided value.
    """
    try:
        current = pheromone_system.current_signal(surface_key)
    except KeyError:
        current = 0.0
    new_signal = max(0.0, current + reward)
    pheromone_system.set_signal(surface_key, new_signal, half_life_seconds)

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data dimensions
    d = 8   # dimension of brain‑map / context
    n = 5   # number of RBF centres
    m = 4   # output dimension (aligned with GROUPS)

    # Random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Generate random Krampus brain‑map
    krampus_brainmap = np.random.rand(d)

    # Random RBF centres
    centres = np.random.rand(n, d)

    # Random RBF weight matrix (m × n)
    rbf_weights = np.random.randn(m, n)

    # Initialise pheromone system and set an initial signal
    pher_sys = PheromoneSystem()
    surface = "global"
    pher_sys.set_signal(surface_key=surface, signal_value=1.0, half_life_seconds=600.0)

    # Compute hybrid scores
    scores = hybrid_score(
        rbf_weights=rbf_weights,
        centers=centres,
        krampus_brainmap=krampus_brainmap,
        groups=list(GROUPS),
        pheromone_system=pher_sys,
        surface_key=surface,
        epsilon=0.5,
    )
    print("Hybrid scores:", scores)

    # Choose an action based on scores
    actions = ["act_A", "act_B", "act_C", "act_D"]
    chosen = select_action(scores, actions, temperature=0.3)
    print("Chosen action:", chosen)

    # Simulate a reward and update pheromone
    reward = random.uniform(-0.2, 0.8)
    print("Simulated reward:", reward)
    update_pheromone(pheromone_system=pher_sys, surface_key=surface, reward=reward)

    # Show decayed signal after a short wait
    later = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=120)
    decayed = pher_sys.decay_factor(surface, at_time=later)
    print(f"Pheromone decay factor after 2 min: {decayed:.4f}")