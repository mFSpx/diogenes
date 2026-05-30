# DARWIN HAMMER — match 2799, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_diffusion_for_hybrid_hybrid_gliner_m369_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_nlms_o_hybrid_distributed_l_m1145_s0.py (gen3)
# born: 2026-05-29T23:45:57Z

"""
Hybrid Diffusion‑NLMS Algorithm
================================

This module fuses the core mathematics of:

* **Parent A** – *hybrid_hybrid_diffusion_for_hybrid_hybrid_gliner_m369_s0.py*  
  – Uses a **noise schedule** to corrupt input vectors and a **pheromone store**
    whose entries decay exponentially (half‑life) and guide decisions.

* **Parent B** – *hybrid_hybrid_hybrid_nlms_o_hybrid_distributed_l_m1145_s0.py*  
  – Implements the **Normalized Least‑Mean‑Squares (NLMS)** adaptive update.
    The step‑size ``mu`` is modulated by a **broadcast probability** derived
    from a leader‑election routine.

**Mathematical bridge**

Both parents maintain a mutable vector (weights / pheromone strength) that is
updated iteratively:

* In A the pheromone signal ``s`` decays as ``s·0.5^{age/half_life}``.
* In B the NLMS weight update is  

  ``w_{k+1} = w_k + μ·e_k·x_k / (‖x_k‖² + ε)``  

  where ``μ`` is a scalar step‑size.

The hybrid treats the **aggregated pheromone strength** as a *dynamic scaling
factor* for the NLMS step‑size and also feeds the current ``μ`` into the noise
schedule.  Concretely:


σ_t   = base_sigma * (1 - t/T)                     # diffusion schedule
x̂_t   = x_t + 𝒩(0, σ_t²)                           # corrupted input
p_t    = broadcast_probability(phase, step)       # leader‑election factor
φ_t    = aggregated_pheromone(store, key)          # exponential decay sum
μ_t    = base_mu * (1 + φ_t) * p_t                 # hybrid step‑size
w_{t+1}= w_t + μ_t * (target - w_t·x̂_t) * x̂_t / (‖x̂_t‖²+ε)


Thus the diffusion, pheromone decay, and NLMS update are mathematically
interlocked in a single iterative scheme.

The module provides three public functions that illustrate this hybrid
operation and a tiny smoke‑test under ``__main__``.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Hashable, Mapping

# ----------------------------------------------------------------------
# Shared data structures (adapted from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
    """A single pheromone signal that decays exponentially."""
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(pathlib.Path('/proc/self/cmdline').stat().st_ino)  # cheap unique id
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = max(1, half_life_seconds)
        now = pathlib.Path('/proc/self/cmdline').stat().st_ctime
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return pathlib.Path('/proc/self/cmdline').stat().st_ctime - self.last_decay

    def decay_factor(self) -> float:
        """Multiplicative decay since the last decay timestamp."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = pathlib.Path('/proc/self/cmdline').stat().st_ctime


class PheromoneStore:
    """In‑memory container for pheromone entries."""
    def __init__(self):
        self._entries: Dict[str, PheromoneEntry] = {}

    def add(self, entry: PheromoneEntry) -> None:
        self._entries[entry.uuid] = entry

    def decay_all(self) -> None:
        for e in list(self._entries.values()):
            e.apply_decay()
            if e.signal_value < 1e-12:   # prune near‑zero signals
                del self._entries[e.uuid]

    def aggregate(self, surface_key: str) -> float:
        """Sum of decayed signal values for a given surface key."""
        self.decay_all()
        total = 0.0
        for e in self._entries.values():
            if e.surface_key == surface_key:
                total += e.signal_value
        return total


# ----------------------------------------------------------------------
# Parent B utilities (NLMS core & broadcast probability)
# ----------------------------------------------------------------------
def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction y = w·x."""
    return float(np.dot(weights, x))


def nlms_update(weights: np.ndarray, x: np.ndarray, target: float,
                mu: float, eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS weight update.

    Returns the new weight vector and the instantaneous error.
    """
    y = predict(weights, x)
    error = target - y
    power = float(np.dot(x, x) + eps)
    new_weights = weights + mu * error * x / power
    return new_weights, error


def broadcast_probability(phase: int, step: int) -> float:
    """
    Simple broadcast probability inspired by the leader‑election parent.
    It monotonically decreases with phase and step, never exceeding 1.
    """
    if phase < 1 or step < 1:
        raise ValueError("phase and step must be positive integers")
    # Example formulation: p = min(1, 1 / (phase * sqrt(step)))
    p = 1.0 / (phase * math.sqrt(step))
    return min(1.0, p)


# ----------------------------------------------------------------------
# Hybrid components (mathematical bridge)
# ----------------------------------------------------------------------
def diffusion_schedule(t: int, T: int, sigma_0: float = 1.0) -> float:
    """
    Linear decay noise schedule used by the diffusion side.
    sigma_t = sigma_0 * (1 - t/T),  clamped to >= 0.
    """
    if T <= 0:
        raise ValueError("Total steps T must be positive")
    fraction = max(0.0, 1.0 - t / T)
    return sigma_0 * fraction


def corrupt_input(x: np.ndarray, sigma: float) -> np.ndarray:
    """Add isotropic Gaussian noise with standard deviation sigma."""
    if sigma <= 0.0:
        return x.copy()
    noise = np.random.normal(loc=0.0, scale=sigma, size=x.shape)
    return x + noise


def hybrid_step(weights: np.ndarray,
                x: np.ndarray,
                target: float,
                store: PheromoneStore,
                surface_key: str,
                t: int,
                T: int,
                phase: int,
                step: int,
                base_mu: float = 0.5,
                sigma_0: float = 1.0,
                eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    """
    Execute a single hybrid iteration:

    1. Compute diffusion noise sigma_t and corrupt the input.
    2. Retrieve aggregated pheromone strength φ_t for *surface_key*.
    3. Compute broadcast probability p_t.
    4. Form hybrid step‑size μ_t = base_mu * (1 + φ_t) * p_t.
    5. Perform NLMS update with the corrupted input and μ_t.
    6. Feed the absolute error back into the pheromone store as a new signal.
    """
    # 1. Diffusion noise & corruption
    sigma_t = diffusion_schedule(t, T, sigma_0)
    x_hat = corrupt_input(x, sigma_t)

    # 2. Pheromone aggregation
    phi_t = store.aggregate(surface_key)

    # 3. Broadcast probability
    p_t = broadcast_probability(phase, step)

    # 4. Hybrid step‑size
    mu_t = base_mu * (1.0 + phi_t) * p_t
    mu_t = min(1.0, mu_t)   # keep NLMS stable

    # 5. NLMS weight update
    new_weights, error = nlms_update(weights, x_hat, target, mu_t, eps)

    # 6. Push error‑derived pheromone (larger error → stronger signal)
    signal = PheromoneEntry(
        surface_key=surface_key,
        signal_kind="error_feedback",
        signal_value=abs(error),
        half_life_seconds=30  # arbitrary half‑life
    )
    store.add(signal)

    return new_weights, error


# ----------------------------------------------------------------------
# Demonstration utilities
# ----------------------------------------------------------------------
def initialize_hybrid(dim: int) -> Tuple[np.ndarray, PheromoneStore]:
    """Create a zero‑initialized weight vector and an empty pheromone store."""
    weights = np.zeros(dim, dtype=float)
    store = PheromoneStore()
    return weights, store


def run_demo(iterations: int = 20, dim: int = 5) -> None:
    """
    Small synthetic demo:
    - Random target vector is generated.
    - Input vectors are drawn from a standard normal distribution.
    - The hybrid algorithm tries to learn the target.
    """
    # Fixed random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Synthetic target and initial state
    target_weights = np.random.uniform(-1, 1, size=dim)
    weights, store = initialize_hybrid(dim)

    for t in range(iterations):
        # Simulated input and scalar target (dot product of true weights)
        x = np.random.normal(0, 1, size=dim)
        scalar_target = float(np.dot(target_weights, x))

        # Phase and step are simple counters for the demo
        phase = 1 + (t // 5)      # increase phase every 5 iterations
        step = t + 1

        weights, err = hybrid_step(
            weights=weights,
            x=x,
            target=scalar_target,
            store=store,
            surface_key="demo_key",
            t=t,
            T=iterations,
            phase=phase,
            step=step,
            base_mu=0.5,
            sigma_0=0.8
        )

        # Simple progress print (optional)
        if (t + 1) % 5 == 0 or t == iterations - 1:
            mse = np.mean((weights - target_weights) ** 2)
            print(f"Iter {t+1:02d} | Err {err: .4f} | MSE {mse: .6f}")

    print("\nFinal learned weights vs. true target:")
    print("Learned :", np.round(weights, 4))
    print("Target  :", np.round(target_weights, 4))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_demo(iterations=30, dim=8)