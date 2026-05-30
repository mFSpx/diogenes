# DARWIN HAMMER — match 4898, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_privac_m1523_s1.py (gen6)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s1.py (gen3)
# born: 2026-05-29T23:58:54Z

"""Hybrid NLMS‑Pheromone Scheduler
This module fuses the core mathematics of two parent algorithms:

* **Parent A** – Normalised Least‑Mean‑Squares (NLMS) adaptive filtering used for
  model‑loading / VRAM scheduling decisions.
* **Parent B** – Lead‑lag path transformation together with a decaying pheromone
  signal model.

**Mathematical bridge** – The lead‑lag transformation converts a time‑ordered
path into a high‑dimensional feature vector `x`.  This vector is fed to the NLMS
update rule, where the *pheromone signal value* plays the role of the desired
scalar output `d`.  The NLMS error `e = d – wᵀx` drives the weight adaptation,
while the pheromone decay mechanism supplies a Bayesian‑like prior that
modulates the learning‑rate `μ`.  Thus the adaptive filter continuously learns
how path signatures map to resource‑pressure signals, and the learned weights
are later used to decide VRAM loading/eviction actions. """

import math
import random
import sys
import pathlib
import time
from dataclasses import dataclass, field
import numpy as np

# ----------------------------------------------------------------------
# NLMS core (from Parent A)
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Dot‑product prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """
    One Normalised LMS weight update.

    Parameters
    ----------
    weights : np.ndarray
        Current weight vector (1‑D).
    x : np.ndarray
        Input feature vector.
    target : float
        Desired scalar output.
    mu : float
        Base learning rate (0, 2).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    (weights, error) : tuple
        Updated weights and the prediction error.
    """
    error = target - nlms_predict(weights, x)
    norm_factor = (x @ x) + eps
    weights = weights + mu * error * x / norm_factor
    return weights, error


# ----------------------------------------------------------------------
# Lead‑lag transform (from Parent B)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Convert a T×d path into a (2T‑1)×(2d) lead‑lag representation.
    The output can be used directly as the NLMS feature vector.
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array of shape (T, d)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


# ----------------------------------------------------------------------
# Pheromone handling (from Parent B)
# ----------------------------------------------------------------------
def _current_timestamp() -> float:
    """Return a monotonic timestamp (seconds)."""
    # pathlib.Path('/proc/self/cmdline').stat().st_ctime works on Linux,
    # but we fall back to time.time() for portability.
    try:
        return pathlib.Path("/proc/self/cmdline").stat().st_ctime
    except Exception:
        return time.time()


def decay_pheromone(entry: dict, now: float) -> float:
    """
    Compute the decayed pheromone value given a stored entry and the current
    timestamp.
    """
    elapsed = now - entry["created_time"]
    half_life = entry["half_life_seconds"]
    if half_life <= 0:
        return entry["signal_value"]
    decayed = entry["signal_value"] * math.pow(0.5, elapsed / half_life)
    return decayed


# ----------------------------------------------------------------------
# Hybrid system definition
# ----------------------------------------------------------------------
@dataclass
class HybridNLMSPheromone:
    """
    Adaptive scheduler that learns from lead‑lag path signatures and
    decaying pheromone signals using NLMS.
    """
    weight_dim: int
    mu_base: float = 0.5
    eps: float = 1e-9
    weights: np.ndarray = field(init=False)
    pheromones: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        # Initialise weights to small random values for symmetry breaking.
        self.weights = np.random.randn(self.weight_dim) * 0.01

    # ------------------------------------------------------------------
    # Pheromone API
    # ------------------------------------------------------------------
    def update_pheromone(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> None:
        """
        Insert or refresh a pheromone entry.  If the key already exists,
        the stored value is decayed before being overwritten.
        """
        now = _current_timestamp()
        if surface_key in self.pheromones:
            prev = self.pheromones[surface_key]
            decayed = decay_pheromone(prev, now)
            # Blend old decayed value with new one (simple average)
            signal_value = 0.5 * (decayed + signal_value)
        self.pheromones[surface_key] = {
            "signal_kind": signal_kind,
            "signal_value": signal_value,
            "half_life_seconds": half_life_seconds,
            "created_time": now,
        }

    def get_pheromone(self, surface_key: str) -> float | None:
        """Return the current decayed pheromone value, or None if missing."""
        if surface_key not in self.pheromones:
            return None
        now = _current_timestamp()
        entry = self.pheromones[surface_key]
        return decay_pheromone(entry, now)

    # ------------------------------------------------------------------
    # Hybrid learning step
    # ------------------------------------------------------------------
    def process_path_and_signal(
        self,
        path: np.ndarray,
        surface_key: str,
        signal_kind: str,
        raw_signal: float,
        half_life_seconds: float,
    ) -> float:
        """
        1. Transform the path → feature matrix X (lead‑lag).
        2. Reduce X to a single feature vector by averaging over rows.
           (Any linear reduction works; averaging keeps the interface simple.)
        3. Retrieve the decayed pheromone value for `surface_key`.
        4. Use the pheromone value as the desired output `d`.  If absent,
           fall back to the raw signal supplied by the caller.
        5. Perform an NLMS weight update with an adaptive learning‑rate:
           μ = μ_base * (1 + decay_factor), where decay_factor reflects
           pheromone freshness.
        6. Store the (possibly) updated pheromone entry.
        Returns the prediction error after the update.
        """
        # 1‑2. Feature extraction
        transformed = lead_lag_transform(path)          # (2T‑1, 2d)
        x = transformed.mean(axis=0)                    # (2d,)

        # Ensure dimensionality matches the internal weight vector
        if x.shape[0] != self.weight_dim:
            raise ValueError(
                f"Feature dimension {x.shape[0]} does not match weight dimension {self.weight_dim}"
            )

        # 3‑4. Desired output
        decayed = self.get_pheromone(surface_key)
        target = decayed if decayed is not None else raw_signal

        # 5. Adaptive learning rate based on pheromone freshness
        now = _current_timestamp()
        freshness = 1.0
        if surface_key in self.pheromones:
            entry = self.pheromones[surface_key]
            elapsed = now - entry["created_time"]
            # Fresh pheromones (small elapsed) increase μ, stale ones reduce it.
            freshness = math.exp(-elapsed / max(entry["half_life_seconds"], 1e-6))
        mu_adapt = self.mu_base * (1.0 + freshness)

        # NLMS update
        self.weights, error = nlms_update(
            self.weights, x, target, mu=mu_adapt, eps=self.eps
        )

        # 6. Update pheromone store with the new (raw) signal
        self.update_pheromone(surface_key, signal_kind, raw_signal, half_life_seconds)

        return error

    # ------------------------------------------------------------------
    # Decision logic (VRAM‑like scheduling)
    # ------------------------------------------------------------------
    def decide_action(self, path: np.ndarray, threshold: float = 0.0) -> str:
        """
        Predict the resource pressure for a given path and return a textual
        action:
        - ``'load'``   if prediction > threshold,
        - ``'evict'``  if prediction < -threshold,
        - ``'hold'``  otherwise.
        """
        x = lead_lag_transform(path).mean(axis=0)
        if x.shape[0] != self.weight_dim:
            raise ValueError("Dimension mismatch in decide_action")
        pred = nlms_predict(self.weights, x)
        if pred > threshold:
            return "load"
        if pred < -threshold:
            return "evict"
        return "hold"


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
def _random_path(T: int = 5, d: int = 3) -> np.ndarray:
    """Generate a random T×d path."""
    return np.random.randn(T, d)


def main() -> None:
    # Determine feature dimension: lead‑lag doubles both time and space.
    T, d = 6, 4
    feature_dim = 2 * d  # because we later average rows → length 2d

    hybrid = HybridNLMSPheromone(weight_dim=feature_dim, mu_base=0.4)

    # Simulate a stream of paths with associated pheromone signals.
    for step in range(10):
        path = _random_path(T, d)
        surface = f"surface_{step % 3}"          # three repeating surfaces
        kind = "cpu_pressure"
        raw_signal = random.uniform(-1.0, 1.0)   # synthetic pressure metric
        half_life = random.uniform(5.0, 20.0)   # seconds

        err = hybrid.process_path_and_signal(
            path, surface, kind, raw_signal, half_life
        )
        action = hybrid.decide_action(path, threshold=0.2)

        print(
            f"Step {step:02d} | surface={surface} | raw={raw_signal:+.3f} "
            f"| err={err:+.3f} | decision={action}"
        )

    # Final weight snapshot
    print("\nFinal weight vector (first 10 entries):")
    print(hybrid.weights[:10])


if __name__ == "__main__":
    main()