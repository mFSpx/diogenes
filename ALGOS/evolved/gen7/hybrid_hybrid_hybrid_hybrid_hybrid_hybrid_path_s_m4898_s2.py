# DARWIN HAMMER — match 4898, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_privac_m1523_s1.py (gen6)
# parent_b: hybrid_hybrid_path_signatur_hybrid_hybrid_pherom_m266_s1.py (gen3)
# born: 2026-05-29T23:58:54Z

import math
import time
import pathlib
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------
# Utility: monotonic timestamp (seconds)
# ----------------------------------------------------------------------
def _now() -> float:
    """Return a monotonic timestamp (seconds)."""
    try:
        # Linux specific, fast and monotonic
        return pathlib.Path("/proc/self/cmdline").stat().st_ctime
    except Exception:
        return time.time()


# ----------------------------------------------------------------------
# NLMS core (Parent A) – extended to batch updates
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Scalar prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """
    Single‑sample Normalised LMS weight update.

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


def nlms_batch_update(
    weights: np.ndarray,
    X: np.ndarray,
    d: np.ndarray,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch version of NLMS that processes a matrix X (N×D) and a vector d (N,).

    Returns the updated weights and the vector of errors.
    """
    # Predictions for all samples
    y = X @ weights
    error = d - y
    # Normalisation per sample
    norm = np.sum(X * X, axis=1) + eps
    # Weighted sum of updates
    update = (mu * (error / norm)[:, None]) * X
    weights = weights + np.sum(update, axis=0)
    return weights, error


# ----------------------------------------------------------------------
# Lead‑lag transform (Parent B) – vectorised implementation
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Convert a T×d path into a (2T‑1)×(2d) lead‑lag representation.

    For each time step t (0 ≤ t < T‑1) we emit two rows:
        [p_t, p_t]      – “lead”
        [p_{t+1}, p_t]  – “lag”

    The final row repeats the last point:
        [p_{T‑1}, p_{T‑1}]
    """
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be a 2‑D array of shape (T, d)")

    T, d = path.shape
    if T == 0:
        raise ValueError("path must contain at least one time step")

    # Pre‑allocate output
    out = np.empty((2 * T - 1, 2 * d), dtype=float)

    # Lead rows (repeat each point)
    out[0::2, :d] = path
    out[0::2, d:] = path

    # Lag rows (shifted forward, except last)
    out[1::2, :d] = path[1:]
    out[1::2, d:] = path[:-1]

    return out


# ----------------------------------------------------------------------
# Pheromone handling (Parent B)
# ----------------------------------------------------------------------
def decay_pheromone(entry: dict, now: float) -> float:
    """
    Compute the decayed pheromone value for a stored entry.

    The decay follows a half‑life law:
        v(t) = v0 * 0.5^{elapsed / half_life}
    """
    elapsed = now - entry["created_time"]
    half_life = entry["half_life_seconds"]
    if half_life <= 0:
        return entry["signal_value"]
    return entry["signal_value"] * math.pow(0.5, elapsed / half_life)


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
    prune_seconds: float = 3600.0  # optional automatic cleanup

    # Runtime state (filled in __post_init__)
    weights: np.ndarray = field(init=False)
    pheromones: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        # Small random initialisation to break symmetry.
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
        Insert or refresh a pheromone entry.

        If the key already exists, the stored value is decayed before being
        blended with the new observation (simple exponential moving average).
        """
        now = _now()
        if surface_key in self.pheromones:
            prev = self.pheromones[surface_key]
            decayed = decay_pheromone(prev, now)
            # EMA with α = 0.5 (can be tuned)
            signal_value = 0.5 * decayed + 0.5 * signal_value

        self.pheromones[surface_key] = {
            "signal_kind": signal_kind,
            "signal_value": signal_value,
            "half_life_seconds": max(half_life_seconds, 1e-6),
            "created_time": now,
        }

    def get_pheromone(self, surface_key: str) -> float | None:
        """Return the current decayed pheromone value, or None if missing."""
        entry = self.pheromones.get(surface_key)
        if entry is None:
            return None
        return decay_pheromone(entry, _now())

    def _freshness_factor(self, entry: dict, now: float) -> float:
        """
        Compute a freshness factor in [0, 1] based on elapsed time and half‑life.
        Fresh pheromones (elapsed << half‑life) → ≈1, stale → ≈0.
        """
        half_life = entry["half_life_seconds"]
        if half_life <= 0:
            return 1.0
        elapsed = now - entry["created_time"]
        return math.exp(-elapsed / half_life)

    def _prune_stale(self, now: float) -> None:
        """Remove entries that have decayed below a negligible threshold."""
        to_del = [
            k
            for k, v in self.pheromones.items()
            if decay_pheromone(v, now) < 1e-6
        ]
        for k in to_del:
            del self.pheromones[k]

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
        Perform a full learning step:

        1. Transform ``path`` → lead‑lag matrix ``X``.
        2. Reduce ``X`` to a single feature vector ``x`` via *energy‑weighted*
           averaging (rows weighted by their Euclidean norm). This preserves
           more information than a plain mean.
        3. Retrieve the current decayed pheromone value for ``surface_key``.
        4. Use the pheromone value as the desired output ``d``; fall back to
           ``raw_signal`` when absent.
        5. Compute an adaptive learning rate:
               μ = μ_base * (1 + freshness)
           where *freshness* ∈ [0, 1] reflects how recent the pheromone is.
        6. Apply the NLMS update.
        7. Store (or refresh) the pheromone entry with the new observation.
        8. Optionally prune very stale entries.

        Returns
        -------
        error : float
            The scalar prediction error after the weight update.
        """
        now = _now()

        # ------------------------------------------------------------------
        # 1‑2. Feature extraction with energy‑weighted reduction
        # ------------------------------------------------------------------
        X = lead_lag_transform(path)                     # (2T‑1, 2d)
        row_norms = np.linalg.norm(X, axis=1) + self.eps
        weights_rows = row_norms / np.sum(row_norms)    # normalized energy weights
        x = np.average(X, axis=0, weights=weights_rows)  # (2d,)

        if x.shape[0] != self.weight_dim:
            raise ValueError(
                f"Feature dimension {x.shape[0]} does not match weight dimension {self.weight_dim}"
            )

        # ------------------------------------------------------------------
        # 3‑4. Desired output (target)
        # ------------------------------------------------------------------
        decayed = self.get_pheromone(surface_key)
        target = decayed if decayed is not None else raw_signal

        # ------------------------------------------------------------------
        # 5. Adaptive learning rate based on pheromone freshness
        # ------------------------------------------------------------------
        freshness = 1.0
        if surface_key in self.pheromones:
            entry = self.pheromones[surface_key]
            freshness = self._freshness_factor(entry, now)
        mu = self.mu_base * (1.0 + freshness)   # μ ∈ [μ_base, 2·μ_base]

        # ------------------------------------------------------------------
        # 6. NLMS weight update
        # ------------------------------------------------------------------
        self.weights, error = nlms_update(
            self.weights, x, target, mu=mu, eps=self.eps
        )

        # ------------------------------------------------------------------
        # 7. Refresh pheromone entry (store the raw observation for future steps)
        # ------------------------------------------------------------------
        self.update_pheromone(
            surface_key,
            signal_kind,
            raw_signal,
            half_life_seconds,
        )

        # ------------------------------------------------------------------
        # 8. Optional housekeeping
        # ------------------------------------------------------------------
        if self.prune_seconds > 0:
            self._prune_stale(now)

        return error


# ----------------------------------------------------------------------
# Example usage (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy path: 5 time steps, 3‑D coordinates
    dummy_path = np.random.rand(5, 3)

    # Initialise hybrid scheduler; feature dimension = 2 * d = 6
    scheduler = HybridNLMSPheromone(weight_dim=6)

    err = scheduler.process_path_and_signal(
        path=dummy_path,
        surface_key="texture_01",
        signal_kind="load_pressure",
        raw_signal=0.73,
        half_life_seconds=30.0,
    )
    print(f"Prediction error after update: {err:.6f}")