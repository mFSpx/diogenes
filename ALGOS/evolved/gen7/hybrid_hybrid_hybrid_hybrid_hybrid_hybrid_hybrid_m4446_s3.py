# DARWIN HAMMER — match 4446, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2249_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_sketch_m1355_s0.py (gen5)
# born: 2026-05-29T23:55:45Z

"""Hybrid NLMS‑RLCT‑Pheromone Algorithm
Integrates:
- Parent A: Normalized Least‑Mean‑Squares (NLMS) update with a fractional‑memory kernel
  (using Lanczos gamma for fractional differencing weights) and MinHash sketching.
- Parent B: Real Log Canonical Threshold (RLCT) estimation from training losses,
  pheromone‑based decay, and Bayesian‑style sketch updates.

Mathematical bridge:
The RLCT estimate provides a data‑driven scaling factor μ̂ for the NLMS step‑size,
while the fractional‑memory kernel weights past prediction errors.  Pheromone
signals modulate the kernel decay (half‑life) and are updated using a Bayesian
posterior over the MinHash sketch of the input vector.  Thus the prediction
error eₜ influences future updates through both fractional memory and a
RLCT‑adapted learning rate, yielding a unified adaptive learning rule."""
import numpy as np
import math
import random
import sys
import pathlib
from collections import defaultdict
from datetime import datetime

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.139216000391,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
])

def lanczos_gamma(z: np.ndarray) -> np.ndarray:
    """Lanczos approximation of the Gamma function (vectorised)."""
    z = z - 1.0
    x = 1.0 / (z * z + 5.0 * z - 6.0)
    series = _LANCZOS_C / (z + np.arange(_LANCZOS_G) + 1.0)
    # dot with a vector of ones gives the sum of the series
    result = np.sqrt(2 * math.pi) * np.power(z, z + 0.5) * np.exp(-z) * x * np.dot(series, np.ones_like(series))
    return result

def minhash(x: np.ndarray, num_buckets: int = 10) -> np.ndarray:
    """Simple MinHash sketch of a vector."""
    rng = np.random.default_rng()
    # random permutations are emulated by hashing with random seeds
    sketches = []
    for _ in range(num_buckets):
        perm = rng.permutation(len(x))
        sketches.append(np.min(x[perm] % num_buckets))
    return np.array(sketches)

# ----------------------------------------------------------------------
# Fractional‑memory kernel (uses Lanczos Gamma for fractional differencing)
# ----------------------------------------------------------------------
def fractional_memory_weights(alpha: float, length: int) -> np.ndarray:
    """
    Compute fractional differencing weights w_k = Gamma(k - alpha) / (Gamma(-alpha) Gamma(k+1))
    for k = 0 .. length‑1.  The weights sum to 1 after normalization.
    """
    ks = np.arange(length, dtype=np.float64)
    numer = lanczos_gamma(ks - alpha + 1)          # Gamma(k - alpha + 1)
    denom = lanczos_gamma(-alpha + 1) * lanczos_gamma(ks + 2)  # Gamma(1-alpha) * Gamma(k+2)
    w = numer / denom
    w = np.where(np.isfinite(w), w, 0.0)
    w_sum = w.sum()
    if w_sum == 0:
        return np.full_like(w, 1.0 / length)
    return w / w_sum

# ----------------------------------------------------------------------
# NLMS core (Parent A) with RLCT‑scaled step size
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction ŷ = w·x."""
    return float(np.dot(weights, x))

def nlms_update(weights: np.ndarray,
                x: np.ndarray,
                target: float,
                mu: float,
                epsilon: float = 1e-8) -> np.ndarray:
    """
    Standard NLMS weight update:
        w ← w + (mu / (ε + ||x||²)) * e * x
    where e = target – w·x.
    """
    pred = nlms_predict(weights, x)
    error = target - pred
    norm_sq = np.dot(x, x)
    step = (mu / (epsilon + norm_sq)) * error
    return weights + step * x

# ----------------------------------------------------------------------
# RLCT estimator (Parent B)
# ----------------------------------------------------------------------
def estimate_rlct_from_losses(train_losses: list[float], n_vals: list[int]) -> float:
    """
    Linear regression of log(loss) on log(log(n)) as described in Parent B.
    Returns the slope, which is the RLCT estimate.
    """
    losses = np.asarray(train_losses, dtype=np.float64)
    ns = np.asarray(n_vals, dtype=np.float64)
    if np.any(ns <= math.e):
        raise ValueError("All n_values must be > e for log(log(n)) to be defined")
    if losses.shape != ns.shape:
        raise ValueError("train_losses and n_vals must have the same length")
    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    var_x = (x_centered ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    slope = float((x_centered * y_centered).sum() / var_x)
    return slope

# ----------------------------------------------------------------------
# Pheromone system (Parent B) – decay modulates fractional memory
# ----------------------------------------------------------------------
def decay_factor(half_life_seconds: float, elapsed_seconds: float = 1.0) -> float:
    """Exponential decay based on half‑life."""
    if half_life_seconds <= 0:
        return 1.0
    return 0.5 ** (elapsed_seconds / half_life_seconds)

def update_pheromone(pheromones: dict,
                    key: str,
                    signal_kind: str,
                    value: float,
                    half_life_seconds: float) -> None:
    """
    Store or update a pheromone signal and apply decay.
    """
    if key not in pheromones:
        pheromones[key] = {}
    if signal_kind not in pheromones[key]:
        pheromones[key][signal_kind] = value
    else:
        # decay previous value then add new contribution
        decay = decay_factor(half_life_seconds)
        pheromones[key][signal_kind] = pheromones[key][signal_kind] * decay + value

def get_pheromone(pheromones: dict, key: str, signal_kind: str, default: float = 0.0) -> float:
    return pheromones.get(key, {}).get(signal_kind, default)

# ----------------------------------------------------------------------
# Hybrid system class
# ----------------------------------------------------------------------
class HybridNLMSRLCT:
    """
    Unified learner that:
    1. Keeps an NLMS weight vector.
    2. Stores a history of recent errors for fractional‑memory weighting.
    3. Adapts the NLMS step size μ using the RLCT estimate from a loss log.
    4. Modulates the fractional‑memory kernel via pheromone‑controlled decay.
    5. Optionally sketches inputs with MinHash for Bayesian‑style updates.
    """
    def __init__(self,
                 input_dim: int,
                 alpha: float = 0.6,
                 base_mu: float = 0.1,
                 pheromone_half_life: float = 30.0,
                 max_history: int = 20):
        self.input_dim = input_dim
        self.weights = np.zeros(input_dim, dtype=np.float64)
        self.alpha = alpha                      # fractional order
        self.base_mu = base_mu                  # baseline NLMS step size
        self.pheromone_half_life = pheromone_half_life
        self.max_history = max_history
        self.error_history: list[float] = []
        self.loss_log: list[float] = []         # training loss per iteration
        self.n_log: list[int] = []              # corresponding sample sizes
        self.pheromones: dict = {}
        self.sketch_hist: defaultdict = defaultdict(list)   # key → list of sketches

    # ------------------------------------------------------------------
    # Core hybrid operations
    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> float:
        """Linear prediction using current NLMS weights."""
        return nlms_predict(self.weights, x)

    def register_loss(self, loss: float, n_samples: int) -> None:
        """Append loss and sample size for RLCT estimation."""
        self.loss_log.append(loss)
        self.n_log.append(n_samples)

    def _current_mu(self) -> float:
        """
        Compute the effective NLMS step size:
            μ_eff = base_mu * (1 + |RLCT|)⁻¹
        If insufficient data for RLCT, fall back to base_mu.
        """
        if len(self.loss_log) < 5:
            return self.base_mu
        try:
            rlct = estimate_rlct_from_losses(self.loss_log, self.n_log)
            # RLCT may be negative; use absolute value to keep μ positive
            scale = 1.0 / (1.0 + abs(rlct))
            return self.base_mu * scale
        except Exception:
            return self.base_mu

    def _fractional_weights(self) -> np.ndarray:
        """
        Compute fractional‑memory weights for the stored error history,
        applying pheromone‑based decay to the effective memory length.
        """
        # Determine effective length via pheromone signal (larger pheromone → longer memory)
        decay_signal = get_pheromone(self.pheromones, "memory", "decay", default=1.0)
        effective_len = min(len(self.error_history),
                            max(1, int(self.max_history * decay_signal)))
        if effective_len == 0:
            return np.array([])
        recent_errors = np.array(self.error_history[-effective_len:], dtype=np.float64)
        mem_weights = fractional_memory_weights(self.alpha, effective_len)
        return mem_weights * recent_errors

    def update(self, x: np.ndarray, target: float) -> None:
        """
        Perform a full hybrid update:
        1. Compute NLMS prediction and error.
        2. Store error in history.
        3. Compute fractional‑memory weighted error term.
        4. Adapt step size via RLCT.
        5. Update NLMS weights using the combined error.
        6. Update pheromone signal based on magnitude of the weighted error.
        7. Update MinHash sketch history (Bayesian‑style placeholder).
        """
        # 1. Prediction & raw error
        pred = self.predict(x)
        raw_error = target - pred

        # 2. Store raw error
        self.error_history.append(float(raw_error))
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)

        # 3. Fractional‑memory weighted error
        weighted_error = self._fractional_weights().sum() if self.error_history else raw_error

        # 4. Adaptive step size
        mu_eff = self._current_mu()

        # 5. NLMS weight update using weighted_error as the error term
        #    The NLMS formula uses e = weighted_error
        norm_sq = np.dot(x, x)
        step = (mu_eff / (1e-8 + norm_sq)) * weighted_error
        self.weights += step * x

        # 6. Pheromone update: larger |weighted_error| reinforces memory decay signal
        signal_strength = min(1.0, abs(weighted_error) / (1.0 + np.linalg.norm(x)))
        update_pheromone(self.pheromones,
                         key="memory",
                         signal_kind="decay",
                         value=signal_strength,
                         half_life_seconds=self.pheromone_half_life)

        # 7. Sketch update (store MinHash of x under a key derived from target)
        sketch_key = f"target_{int(target)}"
        sketch = minhash(x)
        self.sketch_hist[sketch_key].append(sketch)

    # ------------------------------------------------------------------
    # Helper introspection utilities
    # ------------------------------------------------------------------
    def current_state(self) -> dict:
        """Return a snapshot of internal variables for debugging / testing."""
        return {
            "weights": self.weights.copy(),
            "error_history": list(self.error_history),
            "mu_eff": self._current_mu(),
            "pheromones": {k: v.copy() for k, v in self.pheromones.items()},
            "rlct": (estimate_rlct_from_losses(self.loss_log, self.n_log)
                     if len(self.loss_log) >= 5 else None),
            "sketch_counts": {k: len(v) for k, v in self.sketch_hist.items()}
        }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    dim = 5
    hybrid = HybridNLMSRLCT(input_dim=dim, alpha=0.7, base_mu=0.2)

    # Generate synthetic data: y = 2*x0 - 3*x2 + noise
    true_w = np.array([2.0, 0.0, -3.0, 0.0, 0.0])
    for epoch in range(30):
        x = np.random.randn(dim)
        noise = np.random.normal(scale=0.1)
        y = np.dot(true_w, x) + noise

        hybrid.update(x, y)

        # Log a synthetic loss (MSE) every few steps for RLCT
        if epoch % 5 == 0:
            pred = hybrid.predict(x)
            loss = (y - pred) ** 2
            hybrid.register_loss(loss, n_samples=epoch + 1)

    state = hybrid.current_state()
    print("Final NLMS weights:", state["weights"])
    print("Effective mu:", state["mu_eff"])
    print("RLCT estimate (if available):", state["rlct"])
    print("Pheromone memory decay signal:", state["pheromones"].get("memory", {}))
    print("Sketch entry counts:", state["sketch_counts"])
    sys.exit(0)