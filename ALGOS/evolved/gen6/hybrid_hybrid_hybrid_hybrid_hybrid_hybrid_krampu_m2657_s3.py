# DARWIN HAMMER — match 2657, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m416_s4.py (gen5)
# parent_b: hybrid_hybrid_krampus_brain_hybrid_indy_learning_m273_s0.py (gen4)
# born: 2026-05-29T23:43:20Z

"""Hybrid algorithm merging:
- Parent A: tropical network (linear + ReLU), morphology, endpoint circuit‑breaker.
- Parent B: pheromone‐based stochastic store.

Mathematical bridge:
Both systems can be expressed as probability vectors.
The TropicalNetwork produces a logit vector `z = W·x + b`. After a softmax
`p_i = exp(z_i)/Σ_j exp(z_j)` we obtain a categorical distribution over
engine endpoints.  
The PheromoneStore maintains a set of scalar pheromone values `φ_k`. By
normalising these values we obtain a second distribution `q_k`.

The hybrid loss is the Kullback‑Leibler divergence  

    D_KL(p‖q) = Σ_i p_i log(p_i / q_i)

which quantifies the information‑theoretic gap between the network’s
prediction and the pheromone landscape.  The gradient of `D_KL` w.r.t.
the logits is `∂D/∂z_i = p_i – q_i`.  This gradient is used to update the
network weights (a simple stochastic‑gradient step) while the pheromone
store is nudged in the opposite direction, closing the gap iteratively.
The resulting hybrid system therefore fuses the linear‑algebraic core of
Parent A with the adaptive probabilistic core of Parent B."""

import math
import random
import sys
import pathlib
import uuid
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np


# ---------- Parent A components ----------

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"


class TropicalNetwork:
    """Linear network with ReLU activation per output dimension."""
    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        """
        weights: shape (n_out, n_in)
        biases: shape (n_out,)
        """
        self.weights = weights.astype(float)
        self.biases = biases.astype(float)

    def logits(self, input_vector: np.ndarray) -> np.ndarray:
        """Raw linear scores (no non‑linearity)."""
        return self.weights @ input_vector + self.biases

    def evaluate(self, input_vector: np.ndarray) -> np.ndarray:
        """ReLU‑clipped output used by Parent A."""
        raw = self.logits(input_vector)
        return np.maximum(0.0, raw)


def ssim(x: List[float], y: List[float],
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index Metric (unchanged)."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((np.array(x) - mu_x) * (np.array(y) - mu_y))
    return ((2 * mu_x * mu_y + k1 * dynamic_range ** 2) *
            (2 * sigma_xy + k2 * dynamic_range ** 2) /
            ((mu_x ** 2 + mu_y ** 2 + k1 * dynamic_range ** 2) *
             (sigma_x ** 2 + sigma_y ** 2 + k2 * dynamic_range ** 2)))


class EndpointCircuitBreaker:
    """Simple failure‑count circuit breaker."""
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = ""

    def allow(self) -> bool:
        return not self.open

    def as_dict(self) -> dict:
        return {
            "failure_threshold": self.failure_threshold,
            "failures": self.failures,
            "open": self.open,
            "last_event_at": self.last_event_at,
        }


# ---------- Parent B components ----------

class PheromoneEntry:
    """Single pheromone datum with exponential decay."""
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid1())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = float(signal_value)
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative factor since last_decay."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """Container for many PheromoneEntry objects."""
    def __init__(self):
        self._entries: Dict[str, PheromoneEntry] = {}

    def add(self, entry: PheromoneEntry) -> None:
        self._entries[entry.surface_key] = entry

    def decay_all(self) -> None:
        for e in self._entries.values():
            e.apply_decay()

    def distribution(self) -> np.ndarray:
        """Normalised pheromone values as a probability vector."""
        self.decay_all()
        values = np.array([e.signal_value for e in self._entries.values()], dtype=float)
        total = values.sum()
        if total <= 0.0:
            # Uniform fallback if everything decayed away
            return np.full_like(values, 1.0 / len(values)) if len(values) > 0 else np.array([])
        return values / total

    def keys(self) -> List[str]:
        return list(self._entries.keys())

    def update_with_delta(self, delta: np.ndarray) -> None:
        """Adjust pheromone values by adding delta (same order as keys())."""
        keys = self.keys()
        for i, k in enumerate(keys):
            entry = self._entries[k]
            entry.signal_value = max(0.0, entry.signal_value + delta[i])


# ---------- Hybrid mathematical core ----------

def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z_max = np.max(z)
    exp_z = np.exp(z - z_max)
    sum_exp = np.sum(exp_z)
    return exp_z / sum_exp if sum_exp > 0 else np.full_like(z, 1.0 / z.size)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p‖q) with safe handling of zeros."""
    eps = np.finfo(float).eps
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return np.sum(p_safe * np.log(p_safe / q_safe))


def hybrid_step(network: TropicalNetwork,
                store: PheromoneStore,
                input_vec: np.ndarray,
                lr: float = 0.01) -> float:
    """
    Perform a single hybrid iteration:
      1. Compute network distribution p (softmax of logits).
      2. Obtain pheromone distribution q.
      3. Compute KL(p‖q) as loss.
      4. Gradient‑step network weights using (p‑q).
      5. Adjust pheromones in the opposite direction.
    Returns the KL loss value.
    """
    # 1. Network forward pass
    logits = network.logits(input_vec)
    p = softmax(logits)

    # 2. Pheromone distribution (ensure same dimensionality)
    q = store.distribution()
    if q.size == 0:
        # Initialise a uniform pheromone vector matching network size
        q = np.full(p.shape, 1.0 / p.size)
        # Populate store with dummy entries
        for i in range(p.size):
            store.add(PheromoneEntry(surface_key=f"node_{i}",
                                     signal_kind="generic",
                                     signal_value=1.0,
                                     half_life_seconds=3600))
    elif q.size != p.size:
        # Align sizes by truncating or padding with small epsilon
        if q.size > p.size:
            q = q[:p.size]
        else:
            pad = np.full(p.size - q.size, 1e-9)
            q = np.concatenate([q, pad])

    # 3. Loss
    loss = kl_divergence(p, q)

    # 4. Weight update (gradient descent on KL)
    grad_logits = p - q  # ∂D/∂z_i
    grad_w = np.outer(grad_logits, input_vec)  # shape (n_out, n_in)
    grad_b = grad_logits

    network.weights -= lr * grad_w
    network.biases -= lr * grad_b

    # 5. Pheromone update (move q towards p)
    delta = (p - q) * lr * 10.0  # amplify a bit for visibility
    store.update_with_delta(delta)

    return loss


# ---------- Convenience wrapper ----------

class HybridEngine:
    """Encapsulates the fused system."""
    def __init__(self, n_in: int, n_out: int):
        w = np.random.randn(n_out, n_in) * 0.1
        b = np.zeros(n_out)
        self.network = TropicalNetwork(w, b)
        self.store = PheromoneStore()
        # Seed pheromone store with matching keys
        for i in range(n_out):
            self.store.add(PheromoneEntry(
                surface_key=f"node_{i}",
                signal_kind="init",
                signal_value=random.random(),
                half_life_seconds=7200))

    def step(self, input_vec: np.ndarray) -> float:
        return hybrid_step(self.network, self.store, input_vec)

    def status(self) -> dict:
        return {
            "weights": self.network.weights.tolist(),
            "biases": self.network.biases.tolist(),
            "pheromones": {k: e.signal_value for k, e in self.store._entries.items()}
        }


# ---------- Smoke test ----------

if __name__ == "__main__":
    # Simple deterministic seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    n_features = 5
    n_outputs = 3
    engine = HybridEngine(n_in=n_features, n_out=n_outputs)

    # Random input vector
    x = np.random.rand(n_features)

    # Run a few hybrid steps and print losses
    for step in range(5):
        loss = engine.step(x)
        print(f"Step {step+1}, KL loss = {loss:.6f}")

    # Show final internal state (truncated)
    st = engine.status()
    print("\nFinal network biases:", st["biases"])
    print("Final pheromone values:", st["pheromones"])