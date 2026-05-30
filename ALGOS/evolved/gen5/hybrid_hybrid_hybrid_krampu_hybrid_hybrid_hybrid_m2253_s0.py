# DARWIN HAMMER — match 2253, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_krampus_brain_hybrid_hybrid_fisher_m1097_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_percyp_hybrid_caputo_fracti_m610_s0.py (gen4)
# born: 2026-05-29T23:41:39Z

"""Hybrid Algorithm: Caputo-Pheromone-Infotaxis with CircuitBreaker

Parents:
- hybrid_hybrid_krampus_brain_hybrid_hybrid_fisher_m1097_s3.py (HybridPheromoneFisher)
- hybrid_hybrid_hybrid_percyp_hybrid_caputo_fracti_m610_s0.py (Caputo Fractional Derivative + EndpointCircuitBreaker)

Mathematical Bridge:
Both parents manipulate a pheromone/weight vector.  The Fisher/Infotaxis side supplies
entropy‑based information gain, while the Caputo side supplies a power‑law memory
kernel (fractional derivative) that modulates pheromone decay.  By applying a
Caputo fractional derivative to the entropy‑scaled pheromone vector we obtain a
long‑range, history‑aware decay term.  This term feeds the EndpointCircuitBreaker,
which opens when the fractional decay exceeds a configurable failure threshold.
The resulting system simultaneously respects information‑theoretic weighting and
fractional‑order temporal dynamics, yielding a unified decision metric."""
import sys
import math
import random
import pathlib
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass

# ----------------------------------------------------------------------
# Utility functions (shared by both parent concepts)
# ----------------------------------------------------------------------
def calculate_entropy(vec: np.ndarray) -> float:
    """Shannon entropy of a non‑negative vector (treated as a probability distribution)."""
    prob = vec / np.sum(vec) if np.sum(vec) != 0 else np.full_like(vec, 1 / vec.size)
    prob = np.clip(prob, 1e-12, 1)  # avoid log(0)
    return -np.sum(prob * np.log2(prob))


def calculate_information_gain(prev_vec: np.ndarray, new_vec: np.ndarray) -> float:
    """Kullback‑Leibler divergence as a proxy for information gain."""
    p = prev_vec / np.sum(prev_vec) if np.sum(prev_vec) != 0 else np.full_like(prev_vec, 1 / prev_vec.size)
    q = new_vec / np.sum(new_vec) if np.sum(new_vec) != 0 else np.full_like(new_vec, 1 / new_vec.size)
    p = np.clip(p, 1e-12, 1)
    q = np.clip(q, 1e-12, 1)
    return np.sum(p * np.log2(p / q))


def sinusoidal_weekday_weights(date: datetime) -> np.ndarray:
    """
    Produce a 7‑element row‑stochastic vector using sinusoidal rotation.
    The vector sums to 1 and varies smoothly across weekdays.
    """
    base = np.arange(7)
    angle = 2 * math.pi * (date.timetuple().tm_yday % 7) / 7.0
    weights = np.sin(base + angle) ** 2
    weights /= weights.sum()
    return weights


# ----------------------------------------------------------------------
# Fractional calculus utilities (Caputo side)
# ----------------------------------------------------------------------
def fractional_binomial(alpha: float, k: int) -> float:
    """Generalized binomial coefficient (α choose k) using Gamma function."""
    return math.gamma(alpha + 1) / (math.gamma(k + 1) * math.gamma(alpha - k + 1))


def caputo_fractional_derivative(signal: np.ndarray, order: float, dt: float) -> np.ndarray:
    """
    Approximate the Caputo fractional derivative of order `order` for a discrete signal.
    Uses the Grünwald‑Letnikov representation:
        D^α f(t_n) ≈ (1 / dt^α) * Σ_{k=0}^{n} (-1)^k * C(α, k) * f(t_{n‑k})
    """
    if order <= 0 or order >= 1:
        raise ValueError("Order must be in (0,1) for this simple implementation.")
    n = len(signal)
    derivative = np.zeros_like(signal, dtype=float)
    coeffs = np.array([(-1) ** k * fractional_binomial(order, k) for k in range(n)])
    for i in range(n):
        derivative[i] = np.sum(coeffs[: i + 1][::-1] * signal[: i + 1])
    derivative /= dt ** order
    return derivative


# ----------------------------------------------------------------------
# Core hybrid classes
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


class EndpointCircuitBreaker:
    """Simple circuit breaker tracking consecutive failures."""
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class HybridCaputoPheromone:
    """
    Combines entropy‑based pheromone weighting (Infotaxis) with a Caputo
    fractional decay kernel.  The fractional derivative modulates the pheromone
    vector, and the result drives an EndpointCircuitBreaker.
    """
    def __init__(self, order: float = 0.5, dt: float = 1.0, failure_threshold: int = 3):
        self.order = order
        self.dt = dt
        self.circuit_breaker = EndpointCircuitBreaker(failure_threshold)

    # ------------------------------------------------------------------
    # 1️⃣ Entropy‑scaled pheromone update
    # ------------------------------------------------------------------
    def entropy_scaled_pheromone(self, vector: np.ndarray) -> np.ndarray:
        """Scale a raw pheromone vector by its Shannon entropy."""
        ent = calculate_entropy(vector)
        return vector * ent

    # ------------------------------------------------------------------
    # 2️⃣ Fractional decay of the scaled pheromone
    # ------------------------------------------------------------------
    def fractional_decay(self, scaled_vec: np.ndarray) -> np.ndarray:
        """Apply Caputo fractional derivative as a decay operator."""
        deriv = caputo_fractional_derivative(scaled_vec, self.order, self.dt)
        # Decay is interpreted as subtraction of the derivative (memory‑driven loss)
        decayed = scaled_vec - deriv
        # Ensure non‑negative pheromone levels
        decayed = np.clip(decayed, 0, None)
        return decayed

    # ------------------------------------------------------------------
    # 3️⃣ Decision metric incorporating weekday weighting
    # ------------------------------------------------------------------
    def decision_score(self, pheromone_vec: np.ndarray, date: datetime) -> float:
        """
        Compute a scalar decision score:
            score = (entropy * info_gain) * weekday_weight
        where weekday_weight is the component of the sinusoidal vector that
        corresponds to the current weekday.
        """
        entropy = calculate_entropy(pheromone_vec)
        # Simulate a previous state by a small random perturbation
        prev = pheromone_vec * (1 + 0.01 * (np.random.rand(*pheromone_vec.shape) - 0.5))
        info_gain = calculate_information_gain(prev, pheromone_vec)

        weekday_weights = sinusoidal_weekday_weights(date)
        today_idx = date.weekday()  # Monday=0 … Sunday=6
        weight = weekday_weights[today_idx]

        return (entropy * info_gain) * weight

    # ------------------------------------------------------------------
    # 4️⃣ Full hybrid step
    # ------------------------------------------------------------------
    def hybrid_step(self, raw_vector: np.ndarray, date: datetime) -> float:
        """
        Execute a full hybrid iteration:
        1. Scale by entropy.
        2. Apply fractional decay.
        3. Compute decision score.
        4. Update circuit breaker based on the magnitude of decay.
        Returns the decision score.
        """
        scaled = self.entropy_scaled_pheromone(raw_vector)
        decayed = self.fractional_decay(scaled)

        # Failure condition: any element decays more than 30% in a single step
        decay_ratio = np.max((scaled - decayed) / (scaled + 1e-12))
        if decay_ratio > 0.3:
            self.circuit_breaker.record_failure()
        else:
            self.circuit_breaker.record_success()

        score = self.decision_score(decayed, date)
        return score


# ----------------------------------------------------------------------
# Additional helper: simple Minimum‑Cost Tree using Prim's algorithm
# (acts as the "graph‑based optimization" bridge)
# ----------------------------------------------------------------------
def prim_mst(weight_matrix: np.ndarray) -> float:
    """
    Compute total weight of a Minimum Spanning Tree (MST) for a fully
    connected undirected graph given by a symmetric weight matrix.
    Returns the sum of edge weights in the MST.
    """
    n = weight_matrix.shape[0]
    selected = np.zeros(n, dtype=bool)
    min_edge = np.full(n, np.inf)
    min_edge[0] = 0.0
    total = 0.0

    for _ in range(n):
        # select the unselected vertex with smallest edge weight
        u = np.argmin(min_edge + selected * np.inf)
        total += min_edge[u]
        selected[u] = True
        # update candidate edges
        for v in range(n):
            if not selected[v] and weight_matrix[u, v] < min_edge[v]:
                min_edge[v] = weight_matrix[u, v]
    return total


# ----------------------------------------------------------------------
# High‑level hybrid operation exposing at least three functions
# ----------------------------------------------------------------------
def hybrid_entropy_information(vector: np.ndarray) -> float:
    """Function 1: Returns combined entropy * information gain for a vector."""
    entropy = calculate_entropy(vector)
    perturbed = vector * (1 + 0.02 * (np.random.rand(*vector.shape) - 0.5))
    info_gain = calculate_information_gain(vector, perturbed)
    return entropy * info_gain


def hybrid_fractional_pheromone(vector: np.ndarray, order: float = 0.5) -> np.ndarray:
    """Function 2: Applies entropy scaling then Caputo fractional decay."""
    hy = HybridCaputoPheromone(order=order, dt=1.0)
    scaled = hy.entropy_scaled_pheromone(vector)
    return hy.fractional_decay(scaled)


def hybrid_decision_pipeline(vector: np.ndarray, morph: Morphology, date: datetime) -> dict:
    """
    Function 3: Full pipeline returning a dict with:
      - decision_score
      - circuit_breaker_state
      - mst_cost (graph optimization using the vector as node weights)
    """
    # 1. Build a synthetic weight matrix from the vector (outer product)
    w = np.outer(vector, vector)
    mst_cost = prim_mst(w)

    # 2. Hybrid step (includes circuit breaker)
    hy = HybridCaputoPheromone(order=0.5, dt=1.0, failure_threshold=2)
    score = hy.hybrid_step(vector, date)

    return {
        "decision_score": score,
        "circuit_breaker_open": hy.circuit_breaker.open,
        "mst_total_weight": mst_cost,
        "morphology_volume": morph.length * morph.width * morph.height,
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # synthetic high‑dimensional vector (e.g., 10‑dimensional)
    vec = np.abs(np.random.randn(10))

    # simple morphology instance
    morph = Morphology(length=2.0, width=1.5, height=0.8, mass=3.4)

    today = datetime.now(timezone.utc)

    # Run the three exposed functions
    ei = hybrid_entropy_information(vec)
    print(f"Entropy * InfoGain: {ei:.6f}")

    decayed_vec = hybrid_fractional_pheromone(vec, order=0.6)
    print(f"Decayed vector (first 3 entries): {decayed_vec[:3]}")

    result = hybrid_decision_pipeline(vec, morph, today)
    print("Hybrid pipeline result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # Ensure circuit breaker logic does not raise
    assert isinstance(result["circuit_breaker_open"], bool)
    sys.exit(0)