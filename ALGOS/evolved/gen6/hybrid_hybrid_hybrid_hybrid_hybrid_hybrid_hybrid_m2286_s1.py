# DARWIN HAMMER — match 2286, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_hybrid_hybrid_m606_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m480_s4.py (gen4)
# born: 2026-05-29T23:41:50Z

"""Hybrid Algorithm integrating privacy‑risk (Parent A) with morphology‑driven recovery priority and RLCT estimation (Parent B).

Mathematical Bridge:
- The Fisher score from Parent A is treated as a *sensitivity* factor.
- This sensitivity is scaled by the RLCT estimate (log‑log regression) from Parent B, yielding a **Fisher‑RLCT weight**.
- Morphology‑derived `recovery_priority` modulates the contribution of each model to the total system load.
- The total load for a selection vector **x** (binary model activation) is:

    load_i = (ram_i + privacy_i * fisher_rlct_weight_i) * (1 - recovery_priority_i)

where `privacy_i` is the reconstruction risk score, and `fisher_rlct_weight_i` = fisher_score_i * rlct_estimate.

The `HybridCircuitBreaker` opens the circuit when either the failure count exceeds a threshold
or the aggregated load surpasses a configurable capacity, thus unifying the endpoint circuit‑breaker
logic of Parent A with the morphology‑driven priority of Parent B.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np

# ------------------- Parent A components ------------------- #

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Normalized reconstruction risk in [0,1]."""
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.failures = 0
        self.open = False
        self.last_event_at = "success"

    def record_failure(self) -> None:
        self.failures += 1
        self.last_event_at = "failure"
        if self.failures >= self.failure_threshold:
            self.open = True

    def is_open(self) -> bool:
        return self.open

# ------------------- Parent B components ------------------- #

EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever

def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1]; higher righting time => lower priority."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))

class PheromoneRLCTSystem:
    def __init__(self):
        self.pheromone_signals: Dict[str, float] = {}

    @staticmethod
    def estimate_rlct_from_losses(train_losses_per_n: List[float], n_values: List[float]) -> float:
        losses = np.asarray(train_losses_per_n, dtype=np.float64)
        ns = np.asarray(n_values, dtype=np.float64)

        if np.any(ns <= np.e):
            raise ValueError("all n_values must be > e for log(log(n)) to be positive")
        if losses.shape != ns.shape:
            raise ValueError("train_losses_per_n and n_values must have the same length")

        y = np.log(np.maximum(losses, 1e-300))
        x = np.log(np.log(ns))

        x_c = x - x.mean()
        y_c = y - y.mean()
        var_x = (x_c ** 2).sum()
        if var_x < 1e-15:
            raise ValueError("n_values have no variance in log(log(n)) space")
        return float((x_c * y_c).sum() / var_x)

    def deposit(self, key: str, amount: float) -> None:
        self.pheromone_signals[key] = self.pheromone_signals.get(key, 0.0) + amount

    def evaporate(self, factor: float = 0.9) -> None:
        for k in list(self.pheromone_signals.keys()):
            self.pheromone_signals[k] *= factor
            if self.pheromone_signals[k] < 1e-6:
                del self.pheromone_signals[k]

# ------------------- Hybrid Structures ------------------- #

@dataclass
class ModelDescriptor:
    name: str
    ram_mb: float
    uqis: int                 # unique quasi‑identifiers
    total_records: int
    theta: float              # parameter for Fisher score
    center: float
    width: float
    morphology: Morphology

def hybrid_fisher_rlct_weight(theta: float, center: float, width: float, rlct_estimate: float) -> float:
    """Fisher score scaled by the RLCT estimate (the bridge)."""
    return fisher_score(theta, center, width) * rlct_estimate

def compute_hybrid_load(models: List[ModelDescriptor],
                        selection_vector: np.ndarray,
                        rlct_estimate: float) -> float:
    """
    Vectorised computation of total system load.
    load_i = (ram_i + privacy_i * fisher_rlct_weight_i) * (1 - priority_i)
    """
    if selection_vector.shape[0] != len(models):
        raise ValueError("selection_vector length must match number of models")
    # Extract arrays
    ram = np.array([m.ram_mb for m in models], dtype=np.float64)
    privacy = np.array([reconstruction_risk_score(m.uqis, m.total_records) for m in models],
                       dtype=np.float64)
    fisher_weights = np.array([hybrid_fisher_rlct_weight(m.theta, m.center, m.width, rlct_estimate)
                               for m in models], dtype=np.float64)
    priority = np.array([recovery_priority(m.morphology) for m in models], dtype=np.float64)

    # Component calculations
    base_load = ram + privacy * fisher_weights
    adjusted_load = base_load * (1.0 - priority)

    total_load = float(np.dot(adjusted_load, selection_vector))
    return total_load

class HybridCircuitBreaker:
    """Combines EndpointCircuitBreaker with load‑capacity monitoring."""
    def __init__(self,
                 failure_threshold: int = 3,
                 load_capacity: float = 1024.0):
        self.endpoint_cb = EndpointCircuitBreaker(failure_threshold)
        self.load_capacity = load_capacity

    def evaluate(self, current_load: float) -> None:
        if current_load > self.load_capacity:
            self.endpoint_cb.record_failure()
        else:
            self.endpoint_cb.record_success()

    def is_open(self) -> bool:
        return self.endpoint_cb.is_open()

# ------------------- Demonstration Functions ------------------- #

def hybrid_model_selection(models: List[ModelDescriptor],
                           rlct_system: PheromoneRLCTSystem,
                           load_capacity: float = 1024.0) -> Tuple[List[str], float]:
    """
    Select a subset of models that keeps load under capacity while
    favouring higher pheromone signals. Returns selected model names and final load.
    """
    # Estimate RLCT from a dummy loss curve (could be replaced by real data)
    dummy_losses = [random.random() + 0.1 for _ in models]
    dummy_ns = [i + math.e + 1 for i in range(len(models))]
    rlct = rlct_system.estimate_rlct_from_losses(dummy_losses, dummy_ns)

    # Simple greedy heuristic: sort by (pheromone / (ram+privacy)) descending
    scores = []
    for m in models:
        privacy = reconstruction_risk_score(m.uqis, m.total_records)
        base = m.ram_mb + privacy
        pheromone = rlct_system.pheromone_signals.get(m.name, 1.0)
        scores.append((pheromone / (base + 1e-6), m))
    scores.sort(key=lambda x: x[0], reverse=True)

    selection = np.zeros(len(models), dtype=np.float64)
    cb = HybridCircuitBreaker(load_capacity=load_capacity)

    for _, m in scores:
        idx = models.index(m)
        trial_selection = selection.copy()
        trial_selection[idx] = 1.0
        load = compute_hybrid_load(models, trial_selection, rlct)
        cb.evaluate(load)
        if not cb.is_open():
            selection[idx] = 1.0
        else:
            # stop adding more models once capacity is breached
            break

    selected_names = [m.name for i, m in enumerate(models) if selection[i] == 1.0]
    final_load = compute_hybrid_load(models, selection, rlct)
    return selected_names, final_load

def update_pheromones_based_on_load(rlct_system: PheromoneRLCTSystem,
                                    models: List[ModelDescriptor],
                                    load: float,
                                    decay_factor: float = 0.95) -> None:
    """
    Deposit pheromones inversely proportional to load for each model.
    Then evaporate globally.
    """
    for m in models:
        reward = max(0.0, 1.0 - load / 2000.0)  # arbitrary scaling
        rlct_system.deposit(m.name, reward)
    rlct_system.evaporate(decay_factor)

# ------------------- Smoke Test ------------------- #

if __name__ == "__main__":
    # Create synthetic models
    models = []
    for i in range(5):
        morph = Morphology(length= random.uniform(1.0, 5.0),
                           width= random.uniform(0.5, 3.0),
                           height= random.uniform(0.5, 2.5),
                           mass= random.uniform(10.0, 50.0))
        models.append(ModelDescriptor(
            name=f"model_{i}",
            ram_mb= random.uniform(100.0, 300.0),
            uqis= random.randint(10, 500),
            total_records= random.randint(1000, 5000),
            theta= random.uniform(-2.0, 2.0),
            center=0.0,
            width= random.uniform(0.5, 2.0),
            morphology=morph
        ))

    rlct_system = PheromoneRLCTSystem()
    selected, final_load = hybrid_model_selection(models, rlct_system, load_capacity=800.0)
    print("Selected models:", selected)
    print("Final load:", final_load)

    update_pheromones_based_on_load(rlct_system, models, final_load)
    print("Pheromone signals after update:", rlct_system.pheromone_signals)