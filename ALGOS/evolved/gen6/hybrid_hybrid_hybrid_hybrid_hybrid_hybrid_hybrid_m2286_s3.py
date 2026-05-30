# DARWIN HAMMER — match 2286, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_privac_hybrid_hybrid_hybrid_m606_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_rlct_g_hybrid_hybrid_hybrid_m480_s4.py (gen4)
# born: 2026-05-29T23:41:50Z

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
    if total_records <= 0:
        raise ValueError("total_records must be positive")
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
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

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
    rti = righting_time_index(m)
    return max(0.0, min(1.0, rti / max_index))

class PheromoneRLCTSystem:
    def __init__(self):
        self.pheromone_signals: Dict[str, float] = {}

    @staticmethod
    def estimate_rlct_from_losses(train_losses_per_n: List[float], n_values: List[float]) -> float:
        if len(train_losses_per_n) != len(n_values):
            raise ValueError("train_losses_per_n and n_values must have the same length")
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
    dummy_ns = np.logspace(1, 5, len(models)).tolist()

    rlct_estimate = PheromoneRLCTSystem.estimate_rlct_from_losses(dummy_losses, dummy_ns)

    # Sort models by pheromone signal (highest first)
    model_signals = [(m.name, rlct_system.pheromone_signals.get(m.name, 0.0)) for m in models]
    model_signals.sort(key=lambda x: x[1], reverse=True)

    # Select models greedily
    selected_models = []
    total_load = 0.0
    selection_vector = np.zeros(len(models), dtype=np.float64)
    for model_name, _ in model_signals:
        model_idx = next((i for i, m in enumerate(models) if m.name == model_name), None)
        if model_idx is not None:
            selection_vector[model_idx] = 1.0
            current_load = compute_hybrid_load(models, selection_vector, rlct_estimate)
            if current_load <= load_capacity:
                selected_models.append(model_name)
                total_load = current_load
            else:
                break

    return selected_models, total_load