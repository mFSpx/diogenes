# DARWIN HAMMER — match 1660, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_semant_hybrid_hybrid_krampu_m787_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_endpoint_circ_m199_s6.py (gen3)
# born: 2026-05-29T23:38:11Z

import numpy as np
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


# ---------- Unified Core Components ----------
@dataclass(frozen=True)
class Morphology:
    """Unified morphology descriptor used by both parent systems."""
    name: str
    length: float
    width: float
    height: float
    mass: float          # physical mass (kg)
    ram_mb: float = 0.0  # optional RAM footprint for model‑pool logic


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def righting_time_index(m: Morphology, b: float = 1.0 / 3.0,
                        k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    fi = flatness_index(m.length, m.width, m.height)
    return (m.mass ** b) * math.exp(k * fi) / neck_lever


def recovery_priority(m: Morphology, max_index: float = 10.0) -> float:
    """Normalized priority in [0,1] derived from righting time."""
    return max(0.0, min(1.0, righting_time_index(m) / max_index))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if den == 0 else float(np.dot(a, b) / den)


# ---------- Model Pool (Parent B) ----------
class ModelPool:
    """Manages loaded models respecting a RAM ceiling."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, Morphology] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used_ram(self) -> float:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, model: Morphology) -> bool:
        """True if the model fits within remaining RAM."""
        return (self._used_ram() + model.ram_mb) <= self.ram_ceiling_mb

    def load(self, model: Morphology) -> None:
        """Load a model if RAM permits; raise otherwise."""
        if self.is_loaded(model.name):
            return
        if not self.can_load(model):
            raise RuntimeError(
                f"Insufficient RAM to load {model.name}: "
                f"required {model.ram_mb} MB, available {self.ram_ceiling_mb - self._used_ram():.1f} MB."
            )
        self.loaded[model.name] = model

    def unload(self, name: str) -> None:
        self.loaded.pop(name, None)


# ---------- Graph Sheaf (Hybrid Structure) ----------
class GraphSheaf:
    """Encapsulates feature vectors, edge‑weights, and curvature."""
    def __init__(self, morphologies: List[Morphology]):
        self.num_nodes = len(morphologies)
        self.feature_vectors = np.zeros((self.num_nodes, 3))
        self.edge_weights = np.zeros((self.num_nodes, self.num_nodes))
        self.curvature = np.zeros((self.num_nodes, self.num_nodes))
        self._populate(morphologies)

    def _populate(self, morphologies: List[Morphology]) -> None:
        for i, m in enumerate(morphologies):
            self.feature_vectors[i] = np.array([m.length, m.width, m.height])
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                # Edge weight from cosine similarity (semantic proximity)
                self.edge_weights[i, j] = 1.0 - cosine_similarity(
                    self.feature_vectors[i], self.feature_vectors[j]
                )
                # Dummy curvature metric – can be replaced by a true geometric computation
                self.curvature[i, j] = abs(self.feature_vectors[i, 0] - self.feature_vectors[j, 0]) / (
                    max(self.feature_vectors[i, 0], self.feature_vectors[j, 0]) + 1e-9
                )


def discrete_caputo_fractional_diffusion(
    edge_matrix: np.ndarray,
    priority_vec: np.ndarray,
    alpha: float = 0.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Implements a simple Caputo‑type fractional diffusion on a graph:
        (I - α·W)⁻¹ p
    where W is the edge‑weight matrix and p the priority vector.
    The inversion is regularised to avoid singularities.
    """
    n = edge_matrix.shape[0]
    I = np.eye(n)
    A = I - alpha * edge_matrix
    # Regularise diagonal if matrix is near‑singular
    diag_reg = np.diag(np.where(np.abs(np.diag(A)) < eps, eps, 0.0))
    A_reg = A + diag_reg
    return np.linalg.solve(A_reg, priority_vec)


# ---------- Hybrid Functions ----------
def hybrid_recovery_priority(
    morphologies: List[Morphology],
    max_index: float = 10.0,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Computes a diffusion‑enhanced recovery priority vector.
    Steps:
        1. Compute raw priorities from Parent A.
        2. Build GraphSheaf (features + edge weights).
        3. Apply Caputo fractional diffusion using the edge matrix.
    Returns the diffused priority vector (still in [0,1] after re‑normalisation).
    """
    raw_priorities = np.array([recovery_priority(m, max_index) for m in morphologies])
    sheaf = GraphSheaf(morphologies)
    diffused = discrete_caputo_fractional_diffusion(sheaf.edge_weights, raw_priorities, alpha)
    # Re‑scale to [0,1] to preserve interpretability
    min_val, max_val = diffused.min(), diffused.max()
    if max_val - min_val < 1e-9:
        return np.clip(diffused, 0.0, 1.0)
    return np.clip((diffused - min_val) / (max_val - min_val), 0.0, 1.0)


def hybrid_model_pool(
    morphologies: List[Morphology],
    ram_ceiling_mb: int = 6000,
) -> ModelPool:
    """
    Loads all morphologies into a ModelPool, using the explicit `ram_mb`
    attribute for RAM accounting (instead of the ambiguous `mass` field).
    """
    pool = ModelPool(ram_ceiling_mb)
    for m in morphologies:
        if not pool.is_loaded(m.name):
            pool.load(m)
    return pool


def hybrid_endpoint_health(
    morphologies: List[Morphology],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Provides a health metric derived from the same fractional diffusion
    used for recovery priority, but applied to the curvature‑filtered
    edge matrix to emphasise geometric constraints.
    """
    raw_priorities = np.array([recovery_priority(m) for m in morphologies])
    sheaf = GraphSheaf(morphologies)
    # Use curvature matrix instead of pure edge weights for a stricter diffusion
    health_vec = discrete_caputo_fractional_diffusion(sheaf.curvature, raw_priorities, alpha)
    # Normalise for interpretability
    health_vec -= health_vec.min()
    if health_vec.max() > 0:
        health_vec /= health_vec.max()
    return health_vec


# ---------- Example Usage ----------
if __name__ == "__main__":
    # Define a small catalogue with explicit RAM footprints
    catalog = [
        Morphology(name="A1", length=10.0, width=5.0, height=2.0, mass=50.0, ram_mb=1200.0),
        Morphology(name="B2", length=20.0, width=10.0, height=4.0, mass=100.0, ram_mb=2500.0),
        Morphology(name="C3", length=15.0, width=7.5, height=3.0, mass=75.0, ram_mb=1800.0),
    ]

    print("Diffused Recovery Priorities:")
    print(hybrid_recovery_priority(catalog))

    print("\nModelPool State:")
    pool = hybrid_model_pool(catalog, ram_ceiling_mb=6000)
    for name, model in pool.loaded.items():
        print(f"{name}: RAM {model.ram_mb} MB, Mass {model.mass} kg")

    print("\nEndpoint Health Vector:")
    print(hybrid_endpoint_health(catalog))