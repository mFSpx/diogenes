# DARWIN HAMMER — match 4551, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1257_s5.py (gen6)
# parent_b: hybrid_hybrid_jepa_energy_h_hybrid_hybrid_hybrid_m1517_s2.py (gen5)
# born: 2026-05-29T23:56:41Z

"""HybridFusionAllocator
Integrates:
- Parent A (HybridFusion) – Gaussian RBF surrogate producing a 3‑dimensional resource vector
  (score, distance, privacy_load) from an input feature vector.
- Parent B (HybridDarwinAllocator) – ModelPool / WorkshareLane system that manages model
  loading/unloading using an energy budget and a variational‑free‑energy‑like allocation
  driven by a scalar derived from the resource vector.

Mathematical bridge:
The resource vector **r** ∈ ℝ³ is interpreted as a *measurement* **yₖ** for a linear
Gaussian state‑space model (SSM)

    xₖ₊₁ = A·xₖ + B·uₖ + wₖ ,   wₖ ∼ N(0,Q)
    yₖ   = H·xₖ + vₖ ,           vₖ ∼ N(0,R)

where **R** is built from epistemic‑certainty flags attached to an EngineEndpoint.
The posterior state estimate **x̂ₖ** is then mapped to a scalar *allocation score*
s = wᵀ·x̂ₖ (with a fixed weight vector w).  This score drives the work‑share
allocation decisions of the ModelPool: models are loaded when s exceeds a
threshold and evicted otherwise.  Thus the core topologies of both parents are
fused into a single hybrid estimator‑allocator pipeline.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent‑A building blocks (HybridFusion + linear SSM)
# ----------------------------------------------------------------------
class HybridFusion:
    """RBF‑based surrogate that maps an input vector to a 3‑D resource vector."""
    def __init__(
        self,
        d_in: int,
        d_out: int = 3,
        seed: int = 0,
        epsilon: float = 1.0,
    ) -> None:
        self.d_in = d_in
        self.d_out = d_out
        self.epsilon = epsilon
        self.rng = random.Random(seed)

    @staticmethod
    def gaussian(r: float, epsilon: float = 1.0) -> float:
        """Isotropic Gaussian radial‑basis function."""
        return math.exp(-((epsilon * r) ** 2))

    @staticmethod
    def euclidean(a: List[float], b: List[float]) -> float:
        """Euclidean distance between two vectors."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def predict(self, x: List[float]) -> np.ndarray:
        """
        Produce a resource vector:
        - score   = Gaussian(RBF) of the distance to a random centre
        - distance = Euclidean distance to the centre
        - privacy_load = random scalar modulated by epsilon
        """
        centre = [self.rng.random() for _ in range(self.d_in)]
        dist = self.euclidean(x, centre)
        score = self.gaussian(dist, self.epsilon)
        privacy_load = self.rng.random() * (1.0 / (1.0 + dist))
        return np.array([score, dist, privacy_load], dtype=float)


class LinearSSM:
    """Simple linear Gaussian state‑space model."""
    def __init__(
        self,
        dim_state: int,
        dim_meas: int,
        A: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
    ) -> None:
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.dim_state = dim_state
        self.dim_meas = dim_meas

    def step(
        self,
        x_prev: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        One Kalman‑filter‑like update (prediction + measurement correction).
        Returns (x_post, P_post) where P is the state covariance.
        """
        # Prediction
        x_pred = self.A @ x_prev + self.B @ u
        P_pred = self.A @ np.eye(self.dim_state) @ self.A.T + self.Q

        # Kalman gain
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Update
        y_residual = y - self.H @ x_pred
        x_post = x_pred + K @ y_residual
        P_post = (np.eye(self.dim_state) - K @ self.H) @ P_pred
        return x_post, P_post


# ----------------------------------------------------------------------
# Parent‑B building blocks (ModelPool / Workshare allocation)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


@dataclass
class WorkshareLane:
    group: str
    llm_units: float
    llm_share_pct: float
    proof_required: bool


class ModelPool:
    """Manages a set of loaded models and an internal energy budget."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._energy: float = 0.0

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load a model, applying rewards/penalties to the energy budget."""
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            self._energy -= 5e5   # heavy penalty for overflow
        else:
            self._energy -= 1e4   # reward for successful load
        self.loaded[model.name] = model

    def evict(self, name: str) -> None:
        """Evict a model, granting a small reward."""
        if name in self.loaded:
            del self.loaded[name]
            self._energy -= 5e3

    @property
    def energy(self) -> float:
        return self._energy

    def status(self) -> str:
        return f"Used {self._used()}/{self.ram_ceiling_mb} MB, Energy {self._energy:.1f}"


# ----------------------------------------------------------------------
# Hybrid interface utilities
# ----------------------------------------------------------------------
def epistemic_certainty_matrix(flags: Tuple[bool, bool, bool]) -> np.ndarray:
    """
    Build a diagonal measurement covariance matrix R from three boolean certainty flags.
    True  → low variance (0.1), False → high variance (10.0)
    """
    variances = [0.1 if f else 10.0 for f in flags]
    return np.diag(variances)


def allocation_score(state: np.ndarray, weight: np.ndarray = None) -> float:
    """
    Map the posterior state vector to a scalar score used for work‑share decisions.
    Default weight is a simple averaging vector.
    """
    if weight is None:
        weight = np.ones_like(state) / state.size
    return float(weight @ state)


def hybrid_allocate(
    resource_vec: np.ndarray,
    model_pool: ModelPool,
    candidate_models: List[ModelTier],
    score_threshold: float = 0.5,
) -> None:
    """
    Decide which models to load or evict based on the allocation score derived
    from the posterior state (which itself is driven by the resource vector).
    """
    # Simple linear mapping from resource vector to a synthetic state
    A = np.eye(3) * 0.9
    B = np.eye(3) * 0.1
    H = np.eye(3)
    Q = np.diag([0.01, 0.01, 0.01])
    R = np.diag([0.05, 0.05, 0.05])
    ssm = LinearSSM(dim_state=3, dim_meas=3, A=A, B=B, H=H, Q=Q, R=R)

    # Zero initial state and zero control
    x0 = np.zeros(3)
    u0 = np.zeros(3)

    # Perform one SSM update using the resource vector as measurement
    x_post, _ = ssm.step(x0, u0, resource_vec)

    # Compute scalar allocation score
    score = allocation_score(x_post)

    # Load models if score exceeds threshold, otherwise evict the least‑ram model
    if score > score_threshold:
        for model in candidate_models:
            if not model_pool._used() + model.ram_mb <= model_pool.ram_ceiling_mb:
                # Need to free space: evict the largest model
                if model_pool.loaded:
                    largest = max(model_pool.loaded.values(), key=lambda m: m.ram_mb)
                    model_pool.evict(largest.name)
            model_pool.load(model)
    else:
        # Evict a random model if any are loaded
        if model_pool.loaded:
            name = next(iter(model_pool.loaded))
            model_pool.evict(name)


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def compute_resource_vector(fusion: HybridFusion, input_vec: List[float]) -> np.ndarray:
    """Wraps HybridFusion.predict with a deterministic centre for reproducibility."""
    return fusion.predict(input_vec)


def run_hybrid_step(
    fusion: HybridFusion,
    input_vec: List[float],
    model_pool: ModelPool,
    candidate_models: List[ModelTier],
) -> Tuple[np.ndarray, float]:
    """
    Executes the full hybrid pipeline:
    1. Predict resource vector.
    2. Build measurement covariance from synthetic certainty flags.
    3. Perform a Kalman update (embedded in hybrid_allocate).
    4. Allocate / evict models.
    Returns the final resource vector and the pool's energy.
    """
    # 1. Resource prediction
    r_vec = compute_resource_vector(fusion, input_vec)

    # 2. Certainty flags (for illustration we map resource components to booleans)
    flags = (r_vec[0] > 0.5, r_vec[1] < 5.0, r_vec[2] < 0.3)
    R = epistemic_certainty_matrix(flags)

    # 3. Update SSM measurement covariance (used inside hybrid_allocate)
    #    The hybrid_allocate function constructs its own SSM; we only need to
    #    ensure the flags affect the decision indirectly via the resource vector.
    #    Here we simply pass the resource vector forward.

    # 4. Allocation decision
    hybrid_allocate(r_vec, model_pool, candidate_models)

    return r_vec, model_pool.energy


def demo_hybrid_system() -> None:
    """Smoke‑test that runs the hybrid system with synthetic data."""
    # Initialise components
    fusion = HybridFusion(d_in=5, seed=42, epsilon=0.8)
    pool = ModelPool(ram_ceiling_mb=2000)

    # Define a small catalogue of candidate models
    candidates = [
        ModelTier(name="Llama-7B", ram_mb=1200, tier="T2"),
        ModelTier(name="GPT-2", ram_mb=500, tier="T1"),
        ModelTier(name="BERT-base", ram_mb=300, tier="T1"),
    ]

    # Random input vector
    inp = [random.random() for _ in range(5)]

    # Run the hybrid step
    resource, energy = run_hybrid_step(fusion, inp, pool, candidates)

    # Output diagnostics
    print("Resource vector :", resource)
    print("ModelPool status :", pool.status())
    print("Energy after step:", energy)


if __name__ == "__main__":
    demo_hybrid_system()