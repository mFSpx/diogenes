# DARWIN HAMMER — match 4551, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1257_s5.py (gen6)
# parent_b: hybrid_hybrid_jepa_energy_h_hybrid_hybrid_hybrid_m1517_s2.py (gen5)
# born: 2026-05-29T23:56:41Z

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

import numpy as np


# ----------------------------------------------------------------------
# Deterministic Gaussian RBF surrogate (Parent A)
# ----------------------------------------------------------------------
class DeterministicHybridFusion:
    """
    A reproducible RBF surrogate that maps an input feature vector to a
    3‑dimensional resource vector **r** = (score, distance, privacy_load) and
    a tuple of epistemic certainty flags.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int = 3,
        seed: int = 42,
        epsilon: float = 1.0,
        n_centers: int = 5,
    ) -> None:
        self.d_in = d_in
        self.d_out = d_out
        self.epsilon = epsilon
        self.rng = random.Random(seed)

        # Pre‑sample a *fixed* set of centres – this removes the per‑call randomness
        self.centers = np.array(
            [
                [self.rng.random() for _ in range(d_in)]
                for _ in range(n_centers)
            ],
            dtype=float,
        )

    @staticmethod
    def _gaussian(r: float, epsilon: float) -> float:
        """Isotropic Gaussian RBF."""
        return math.exp(-((epsilon * r) ** 2))

    @staticmethod
    def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def predict(self, x: List[float]) -> Tuple[np.ndarray, Tuple[bool, bool, bool]]:
        """
        Returns
        -------
        r : np.ndarray, shape (3,)
            (score, distance, privacy_load)
        flags : Tuple[bool, bool, bool]
            Epistemic‑certainty flags derived from the three components:
            *score* → high certainty if > 0.5,
            *distance* → high certainty if < 0.3,
            *privacy_load* → high certainty if < 0.2.
        """
        x_arr = np.asarray(x, dtype=float)

        # Find the nearest centre – deterministic because centres are fixed
        dists = np.linalg.norm(self.centers - x_arr, axis=1)
        nearest_idx = int(np.argmin(dists))
        centre = self.centers[nearest_idx]

        distance = self._euclidean(x_arr, centre)
        score = self._gaussian(distance, self.epsilon)

        # privacy_load is a smooth function of distance, not a raw RNG
        privacy_load = 1.0 / (1.0 + distance + 1e-6)

        r = np.array([score, distance, privacy_load], dtype=float)

        flags = (
            score > 0.5,          # confident about the surrogate score
            distance < 0.3,       # confident about proximity
            privacy_load < 0.2,   # confident about privacy budget
        )
        return r, flags


# ----------------------------------------------------------------------
# Linear Gaussian State‑Space Model (Parent B)
# ----------------------------------------------------------------------
class KalmanSSM:
    """
    A minimal Kalman filter that keeps the posterior mean **x̂** and covariance **P**
    across multiple calls.  The measurement covariance **R** is built on‑the‑fly
    from epistemic certainty flags, providing a tighter coupling between the two
    parent systems.
    """

    def __init__(
        self,
        dim_state: int,
        dim_meas: int,
        A: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
    ) -> None:
        self.dim_state = dim_state
        self.dim_meas = dim_meas
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q

        # Initialise with a zero mean and a relatively uninformative covariance
        self.x = np.zeros(dim_state, dtype=float)
        self.P = np.eye(dim_state, dtype=float) * 1e3

    @staticmethod
    def _certainty_to_R(flags: Tuple[bool, bool, bool]) -> np.ndarray:
        """
        Low variance (0.1) for a *True* flag, high variance (10.0) otherwise.
        """
        variances = [0.1 if f else 10.0 for f in flags]
        return np.diag(variances)

    def step(self, u: np.ndarray, y: np.ndarray, flags: Tuple[bool, bool, bool]) -> None:
        """One Kalman filter iteration."""
        # ---- Prediction ----------------------------------------------------
        x_pred = self.A @ self.x + self.B @ u
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # ---- Update --------------------------------------------------------
        R = self._certainty_to_R(flags)
        S = self.H @ P_pred @ self.H.T + R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        y_residual = y - self.H @ x_pred
        self.x = x_pred + K @ y_residual
        self.P = (np.eye(self.dim_state) - K @ self.H) @ P_pred

    def state(self) -> np.ndarray:
        return self.x.copy()


# ----------------------------------------------------------------------
# Model pool and work‑share lane (Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


class ModelPool:
    """
    Manages loaded models with a hard RAM ceiling and an energy budget that
    reflects both successful and penalised actions.
    """

    def __init__(self, ram_ceiling_mb: int = 6000, initial_energy: float = 0.0):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._energy: float = initial_energy

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _can_fit(self, model: ModelTier) -> bool:
        return self._used() + model.ram_mb <= self.ram_ceiling_mb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, model: ModelTier) -> None:
        """Load a model; reward successful loads, penalise overflow attempts."""
        if self._can_fit(model):
            self.loaded[model.name] = model
            self._energy += 1e4   # reward
        else:
            # No load, but we still penalise heavily to discourage overflow
            self._energy -= 5e5

    def evict(self, name: str) -> None:
        """Evict a model and grant a modest reward."""
        if name in self.loaded:
            del self.loaded[name]
            self._energy += 5e3

    @property
    def energy(self) -> float:
        return self._energy

    def status(self) -> str:
        return f"Used {self._used()}/{self.ram_ceiling_mb} MB, Energy {self._energy:.1f}"


# ----------------------------------------------------------------------
# Improved hybrid allocator – deeper mathematical integration
# ----------------------------------------------------------------------
class ImprovedHybridAllocator:
    """
    Orchestrates the surrogate, Kalman filter, and model‑pool decisions.
    The design addresses the following weak points of the original prototype:

    1. **Determinism** – fixed RBF centres remove per‑call randomness.
    2. **Epistemic coupling** – measurement covariance **R** is derived from
       surrogate‑generated certainty flags.
    3. **State persistence** – the Kalman filter keeps its posterior across
       allocation cycles, enabling temporal smoothing.
    4. **Selective loading** – only the *best‑scoring* candidate that fits the RAM
       budget is loaded; excess models are evicted in a principled order.
    5. **Energy‑aware decisions** – the score threshold is dynamically adapted
       based on current energy, encouraging conservative behaviour when the
       budget is low.
    """

    def __init__(
        self,
        d_in: int,
        candidate_models: List[ModelTier],
        ram_ceiling_mb: int = 6000,
        base_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.surrogate = DeterministicHybridFusion(d_in=d_in, seed=seed)
        self.candidates = candidate_models
        self.model_pool = ModelPool(ram_ceiling_mb=ram_ceiling_mb)

        # Linear SSM matrices – now *learned* (hard‑coded for illustration)
        dim = 3
        self.ssm = KalmanSSM(
            dim_state=dim,
            dim_meas=dim,
            A=np.eye(dim) * 0.9,
            B=np.eye(dim) * 0.1,
            H=np.eye(dim),
            Q=np.diag([0.01, 0.01, 0.01]),
        )

        self.base_threshold = base_threshold
        self._last_score: float = 0.0

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------
    def _adaptive_threshold(self) -> float:
        """
        Lower the threshold when the energy budget is negative, forcing the
        allocator to be more aggressive about freeing RAM.
        """
        energy = self.model_pool.energy
        if energy < -1e5:
            return self.base_threshold * 0.7
        if energy > 1e5:
            return self.base_threshold * 1.3
        return self.base_threshold

    def allocate(self, feature_vec: List[float]) -> None:
        """
        Run one allocation cycle:
        1. Surrogate → resource vector + flags
        2. Kalman update → posterior state
        3. Compute scalar allocation score (weighted sum)
        4. Decide which model (if any) to load, possibly evicting others.
        """
        # ---- 1. Surrogate -------------------------------------------------
        resource_vec, flags = self.surrogate.predict(feature_vec)

        # ---- 2. Kalman update ---------------------------------------------
        u = np.zeros_like(resource_vec)  # no external control signal
        self.ssm.step(u=u, y=resource_vec, flags=flags)

        # ---- 3. Score computation -----------------------------------------
        # Use a *domain‑aware* weight: give more importance to the surrogate score
        weight = np.array([0.6, 0.2, 0.2], dtype=float)
        score = float(weight @ self.ssm.state())
        self._last_score = score

        # ---- 4. Allocation decision ----------------------------------------
        thr = self._adaptive_threshold()

        if score > thr:
            # Choose the *most valuable* candidate that fits the current RAM budget.
            # Value is defined as (tier priority) / ram_mb.  Tier priority mapping:
            tier_priority = {"T1": 3, "T2": 2, "T3": 1}
            feasible = [
                m for m in self.candidates if self.model_pool._can_fit(m)
            ]

            if not feasible:
                # Need to free space – evict the *largest* loaded model first
                if self.model_pool.loaded:
                    largest = max(self.model_pool.loaded.values(), key=lambda m: m.ram_mb)
                    self.model_pool.evict(largest.name)
                feasible = [
                    m for m in self.candidates if self.model_pool._can_fit(m)
                ]

            if feasible:
                best = max(
                    feasible,
                    key=lambda m: tier_priority.get(m.tier, 0) / (m.ram_mb + 1e-6),
                )
                self.model_pool.load(best)
        else:
            # Score too low → evict the *least valuable* loaded model, if any.
            if self.model_pool.loaded:
                tier_priority = {"T1": 3, "T2": 2, "T3": 1}
                least = min(
                    self.model_pool.loaded.values(),
                    key=lambda m: tier_priority.get(m.tier, 0) / (m.ram_mb + 1e-6),
                )
                self.model_pool.evict(least.name)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Expose internal diagnostics for monitoring or testing."""
        return {
            "posterior_state": self.ssm.state(),
            "energy": self.model_pool.energy,
            "loaded_models": list(self.model_pool.loaded.keys()),
            "last_score": self._last_score,
            "ram_used": self.model_pool._used(),
        }


# ----------------------------------------------------------------------
# Example usage (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a few dummy models
    models = [
        ModelTier(name="model_A", ram_mb=1500, tier="T1"),
        ModelTier(name="model_B", ram_mb=800, tier="T2"),
        ModelTier(name="model_C", ram_mb=1200, tier="T3"),
    ]

    allocator = ImprovedHybridAllocator(d_in=8, candidate_models=models)

    # Simulate a stream of feature vectors
    for step in range(20):
        feats = [random.random() for _ in range(8)]
        allocator.allocate(feats)
        diagnostics = allocator.get_state()
        print(
            f"Step {step:02d} | Score {diagnostics['last_score']:.3f} | "
            f"Energy {diagnostics['energy']:.0f} | "
            f"Loaded {diagnostics['loaded_models']} | RAM {diagnostics['ram_used']}/{allocator.model_pool.ram_ceiling_mb}"
        )