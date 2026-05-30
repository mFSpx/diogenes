# DARWIN HAMMER — match 3242, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1223_s4.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m1260_s0.py (gen4)
# born: 2026-05-29T23:48:39Z

"""Hybrid Algorithm Fusion of Ternary Lens Koopman Dynamics (Parent A) and
Entropy‑Driven Model Pool Management (Parent B).

Mathematical Bridge
-------------------
Parent A provides a Koopman operator **K** that linearly propagates a
state‑vector **xₜ** (e.g. resource‑usage descriptors) to the next time step:
 xₜ₊₁ = K·xₜ.

Parent B manages a pool of models and uses Shannon entropy of feature‑usage
counts to decide which models to load or evict.

The fusion treats the **resource‑usage vector** as the common state:
 x = [r, n, e]ᵀ
where *r* is total RAM used by the pool, *n* the number of loaded models,
and *e* the current entropy of feature counts.  The Koopman operator predicts
future resource pressure; the predicted *r* and *e* feed the entropy‑based
eviction policy, creating a closed loop that couples dynamics (Parent A) with
decision hygiene (Parent B)."""

import sys
import math
import random
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable
import numpy as np

# ----------------------------------------------------------------------
# Parent B – Model pool and entropy utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str  # e.g. "T1", "T2", "T3"


class ModelPool:
    """Manages a RAM‑bounded pool of models with simple eviction."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        """Load a model respecting tier exclusivity and RAM ceiling."""
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        """Load a model, evicting oldest entries until enough RAM is free."""
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # FIFO eviction
            evict_name = next(iter(self.loaded))
            del self.loaded[evict_name]
        self.load(model)

    def snapshot(self) -> Tuple[int, int]:
        """Return (total_ram_used, number_of_models)."""
        return self._used(), len(self.loaded)


def shannon_entropy(feature_counts: Dict[str, int]) -> float:
    """Standard Shannon entropy over a histogram of feature occurrences."""
    total = sum(feature_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in feature_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# ----------------------------------------------------------------------
# Parent A – Minimal Koopman operator utilities
# ----------------------------------------------------------------------
class KoopmanOperator:
    """
    Constructs a Koopman matrix K from paired snapshots (x_t, x_{t+1})
    using least‑squares regression:  X' ≈ K·X  →  K = X'·X⁺ .
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.K = np.eye(dim)  # identity as a safe default

    def fit(self, snapshots: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Fit K from a list of (x_t, x_{t+1}) pairs."""
        if not snapshots:
            return
        X = np.column_stack([s[0] for s in snapshots])
        Xp = np.column_stack([s[1] for s in snapshots])
        # Pseudo‑inverse for robustness
        X_pinv = np.linalg.pinv(X)
        self.K = Xp @ X_pinv

    def predict(self, x: np.ndarray) -> np.ndarray:
        """One‑step forward prediction."""
        return self.K @ x


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def build_state_vector(pool: ModelPool, feature_counts: Dict[str, int]) -> np.ndarray:
    """
    Assemble the shared state vector x = [ram_used, num_models, entropy]^T.
    """
    ram_used, n_models = pool.snapshot()
    entropy = shannon_entropy(feature_counts)
    return np.array([float(ram_used), float(n_models), float(entropy)], dtype=np.float64)


def hybrid_predict_and_decide(
    koop: KoopmanOperator,
    current_state: np.ndarray,
    pool: ModelPool,
    candidate: ModelTier,
    feature_counts: Dict[str, int],
) -> Tuple[bool, np.ndarray]:
    """
    Predict the next state, then decide whether the *candidate* model can be
    loaded without violating the RAM ceiling.  If the prediction indicates
    an overload, evict the model with the lowest contribution to entropy
    (heuristic: smallest RAM among loaded models).

    Returns
    -------
    loaded : bool
        True if the candidate ends up loaded, False otherwise.
    new_state : np.ndarray
        Updated state after the load/eviction step.
    """
    # 1. Predict next resource pressure
    pred_state = koop.predict(current_state)

    # 2. Simple policy: if predicted RAM exceeds 90 % of ceiling, evict.
    ram_ceiling = pool.ram_ceiling_mb
    predicted_ram = pred_state[0]

    if predicted_ram > 0.9 * ram_ceiling:
        # Evict the loaded model with smallest RAM (least impact on entropy)
        if pool.loaded:
            evict_name = min(pool.loaded, key=lambda n: pool.loaded[n].ram_mb)
            del pool.loaded[evict_name]

    # 3. Attempt to load the candidate (with eviction fallback)
    try:
        pool.load(candidate)
        loaded = True
    except RuntimeError:
        # Forced eviction of the smallest‑RAM model and retry
        if pool.loaded:
            evict_name = min(pool.loaded, key=lambda n: pool.loaded[n].ram_mb)
            del pool.loaded[evict_name]
        try:
            pool.load(candidate)
            loaded = True
        except RuntimeError:
            loaded = False

    # 4. Update feature counts (mock: increment a random feature)
    if loaded:
        feat = random.choice(list(feature_counts.keys())) if feature_counts else "default"
        feature_counts[feat] = feature_counts.get(feat, 0) + 1

    new_state = build_state_vector(pool, feature_counts)
    return loaded, new_state


def hybrid_training_step(
    history: List[np.ndarray],
    pool: ModelPool,
    feature_counts: Dict[str, int],
    candidate_models: Iterable[ModelTier],
) -> None:
    """
    Perform one training iteration:
    * Fit a Koopman operator on the supplied *history* of state vectors.
    * Predict the next state.
    * Attempt to load the next candidate model using the hybrid decision rule.
    """
    dim = history[0].shape[0] if history else 3
    koop = KoopmanOperator(dim)

    # Build snapshot pairs (x_t, x_{t+1})
    snapshots = [(history[i], history[i + 1]) for i in range(len(history) - 1)]
    koop.fit(snapshots)

    current_state = history[-1] if history else build_state_vector(pool, feature_counts)

    for cand in candidate_models:
        loaded, new_state = hybrid_predict_and_decide(
            koop, current_state, pool, cand, feature_counts
        )
        # Advance the simulation with the new state
        current_state = new_state
        # Append to history for the next iteration
        history.append(new_state)
        if loaded:
            break  # stop after first successful load


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a modest model pool
    pool = ModelPool(ram_ceiling_mb=2000)

    # Pre‑populate with a few low‑tier models
    initial_models = [
        ModelTier(name="mA", ram_mb=200, tier="T1"),
        ModelTier(name="mB", ram_mb=350, tier="T1"),
        ModelTier(name="mC", ram_mb=150, tier="T2"),
    ]
    for m in initial_models:
        pool.load(m)

    # Feature histogram (mock)
    feature_counts = {"feat1": 10, "feat2": 5, "feat3": 2}

    # Generate an initial history of 5 state vectors using random perturbations
    history: List[np.ndarray] = []
    base_state = build_state_vector(pool, feature_counts)
    for _ in range(5):
        perturb = np.random.normal(scale=20.0, size=base_state.shape)
        history.append(base_state + perturb)

    # Candidate models to attempt loading
    candidates = [
        ModelTier(name="bigT3", ram_mb=800, tier="T3"),
        ModelTier(name="midT2", ram_mb=500, tier="T2"),
        ModelTier(name="smallT1", ram_mb=120, tier="T1"),
    ]

    # Run a hybrid training step
    hybrid_training_step(history, pool, feature_counts, candidates)

    # Print final pool status
    print("Final loaded models:")
    for name, mdl in pool.loaded.items():
        print(f" - {name}: {mdl.ram_mb} MB, tier {mdl.tier}")

    print(f"Final entropy: {shannon_entropy(feature_counts):.4f}")
    print(f"Final state vector: {build_state_vector(pool, feature_counts)}")