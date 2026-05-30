# DARWIN HAMMER — match 1284, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_sparse_wta_hy_m173_s3.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m824_s0.py (gen5)
# born: 2026-05-29T23:34:58Z

"""Hybrid algorithm merging Parent A (resource‑aware ModelPool with MinHash signatures) 
and Parent B (Regret‑Weighted Strategy modulated by a variational free‑energy term).

Mathematical bridge:
- The hidden state of the regret‑weighted bandit is represented by a MinHash
  signature vector `h ∈ ℤ^k` computed from the current set of action identifiers.
- This signature is fed into a variational free‑energy functional  
  `F(h;θ) = Σ_i log(1 + exp(-θ_i * h_i))`, which yields a scalar `F`.
- The free‑energy `F` rescales the regret‑based weight update:
  `w_i ← w_i * exp(-η * (regret_i + λ * F))`,
  where `η` is a learning rate and `λ` couples the information‑theoretic
  cost (free energy) with the regret term.

The resulting system respects the memory‑budget constraints of `ModelPool`
while performing regret‑aware updates informed by an information‑theoretic
penalty derived from the MinHash representation of the action set.
"""

import sys
import math
import random
import hashlib
import pathlib
from dataclasses import dataclass, field
from typing import Iterable, List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Parent A – ModelPool & MinHash utilities
# ----------------------------------------------------------------------
class ModelPool:
    """Memory‑constrained pool that can load and evict ModelTier objects."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.load_time: Dict[str, float] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise Exception("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise Exception("RAM ceiling exceeded")
        self.loaded[model.name] = model
        self.load_time[model.name] = float(np.datetime64('now'))

    def load_with_eviction(self, model: ModelTier) -> None:
        """Load `model`, evicting least‑recently‑used entries until there is space."""
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            lru_name = min(self.load_time, key=self.load_time.get)
            del self.loaded[lru_name]
            del self.load_time[lru_name]
        self.load(model)


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit integer hash (Blake2b)."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """Compute a MinHash signature of length `k` for the iterable `tokens`."""
    seeds = list(range(k))
    signature = []
    for seed in seeds:
        min_h = sys.maxsize
        for token in tokens:
            h = _hash(seed, token)
            if h < min_h:
                min_h = h
        signature.append(min_h)
    return signature


# ----------------------------------------------------------------------
# Parent B – Regret‑Weighted Strategy & Variational Free Energy
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def extract_master_vector(text: str, dim: int = 12) -> np.ndarray:
    """Deterministic pseudo‑random vector derived from `text`."""
    seed = _hash(123, text)
    rng = random.Random(seed)
    return np.array([rng.random() for _ in range(dim)], dtype=np.float64)


def variational_free_energy(hidden_vec: np.ndarray, theta: np.ndarray) -> float:
    """
    Simple variational free‑energy functional:
        F = Σ_i log(1 + exp(-θ_i * h_i))

    Parameters
    ----------
    hidden_vec : np.ndarray
        The hidden representation (e.g., MinHash signature normalized).
    theta : np.ndarray
        Parameter vector of the same shape as `hidden_vec`.

    Returns
    -------
    float
        Scalar free‑energy value.
    """
    if hidden_vec.shape != theta.shape:
        raise ValueError("Shapes of hidden_vec and theta must match")
    return float(np.sum(np.log1p(np.exp(-theta * hidden_vec))))


@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))


# ----------------------------------------------------------------------
# Hybrid core – three public functions
# ----------------------------------------------------------------------
def compute_hidden_state(actions: List[MathAction], k: int = 64) -> np.ndarray:
    """
    Build a normalized hidden state vector from the MinHash signature of the
    current action identifiers.
    """
    ids = [a.id for a in actions]
    raw_sig = minhash_signature(ids, k=k)                 # List[int] length k
    sig_arr = np.array(raw_sig, dtype=np.float64)
    # Normalization to (0,1) for numerical stability
    sig_norm = (sig_arr - sig_arr.min()) / (sig_arr.ptp() + 1e-12)
    return sig_norm


def regret_weighted_update(
    actions: List[MathAction],
    outcomes: Dict[str, float],
    store: StoreState,
    eta: float = 0.1,
    lambda_fe: float = 0.05,
    theta: np.ndarray = None,
) -> List[MathAction]:
    """
    Perform a regret‑weighted update where the free‑energy term derived from the
    MinHash hidden state modulates the weight scaling.

    Parameters
    ----------
    actions : List[MathAction]
        Current actions with their expected values.
    outcomes : Dict[str, float]
        Observed outcomes keyed by action id.
    store : StoreState
        Reservoir‑style store used to track inflow/outflow.
    eta : float
        Learning rate for regret scaling.
    lambda_fe : float
        Coupling coefficient for the free‑energy term.
    theta : np.ndarray | None
        Parameter vector for the free‑energy functional; if None a default
        vector of ones is used.

    Returns
    -------
    List[MathAction]
        Updated actions with new expected values.
    """
    # 1. Compute regret = expected - observed (positive regret means over‑optimism)
    regrets = {}
    inflow, outflow = [], []
    for a in actions:
        obs = outcomes.get(a.id, a.expected_value)  # fallback to expected if missing
        regret = a.expected_value - obs
        regrets[a.id] = regret
        # Use regret sign to decide inflow/outflow for StoreState
        if regret > 0:
            outflow.append(regret)
        else:
            inflow.append(-regret)

    # 2. Update the store (captures a global resource‑level analogue)
    store.update(inflow, outflow)

    # 3. Build hidden state from action ids and compute free energy
    hidden = compute_hidden_state(actions, k=64)
    if theta is None:
        theta = np.ones_like(hidden)
    fe = variational_free_energy(hidden, theta)

    # 4. Rescale regrets with free‑energy coupling and apply exponential weighting
    updated_actions = []
    for a in actions:
        r = regrets[a.id]
        scaling = math.exp(-eta * (r + lambda_fe * fe))
        new_exp = a.expected_value * scaling
        # Keep cost/risk unchanged; they could be updated similarly if desired
        updated_actions.append(MathAction(id=a.id,
                                          expected_value=new_exp,
                                          cost=a.cost,
                                          risk=a.risk))
    return updated_actions


def hybrid_step(
    actions: List[MathAction],
    outcomes: Dict[str, float],
    model_pool: ModelPool,
    store: StoreState,
    models_to_load: List[ModelTier],
) -> List[MathAction]:
    """
    One iteration of the hybrid algorithm:
      1. Load required models respecting the RAM ceiling.
      2. Perform regret‑weighted update with free‑energy modulation.
      3. Return the refreshed action list.
    """
    # Load models, evicting if necessary
    for mt in models_to_load:
        if not model_pool.is_loaded(mt.name):
            model_pool.load_with_eviction(mt)

    # Core hybrid update
    updated = regret_weighted_update(actions, outcomes, store)

    return updated


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny model pool
    pool = ModelPool(ram_ceiling_mb=500)

    # Dummy models
    models = [
        ModelTier(name="tiny_A", ram_mb=100, tier="T1"),
        ModelTier(name="tiny_B", ram_mb=150, tier="T2"),
    ]

    # Initial actions
    actions = [
        MathAction(id="a1", expected_value=1.0),
        MathAction(id="a2", expected_value=0.5),
        MathAction(id="a3", expected_value=0.2),
    ]

    # Simulated outcomes (some stochasticity)
    outcomes = {
        "a1": 0.8,
        "a2": 0.6,
        "a3": 0.1,
    }

    # Store state
    store = StoreState()

    # Run a single hybrid step
    new_actions = hybrid_step(
        actions=actions,
        outcomes=outcomes,
        model_pool=pool,
        store=store,
        models_to_load=models,
    )

    # Print results to verify execution
    for act in new_actions:
        print(f"Action {act.id}: expected_value={act.expected_value:.4f}")

    # Show store state
    print(f"Store level: {store.level:.4f}, dance: {store.dance:.4f}")