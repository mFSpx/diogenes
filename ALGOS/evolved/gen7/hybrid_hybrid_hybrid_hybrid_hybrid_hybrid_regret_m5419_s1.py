# DARWIN HAMMER — match 5419, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_physar_m1288_s0.py (gen6)
# parent_b: hybrid_hybrid_regret_engine_hybrid_hybrid_hybrid_m2704_s2.py (gen5)
# born: 2026-05-30T00:01:46Z

"""Hybrid algorithm integrating:
- Parent A: Ternary-Tree Transform (TTT) conductance updates and ModelPool management.
- Parent B: Regret‑weighted strategy, MinHash signatures, and similarity measures.

Mathematical bridge:
The TTT weight matrix `W` is interpreted as edge conductances of a
minimum‑cost tree.  Regret values computed from actions (Parent B) are
used as a scalar field that modulates these conductances:
    ΔW = -η · (regret ⊗ 1) ∘ σ(similarity(sig_action, sig_model))
where `η` is a learning rate, `⊗` denotes outer product, `∘` element‑wise
multiplication, `σ` is a sigmoid squashing function, and the similarity
term fuses the MinHash signatures of actions with a (placeholder)
signature of the current model.  The updated conductances then affect
both the TTT self‑supervised loss and the eviction policy of the
ModelPool.  This single system therefore couples the self‑supervised
learning dynamics of Parent A with the regret‑aware decision logic of
Parent B."""

import numpy as np
import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, asdict
from typing import Iterable, List, Tuple, Dict

# ----------------------------------------------------------------------
# Shared data structures
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
    vram_mb: int = 0          # optional, kept for compatibility with Parent B

# ----------------------------------------------------------------------
# Model pool (from Parent A)
# ----------------------------------------------------------------------
class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

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

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            # evict the oldest entry (FIFO policy)
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)

# ----------------------------------------------------------------------
# TTT utilities (from Parent A)
# ----------------------------------------------------------------------
def init_ttt(d_in: int, d_out: int = None, scale: float = 0.01, seed: int = 0) -> np.ndarray:
    """Initialize weight matrix `W` of shape (d_out, d_in)."""
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    """
    Self‑supervised loss for TTT.

    If `target` is None, use reconstruction loss:
        ||W x - x||²
    Otherwise use supervised squared error:
        ||W x - target||²
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1‑D vector")
    if target is None:
        pred = W @ x
        diff = pred - x
    else:
        if target.shape != (W.shape[0],):
            raise ValueError("target shape mismatch")
        pred = W @ x
        diff = pred - target
    return float(np.linalg.norm(diff) ** 2)

# ----------------------------------------------------------------------
# Signature & similarity utilities (from Parent B)
# ----------------------------------------------------------------------
def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError('k must be positive')
    if not toks:
        return [((1 << 64) - 1)] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError('signatures must have equal length')
    if not sig_a:
        raise ValueError('signatures must not be empty')
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )

# ----------------------------------------------------------------------
# Regret‑weighted conductance update (the fusion core)
# ----------------------------------------------------------------------
def compute_regret_weighted_conductance(
    W: np.ndarray,
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    model_sig: List[int],
    learning_rate: float = 0.01,
) -> np.ndarray:
    """
    Fuse Parent A and Parent B:

    1. Compute regret per action (Parent B).
    2. Build a MinHash signature for each action id.
    3. Measure similarity between each action signature and the
       provided model signature.
    4. Modulate the regret signal with the similarity (via sigmoid).
    5. Apply the resulting scalar field as a conductance update to rows of `W`.

    The returned matrix has the same shape as `W`.
    """
    if W.shape[0] != len(actions):
        raise ValueError("Number of rows of W must equal number of actions")

    # ---- 1. Regret computation ------------------------------------------------
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    regret = np.array([a.expected_value - cf.get(a.id, 0.0) for a in actions], dtype=float)

    # ---- 2‑4. Similarity‑modulated scaling ------------------------------------
    # Build signatures for actions (one per id)
    action_sigs = [signature([a.id]) for a in actions]   # each is length 128
    sims = np.array([similarity(sig, model_sig) for sig in action_sigs], dtype=float)
    # Pass similarity through sigmoid to obtain a smooth weighting in (0,1)
    sim_weights = sigmoid(sims * 10.0)  # amplify contrast

    # Final modulation factor per row
    modulation = regret * sim_weights

    # ---- 5. Conductance update -------------------------------------------------
    # Simple conductance rule: ΔW = -η * modulation ⊗ 1 (broadcast over columns)
    delta = -learning_rate * modulation[:, None] * np.ones_like(W)
    return W + delta

# ----------------------------------------------------------------------
# High‑level hybrid step
# ----------------------------------------------------------------------
def hybrid_step(
    pool: ModelPool,
    model: ModelTier,
    W: np.ndarray,
    x: np.ndarray,
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    load_if_loss_above: float = 1.0,
) -> Tuple[ModelPool, np.ndarray, float]:
    """
    Perform one hybrid iteration:

    * Compute TTT loss.
    * Update conductances using regret‑weighted rule.
    * If loss exceeds a threshold, attempt to load the model (with eviction).

    Returns the (potentially mutated) pool, the updated weight matrix,
    and the scalar loss.
    """
    # 1. loss
    loss = ttt_loss(W, x)

    # 2. model signature (using its name as token)
    model_sig = signature([model.name])

    # 3. conductance update
    W_new = compute_regret_weighted_conductance(
        W, actions, counterfactuals, model_sig, learning_rate=0.005
    )

    # 4. conditional loading
    if loss > load_if_loss_above and not pool.is_loaded(model.name):
        try:
            pool.load_with_eviction(model)
        except Exception as e:
            # In a real system we would log; here we simply ignore loading failures.
            pass

    return pool, W_new, loss

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(0)
    np.random.seed(0)

    # Dummy model and pool
    model = ModelTier(name="demo_model", ram_mb=500, tier="T1", vram_mb=200)
    pool = ModelPool(ram_ceiling_mb=2000)

    # Initialise TTT weight matrix (5 actions, input dim 3)
    W = init_ttt(d_in=3, d_out=5, scale=0.02, seed=42)

    # Random input vector
    x = np.random.randn(3)

    # Create actions and counterfactuals
    actions = [
        MathAction(id=f"a{i}", expected_value=random.uniform(-1, 1))
        for i in range(5)
    ]
    counterfactuals = [
        MathCounterfactual(action_id=actions[i].id,
                           outcome_value=random.uniform(-1, 1),
                           probability=random.uniform(0.5, 1.0))
        for i in range(5)
    ]

    # Run a single hybrid step
    pool, W, loss = hybrid_step(pool, model, W, x, actions, counterfactuals)

    # Simple assertions to ensure no runtime errors
    assert isinstance(pool, ModelPool)
    assert W.shape == (5, 3)
    assert isinstance(loss, float)

    # Print a concise summary
    print(f"Loss: {loss:.4f}")
    print(f"Model loaded: {pool.is_loaded(model.name)}")
    print(f"Updated conductances (first row): {W[0][:5]}")