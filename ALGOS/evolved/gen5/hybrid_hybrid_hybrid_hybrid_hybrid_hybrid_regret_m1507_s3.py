# DARWIN HAMMER — match 1507, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_hybrid_m795_s0.py (gen4)
# parent_b: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s0.py (gen3)
# born: 2026-05-29T23:36:52Z

"""Hybrid Risk‑Regret Bandit Algorithm
Integrates:
- Parent A: risk assessment, differential‑privacy aggregation, sphericity index, VRAM load estimation.
- Parent B: regret‑weighted strategy, MinHash similarity, hybrid bandit store dynamics.

Mathematical bridge:
Risk scores (Parent A) are used as a continuous similarity weight for the MinHash‑based
projection in the regret‑weighted update (Parent B).  The DP‑aggregate of the weighted
regrets provides a privacy‑preserving global signal that modulates the store’s
inflow/outflow dynamics.  Additionally, the geometric sphericity index scales the
confidence bound of each bandit action, linking the physical‑resource view of Parent A
with the decision‑theoretic view of Parent B.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np

# ---------- Parent A components ----------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Probability that a record can be re‑identified."""
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))

def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """Differential‑privacy aggregate (Laplace‑like exponential mechanism)."""
    return np.sum([x * math.exp(epsilon) for x in values]) / sensitivity

def sphericity_index(length: float, width: float, height: float) -> float:
    """Geometric factor linking shape to confidence bounds."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / max(length, width, height)

def expected_vram_load(risk_scores: Iterable[float], model_ram_mb: Iterable[int]) -> float:
    """Expected VRAM load given risk‑weighted models."""
    return np.sum([r * m for r, m in zip(risk_scores, model_ram_mb)])

# ---------- Parent B components ----------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

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
        """Standard reservoir update; inflow/outflow are regret‑derived signals."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """A bounded, gain‑scaled version of the last delta."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))

def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, 'big') + token.encode('utf-8', errors='ignore')
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'big')

def minhash_similarity(token: str, reference_tokens: List[str], seed: int = 42) -> float:
    """Jaccard‑like similarity using MinHash of integer hashes."""
    token_hash = _hash(seed, token)
    matches = sum(1 for ref in reference_tokens if _hash(seed, ref) == token_hash)
    return matches / len(reference_tokens) if reference_tokens else 0.0

# ---------- Hybrid Functions ----------
def compute_risk_scores(models: List[ModelTier],
                        unique_ids: int,
                        total_records: int) -> List[float]:
    """Map each model tier to a reconstruction risk score."""
    base_score = reconstruction_risk_score(unique_ids, total_records)
    # Distribute the base score proportionally to model RAM (larger models get higher risk)
    ram_array = np.array([m.ram_mb for m in models], dtype=float)
    if ram_array.sum() == 0:
        return [0.0] * len(models)
    weights = ram_array / ram_array.sum()
    return list(base_score * weights)

def risk_weighted_regret(actions: List[MathAction],
                         risk_scores: List[float]) -> List[float]:
    """
    Regret for each action weighted by its associated risk score.
    Regret = (expected_value - cost) * (1 + risk_score)
    """
    if len(actions) != len(risk_scores):
        raise ValueError("actions and risk_scores must be same length")
    regrets = []
    for act, r in zip(actions, risk_scores):
        base = act.expected_value - act.cost
        regrets.append(base * (1.0 + r))
    return regrets

def hybrid_store_update(store: StoreState,
                        actions: List[MathAction],
                        risk_scores: List[float],
                        reference_ids: List[str]) -> StoreState:
    """
    Update the StoreState using a privacy‑preserving aggregate of risk‑weighted regrets.
    Inflow = dp_aggregate of regrets * similarity; Outflow = dp_aggregate of costs.
    """
    regrets = risk_weighted_regret(actions, risk_scores)
    # similarity of each action id to a reference set via MinHash
    similarities = [minhash_similarity(a.id, reference_ids) for a in actions]
    inflow = [dp_aggregate([reg * sim]) for reg, sim in zip(regrets, similarities)]
    outflow = [dp_aggregate([a.cost]) for a in actions]
    store.update(inflow, outflow)
    return store

def hybrid_select_action(actions: List[MathAction],
                         store: StoreState,
                         risk_scores: List[float],
                         geom_dims: Tuple[float, float, float],
                         reference_ids: List[str]) -> MathAction:
    """
    Choose an action using a blend of:
    - risk‑weighted expected value,
    - confidence bound scaled by sphericity_index,
    - store.dance as a global exploration bonus,
    - MinHash similarity as a locality modifier.
    """
    sph = sphericity_index(*geom_dims)
    best_score = -math.inf
    chosen = None
    for act, r in zip(actions, risk_scores):
        sim = minhash_similarity(act.id, reference_ids)
        # confidence component (higher sphericity -> tighter confidence)
        conf = sph * (1.0 - sim)
        # global exploration term from store
        explore = store.dance * sim
        # final hybrid utility
        utility = (act.expected_value - act.cost) * (1.0 + r) - conf + explore
        if utility > best_score:
            best_score = utility
            chosen = act
    return chosen

# ---------- Smoke Test ----------
if __name__ == "__main__":
    # Define dummy models
    models = [
        ModelTier(name="tiny", ram_mb=256, tier="A"),
        ModelTier(name="small", ram_mb=1024, tier="B"),
        ModelTier(name="medium", ram_mb=4096, tier="C"),
    ]

    # Risk scores based on a synthetic dataset
    risk_scores = compute_risk_scores(models, unique_ids=120, total_records=1000)

    # Define actions corresponding to models
    actions = [
        MathAction(id="a1", expected_value=0.8, cost=0.1, risk=risk_scores[0]),
        MathAction(id="a2", expected_value=0.6, cost=0.05, risk=risk_scores[1]),
        MathAction(id="a3", expected_value=0.9, cost=0.2, risk=risk_scores[2]),
    ]

    # Reference set for MinHash similarity
    reference_ids = ["ref1", "ref2", "a2", "ref4"]

    # Initialise store
    store = StoreState(alpha=0.7, beta=0.3, gain=2.0, limit=5.0)

    # Perform hybrid store update
    store = hybrid_store_update(store, actions, risk_scores, reference_ids)

    # Geometry for sphericity (arbitrary)
    geom = (1.2, 0.8, 0.5)

    # Select best action
    chosen = hybrid_select_action(actions, store, risk_scores, geom, reference_ids)

    print("Risk scores:", risk_scores)
    print("Store level after update:", store.level)
    print("Chosen action:", chosen)
    sys.exit(0)