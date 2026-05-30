# DARWIN HAMMER — match 2779, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_jepa_e_hybrid_hybrid_hybrid_m2401_s1.py (gen5)
# parent_b: hybrid_hybrid_indy_learning_hybrid_pheromone_inf_m541_s0.py (gen5)
# born: 2026-05-29T23:45:48Z

"""Hybrid System integrating ModelPool energy dynamics (Parent A) with
Bandit‑Pheromone entropy selection (Parent B).

Mathematical bridge:
- Parent A defines an energy scalar 𝔈 that penalises memory conflicts and rewards
  model loading/eviction.
- Parent B defines a pheromone distribution 𝑝_i ∝ exp(σ_i) where σ_i are surface
  signals, and uses entropy 𝐻(𝑝) for action selection.
- The hybrid maps the energy 𝔈 to a *risk weight* w = 1/(1+𝔈) ∈ (0,1] and
  modulates each pheromone probability: 𝑝_i′ = w·𝑝_i + (1‑w)·u_i where u_i is a uniform
  baseline. The resulting distribution is renormalised and its entropy is used to
  bias the bandit action scores. Thus the two topologies are fused through a
  shared scalar (energy ↔ risk weight) that directly influences the probabilistic
  decision process.
"""

import math
import random
import sys
import pathlib
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any

import numpy as np

# ---------- Parent A components ----------
@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str


class ModelPool:
    """Manages a pool of models with an energy bookkeeping system."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self._energy: float = 0.0

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def add_model(self, model: ModelTier) -> None:
        # Tier conflict penalty
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            self._energy += 1e10
        # Memory overflow penalty
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            self._energy += 1e6
        self.loaded[model.name] = model

    def load(self, model: ModelTier) -> None:
        self._energy -= 1e4   # reward for loading
        self.add_model(model)

    def load_with_eviction(self, model: ModelTier) -> None:
        self._energy -= 1e3   # reward for eviction attempt
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            evicted_name = max(self.loaded, key=lambda n: self.loaded[n].ram_mb)
            self.loaded.pop(evicted_name)
            self._energy += 1e2   # penalty for evicting
        self.load(model)

    def free_energy(self) -> float:
        return self._energy


def entropy(probabilities: List[float], eps: float = 1e-10) -> float:
    """Shannon entropy of a probability vector."""
    probs = np.asarray(probabilities, dtype=float)
    probs = np.clip(probs, eps, 1.0)
    probs /= probs.sum()
    return -float(np.sum(probs * np.log(probs)))


# ---------- Parent B components ----------
class BanditAction:
    def __init__(self, action_id: str, propensity: float,
                 expected_reward: float, confidence_bound: float,
                 algorithm: str):
        self.action_id = action_id
        self.propensity = propensity
        self.expected_reward = expected_reward
        self.confidence_bound = confidence_bound
        self.algorithm = algorithm


class BanditUpdate:
    def __init__(self, context_id: str, action_id: str,
                 reward: float, propensity: float):
        self.context_id = context_id
        self.action_id = action_id
        self.reward = reward
        self.propensity = propensity


class Pheromone:
    def __init__(self, surface_key: str, signal_value: float):
        self.surface_key = surface_key
        self.signal_value = signal_value


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def tokenize(text: str) -> List[Dict[str, Any]]:
    WORD_RE = re.compile(r"\S+")
    return [{"token": m.group(0), "start": m.start(), "end": m.end()}
            for m in WORD_RE.finditer(text)]


def chunk_text_tokens(text: str, *, max_tokens: int = 500,
                     overlap_tokens: int = 0,
                     source_ref: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be >=0 and < max_tokens")
    toks = tokenize(text)
    source_ref = dict(source_ref or {})
    if not toks:
        return []
    chunks = []
    for i in range(0, len(toks), max_tokens - overlap_tokens):
        chunk = toks[i:i + max_tokens]
        chunks.append({"tokens": chunk, "source": source_ref})
    return chunks


# ---------- Hybrid core ----------
def pheromone_distribution(pheromones: List[Pheromone]) -> np.ndarray:
    """Return a normalised probability vector derived from pheromone signal values."""
    if not pheromones:
        raise ValueError("At least one pheromone required")
    signals = np.array([p.signal_value for p in pheromones], dtype=float)
    # Use exponential to ensure positivity and amplify differences
    exp_signals = np.exp(signals - np.max(signals))
    probs = exp_signals / exp_signals.sum()
    return probs


def risk_weighted_distribution(base_probs: np.ndarray, energy: float) -> np.ndarray:
    """
    Blend the base pheromone distribution with a uniform baseline using the
    risk weight w = 1 / (1 + energy). Higher energy → lower weight → more uniform.
    """
    w = 1.0 / (1.0 + energy)
    uniform = np.full_like(base_probs, 1.0 / base_probs.size)
    blended = w * base_probs + (1.0 - w) * uniform
    return blended / blended.sum()


def compute_action_scores(actions: List[BanditAction],
                          blended_probs: np.ndarray) -> List[float]:
    """
    Combine bandit statistics with the blended pheromone probabilities.
    Score_i = (expected_reward_i + confidence_bound_i) * blended_prob_i
    """
    if len(actions) != blended_probs.size:
        raise ValueError("Number of actions must match probability vector size")
    scores = []
    for act, prob in zip(actions, blended_probs):
        score = (act.expected_reward + act.confidence_bound) * prob
        scores.append(score)
    return scores


def select_best_action(actions: List[BanditAction],
                       scores: List[float]) -> BanditAction:
    """Select the action with the highest hybrid score."""
    idx = int(np.argmax(scores))
    return actions[idx]


# ---------- High‑level hybrid operation ----------
def hybrid_step(model_pool: ModelPool,
                pheromones: List[Pheromone],
                actions: List[BanditAction]) -> BanditAction:
    """
    One iteration of the hybrid algorithm:
    1. Compute current energy from the model pool.
    2. Build a pheromone probability distribution.
    3. Blend it with a uniform distribution using the energy‑derived risk weight.
    4. Compute hybrid scores for each bandit action.
    5. Return the action with maximal score.
    """
    energy = model_pool.free_energy()
    base_probs = pheromone_distribution(pheromones)
    blended = risk_weighted_distribution(base_probs, energy)
    scores = compute_action_scores(actions, blended)
    chosen = select_best_action(actions, scores)
    return chosen


# ---------- Example usage (smoke test) ----------
if __name__ == "__main__":
    # Initialise a model pool and load a couple of models
    pool = ModelPool(ram_ceiling_mb=8000)
    pool.load(ModelTier(name="model_A", ram_mb=1500, tier="T1"))
    pool.load_with_eviction(ModelTier(name="model_B", ram_mb=3000, tier="T2"))
    # Force a conflict to generate energy penalty
    pool.load(ModelTier(name="model_C", ram_mb=2000, tier="T3"))

    # Create pheromones for three surfaces
    pheromones = [
        Pheromone(surface_key="s1", signal_value=0.8),
        Pheromone(surface_key="s2", signal_value=0.1),
        Pheromone(surface_key="s3", signal_value=0.5),
    ]

    # Define three bandit actions
    actions = [
        BanditAction(action_id="a1", propensity=0.3,
                     expected_reward=1.2, confidence_bound=0.4, algorithm="UCB"),
        BanditAction(action_id="a2", propensity=0.5,
                     expected_reward=0.9, confidence_bound=0.6, algorithm="Thompson"),
        BanditAction(action_id="a3", propensity=0.2,
                     expected_reward=1.5, confidence_bound=0.2, algorithm="Epsilon"),
    ]

    # Run a hybrid step
    best = hybrid_step(pool, pheromones, actions)

    print(f"Chosen action: {best.action_id}")
    print(f"Current energy: {pool.free_energy():.2e}")
    print(f"Entropy of blended distribution: {entropy(risk_weighted_distribution(pheromone_distribution(pheromones), pool.free_energy())):.4f}")