# DARWIN HAMMER — match 1896, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_vorono_m622_s1.py (gen5)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s4.py (gen2)
# born: 2026-05-29T23:39:40Z

import math
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 Zulu format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ----------------------------------------------------------------------
# Model Pool (Parent A)
# ----------------------------------------------------------------------
@dataclass
class ModelTier:
    name: str
    ram_mb: int
    tier: str


class ModelPool:
    """
    Tracks models that are currently loaded in memory and enforces a global RAM ceiling.
    """

    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self._loaded: Dict[str, int] = {}  # name → ram_mb

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def current_usage(self) -> int:
        return sum(self._loaded.values())

    def can_load(self, ram_mb: int) -> bool:
        return self.current_usage() + ram_mb <= self.ram_ceiling_mb

    def load(self, name: str, ram_mb: int) -> bool:
        """Attempt to load a model. Returns True on success, False if RAM limit would be exceeded."""
        if self.can_load(ram_mb):
            self._loaded[name] = ram_mb
            return True
        return False

    def unload(self, name: str) -> None:
        self._loaded.pop(name, None)

    def loaded_models(self) -> List[str]:
        return list(self._loaded.keys())

    def __repr__(self) -> str:
        return f"<ModelPool usage={self.current_usage()}/{self.ram_ceiling_mb} MB>"


# ----------------------------------------------------------------------
# Bandit Core (Parent B) – LinUCB with resource‑aware filtering
# ----------------------------------------------------------------------
class LinUCB:
    """
    Contextual bandit using the LinUCB algorithm.
    Each action maintains a d×d matrix A and a d‑vector b.
    """

    def __init__(self, dim: int, alpha: float = 0.1):
        self.dim = dim
        self.alpha = alpha
        self.A: Dict[str, np.ndarray] = defaultdict(
            lambda: np.identity(self.dim)
        )  # action → A matrix
        self.b: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros((self.dim, 1))
        )  # action → b vector

    def _theta(self, action: str) -> np.ndarray:
        """Compute the ridge regression estimate θ = A⁻¹ b."""
        A_inv = np.linalg.inv(self.A[action])
        return A_inv @ self.b[action]

    def predict(self, action: str, x: np.ndarray) -> float:
        """Upper‑confidence bound for a given action and context vector x."""
        theta = self._theta(action)
        A_inv = np.linalg.inv(self.A[action])
        p = float(theta.T @ x + self.alpha * math.sqrt(x.T @ A_inv @ x))
        return p

    def select(self, actions: List[str], x: np.ndarray, rng: random.Random) -> str:
        """Return the action with the highest UCB value."""
        if not actions:
            raise ValueError("actions required")
        scores = {a: self.predict(a, x) for a in actions}
        max_score = max(scores.values())
        # break ties randomly but deterministically via rng
        best = [a for a, s in scores.items() if s == max_score]
        return rng.choice(best)

    def update(self, action: str, x: np.ndarray, reward: float) -> None:
        """Online update of A and b for the chosen action."""
        self.A[action] += x @ x.T
        self.b[action] += reward * x


# ----------------------------------------------------------------------
# Hybrid Model – deeper mathematical integration
# ----------------------------------------------------------------------
class HybridModel:
    """
    Combines a RAM‑aware ModelPool with a contextual LinUCB bandit.
    The bandit’s context vector is enriched with model‑specific features
    (e.g., RAM usage) so that the learning process directly accounts for
    resource constraints.
    """

    def __init__(self, ram_ceiling_mb: int = 6000, alpha: float = 0.1, seed: int | str | None = 7):
        self.model_pool = ModelPool(ram_ceiling_mb)
        self._seed = seed
        self._rng = random.Random(seed)
        self._dim: int | None = None
        self.bandit: LinUCB | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_bandit(self, context: Dict[str, float]) -> None:
        """Initialise LinUCB with the correct dimensionality."""
        if self.bandit is None:
            # Context dimension = user‑provided features + 1 (RAM feature)
            self._dim = len(context) + 1
            self.bandit = LinUCB(dim=self._dim, alpha=0.1)

    def _context_vector(self, context: Dict[str, float], ram_mb: int) -> np.ndarray:
        """Create a column vector [features..., normalized_ram]ᵀ."""
        feats = np.array(list(context.values()), dtype=float).reshape(-1, 1)
        norm_ram = np.array([[ram_mb / self.model_pool.ram_ceiling_mb]], dtype=float)
        return np.vstack([feats, norm_ram])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_model(self, name: str, ram_mb: int) -> bool:
        """Attempt to load a model into the pool."""
        return self.model_pool.load(name, ram_mb)

    def unload_model(self, name: str) -> None:
        self.model_pool.unload(name)

    def update_policy(self, action: str, context: Dict[str, float], reward: float) -> None:
        """Observe a reward for *action* under *context* and update the bandit."""
        if action not in self.model_pool.loaded_models():
            raise ValueError(f"Action '{action}' is not a loaded model")
        self._ensure_bandit(context)
        assert self.bandit is not None and self._dim is not None
        ram_mb = self.model_pool._loaded[action]
        x = self._context_vector(context, ram_mb)
        self.bandit.update(action, x, reward)

    def select_model(self, context: Dict[str, float], candidate_actions: List[str]) -> str:
        """
        Choose a model from *candidate_actions* that is currently loaded and respects the RAM ceiling.
        If none of the candidates are loaded, fall back to the full candidate list.
        """
        # Filter by load state and RAM feasibility
        feasible = [
            a
            for a in candidate_actions
            if self.model_pool.is_loaded(a)
            and self.model_pool.can_load(self.model_pool._loaded[a])
        ]
        actions = feasible or candidate_actions  # graceful degradation

        self._ensure_bandit(context)
        assert self.bandit is not None

        # Build a generic context vector (RAM will be filled per‑action inside select)
        dummy_ram = 0  # placeholder; actual RAM is injected per action inside LinUCB.predict
        x_dummy = self._context_vector(context, dummy_ram)

        # LinUCB.select expects a single x; we therefore compute scores manually
        scores = {}
        for a in actions:
            ram_mb = self.model_pool._loaded.get(a, 0)
            x = self._context_vector(context, ram_mb)
            scores[a] = self.bandit.predict(a, x)

        max_score = max(scores.values())
        best = [a for a, s in scores.items() if s == max_score]
        return self._rng.choice(best)

    def reset(self) -> None:
        """Clear all learned statistics and unload all models."""
        self.model_pool = ModelPool(self.model_pool.ram_ceiling_mb)
        self.bandit = None
        self._rng = random.Random(self._seed)

    # ------------------------------------------------------------------
    # Convenience wrappers for batch training / evaluation
    # ------------------------------------------------------------------
    def train(self, actions: List[str], rewards: List[float], contexts: List[Dict[str, float]]) -> None:
        """Batch‑train the hybrid model on parallel lists."""
        if not (len(actions) == len(rewards) == len(contexts)):
            raise ValueError("actions, rewards, and contexts must have the same length")
        for act, rew, ctx in zip(actions, rewards, contexts):
            self.update_policy(act, ctx, rew)

    def evaluate(self, context: Dict[str, float], actions: List[str]) -> str:
        """Select a model for a single evaluation round."""
        return self.select_model(context, actions)


# ----------------------------------------------------------------------
# Simple sanity test (executed when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    hybrid = HybridModel(ram_ceiling_mb=8000, seed=42)

    # Pretend we have three models with different RAM footprints
    hybrid.load_model("modelA", ram_mb=2000)
    hybrid.load_model("modelB", ram_mb=3500)
    hybrid.load_model("modelC", ram_mb=1500)

    # Training data: contexts + observed rewards
    training_contexts = [
        {"feat1": 0.1, "feat2": 0.3},
        {"feat1": 0.4, "feat2": 0.2},
        {"feat1": 0.2, "feat2": 0.5},
    ]
    training_actions = ["modelA", "modelB", "modelC"]
    training_rewards = [0.6, 0.8, 0.4]

    hybrid.train(training_actions, training_rewards, training_contexts)

    # Evaluation
    test_context = {"feat1": 0.25, "feat2": 0.35}
    chosen = hybrid.evaluate(test_context, ["modelA", "modelB", "modelC"])
    print(f"{now_z()} – Selected model: {chosen}")