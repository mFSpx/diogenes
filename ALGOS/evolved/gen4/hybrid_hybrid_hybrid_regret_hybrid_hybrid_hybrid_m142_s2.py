# DARWIN HAMMER — match 142, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_bandit_m40_s1.py (gen3)
# born: 2026-05-29T23:27:10Z

"""hybrid_regret_pheromone_engine.py
Combines:
- Parent A: Regret‑weighted action selection with MinHash similarity.
- Parent B: Pheromone‑modulated honeybee store dynamics.

Mathematical bridge:
For each candidate action *a* we compute a MinHash similarity 𝜎(a, c) between the
current context *c* and a reference feature vector of *a*.  The regret
ΔRₐ(t) accumulated for *a* is weighted by the store’s dance signal *D(t)*
and by a pheromone level ϕₐ(t).  The hybrid score used for selection is

    Sₐ(t) = (Eₐ + D(t)·σ(a,c)) · (1 + ϕₐ(t)) – ΔRₐ(t)

where Eₐ is the baseline expected reward.  Updates of regret and pheromone are
performed jointly after observing a reward, and the store state is updated
with inflow/outflow derived from the aggregated weighted regrets.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (unified)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class HybridAction:
    """Result of an action selection."""
    id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class HybridUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Honeybee‑style store with a bounded control signal (dance)."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0  # internal cache for dance

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the last Δ."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))


# ----------------------------------------------------------------------
# MinHash utilities (lightweight, deterministic for reproducibility)
# ----------------------------------------------------------------------


def _hash_functions(num_hashes: int, max_val: int = 2 ** 31 - 1) -> List[Tuple[int, int]]:
    """Generate a list of (a, b) coefficients for universal hash functions."""
    random.seed(42)
    coeffs = []
    for _ in range(num_hashes):
        a = random.randint(1, max_val - 1)
        b = random.randint(0, max_val - 1)
        coeffs.append((a, b))
    return coeffs


_HASH_FUNCS = _hash_functions(64)


def minhash_signature(vector: np.ndarray) -> np.ndarray:
    """
    Compute a MinHash signature for a binary feature vector.

    Parameters
    ----------
    vector : np.ndarray
        1‑D binary array (0/1) representing the item.

    Returns
    -------
    np.ndarray
        Signature of length ``len(_HASH_FUNCS)`` with integer hash values.
    """
    if vector.ndim != 1:
        raise ValueError("vector must be 1‑D")
    indices = np.where(vector)[0]
    sig = np.full(len(_HASH_FUNCS), np.iinfo(np.int64).max, dtype=np.int64)
    for a, b in _HASH_FUNCS:
        # h(x) = (a * x + b) % prime
        hashes = (a * indices + b) % (2 ** 31 - 1)
        min_hash = hashes.min() if hashes.size else np.iinfo(np.int64).max
        sig[_HASH_FUNCS.index((a, b))] = min_hash
    return sig


def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    Estimate Jaccard similarity from two MinHash signatures.

    Returns a value in [0, 1].
    """
    if sig1.shape != sig2.shape:
        raise ValueError("signatures must have the same shape")
    return np.mean(sig1 == sig2)


# ----------------------------------------------------------------------
# Hybrid engine that merges regret, pheromone, and store dynamics
# ----------------------------------------------------------------------


class HybridRegretPheromoneEngine:
    """
    Core engine implementing the fused algorithm.

    * regret[a]      – cumulative regret for action a
    * pheromone[a]   – scalar pheromone level for action a
    * expected[a]    – baseline expected reward for action a
    * feature_sig[a] – MinHash signature of the action's feature vector
    """

    def __init__(self,
                 actions: Dict[str, np.ndarray],
                 init_expected: Callable[[str], float] | None = None):
        """
        Parameters
        ----------
        actions : dict
            Mapping from action identifier to a binary feature vector (numpy array).
        init_expected : callable, optional
            Function ``f(action_id) -> float`` providing an initial expected reward.
            If omitted, a uniform value of 0.5 is used.
        """
        self.store = StoreState()
        self.regret: Dict[str, float] = {aid: 0.0 for aid in actions}
        self.pheromone: Dict[str, float] = {aid: 0.0 for aid in actions}
        self.expected: Dict[str, float] = {
            aid: (init_expected(aid) if init_expected else 0.5) for aid in actions
        }
        # Pre‑compute MinHash signatures for all actions
        self.signatures: Dict[str, np.ndarray] = {
            aid: minhash_signature(vec) for aid, vec in actions.items()
        }

    # ------------------------------------------------------------------
    # 1️⃣ Compute hybrid score for a single action
    # ------------------------------------------------------------------
    def _hybrid_score(self, action_id: str, context_vec: np.ndarray) -> float:
        """
        Hybrid score Sₐ(t) = (Eₐ + D·σ)·(1+ϕₐ) – ΔRₐ

        where
        - D   = store.dance
        - σ   = MinHash similarity between context and action
        - ϕₐ = pheromone level
        - ΔRₐ = cumulative regret
        """
        sig_action = self.signatures[action_id]
        sig_context = minhash_signature(context_vec)
        sigma = minhash_similarity(sig_action, sig_context)

        dance = self.store.dance
        phi = self.pheromone[action_id]
        regret = self.regret[action_id]
        expected = self.expected[action_id]

        score = (expected + dance * sigma) * (1.0 + phi) - regret
        return score

    # ------------------------------------------------------------------
    # 2️⃣ Action selection (softmax over hybrid scores)
    # ------------------------------------------------------------------
    def select_action(self, context_vec: np.ndarray, temperature: float = 0.1) -> HybridAction:
        """
        Returns a HybridAction with propensity proportional to exp(score / T).
        """
        scores = {aid: self._hybrid_score(aid, context_vec) for aid in self.signatures}
        max_score = max(scores.values())
        # Stabilize softmax
        exp_vals = {
            aid: math.exp((s - max_score) / temperature) for aid, s in scores.items()
        }
        total = sum(exp_vals.values())
        propensities = {aid: v / total for aid, v in exp_vals.items()}

        # Sample according to propensities
        chosen_id = random.choices(
            population=list(propensities.keys()),
            weights=list(propensities.values()),
            k=1
        )[0]

        chosen_propensity = propensities[chosen_id]
        expected_reward = self.expected[chosen_id]
        confidence = math.sqrt(-math.log(chosen_propensity + 1e-12))  # simple UCB‑like term
        expected_value = scores[chosen_id]

        return HybridAction(
            id=chosen_id,
            propensity=chosen_propensity,
            expected_reward=expected_reward,
            confidence_bound=confidence,
            algorithm="HybridRegretPheromone",
            expected_value=expected_value
        )

    # ------------------------------------------------------------------
    # 3️⃣ Joint update of regret, pheromone, and store state
    # ------------------------------------------------------------------
    def observe(self, update: HybridUpdate, context_vec: np.ndarray) -> None:
        """
        Perform a single learning step.

        - Regret update: ΔRₐ ← ΔRₐ + (max_expected - reward) * propensity
        - Pheromone evaporation & deposition:
            ϕₐ ← (1‑λ)·ϕₐ + λ·reward·propensity
        - Store inflow/outflow are derived from the regret change.
        """
        # Compute max expected reward across actions for regret baseline
        max_exp = max(self.expected.values())

        # Regret contribution for the observed action
        regret_increment = (max_exp - update.reward) * update.propensity
        self.regret[update.action_id] += regret_increment

        # Pheromone dynamics
        evaporation = 0.1  # λ_e
        deposition = 0.9    # λ_d
        self.pheromone[update.action_id] = (
            (1 - evaporation) * self.pheromone[update.action_id]
            + deposition * update.reward * update.propensity
        )

        # Store dynamics: treat regret increment as inflow, reward as outflow
        inflow = [regret_increment]
        outflow = [update.reward]
        self.store.update(inflow, outflow)

    # ------------------------------------------------------------------
    # 4️⃣ Utility: expose current internal state for inspection
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return a shallow copy of regret, pheromone and expected values."""
        return {
            "regret": dict(self.regret),
            "pheromone": dict(self.pheromone),
            "expected": dict(self.expected),
            "store_level": self.store.level,
            "dance": self.store.dance,
        }


# ----------------------------------------------------------------------
# Helper functions demonstrating the hybrid operation
# ----------------------------------------------------------------------


def generate_random_binary_vectors(num: int, dim: int) -> Dict[str, np.ndarray]:
    """Create ``num`` random binary feature vectors of dimension ``dim``."""
    random.seed(0)
    vectors = {}
    for i in range(num):
        vec = np.random.randint(0, 2, size=dim, dtype=np.int8)
        vectors[f"action_{i}"] = vec
    return vectors


def run_hybrid_demo(steps: int = 20) -> None:
    """Simple demonstration loop that selects actions and updates the engine."""
    # 1. Build a synthetic action pool
    actions = generate_random_binary_vectors(num=5, dim=32)

    # 2. Initialise the engine
    engine = HybridRegretPheromoneEngine(actions)

    for t in range(steps):
        # Simulate a random context vector
        context = np.random.randint(0, 2, size=32, dtype=np.int8)

        # 3. Select an action
        chosen = engine.select_action(context)

        # 4. Simulate stochastic reward (higher for actions with higher id index)
        base_reward = int(chosen.id.split('_')[1]) / 4.0  # range [0,1]
        reward = base_reward + random.gauss(0, 0.1)      # add noise
        reward = max(0.0, min(1.0, reward))

        # 5. Feed observation back
        upd = HybridUpdate(
            context_id=f"ctx_{t}",
            action_id=chosen.id,
            reward=reward,
            propensity=chosen.propensity,
        )
        engine.observe(upd, context)

        # Optional: print a concise trace every 5 steps
        if (t + 1) % 5 == 0:
            snap = engine.snapshot()
            print(f"Step {t+1:02d} | chosen={chosen.id} | reward={reward:.3f} | "
                  f"dance={snap['dance']:.2f} | store={snap['store_level']:.2f}")

    # Final snapshot
    print("\nFinal internal state:")
    for k, v in engine.snapshot().items():
        print(f"{k}: {v}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_hybrid_demo(steps=15)