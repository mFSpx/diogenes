# DARWIN HAMMER — match 4279, survivor 4
# gen: 5
# parent_a: honeybee_store.py (gen0)
# parent_b: hybrid_hybrid_hybrid_regret_regret_engine_m822_s5.py (gen4)
# born: 2026-05-29T23:54:36Z

"""HybridStoreBandit
Integration of the honeybee‐style store (parent A) with the regret‑weighted hybrid bandit router
(parent B). The mathematical bridge is the store’s Δ (delta) which drives a control signal
(dance duration). This signal rescales the propensity scores used by the bandit router,
effectively coupling resource accumulation with exploration‑exploitation dynamics.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np


# ---------- Parent A core (store) ----------
def update_store(store: float, inflow: List[float], outflow: List[float],
                 alpha: float = 1.0, beta: float = 1.0, dt: float = 1.0) -> Tuple[float, float]:
    """Honeybee store update."""
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta


def dance_duration(delta_store: float, base: float = 1.0,
                   gain: float = 1.0, limit: float = 10.0) -> float:
    """Control signal derived from Δ."""
    return max(0.0, min(limit, base + gain * delta_store))


# ---------- Parent B core (regret‑weighted bandit) ----------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridStoreBandit"


@dataclass(frozen=True)
class BanditUpdate:
    """Observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ---------- Mathematical bridge utilities ----------
def _minhash_signature(text: str, seed: int = 0) -> int:
    """Deterministic integer signature for a string using a simple hash."""
    return (hash((text, seed)) & ((1 << 64) - 1))


def minhash_similarity(sig_a: int, sig_b: int) -> float:
    """
    Approximate Jaccard‑like similarity between two MinHash signatures.
    Uses normalized Hamming distance on the 64‑bit representation.
    """
    xor = sig_a ^ sig_b
    # count differing bits
    diff = bin(xor).count("1")
    return 1.0 - diff / 64.0


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ---------- Hybrid structures ----------
@dataclass
class StoreState:
    """Honeybee‑style store with lazy dance accessor."""
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply store dynamics and cache Δ."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the most recent Δ."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))


class HybridRegretBandit:
    """
    Regret‑weighted bandit whose propensities are modulated by the store’s dance signal.
    """

    def __init__(self,
                 actions: List[MathAction],
                 store: StoreState,
                 similarity_cache: Callable[[str], int] = None):
        self.actions: Dict[str, MathAction] = {a.id: a for a in actions}
        self.store = store
        # Cumulative regret per action
        self.regret: Dict[str, float] = {a.id: 0.0 for a in actions}
        # Cached MinHash signatures for contexts
        self._sig_cache: Callable[[str], int] = similarity_cache or (lambda ctx: _minhash_signature(ctx))

    # ---------- Core hybrid functions ----------
    def compute_propensities(self,
                             context_id: str,
                             reference_contexts: List[str]) -> Dict[str, float]:
        """
        Compute soft‑max propensities.
        The score for each action = expected_value + γ·regret + λ·similarity,
        then scaled by the store's dance signal.
        """
        # similarity term: average similarity between current context and references
        cur_sig = self._sig_cache(context_id)
        sims = [minhash_similarity(cur_sig, self._sig_cache(ref)) for ref in reference_contexts]
        avg_sim = float(np.mean(sims)) if sims else 0.0

        scores = []
        ids = []
        for aid, act in self.actions.items():
            # Regret term encourages less‑explored actions
            r = self.regret.get(aid, 0.0)
            score = act.expected_value + 0.5 * r + 0.3 * avg_sim
            scores.append(score)
            ids.append(aid)

        raw = np.array(scores)
        prop = _softmax(raw) * self.store.dance  # bridge: scale by dance
        return dict(zip(ids, prop.tolist()))

    def select_action(self,
                      context_id: str,
                      reference_contexts: List[str]) -> BanditAction:
        """Sample an action according to hybrid propensities."""
        props = self.compute_propensities(context_id, reference_contexts)
        ids, probs = zip(*props.items())
        chosen_id = random.choices(ids, weights=probs, k=1)[0]
        chosen_prop = props[chosen_id]
        act = self.actions[chosen_id]
        # confidence bound: simple std‑dev estimate from regret magnitude
        conf = math.sqrt(abs(self.regret[chosen_id]) + 1e-6)
        return BanditAction(
            action_id=chosen_id,
            propensity=chosen_prop,
            expected_reward=act.expected_value,
            confidence_bound=conf,
        )

    def observe(self, update: BanditUpdate) -> None:
        """
        Update regret and store based on observed reward.
        Inflow = reward, outflow = cost of chosen action.
        """
        # Regret update (standard regret‑matching)
        chosen_ev = self.actions[update.action_id].expected_value
        for aid, act in self.actions.items():
            regret_increment = (chosen_ev - act.expected_value) * update.propensity
            self.regret[aid] += regret_increment

        # Store update
        inflow = [update.reward]
        outflow = [self.actions[update.action_id].cost]
        self.store.update(inflow, outflow)


# ---------- Demonstration functions ----------
def hybrid_update_store(store: StoreState,
                        inflow: List[float],
                        outflow: List[float]) -> Tuple[float, float]:
    """
    Wrapper that forwards to StoreState.update, exposing the same signature
    as the original parent A function.
    """
    return store.update(inflow, outflow)


def hybrid_select_action(bandit: HybridRegretBandit,
                         context_id: str,
                         reference_contexts: List[str]) -> BanditAction:
    """
    High‑level API mirroring parent B’s action selection but internally
    incorporates the store’s dance signal.
    """
    return bandit.select_action(context_id, reference_contexts)


def hybrid_process_observation(bandit: HybridRegretBandit,
                               ctx: str,
                               act_id: str,
                               reward: float) -> None:
    """
    Record an observation, update regrets and the store in one step.
    """
    # Retrieve the propensity used during selection (re‑compute for simplicity)
    prop = bandit.compute_propensities(ctx, [ctx])[act_id]
    upd = BanditUpdate(context_id=ctx, action_id=act_id, reward=reward, propensity=prop)
    bandit.observe(upd)


# ---------- Smoke test ----------
if __name__ == "__main__":
    # Define actions
    actions = [
        MathAction(id="a", expected_value=1.0, cost=0.2),
        MathAction(id="b", expected_value=0.8, cost=0.1),
        MathAction(id="c", expected_value=0.5, cost=0.05),
    ]

    # Initialise store and bandit
    store = StoreState(level=5.0, alpha=0.6, beta=0.4, dt=1.0, base=1.0, gain=2.0, limit=8.0)
    bandit = HybridRegretBandit(actions=actions, store=store)

    # Simulate a few steps
    contexts = ["ctx1", "ctx2", "ctx3"]
    for step in range(5):
        ctx = random.choice(contexts)
        ref_ctxs = [c for c in contexts if c != ctx]
        act = hybrid_select_action(bandit, ctx, ref_ctxs)
        # Simulated stochastic reward
        reward = act.expected_reward + random.gauss(0, 0.1) - act.propensity * 0.05
        hybrid_process_observation(bandit, ctx, act.action_id, reward)

        # Print internal state
        print(f"Step {step+1}: ctx={ctx}, chosen={act.action_id}, prop={act.propensity:.3f}, "
              f"store_level={store.level:.3f}, dance={store.dance:.3f}")

    sys.exit(0)