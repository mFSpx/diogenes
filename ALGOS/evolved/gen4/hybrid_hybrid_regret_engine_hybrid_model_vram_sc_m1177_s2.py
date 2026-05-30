# DARWIN HAMMER — match 1177, survivor 2
# gen: 4
# parent_a: hybrid_regret_engine_hybrid_liquid_time_c_m13_s0.py (gen2)
# parent_b: hybrid_model_vram_scheduler_hybrid_hybrid_bandit_m188_s0.py (gen3)
# born: 2026-05-29T23:33:23Z

import numpy as np
import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterable, List, Dict, Tuple, Optional


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MathAction:
    """A concrete action with a deterministic expected value."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0
    tokens: Tuple[str, ...] = field(default_factory=tuple)  # semantic tokens for MinHash


@dataclass(frozen=True)
class MathCounterfactual:
    """What would have happened if a given action had been taken."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class BanditAction:
    """Result of the bandit decision."""
    action_id: str
    propensity: float          # probability of being selected (0‑1)
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretBandit"


@dataclass(frozen=True)
class VramSlotPlan:
    """Allocation plan for a VRAM slot."""
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: Dict[str, float]


# --------------------------------------------------------------------------- #
# Core utilities – deterministic MinHash
# --------------------------------------------------------------------------- #
def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on Blake2b."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: Iterable[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not toks:
        # all‑ones signature – maximal distance to any non‑empty set
        return [(1 << 64) - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity between two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# --------------------------------------------------------------------------- #
# Hybrid Regret‑Bandit Scheduler
# --------------------------------------------------------------------------- #
class HybridRegretBanditScheduler:
    """
    A stateful scheduler that fuses regret‑weighted decision making with a
    linear contextual bandit.  The integration is deep:
      * Regret weights drive the bandit’s *propensity*.
      * MinHash similarity between actions modulates the *confidence bound*.
      * An exponential moving average stores a latent “resource pressure”
        variable that is used to compute VRAM allocation.
    """

    def __init__(
        self,
        k_hash: int = 128,
        alpha: float = 0.05,
        beta: float = 0.05,
        gamma: float = 0.9,
        dt: float = 1.0,
    ):
        """
        Parameters
        ----------
        k_hash : int
            Length of the MinHash signature.
        alpha, beta : float
            Learning‑rate coefficients for the internal store dynamics.
        gamma : float
            EMA decay factor (0 < gamma < 1).
        dt : float
            Discrete‑time step size.
        """
        if not (0 < gamma < 1):
            raise ValueError("gamma must be in (0,1)")
        self.k_hash = k_hash
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dt = dt
        self._store: float = 0.0  # latent resource pressure (0‑1)

    # ------------------------------------------------------------------- #
    # Regret‑weighted computation
    # ------------------------------------------------------------------- #
    @staticmethod
    def _regret_weights(
        actions: List[MathAction],
        counterfactuals: List[MathCounterfactual],
    ) -> Dict[str, float]:
        """
        Compute per‑action regret: expected_value - (counterfactual outcome * prob).
        Missing counterfactuals are treated as zero contribution.
        """
        cf_map = {
            cf.action_id: cf.outcome_value * cf.probability for cf in counterfactuals
        }
        return {
            a.id: a.expected_value - cf_map.get(a.id, 0.0) for a in actions
        }

    # ------------------------------------------------------------------- #
    # Bandit decision
    # ------------------------------------------------------------------- #
    def _bandit_decision(
        self,
        regret_weights: Dict[str, float],
        target_id: str,
        actions: List[MathAction],
    ) -> BanditAction:
        """
        Produce a BanditAction where:
          * propensity = softmax‑scaled regret (ensures 0‑1 range).
          * expected_reward = regret weight of the target action.
          * confidence_bound = variance‑adjusted by MinHash similarity to the
            target action (more similar actions shrink the bound).
        """
        if not regret_weights:
            # Degenerate case – return a neutral action.
            return BanditAction(
                action_id=target_id,
                propensity=0.0,
                expected_reward=0.0,
                confidence_bound=0.0,
            )

        # Softmax over regrets to obtain a proper probability distribution.
        regrets = np.array(list(regret_weights.values()), dtype=float)
        max_r = regrets.max()
        exp_vals = np.exp(regrets - max_r)  # numerical stability
        probs = exp_vals / exp_vals.sum()
        # Map back to ids preserving order.
        ids = list(regret_weights.keys())
        propensity_map = dict(zip(ids, probs))

        # Expected reward for the requested artifact/action.
        expected_reward = regret_weights.get(target_id, 0.0)

        # Compute similarity‑adjusted variance.
        target_action = next((a for a in actions if a.id == target_id), None)
        if target_action is None:
            # No token information – fall back to plain variance.
            similarity_factor = 1.0
        else:
            target_sig = signature(target_action.tokens, self.k_hash)
            sims = []
            for a in actions:
                if a.id == target_id:
                    continue
                sig = signature(a.tokens, self.k_hash)
                sims.append(similarity(target_sig, sig))
            # Average similarity (0‑1). Higher similarity → tighter bound.
            similarity_factor = 1.0 - (np.mean(sims) if sims else 0.0)

        variance = np.var(regrets, ddof=0)
        confidence_bound = math.sqrt(variance) * similarity_factor

        return BanditAction(
            action_id=target_id,
            propensity=propensity_map.get(target_id, 0.0),
            expected_reward=expected_reward,
            confidence_bound=confidence_bound,
        )

    # ------------------------------------------------------------------- #
    # Store dynamics → VRAM allocation
    # ------------------------------------------------------------------- #
    def _update_store(self, bandit: BanditAction) -> float:
        """
        Exponential moving average that integrates propensity (push) and
        confidence bound (pull).  The result stays in [0,1].
        """
        push = self.alpha * bandit.propensity * self.dt
        pull = self.beta * bandit.confidence_bound * self.dt
        raw = self._store + push - pull
        # Clamp and EMA‑smooth.
        self._store = max(0.0, min(1.0, raw))
        self._store = self.gamma * self._store + (1 - self.gamma) * raw
        return self._store

    def _vram_plan(
        self,
        bandit: BanditAction,
        budget_mb: int,
        store: float,
    ) -> VramSlotPlan:
        """
        Convert the latent store (0‑1) into an absolute MB allocation.
        The plan also reports the underlying drivers for transparency.
        """
        allocated = int(budget_mb * store)
        reason = (
            f"store={store:.3f} (propensity={bandit.propensity:.3f}, "
            f"conf_bound={bandit.confidence_bound:.3f})"
        )
        return VramSlotPlan(
            artifact_id=bandit.action_id,
            artifact_kind="artifact",
            action="allocate",
            estimated_mb=allocated,
            reason=reason,
            detail={
                "store": store,
                "propensity": bandit.propensity,
                "confidence_bound": bandit.confidence_bound,
            },
        )

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #
    def schedule(
        self,
        actions: List[MathAction],
        counterfactuals: List[MathCounterfactual],
        artifact_id: str,
        budget_mb: int,
    ) -> VramSlotPlan:
        """
        End‑to‑end hybrid operation:
          1. Compute regret weights.
          2. Choose a bandit action using similarity‑aware confidence.
          3. Update internal store.
          4. Emit a VRAM allocation plan.
        """
        regret_weights = self._regret_weights(actions, counterfactuals)
        bandit_action = self._bandit_decision(regret_weights, artifact_id, actions)
        store = self._update_store(bandit_action)
        return self._vram_plan(bandit_action, budget_mb, store)


# --------------------------------------------------------------------------- #
# Example usage (executed only when run as a script)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Define actions with semantic tokens for MinHash.
    actions = [
        MathAction(
            id="action1",
            expected_value=10.0,
            tokens=("linear", "regression", "model"),
        ),
        MathAction(
            id="action2",
            expected_value=20.0,
            tokens=("tree", "ensemble", "boosting"),
        ),
        MathAction(
            id="action3",
            expected_value=15.0,
            tokens=("neural", "network", "deep"),
        ),
    ]

    counterfactuals = [
        MathCounterfactual(action_id="action1", outcome_value=5.0),
        MathCounterfactual(action_id="action2", outcome_value=15.0),
        MathCounterfactual(action_id="action3", outcome_value=12.0),
    ]

    scheduler = HybridRegretBanditScheduler(alpha=0.07, beta=0.04, gamma=0.85)

    # Simulate a few scheduling steps.
    for step in range(5):
        plan = scheduler.schedule(
            actions=actions,
            counterfactuals=counterfactuals,
            artifact_id="action2",
            budget_mb=4096,
        )
        print(f"Step {step+1}: {plan}")