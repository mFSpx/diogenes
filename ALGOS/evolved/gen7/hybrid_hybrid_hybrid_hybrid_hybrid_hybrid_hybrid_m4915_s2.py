# DARWIN HAMMER — match 4915, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decrea_m2191_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_decisi_m143_s0.py (gen4)
# born: 2026-05-29T23:58:52Z

"""
Hybrid Algorithm: fusion of hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decrea_m2191_s0.py (Parent A)
and hybrid_hybrid_hybrid_bandit_hybrid_hybrid_decisi_m143_s0.py (Parent B).

Mathematical Bridge
------------------
Parent A contributes a Radial Basis Function (RBF) surrogate model whose prediction
`ŷ(x)` is weighted by an epistemic certainty factor `w_f ∈ [0,1]` derived from
the flag‑to‑weight mapping `_EPISTEMIC_WEIGHT`.  

Parent B contributes a temperature‑dependent developmental rate `ρ(T)` (Schoolfield
model) and a Shannon entropy `H(v)` computed on a feature vector `v`.  The entropy
is used as a decreasing‑rate pruning schedule that selects a subset of RBF centers
based on their contribution to the prediction.

The hybrid fuses the two by:

1. Computing the entropy of the current feature vector and using it to prune the
   surrogate’s centers (`prune_surrogate`).
2. Predicting with the pruned surrogate and scaling the raw prediction by the
   developmental rate `ρ(T)` to obtain a temperature‑adjusted prediction.
3. Multiplying the adjusted prediction by the epistemic weight `w_f` to obtain the
   final utility value `U`.
4. Feeding `U` into a contextual bandit update: the action with the highest
   Upper‑Confidence‑Bound (UCB) is selected, its expected reward is set to `U`,
   and the confidence bound is updated with a simple variance estimate.

Thus the core topology of the RBF surrogate, epistemic weighting, developmental
rate, entropy‑based pruning, and bandit decision making are mathematically
intertwined into a single unified system.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Sequence, Any

import numpy as np

# ----------------------------------------------------------------------
# Shared Data Structures
# ----------------------------------------------------------------------
Vector = Sequence[float]
Point = Tuple[float, float]

_EPISTEMIC_FLAGS: Tuple[str, ...] = (
    "FACT",
    "PROBABLE",
    "POSSIBLE",
    "SURE_MAYBE",
    "BULLSHIT",
)

_EPISTEMIC_WEIGHT: Dict[str, float] = {
    "FACT": 1.0,
    "PROBABLE": 0.85,
    "POSSIBLE": 0.6,
    "SURE_MAYBE": 0.4,
    "BULLSHIT": 0.0,
}


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridBandit"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class SchoolfieldParams:
    rho_25: float = 1.0               # reference rate at 25 °C
    delta_h_activation: float = 12_000.0   # J mol⁻¹
    t_low: float = 283.15             # K
    t_high: float = 307.15            # K
    delta_h_low: float = -45_000.0   # J mol⁻¹
    delta_h_high: float = 65_000.0   # J mol⁻¹
    r_cal: float = 1.987              # gas constant cal mol⁻¹ K⁻¹


@dataclass
class RBFSurrogate:
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Standard RBF prediction: Σ w_i * exp(-ε² * ||x-c_i||²)."""
        total = 0.0
        for w, c in zip(self.weights, self.centers):
            dist_sq = sum((xi - ci) ** 2 for xi, ci in zip(x, c))
            total += w * math.exp(-((self.epsilon ** 2) * dist_sq))
        return total


# ----------------------------------------------------------------------
# Core Mathematical Functions
# ----------------------------------------------------------------------
def epistemic_weight(flag: str) -> float:
    """Return the epistemic weight for a given certainty flag."""
    return _EPISTEMIC_WEIGHT.get(flag.upper(), 0.0)


def developmental_rate(temp_k: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield model for temperature‑dependent developmental rate.
    Returns a dimensionless scaling factor ρ(T) relative to the reference rate.
    """
    if temp_k <= 0:
        raise ValueError("Temperature must be > 0 K")
    # Arrhenius term for activation energy
    act = math.exp(
        -params.delta_h_activation / (params.r_cal * 1000.0) *
        (1.0 / temp_k - 1.0 / 298.15)
    )
    # Low‑temperature deactivation
    low = 1.0 + math.exp(
        params.delta_h_low / (params.r_cal * 1000.0) *
        (1.0 / params.t_low - 1.0 / temp_k)
    )
    # High‑temperature deactivation
    high = 1.0 + math.exp(
        params.delta_h_high / (params.r_cal * 1000.0) *
        (1.0 / params.t_high - 1.0 / temp_k)
    )
    return params.rho_25 * act / (low * high)


def shannon_entropy(vec: Sequence[float]) -> float:
    """Compute Shannon entropy of a non‑negative vector (treated as a probability distribution)."""
    arr = np.array(vec, dtype=float)
    if np.any(arr < 0):
        raise ValueError("Entropy requires non‑negative components")
    total = arr.sum()
    if total == 0:
        return 0.0
    prob = arr / total
    prob = prob[prob > 0]  # avoid log(0)
    return -np.sum(prob * np.log2(prob))


def prune_surrogate(surrogate: RBFSurrogate, keep_ratio: float) -> RBFSurrogate:
    """
    Prune the surrogate's centers/weights based on magnitude of weights.
    `keep_ratio` ∈ (0,1] determines the fraction of centers to keep.
    """
    if not (0 < keep_ratio <= 1):
        raise ValueError("keep_ratio must be in (0,1]")
    # Rank indices by absolute weight magnitude (descending)
    idx_sorted = sorted(
        range(len(surrogate.weights)),
        key=lambda i: abs(surrogate.weights[i]),
        reverse=True,
    )
    keep_n = max(1, int(len(surrogate.weights) * keep_ratio))
    keep_idx = set(idx_sorted[:keep_n])
    new_centers = [c for i, c in enumerate(surrogate.centers) if i in keep_idx]
    new_weights = [w for i, w in enumerate(surrogate.weights) if i in keep_idx]
    return RBFSurrogate(centers=new_centers, weights=new_weights, epsilon=surrogate.epsilon)


def hybrid_utility(
    x: Vector,
    surrogate: RBFSurrogate,
    temp_c: float,
    epistemic_flag: str,
    entropy_prune_factor: float = 0.5,
) -> float:
    """
    Compute the hybrid utility U for a given input vector `x`.

    Steps:
    1. Compute entropy of `x` and map it to a pruning ratio.
    2. Prune the surrogate accordingly.
    3. Predict with the pruned surrogate.
    4. Scale by developmental rate ρ(T) (T in Kelvin).
    5. Multiply by epistemic weight w_f.
    """
    # 1 – entropy based pruning ratio (higher entropy → more pruning)
    ent = shannon_entropy(x)
    # Normalise entropy assuming max possible entropy = log2(len(x))
    max_ent = math.log2(len(x)) if len(x) > 1 else 1.0
    norm_ent = min(1.0, ent / max_ent)
    keep_ratio = max(0.1, 1.0 - entropy_prune_factor * norm_ent)

    # 2 – prune surrogate
    pruned = prune_surrogate(surrogate, keep_ratio)

    # 3 – raw RBF prediction
    raw_pred = pruned.predict(x)

    # 4 – temperature scaling
    temp_k = temp_c + 273.15
    rho = developmental_rate(temp_k)

    # 5 – epistemic weighting
    w_f = epistemic_weight(epistemic_flag)

    return raw_pred * rho * w_f


# ----------------------------------------------------------------------
# Bandit Core
# ----------------------------------------------------------------------
class HybridBandit:
    """
    Contextual bandit that uses the hybrid utility as the expected reward.
    Maintains a simple count of pulls per action to compute a UCB.
    """

    def __init__(self, actions: List[BanditAction]):
        self.actions: Dict[str, BanditAction] = {a.action_id: a for a in actions}
        self.pull_counts: Dict[str, int] = {a.action_id: 0 for a in actions}
        self.total_pulls: int = 0

    def select_action(self, context_id: str, x: Vector, surrogate: RBFSurrogate,
                      temp_c: float, epistemic_flag: str) -> BanditAction:
        """Select action with highest Upper‑Confidence‑Bound (UCB)."""
        utilities = {}
        for aid, act in self.actions.items():
            util = hybrid_utility(
                x, surrogate, temp_c, epistemic_flag
            )
            # Simple variance estimate: 1 / (1 + pulls)
            var_est = 1.0 / (1 + self.pull_counts[aid])
            ucb = util + math.sqrt(2 * math.log(self.total_pulls + 1) * var_est)
            utilities[aid] = (ucb, util)

        # Choose the action with max UCB
        chosen_id = max(utilities, key=lambda k: utilities[k][0])
        ucb_val, expected = utilities[chosen_id]

        # Update internal statistics
        self.pull_counts[chosen_id] += 1
        self.total_pulls += 1

        # Return a new BanditAction with updated expected_reward and confidence bound
        chosen = self.actions[chosen_id]
        updated = BanditAction(
            action_id=chosen.action_id,
            propensity=chosen.propensity,
            expected_reward=expected,
            confidence_bound=ucb_val - expected,
            algorithm=chosen.algorithm,
        )
        # Store back the updated action for future calls
        self.actions[chosen_id] = updated
        return updated

    def receive_update(self, update: BanditUpdate) -> None:
        """Placeholder for learning from external feedback; currently a no‑op."""
        # In a full implementation this would adjust propensity/weights.
        pass


# ----------------------------------------------------------------------
# Demonstration Functions
# ----------------------------------------------------------------------
def create_random_surrogate(dim: int, n_centers: int) -> RBFSurrogate:
    """Utility to generate a random RBF surrogate for testing."""
    centers = [tuple(np.random.rand(dim).tolist()) for _ in range(n_centers)]
    weights = np.random.randn(n_centers).tolist()
    epsilon = random.uniform(0.5, 2.0)
    return RBFSurrogate(centers=centers, weights=weights, epsilon=epsilon)


def demo_hybrid_step():
    """Run a single hybrid decision step and print the chosen action."""
    # Random context
    dim = 5
    x = np.random.rand(dim).tolist()
    temp_c = random.uniform(15.0, 35.0)
    epistemic_flag = random.choice(_EPISTEMIC_FLAGS)

    # Surrogate
    surrogate = create_random_surrogate(dim, n_centers=20)

    # Bandit with three dummy actions
    actions = [
        BanditAction(action_id="A", propensity=0.33, expected_reward=0.0, confidence_bound=0.0),
        BanditAction(action_id="B", propensity=0.33, expected_reward=0.0, confidence_bound=0.0),
        BanditAction(action_id="C", propensity=0.34, expected_reward=0.0, confidence_bound=0.0),
    ]
    bandit = HybridBandit(actions)

    chosen = bandit.select_action(
        context_id="demo_ctx",
        x=x,
        surrogate=surrogate,
        temp_c=temp_c,
        epistemic_flag=epistemic_flag,
    )
    print("Context:", "demo_ctx")
    print("Feature vector:", x)
    print("Temperature (°C):", round(temp_c, 2))
    print("Epistemic flag:", epistemic_flag)
    print("Chosen action:", asdict(chosen))


def batch_hybrid_simulation(steps: int = 10):
    """Run a short simulation of multiple hybrid decisions."""
    dim = 4
    actions = [
        BanditAction(action_id="X", propensity=0.5, expected_reward=0.0, confidence_bound=0.0),
        BanditAction(action_id="Y", propensity=0.5, expected_reward=0.0, confidence_bound=0.0),
    ]
    bandit = HybridBandit(actions)

    surrogate = create_random_surrogate(dim, n_centers=30)

    for step in range(steps):
        x = np.random.rand(dim).tolist()
        temp_c = random.uniform(10.0, 40.0)
        flag = random.choice(_EPISTEMIC_FLAGS)
        chosen = bandit.select_action(
            context_id=f"step_{step}",
            x=x,
            surrogate=surrogate,
            temp_c=temp_c,
            epistemic_flag=flag,
        )
        # In a realistic scenario we would now observe a reward and call `receive_update`.
        # Here we simply print the action id and its expected reward.
        print(f"Step {step:02d} → Action {chosen.action_id}, Expected Reward {chosen.expected_reward:.4f}")


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Single Hybrid Decision Demo ===")
    demo_hybrid_step()
    print("\n=== Batch Hybrid Simulation (5 steps) ===")
    batch_hybrid_simulation(steps=5)