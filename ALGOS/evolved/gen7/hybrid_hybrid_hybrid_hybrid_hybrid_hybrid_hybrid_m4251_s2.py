# DARWIN HAMMER — match 4251, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m1218_s4.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1669_s0.py (gen6)
# born: 2026-05-29T23:54:39Z

"""Hybrid algorithm merging:
- Parent A: contextual bandit router with Gaussian kernel surrogate model.
- Parent B: geometric‑algebra Multivector representation with fractional‑order conductance updates.

Mathematical bridge:
Each *context* is a sparse Multivector 𝑀∈𝔾(n). The Gaussian kernel
k(𝑀_i,𝑀_j)=exp(−ε²‖𝑀_i−𝑀_j‖²) provides a similarity weight that is used
both (i) to form kernel‑weighted reward statistics for a UCB‑style bandit
policy and (ii) to modulate the Caputo‑fractional learning‑rate that updates
the conductance‑Multivector. Thus the same algebraic object drives the
surrogate‑model similarity and the fractional‑order dynamics, yielding a
single unified system.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Multivector utilities (merged from both parents)
# ----------------------------------------------------------------------
class Multivector:
    """
    Sparse multivector for a Euclidean Clifford algebra 𝔾(n).

    * ``components`` maps a frozenset of basis indices to a scalar coefficient.
    * The empty frozenset represents the scalar (grade‑0) part.
    """

    def __init__(self, components: Dict[FrozenSet[int], float] | None = None, n: int = 3):
        self.n = int(n)
        # filter near‑zero entries for sparsity
        self.components: Dict[FrozenSet[int], float] = {
            k: float(v) for k, v in (components or {}).items() if abs(v) > 1e-15
        }

    # ------------------------------------------------------------------
    # Algebraic helpers
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
            if abs(result[blade]) < 1e-15:
                del result[blade]
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        return self + (-other)

    def __neg__(self) -> "Multivector":
        return Multivector({b: -c for b, c in self.components.items()}, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({b: c * scalar for b, c in self.components.items()}, self.n)

    __rmul__ = __mul__

    # ------------------------------------------------------------------
    # Norm and distance (required for the Gaussian kernel)
    # ------------------------------------------------------------------
    def norm(self) -> float:
        """Euclidean norm of the coefficient vector."""
        return math.sqrt(sum(c * c for c in self.components.values()))

    def distance(self, other: "Multivector") -> float:
        """‖self − other‖."""
        return (self - other).norm()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            if blade:
                label = "e" + "".join(str(i) for i in sorted(blade))
            else:
                label = "1"
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"


# ----------------------------------------------------------------------
# Bandit data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass
class ActionStats:
    count: float = 0.0
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0

    def mean(self) -> float:
        return self.reward_sum / self.count if self.count > 0 else 0.0

    def variance(self) -> float:
        if self.count == 0:
            return 0.0
        mean = self.mean()
        return max(self.reward_sq_sum / self.count - mean * mean, 0.0)


# ----------------------------------------------------------------------
# Core mathematical bridge utilities
# ----------------------------------------------------------------------
def gaussian_kernel(mv1: Multivector, mv2: Multivector, epsilon: float) -> float:
    """Gaussian similarity between two multivectors."""
    d = mv1.distance(mv2)
    return math.exp(- (epsilon ** 2) * (d ** 2))


def fractional_learning_rate(base_lr: float, t: int, alpha: float) -> float:
    """
    Approximate the Caputo fractional derivative scaling.
    For 0<α<1, the effective learning rate behaves like
        lr(t) ≈ base_lr * t^{α-1}
    """
    if t <= 0:
        return base_lr
    return base_lr * (t ** (alpha - 1.0))


def update_conductance(
    conductance: Multivector,
    context: Multivector,
    reward: float,
    t: int,
    alpha: float,
    base_lr: float,
) -> Multivector:
    """
    Conductance update using a fractional‑order learning rate.
    ΔC = lr(t) * reward * context
    """
    lr = fractional_learning_rate(base_lr, t, alpha)
    delta = context * (lr * reward)
    return conductance + delta


def select_action_ucb(
    context: Multivector,
    history: List[Tuple[Multivector, Dict[str, ActionStats]]],
    actions: List[str],
    epsilon: float,
    c: float,
) -> Tuple[str, float]:
    """
    Kernel‑weighted UCB action selection.

    Parameters
    ----------
    context : Multivector
        Current context.
    history : list of (past_context, action_stats_dict)
        All previously observed contexts together with per‑action statistics.
    actions : list of str
        Candidate action identifiers.
    epsilon : float
        Bandwidth for the Gaussian kernel.
    c : float
        Exploration coefficient.

    Returns
    -------
    (action_id, propensity) : tuple
        Chosen action and its propensity (the normalized kernel weight sum for that action).
    """
    # Compute kernel weights for each past context
    weights = [gaussian_kernel(context, past_ctx, epsilon) for past_ctx, _ in history]
    total_weight = sum(weights) + 1e-12

    # Aggregate weighted statistics per action
    agg_stats: Dict[str, ActionStats] = {a: ActionStats() for a in actions}
    for (past_ctx, stats_dict), w in zip(history, weights):
        for a in actions:
            s = stats_dict.get(a)
            if s is None:
                continue
            agg = agg_stats[a]
            agg.count += w * s.count
            agg.reward_sum += w * s.reward_sum
            agg.reward_sq_sum += w * s.reward_sq_sum

    # Global count for log term (sum of weighted counts across actions)
    global_count = sum(st.count for st in agg_stats.values()) + 1e-12

    # Compute UCB values
    ucb_values: Dict[str, float] = {}
    for a in actions:
        st = agg_stats[a]
        if st.count == 0:
            # Force exploration of never‑seen actions
            ucb = float("inf")
        else:
            mean = st.mean()
            var = st.variance()
            bonus = c * math.sqrt(var * math.log(global_count) / (st.count + 1e-9))
            ucb = mean + bonus
        ucb_values[a] = ucb

    # Choose action with highest UCB
    chosen_action = max(ucb_values, key=ucb_values.get)

    # Propensity is the normalized kernel‑weighted count for the chosen action
    propensity = agg_stats[chosen_action].count / total_weight

    return chosen_action, propensity


# ----------------------------------------------------------------------
# High‑level hybrid step
# ----------------------------------------------------------------------
def hybrid_step(
    conductance: Multivector,
    context: Multivector,
    history: List[Tuple[Multivector, Dict[str, ActionStats]]],
    actions: List[str],
    epsilon: float,
    c: float,
    t: int,
    alpha: float,
    base_lr: float,
) -> Tuple[Multivector, str, float, Multivector]:
    """
    Perform one interaction step of the hybrid system.

    Returns
    -------
    new_conductance : Multivector
        Updated conductance multivector.
    action_id : str
        Action selected by the kernel‑UCB policy.
    propensity : float
        Propensity associated with the selected action.
    new_context : Multivector
        (In this toy implementation the context is unchanged, but the
        function signature mirrors a realistic pipeline where the context
        could be transformed.)
    """
    # 1. Select action using kernel‑weighted UCB
    action_id, propensity = select_action_ucb(
        context, history, actions, epsilon, c
    )

    # 2. Simulated reward (for demonstration we draw from a simple stochastic model)
    #    In a real system this would come from the environment.
    reward = random.gauss(mu=1.0 if action_id == "A" else 0.0, sigma=0.5)

    # 3. Update conductance with fractional learning rate
    new_conductance = update_conductance(
        conductance, context, reward, t, alpha, base_lr
    )

    # 4. Record statistics for future kernel weighting
    #    Create a fresh stats dict for this step
    step_stats = {a: ActionStats() for a in actions}
    step_stats[action_id] = ActionStats(
        count=1.0,
        reward_sum=reward,
        reward_sq_sum=reward * reward,
    )
    history.append((context, step_stats))

    return new_conductance, action_id, propensity, context


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)

    # Hyper‑parameters
    EPSILON = 0.5          # kernel bandwidth
    C = 1.0                # UCB exploration coefficient
    ALPHA = 0.7            # fractional order (0 < α < 1)
    BASE_LR = 0.1          # base learning rate for conductance update
    ACTIONS = ["A", "B", "C"]

    # Initial objects
    conductance = Multivector({}, n=4)
    history: List[Tuple[Multivector, Dict[str, ActionStats]]] = []

    # Generate a random initial context (sparse multivector)
    def random_context(dim: int, sparsity: float = 0.3) -> Multivector:
        comps = {}
        for i in range(dim):
            if random.random() < sparsity:
                blade = frozenset({i})
                comps[blade] = random.uniform(-1.0, 1.0)
        # add a scalar part with small probability
        if random.random() < 0.2:
            comps[frozenset()] = random.uniform(-0.5, 0.5)
        return Multivector(comps, n=dim)

    # Run a short simulation
    for t in range(1, 11):
        ctx = random_context(dim=5, sparsity=0.4)
        conductance, act, prop, _ = hybrid_step(
            conductance,
            ctx,
            history,
            ACTIONS,
            EPSILON,
            C,
            t,
            ALPHA,
            BASE_LR,
        )
        print(
            f"Step {t:2d} | Action: {act} | Propensity: {prop:.4f} | "
            f"Reward (simulated) ≈ {history[-1][1][act].reward_sum:.3f} | "
            f"Conductance norm: {conductance.norm():.3f}"
        )

    print("\nFinal conductance representation:")
    print(conductance)