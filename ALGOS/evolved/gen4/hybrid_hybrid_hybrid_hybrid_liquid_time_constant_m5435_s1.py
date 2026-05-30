# DARWIN HAMMER — match 5435, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_voronoi_parti_m169_s0.py (gen3)
# parent_b: liquid_time_constant.py (gen0)
# born: 2026-05-30T00:01:51Z

"""Hybrid Bandit‑Voronoi‑LTC Algorithm
===================================

This module fuses two parent algorithms:

* **Parent A** – a contextual multi‑armed bandit that uses Euclidean distances
  (Voronoi partition) and a circuit‑breaker per action.
* **Parent B** – Liquid‑Time‑Constant (LTC) continuous‑time recurrent networks
  whose hidden state evolves according to an ODE whose time‑constant depends on
  a learned gating function *f*.

**Mathematical bridge**

The Voronoi distance `d = ‖context – action‖₂` is fed as the external input
`I(t)` to the LTC. The LTC gating `f(x, I, θ)` modulates the *effective* time
constant `τ_sys(t) = τ / (1 + τ·f)`. The resulting gating value is then used to
scale the bandit’s confidence bound, i.e. the exploration term. In this way
the geometric relationship between context and actions directly shapes the
temporal dynamics of the bandit’s internal state and its action‑selection
policy.

The hybrid therefore consists of three tightly coupled parts:

1. **Voronoi distance computation** – provides the LTC input.
2. **LTC dynamics** – evolve a hidden state per context and output a gating
   vector `g = f(x, I, θ)`.
3. **Bandit decision** – uses `g` to adapt the Upper‑Confidence‑Bound (UCB)
   score for each action; a circuit‑breaker disables actions that exceed a
   failure threshold.

All components are pure NumPy / Python and can be executed without external
deep‑learning libraries.
"""

import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Bandit core (parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float = 0.0          # will be filled by the hybrid selector
    expected_reward: float = 0.0
    confidence_bound: float = 0.0
    algorithm: str = "hybrid_bandit_ltc"
    position: Tuple[float, float] = (0.0, 0.0)   # Voronoi coordinate


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# Global stores (mirroring parent A)
_POLICY: Dict[str, List[float]] = {}          # action_id → [total_reward, count]
_STORE: Dict[str, float] = {}                 # virtual VRAM per key
FAILURES: Dict[str, int] = {}                 # circuit‑breaker counters
FAILURE_THRESHOLD: int = 3                    # open after 3 failures


def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()
    FAILURES.clear()


def _reward(a: str) -> float:
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


# ----------------------------------------------------------------------
# Voronoi helpers (parent A)
# ----------------------------------------------------------------------
Point = Tuple[float, float]


def euclidean_distance(a: Point, b: Point) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# Circuit‑breaker (parent A)
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Failure counter that opens after a configurable threshold."""

    def __init__(self, failure_threshold: int = FAILURE_THRESHOLD):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be > 0")
        self.threshold = failure_threshold

    def record_failure(self, action_id: str) -> None:
        FAILURES[action_id] = FAILURES.get(action_id, 0) + 1

    def is_open(self, action_id: str) -> bool:
        return FAILURES.get(action_id, 0) >= self.threshold


_circuit_breaker = EndpointCircuitBreaker()


# ----------------------------------------------------------------------
# Liquid‑Time‑Constant core (parent B)
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable element‑wise sigmoid σ(x) = 1 / (1 + exp(-x))."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def ltc_f(
    x: np.ndarray,
    I: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Gating function f(x, I, θ) = σ(W·[x;I] + b)."""
    concat = np.concatenate([x, I])
    return sigmoid(W @ concat + b)


def ltc_step(
    x: np.ndarray,
    I: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """
    One explicit Euler step of the LTC ODE:

        dx/dt = -[1/τ + f]·x + f·A

    Returns the updated hidden state.
    """
    τ = params["tau"]
    A = params["A"]
    W = params["W"]
    b = params["b"]
    dt = params["dt"]

    f = ltc_f(x, I, W, b)                # shape (hidden_dim,)
    tau_eff = τ / (1.0 + τ * f)           # element‑wise effective time constant
    dx = -(1.0 / tau_eff) * x + f * A
    return x + dt * dx


def init_ltc(
    input_dim: int,
    hidden_dim: int,
    tau: float = 1.0,
    dt: float = 0.01,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Initialise LTC parameters.

    Returns a dictionary containing:
        - hidden_dim, input_dim
        - τ, dt
        - A (target state, shape (hidden_dim,))
        - W (weight matrix, shape (hidden_dim, hidden_dim+input_dim))
        - b (bias vector, shape (hidden_dim,))
        - x (initial hidden state, shape (hidden_dim,))
    """
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0, size=hidden_dim)
    W = rng.normal(0.0, 1.0, size=(hidden_dim, hidden_dim + input_dim))
    b = rng.normal(0.0, 1.0, size=hidden_dim)
    x = np.zeros(hidden_dim)  # start from rest
    return {
        "hidden_dim": hidden_dim,
        "input_dim": input_dim,
        "tau": float(tau),
        "dt": float(dt),
        "A": A,
        "W": W,
        "b": b,
        "x": x,
    }


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_ltc_update(
    ltc_state: Dict[str, Any],
    context_point: Point,
    action_point: Point,
) -> np.ndarray:
    """
    Update the LTC hidden state using the Euclidean distance between the
    context and the candidate action as the external input.

    Parameters
    ----------
    ltc_state : dict
        Current LTC parameters (including hidden state ``x``).
    context_point : tuple[float, float]
        (x, y) coordinates of the current context.
    action_point : tuple[float, float]
        (x, y) coordinates of the action under consideration.

    Returns
    -------
    np.ndarray
        The updated hidden state (also stored back into ``ltc_state``).
    """
    # Input to the LTC is a 1‑D vector containing the distance
    distance = euclidean_distance(context_point, action_point)
    I = np.array([distance], dtype=np.float64)

    # Perform one integration step
    new_x = ltc_step(ltc_state["x"], I, ltc_state)
    ltc_state["x"] = new_x
    return new_x


def compute_ucb_score(
    action: BanditAction,
    gating: np.ndarray,
    total_counts: int,
    exploration_coeff: float = 1.0,
) -> float:
    """
    Upper‑Confidence‑Bound score that incorporates the LTC gating.

    The gating vector modulates the confidence term; larger gating → more
    exploration (faster response).

    Parameters
    ----------
    action : BanditAction
    gating : np.ndarray
        Output of ``ltc_f`` (shape (hidden_dim,)). We use its mean magnitude.
    total_counts : int
        Total number of pulls across all actions (for UCB normalisation).
    exploration_coeff : float
        Scaling of the confidence bound.

    Returns
    -------
    float
        The UCB score.
    """
    avg_reward = _reward(action.action_id)
    # Confidence bound scaled by gating magnitude
    gate_factor = np.mean(gating) if gating.size else 0.0
    conf = exploration_coeff * gate_factor * math.sqrt(
        math.log(total_counts + 1) / (1 + _POLICY.get(action.action_id, [0.0, 0.0])[1])
    )
    return avg_reward + conf


def hybrid_select_action(
    context_id: str,
    context_point: Point,
    actions: List[BanditAction],
    ltc_state: Dict[str, Any],
) -> Tuple[BanditAction, np.ndarray]:
    """
    Choose an action for the given context using the hybrid bandit‑LTC rule.

    The steps are:

    1. For each *available* action (circuit breaker closed) compute the updated
       LTC hidden state based on its Voronoi distance.
    2. Derive a gating vector from the *current* hidden state (before update) –
       this provides a smooth influence of past dynamics.
    3. Compute a UCB‑style score that blends the expected reward with a
       confidence term scaled by the gating.
    4. Sample proportionally to the softmax of the scores.

    Returns
    -------
    (selected_action, gating_vector)
    """
    total_counts = sum(cnt for _, cnt in (_POLICY.get(a.action_id, [0.0, 0.0]) for a in actions))

    scores = []
    gating_vectors = []
    updated_states = []

    for action in actions:
        if _circuit_breaker.is_open(action.action_id):
            # Skip actions that have been blocked
            scores.append(-float("inf"))
            gating_vectors.append(np.zeros(ltc_state["hidden_dim"]))
            updated_states.append(ltc_state["x"])
            continue

        # 1. Compute gating based on *current* hidden state
        #    Input for gating is the distance as a 1‑D vector
        distance = euclidean_distance(context_point, action.position)
        I = np.array([distance], dtype=np.float64)
        gating = ltc_f(ltc_state["x"], I, ltc_state["W"], ltc_state["b"])

        # 2. UCB score with gating
        score = compute_ucb_score(action, gating, total_counts)
        scores.append(score)
        gating_vectors.append(gating)

        # 3. Update hidden state *after* scoring (so next action sees the effect)
        new_state = hybrid_ltc_update(ltc_state, context_point, action.position)
        updated_states.append(new_state)

    # Softmax over scores for stochastic selection
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) if s > -float("inf") else 0.0 for s in scores]
    sum_exp = sum(exp_scores) or 1.0
    probs = [e / sum_exp for e in exp_scores]

    chosen_idx = random.choices(range(len(actions)), weights=probs, k=1)[0]
    chosen_action = actions[chosen_idx]

    # Store the updated hidden state (the one that resulted from the chosen action)
    ltc_state["x"] = updated_states[chosen_idx]

    # Attach propensity for downstream logging
    chosen_action = BanditAction(
        action_id=chosen_action.action_id,
        propensity=probs[chosen_idx],
        expected_reward=_reward(chosen_action.action_id),
        confidence_bound=0.0,               # not used further
        algorithm=chosen_action.algorithm,
        position=chosen_action.position,
    )

    return chosen_action, gating_vectors[chosen_idx]


def hybrid_update_policy(update: BanditUpdate) -> None:
    """
    Record the observed reward and update the global policy statistics.
    """
    stats = _POLICY.get(update.action_id, [0.0, 0.0])
    total, cnt = stats
    total += update.reward
    cnt += 1
    _POLICY[update.action_id] = [total, cnt]

    # Simple failure detection: if reward is negative, count a failure
    if update.reward < 0:
        _circuit_breaker.record_failure(update.action_id)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset everything
    reset_policy()

    # Define a tiny action set with explicit Voronoi positions
    actions = [
        BanditAction(action_id="A", position=(0.0, 0.0)),
        BanditAction(action_id="B", position=(1.0, 0.0)),
        BanditAction(action_id="C", position=(0.0, 1.0)),
    ]

    # Initialise a 1‑D input (distance) LTC with 4 hidden units
    ltc = init_ltc(input_dim=1, hidden_dim=4, tau=1.0, dt=0.05, seed=42)

    # Simulate 30 interaction steps
    for step in range(30):
        ctx_id = f"ctx_{step}"
        # Random context point inside the unit square
        ctx_point = (random.random(), random.random())

        # Select action using the hybrid mechanism
        chosen, gating = hybrid_select_action(ctx_id, ctx_point, actions, ltc)

        # Simulate a stochastic reward: +1 if distance < 0.5 else -1
        dist = euclidean_distance(ctx_point, chosen.position)
        reward = 1.0 if dist < 0.5 else -1.0

        # Record the update
        upd = BanditUpdate(
            context_id=ctx_id,
            action_id=chosen.action_id,
            reward=reward,
            propensity=chosen.propensity,
        )
        hybrid_update_policy(upd)

        # Optional: print a concise log
        print(
            f"Step {step:02d} | ctx={ctx_point} | chosen={chosen.action_id} "
            f"| dist={dist:.3f} | reward={reward:.1f} | prop={chosen.propensity:.3f}"
        )

    # Final policy summary
    print("\n=== Final Policy Statistics ===")
    for aid, (tot, cnt) in _POLICY.items():
        print(f"Action {aid}: total_reward={tot:.2f}, pulls={int(cnt)}")
    print("\nFailures recorded:", FAILURES)