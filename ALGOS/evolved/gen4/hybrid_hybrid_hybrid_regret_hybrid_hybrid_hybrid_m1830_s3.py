# DARWIN HAMMER — match 1830, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_hybrid_bandit_m83_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_bandit_router_m202_s2.py (gen3)
# born: 2026-05-29T23:39:02Z

"""
Hybrid Regret‑Bandit‑Koopman‑Honeybee‑Ternary Engine
===================================================

Parents
-------
* **Parent A** – ``hybrid_hybrid_regret_engine_hybrid_hybrid_bandit_m83_s4``  
  Provides a regret‑weighted probability distribution `p_t` over actions, a Gini
  coefficient `G_t` that quantifies inequality of `p_t`, a *store* `S_t` that
  scales confidence, and a Koopman operator `K` that linearly forecasts the
  empirical mean reward vector `μ_t`.

* **Parent B** – ``hybrid_hybrid_hybrid_ternar_hybrid_bandit_router_m202_s2``  
  Supplies a honeybee‑style store (`StoreState`) whose dynamics are driven by
  inflow/outflow streams, and a structural‑similarity (`ssim`) metric that can
  be used to compare a context vector with a predicted reward vector.

Mathematical Bridge
-------------------
The regret‑weighted distribution `p_t` is interpreted as the observable vector
`μ_t` for the Koopman operator.  The Koopman forecast `μ̂_{t+h}=K^h μ_t` gives an
exploitation term.  The Gini coefficient `G_t` of `p_t` modulates the store
level `S_t` from Parent B; the product `C_t = S_t·G_t` scales the confidence
bound supplied by the contextual bandit.  Finally, the similarity between the
current context `c_t` and the forecast `μ̂_{t+h}` (via `ssim`) is fed back to
the ternary router to bias the selection toward actions whose predicted
rewards are structurally similar to the context.

The unified decision index for action *a* at time *t* is

    U_a(t) = μ̂_a(t) + η·‖c_t‖·C_t·ssim(c_t, μ̂(t))

where `η` is a tunable exploration coefficient.  The action with maximal `U`
is selected.

The implementation below fuses the core topologies of both parents while
respecting the required constraints.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MathAction:
    """Action definition used by the regret engine."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class BanditAction:
    """Result of a bandit selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretBanditKoopmanTernary"


@dataclass
class StoreState:
    """Honeybee‑style store with simple inflow/outflow dynamics."""
    level: float = 0.0
    alpha: float = 1.0   # inflow gain
    beta: float = 1.0    # outflow decay
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """
        Apply the store equation:
            dS/dt = α·Σ(inflow) – β·Σ(outflow)
        and integrate over a single time step `dt`.  The level is clipped to
        `[0, limit]`.  Returns the new level and the applied delta.
        """
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, min(self.limit, self.level + delta * self.dt))
        return self.level, delta


# ----------------------------------------------------------------------
# Core mathematical utilities (shared by both parents)
# ----------------------------------------------------------------------


def compute_gini(probabilities: List[float]) -> float:
    """Gini coefficient of a probability distribution."""
    if not probabilities:
        return 0.0
    sorted_p = sorted(probabilities)
    n = len(sorted_p)
    cum = sum((i + 1) * p for i, p in enumerate(sorted_p))
    gini = (2 * cum) / (n * sum(sorted_p)) - (n + 1) / n
    return max(0.0, min(1.0, gini))


def koopman_forecast(mu: np.ndarray, K: np.ndarray, horizon: int) -> np.ndarray:
    """
    Linear Koopman forecast: μ̂ = K^h μ.
    """
    if horizon < 0:
        raise ValueError("horizon must be non‑negative")
    K_power = np.linalg.matrix_power(K, horizon)
    return K_power @ mu


def ssim(x: np.ndarray, y: np.ndarray, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """
    Simplified Structural Similarity Index Measure (SSIM) for 1‑D vectors.
    Returns a value in [0, 1] where 1 means perfect similarity.
    """
    if x.shape != y.shape:
        raise ValueError("SSIM requires vectors of equal shape")
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator) if denominator != 0 else 0.0


def softmax(values: List[float], temperature: float = 1.0) -> List[float]:
    """Numerically stable softmax."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    max_val = max(values)
    exps = [math.exp((v - max_val) / temperature) for v in values]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------


def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[Tuple[str, float]],
    temperature: float = 0.5,
) -> Dict[str, float]:
    """
    Simple regret‑weighted softmax:
        w_a = max(0, regret_a)
    where regret_a = (best_expected - expected_value_a) + cost_a.
    Returns a probability distribution `p_t`.
    """
    if not actions:
        return {}
    best_ev = max(a.expected_value for a in actions)
    weights = []
    ids = []
    for a in actions:
        regret = max(0.0, (best_ev - a.expected_value) + a.cost)
        weights.append(regret)
        ids.append(a.id)
    probs = softmax(weights, temperature)
    return dict(zip(ids, probs))


def hybrid_decision_index(
    actions: List[MathAction],
    K: np.ndarray,
    horizon: int,
    context_vec: np.ndarray,
    store: StoreState,
    eta: float = 1.0,
) -> Tuple[BanditAction, Dict[str, float]]:
    """
    Compute the unified index `U_a(t)` for each action and return the selected
    `BanditAction` together with the intermediate probability distribution.
    """
    # 1. Regret‑weighted distribution (Parent A)
    p_dist = compute_regret_weighted_strategy(actions, [])
    probs = list(p_dist.values())
    gini = compute_gini(probs)

    # 2. Observable vector μ_t (expected values) for Koopman (Parent A)
    mu = np.array([a.expected_value for a in actions])
    mu_hat = koopman_forecast(mu, K, horizon)  # exploitation term

    # 3. Store dynamics (Parent B)
    # Use inflow = regret probabilities, outflow = (1‑probability)
    inflow = probs
    outflow = [1.0 - p for p in probs]
    level, _ = store.update(inflow, outflow)

    # 4. Confidence scaling via Gini and store level
    confidence_scale = level * gini

    # 5. Similarity between context and forecast
    sim = ssim(context_vec, mu_hat)

    # 6. Compute final index for each action
    norm_context = np.linalg.norm(context_vec)
    indices = []
    for idx, a in enumerate(actions):
        exploit = mu_hat[idx]
        explore = eta * norm_context * confidence_scale * sim
        U = exploit + explore
        indices.append(U)

    # 7. Select action with maximal index
    best_idx = int(np.argmax(indices))
    best_action = actions[best_idx]
    selected = BanditAction(
        action_id=best_action.id,
        propensity=p_dist.get(best_action.id, 0.0),
        expected_reward=mu_hat[best_idx],
        confidence_bound=confidence_scale * sim,
        algorithm="HybridRegretBanditKoopmanTernary",
    )
    return selected, p_dist


def ternary_route_adjustment(
    current_route: List[int],
    similarity: float,
    threshold: float = 0.6,
) -> List[int]:
    """
    Adjust a ternary routing command based on similarity.
    If similarity exceeds `threshold`, bias the route toward the central
    command (1).  Otherwise, keep the original ternary pattern.
    """
    if similarity >= threshold:
        # push the route toward the middle value (1) in a ternary set {0,1,2}
        return [1 if r != 1 else r for r in current_route]
    return current_route


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Define a tiny action set
    actions = [
        MathAction(id="A", expected_value=1.2, cost=0.1),
        MathAction(id="B", expected_value=0.9, cost=0.2),
        MathAction(id="C", expected_value=1.5, cost=0.05),
    ]

    # Koopman operator (3×3) – simple stochastic matrix
    K = np.array([[0.7, 0.2, 0.1],
                  [0.1, 0.8, 0.1],
                  [0.2, 0.3, 0.5]])

    horizon = 2

    # Random context vector (e.g., feature embedding)
    rng = np.random.default_rng(42)
    context_vec = rng.random(3)

    store = StoreState(level=2.0, alpha=0.9, beta=0.4, dt=1.0, limit=5.0)

    selected_action, prob_dist = hybrid_decision_index(
        actions=actions,
        K=K,
        horizon=horizon,
        context_vec=context_vec,
        store=store,
        eta=0.8,
    )

    print("Probability distribution (regret‑weighted):")
    for aid, p in prob_dist.items():
        print(f"  {aid}: {p:.4f}")

    print("\nStore level after update:", store.level)
    print("\nSelected action:", selected_action)

    # Demonstrate ternary route adjustment
    route = [0, 2, 0]  # example ternary command
    similarity = ssim(context_vec, koopman_forecast(
        np.array([a.expected_value for a in actions]), K, horizon))
    adjusted_route = ternary_route_adjustment(route, similarity)
    print("\nOriginal route:", route)
    print("Similarity:", f"{similarity:.4f}")
    print("Adjusted route:", adjusted_route)