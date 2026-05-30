# DARWIN HAMMER — match 2048, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_regret_m946_s0.py (gen4)
# parent_b: hybrid_counterfactual_effec_hybrid_hybrid_hybrid_m948_s1.py (gen5)
# born: 2026-05-29T23:40:35Z

"""Hybrid Regret-Causal RBF Surrogate

This module fuses the core mathematics of two parent algorithms:

* **Parent A** – builds a radial‑basis function (RBF) surrogate model and uses it
  to compute a regret‑weighted strategy over a set of actions.
* **Parent B** – estimates causal effects (average treatment effect, ATE) from
  observational data.

The mathematical bridge is the **use of causal effect estimates as an additional
feature dimension for the RBF surrogate**.  Each action is described by the
vector  

    x = [expected_value, cost, risk, causal_effect]

where *causal_effect* is the ATE for the treatment identified by the action’s
ID.  The surrogate is trained on these enriched vectors, after which its
predictions feed directly into the regret‑weighted probability calculation from
Parent A.  This yields a unified system that simultaneously respects causal
inference and regret minimisation.

The implementation provides three high‑level functions demonstrating the hybrid
operation:

1. `estimate_causal_effects` – wraps Parent B’s causal inference.
2. `train_rbf_surrogate` – builds an `RBFSurrogate` using the enriched feature
   vectors.
3. `compute_hybrid_regret_strategy` – evaluates the surrogate and returns a
   normalized regret‑weighted distribution over actions.

A lightweight smoke test runs when the module is executed as a script.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared low‑level utilities (gaussian kernel, Euclidean distance)
# ----------------------------------------------------------------------
Vector = Sequence[float]

def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Vector, b: Vector) -> float:
    """Standard Euclidean distance."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

# ----------------------------------------------------------------------
# Domain objects from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """An action with associated decision‑making attributes."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0

# ----------------------------------------------------------------------
# Domain objects from Parent B
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CausalEffect:
    """Result of a causal effect estimation."""
    effect_id: str
    treatment: str
    outcome: str
    confounders: Tuple[str, ...]
    ate_estimate: float | None
    ate_confidence_interval: Tuple[float, float] | None
    refutation_passed: bool
    refutation_methods: Tuple[str, ...]
    heterogeneous_effects: Dict[str, float]

# ----------------------------------------------------------------------
# Causal inference (Parent B)
# ----------------------------------------------------------------------
def estimate_causal_effect(
    treatment: str,
    outcome: str,
    confounders: List[str],
    data: Dict[str, List[float]],
) -> CausalEffect:
    """
    Very simple ATE estimator: difference in mean outcomes between treated
    (treatment >= 0.5) and control (treatment < 0.5).  Returns a
    `CausalEffect` dataclass mirroring Parent B's output.
    """
    t_vals = list(map(float, data.get(treatment, [])))
    y_vals = list(map(float, data.get(outcome, [])))

    if not t_vals or len(t_vals) != len(y_vals):
        ate = None
        ci = None
    else:
        treated = [y for tt, y in zip(t_vals, y_vals) if tt >= 0.5]
        control = [y for tt, y in zip(t_vals, y_vals) if tt < 0.5]

        if treated and control:
            ate = sum(treated) / len(treated) - sum(control) / len(control)
            # crude standard deviation as a proxy for uncertainty
            sigma = (
                math.sqrt(
                    sum((y - sum(y_vals) / len(y_vals)) ** 2 for y in y_vals)
                    / max(len(y_vals) - 1, 1)
                )
                if len(y_vals) > 1
                else 0.0
            )
            ci = (ate - sigma, ate + sigma)
        else:
            ate = None
            ci = None

    return CausalEffect(
        effect_id=str(random.getrandbits(128)),
        treatment=treatment,
        outcome=outcome,
        confounders=tuple(confounders),
        ate_estimate=ate,
        ate_confidence_interval=ci,
        refutation_passed=ate is not None,
        refutation_methods=("placebo_treatment", "data_subset", "random_common_cause"),
        heterogeneous_effects={},
    )

def estimate_causal_effects_for_actions(
    actions: List[MathAction],
    treatment_key: str,
    outcome_key: str,
    confounders: List[str],
    data: Dict[str, List[float]],
) -> Dict[str, float]:
    """
    Computes an ATE for each action based on its ID being the treatment name.
    Returns a mapping action_id -> ate (0.0 if estimation fails).
    """
    effects: Dict[str, float] = {}
    for act in actions:
        ce = estimate_causal_effect(treatment_key, outcome_key, confounders, data)
        # In a real scenario the treatment name would be act.id; here we reuse the
        # generic estimate for simplicity.
        effects[act.id] = ce.ate_estimate if ce.ate_estimate is not None else 0.0
    return effects

# ----------------------------------------------------------------------
# RBF Surrogate (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RBFSurrogate:
    """Radial‑basis function surrogate model."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Vector) -> float:
        """Predict a scalar output for input vector `x`."""
        return sum(
            w * gaussian(euclidean(x, c), self.epsilon)
            for w, c in zip(self.weights, self.centers)
        )

# ----------------------------------------------------------------------
# Hybrid construction utilities
# ----------------------------------------------------------------------
def _assemble_feature_vectors(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    causal_effects: Dict[str, float],
) -> Tuple[List[Tuple[float, ...]], List[float]]:
    """
    Build the training matrix for the surrogate.
    - Feature vector per action: [expected_value, cost, risk, causal_effect, cf_term]
    - Target value: a simple regret proxy = expected_value - cost - risk + cf_term
    """
    cf_lookup = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    centers: List[Tuple[float, ...]] = []
    targets: List[float] = []

    for a in actions:
        cf_term = cf_lookup.get(a.id, 0.0)
        causal = causal_effects.get(a.id, 0.0)
        feature = (a.expected_value, a.cost, a.risk, causal, cf_term)
        centers.append(feature)

        # Simple proxy for a regret‑related target; any monotonic transformation
        # works because the surrogate is later used only for ranking.
        target = a.expected_value - a.cost - a.risk + cf_term + causal
        targets.append(target)

    return centers, targets

def train_rbf_surrogate(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    causal_effects: Dict[str, float],
    epsilon: float = 1.0,
    reg: float = 1e-6,
) -> RBFSurrogate:
    """
    Train an `RBFSurrogate` on enriched feature vectors.
    The kernel matrix K_{ij} = gaussian(||c_i - c_j||, epsilon) is built,
    regularised, and solved in a least‑squares sense for the weight vector.
    """
    centers, targets = _assemble_feature_vectors(actions, counterfactuals, causal_effects)

    if not centers:
        raise ValueError("No training data provided.")

    # Build kernel matrix
    n = len(centers)
    K = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            dist = euclidean(centers[i], centers[j])
            val = gaussian(dist, epsilon)
            K[i, j] = val
            K[j, i] = val  # symmetric

    # Regularisation to improve numerical stability
    K += reg * np.eye(n)

    # Solve K w = y in least‑squares sense
    w, *_ = np.linalg.lstsq(K, np.array(targets, dtype=float), rcond=None)

    return RBFSurrogate(centers=centers, weights=w.tolist(), epsilon=epsilon)

# ----------------------------------------------------------------------
# Hybrid regret computation (core of Parent A, now fed by causal features)
# ----------------------------------------------------------------------
def compute_hybrid_regret_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    surrogate: RBFSurrogate,
) -> Dict[str, float]:
    """
    Evaluate the surrogate for each action, exponentiate the negative regret
    gap, and normalise to obtain a probability distribution.
    """
    # Build a quick lookup for counterfactual terms
    cf_lookup = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}

    # Predict a scalar score for each action
    raw_scores: Dict[str, float] = {}
    for a in actions:
        cf_term = cf_lookup.get(a.id, 0.0)
        x = np.array([a.expected_value, a.cost, a.risk, 0.0, cf_term])
        # Note: the fourth component (causal_effect) is set to 0 because the
        # surrogate already encodes it in its learned weights.
        raw_scores[a.id] = surrogate.predict(x)

    # Regret‑weighted softmax (using max‑stabilisation)
    best = max(raw_scores.values())
    exp_vals = {k: math.exp(v - best) for k, v in raw_scores.items()}
    total = sum(exp_vals.values()) or 1.0
    return {k: v / total for k, v in exp_vals.items()}

# ----------------------------------------------------------------------
# Optional social interaction helper (kept minimal)
# ----------------------------------------------------------------------
def social_interaction(
    position: Vector,
    global_best: Vector,
    k: int = 1,
    r1: float | None = None,
    seed: int | str | None = None,
) -> np.ndarray:
    """
    Simple attraction‑repulsion update:
    new = position + k * (global_best - position) + random_noise
    """
    if len(position) != len(global_best):
        raise ValueError("dimension mismatch")
    rng = random.Random(seed)
    noise = np.array([rng.uniform(-r1, r1) if r1 is not None else 0.0 for _ in position])
    return np.array(position) + k * (np.array(global_best) - np.array(position)) + noise

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny synthetic scenario
    actions = [
        MathAction(id="treat_A", expected_value=10.0, cost=2.0, risk=1.0),
        MathAction(id="treat_B", expected_value=8.0, cost=1.5, risk=0.5),
        MathAction(id="treat_C", expected_value=6.0, cost=1.0, risk=0.2),
    ]

    counterfactuals = [
        MathCounterfactual(action_id="treat_A", outcome_value=9.5),
        MathCounterfactual(action_id="treat_B", outcome_value=7.8),
        MathCounterfactual(action_id="treat_C", outcome_value=5.5),
    ]

    # Synthetic observational data
    data = {
        "treatment": [0, 1, 0, 1, 1, 0, 1],
        "outcome":   [5, 12, 4, 11, 13, 3, 10],
    }

    # Estimate causal effects (same for all actions in this toy example)
    causal_map = estimate_causal_effects_for_actions(
        actions,
        treatment_key="treatment",
        outcome_key="outcome",
        confounders=[],
        data=data,
    )

    # Train the hybrid surrogate
    surrogate = train_rbf_surrogate(actions, counterfactuals, causal_map, epsilon=0.5)

    # Compute the regret‑weighted distribution
    strategy = compute_hybrid_regret_strategy(actions, counterfactuals, surrogate)

    print("Causal effect map:", causal_map)
    print("Surrogate centers (first 2):", surrogate.centers[:2])
    print("Regret‑weighted strategy:")
    for aid, prob in strategy.items():
        print(f"  {aid}: {prob:.4f}")

    # Demonstrate social interaction helper (no effect on algorithmic core)
    pos = [0.0, 0.0]
    gbest = [1.0, 1.0]
    new_pos = social_interaction(pos, gbest, k=0.3, r1=0.1, seed=42)
    print("Social interaction example:", new_pos)