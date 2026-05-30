# DARWIN HAMMER — match 2484, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_nlms_o_fisher_localization_m1155_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_regret_regret_engine_m822_s0.py (gen4)
# born: 2026-05-29T23:42:39Z

"""Hybrid NLMS‑Regret‑Fisher Module
This module fuses the core mathematics of two parent algorithms:

* **Parent A (hybrid_hybrid_hybrid_nlms_o_fisher_localization_m1155_s1.py)**
  – Provides a MinHash `signature` for a token set, a Gaussian beam model,
    and a Fisher information score `fisher_score(theta, center, width)`.
  – The Fisher score quantifies the sensitivity of a directional observation
    and is used as a scalar feature for each action.

* **Parent B (hybrid_hybrid_hybrid_regret_regret_engine_m822_s0.py)**
  – Supplies a regret‑weighted strategy `compute_regret_weighted_strategy`
    that turns expected values, costs, risks and counter‑factual outcomes
    into a probability distribution over actions.
  – Defines a `StoreState` whose internal dynamics produce a learning‑rate
    factor (`dance`) that can modulate adaptive updates.

**Mathematical Bridge**

For every candidate action *a* we compute  


R_a = regret‑weight(a)                     # from Parent B
F_a = fisher_score(θ_a, center, width)    # from Parent A
H_a = R_a * F_a                            # hybrid importance


The vector **H** is normalised to a probability simplex and serves as the
*target* distribution for an NLMS (Normalized Least‑Mean‑Squares) adaptive
filter that maps the MinHash signature **s** (size *k*) to a scalar scaling
factor `γ ∈ [0,1]`.  The NLMS update is


e   = H_target - γ
γ   = tanh(w·s)               # bounded output
w← w + μ * e * s / (ε + s·s)  # μ taken from StoreState.dance


The final propensity for action *a* is


π_a = H_a * γ


which is again normalised.  This construction intertwines the Fisher‑informed
directional sensitivity, the regret‑aware utility, and the NLMS adaptive
learning loop, yielding a single unified decision engine.

The module implements three core hybrid functions:
1. `hybrid_fisher_regret_weights` – builds the H‑vector.
2. `nlms_fisher_update` – performs the NLMS adaptation using the StoreState
   learning‑rate.
3. `hybrid_predict` – returns the final normalized propensities.
"""

import math
import random
import hashlib
import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

MAX64 = (1 << 64) - 1

def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash based on a seed and a token."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")

def signature(tokens: List[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set with k hash functions."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

# ----- Parent B data structures ------------------------------------------------

@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Learning‑rate factor derived from the most recent delta."""
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))

# ----- Regret‑Weighted Strategy (Parent B) ------------------------------------

def compute_regret_weighted_strategy(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual]
) -> Dict[str, float]:
    """Return a softmax‑like distribution over actions."""
    if not actions:
        return {}
    cf = {c.action_id: c.outcome_value * c.probability for c in counterfactuals}
    vals = {
        a.id: a.expected_value - a.cost - a.risk + cf.get(a.id, 0.0)
        for a in actions
    }
    best = max(vals.values())
    w = {k: math.exp(v - best) for k, v in vals.items()}
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}

# ----- Hybrid Core -------------------------------------------------------------

def hybrid_fisher_regret_weights(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    angles: Dict[str, float],
    center: float,
    width: float
) -> Dict[str, float]:
    """
    Combine regret weights with Fisher information.
    `angles` maps action IDs to a directional angle θ (radians).
    Returns a normalized hybrid importance vector H.
    """
    regret = compute_regret_weighted_strategy(actions, counterfactuals)
    # Fisher scores per action
    fisher = {
        a.id: fisher_score(angles.get(a.id, center), center, width)
        for a in actions
    }
    # Element‑wise product
    raw = {aid: regret.get(aid, 0.0) * fisher.get(aid, 0.0) for aid in regret}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}

def nlms_fisher_update(
    w: np.ndarray,
    s: np.ndarray,
    target: float,
    state: StoreState,
    eps: float = 1e-8
) -> Tuple[np.ndarray, float]:
    """
    Perform one NLMS adaptation step.
    * w – current weight vector (size k)
    * s – signature feature vector (size k, normalized to [0,1])
    * target – desired scalar output (the hybrid importance for the chosen action)
    Returns the updated weight vector and the scalar output after the update.
    """
    mu = state.dance  # learning‑rate from StoreState
    # Bounded output via tanh to keep γ ∈ (-1,1)
    gamma = math.tanh(float(np.dot(w, s)))
    error = target - gamma
    norm_sq = float(np.dot(s, s)) + eps
    w_new = w + (mu * error / norm_sq) * s
    # Keep weights finite
    w_new = np.clip(w_new, -1e6, 1e6)
    return w_new, gamma

def hybrid_predict(
    actions: List[MathAction],
    hybrid_weights: Dict[str, float],
    nlms_weights: np.ndarray,
    sig: List[int]
) -> Dict[str, float]:
    """
    Produce final propensities.
    * `hybrid_weights` – H_a from `hybrid_fisher_regret_weights`
    * `nlms_weights` – current NLMS weight vector
    * `sig` – MinHash signature (integer list)
    Returns a normalized dictionary {action_id: propensity}.
    """
    # Normalise signature to [0,1] for NLMS input
    s = np.array(sig, dtype=np.float64) / float(MAX64)
    gamma = math.tanh(float(np.dot(nlms_weights, s)))  # scalar scaling ∈ (-1,1)
    scale = 0.5 * (gamma + 1.0)                         # map to [0,1]
    raw = {aid: hybrid_weights.get(aid, 0.0) * scale for aid in hybrid_weights}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}

# ----- Convenience wrapper -----------------------------------------------------

def hybrid_step(
    state: StoreState,
    nlms_w: np.ndarray,
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    tokens: List[str],
    angles: Dict[str, float],
    center: float,
    width: float
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Execute one full hybrid iteration:
    1. Compute hybrid Fisher‑regret weights H.
    2. Build MinHash signature s from `tokens`.
    3. Pick a random action as the “reference” for NLMS target.
    4. Update NLMS weights using the target H for that action.
    5. Return updated NLMS weights and the full propensity distribution.
    """
    # 1. hybrid importance
    H = hybrid_fisher_regret_weights(actions, counterfactuals, angles, center, width)

    # 2. signature vector
    sig_int = signature(tokens, k=nlms_w.shape[0])
    s = np.array(sig_int, dtype=np.float64) / float(MAX64)

    # 3. choose a reference action (e.g., the one with max H)
    ref_id = max(H, key=H.get)
    target = H[ref_id]  # scalar target for NLMS

    # 4. NLMS adaptation
    nlms_w, _ = nlms_fisher_update(nlms_w, s, target, state)

    # 5. final propensities
    propensities = hybrid_predict(actions, H, nlms_w, sig_int)
    return nlms_w, propensities

# ----- Smoke test --------------------------------------------------------------

if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Sample actions
    actions = [
        MathAction(id="a1", expected_value=5.0, cost=1.0, risk=0.5),
        MathAction(id="a2", expected_value=3.0, cost=0.5, risk=0.2),
        MathAction(id="a3", expected_value=4.0, cost=0.8, risk=0.3),
    ]

    # Counterfactual outcomes (simulated)
    counterfactuals = [
        MathCounterfactual(action_id="a1", outcome_value=0.2, probability=0.9),
        MathCounterfactual(action_id="a2", outcome_value=-0.1, probability=0.7),
        MathCounterfactual(action_id="a3", outcome_value=0.05, probability=0.5),
    ]

    # Assign a random angle to each action (in radians)
    angles = {a.id: random.uniform(-math.pi, math.pi) for a in actions}
    center = 0.0
    width = 1.0

    # Token set for MinHash
    tokens = ["alpha", "beta", "gamma", "delta", "epsilon"]

    # Initialise StoreState and NLMS weight vector
    state = StoreState()
    k = 128  # signature length
    nlms_w = np.zeros(k, dtype=np.float64)

    # Run a few hybrid steps
    for step in range(5):
        nlms_w, prop = hybrid_step(
            state,
            nlms_w,
            actions,
            counterfactuals,
            tokens,
            angles,
            center,
            width,
        )
        print(f"Step {step+1} propensities:", prop)

    print("Final NLMS weights norm:", np.linalg.norm(nlms_w))