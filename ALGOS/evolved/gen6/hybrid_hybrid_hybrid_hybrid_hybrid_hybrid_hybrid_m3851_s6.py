# DARWIN HAMMER — match 3851, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1016_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m871_s3.py (gen5)
# born: 2026-05-29T23:52:01Z

"""Hybrid Fusion Module
====================

This module fuses the core topologies of two parent algorithms:

* **Parent A** – provides a geometric‑algebra based ``Multivector`` class whose
  geometric product can be modulated by a scalar *pheromone* signal.
* **Parent B** – supplies a Liquid‑Time‑Constant (LTC)‑style scalar generator
  (adaptively producing a pheromone value from contextual features) and a
  Count‑Min Sketch that yields fast scalar log‑likelihood estimates for a
  bandit‑router.

**Mathematical Bridge**

Both parents treat a *scalar* as a modulating factor:

* In Parent A the pheromone scales the coefficients of the geometric product.
* In Parent B the LTC network outputs a scalar that scales resource allocation,
  while the Count‑Min Sketch supplies a scalar log‑likelihood estimate.

The fused algorithm therefore:

1. Uses the LTC network to produce an adaptive pheromone value from the current
   context.
2. Applies that pheromone to the geometric product of two multivectors.
3. Reduces the resulting multivector to a scalar (e.g. the sum of its
   coefficients) and feeds it into a Count‑Min Sketch to obtain a fast
   log‑likelihood estimate.
4. Uses the sketch‑derived estimate together with a classic bandit policy to
   select an action.

The three public functions below illustrate this integrated workflow.
"""

import math
import random
import sys
from datetime import date
from pathlib import Path
from collections import defaultdict
import numpy as np
import hashlib

# ----------------------------------------------------------------------
# Shared constants and utilities
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


# ----------------------------------------------------------------------
# Geometric Algebra core (Parent A)
# ----------------------------------------------------------------------
class Multivector:
    """Sparse representation of a multivector.

    ``components`` maps a blade (tuple of basis indices) to a float coefficient.
    Only non‑zero coefficients are stored.
    """

    def __init__(self, components: dict[tuple[int, ...], float], n: int):
        self.n = int(n)
        # store only non‑zero coefficients, coerced to float
        self.components = {
            blade: float(coef) for blade, coef in components.items() if coef != 0.0
        }

    def grade(self, k: int) -> "Multivector":
        """Return a new multivector containing only blades of grade ``k``."""
        return Multivector(
            {
                blade: coef
                for blade, coef in self.components.items()
                if len(blade) == k
            },
            self.n,
        )

    def scalar_part(self) -> "Multivector":
        """Extract the grade‑0 (scalar) part."""
        return self.grade(0)

    def __add__(self, other: "Multivector") -> "Multivector":
        assert self.n == other.n, "Dimension mismatch"
        new_comp = self.components.copy()
        for blade, coef in other.components.items():
            new_comp[blade] = new_comp.get(blade, 0.0) + coef
            if abs(new_comp[blade]) < 1e-12:
                del new_comp[blade]
        return Multivector(new_comp, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product without pheromone scaling (used internally)."""
        return geometric_product(self, other, pheromone=1.0)

    def __repr__(self) -> str:
        return f"Multivector({self.components}, n={self.n})"


def _blade_sign(indices: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    """Return the sorted blade and the sign introduced by swapping indices.

    Identical indices cancel (Grassmann algebra property)."""
    lst = list(indices)
    sign = 1
    # bubble‑sort while counting swaps
    for i in range(len(lst)):
        for j in range(len(lst) - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign = -sign
    # cancel pairs of identical indices
    i = 0
    while i < len(lst) - 1:
        if lst[i] == lst[i + 1]:
            del lst[i : i + 2]
            sign = -sign  # swapping identical indices introduces a sign flip
            i = 0  # restart scan
        else:
            i += 1
    return tuple(lst), sign


def geometric_product(
    a: Multivector, b: Multivector, pheromone: float = 1.0
) -> Multivector:
    """Geometric product of two multivectors modulated by a scalar pheromone.

    The classic geometric product is computed blade‑wise; each resulting
    coefficient is multiplied by ``pheromone`` – the bridge to Parent B.
    """
    result: dict[tuple[int, ...], float] = defaultdict(float)
    for blade_a, coef_a in a.components.items():
        for blade_b, coef_b in b.components.items():
            combined = blade_a + blade_b
            sorted_blade, sign = _blade_sign(combined)
            result[sorted_blade] += sign * coef_a * coef_b * pheromone
    # prune zeros
    result = {b: v for b, v in result.items() if abs(v) > 1e-12}
    return Multivector(result, a.n)


# ----------------------------------------------------------------------
# Liquid‑Time‑Constant (LTC) scalar generator (Parent B)
# ----------------------------------------------------------------------
class LTCNetwork:
    """A minimal LTC implementation that maps a feature vector to a scalar pheromone.

    The dynamics follow a first‑order linear ODE:
        dp/dt = -α·p + w·x
    where ``p`` is the pheromone, ``α`` a decay constant and ``w`` a weight vector.
    Discretisation uses Euler steps with a configurable time step ``dt``.
    """

    def __init__(self, input_dim: int, decay: float = 0.1, dt: float = 0.01):
        self.input_dim = int(input_dim)
        self.decay = float(decay)
        self.dt = float(dt)
        # Randomly initialise weights; seed for reproducibility
        random.seed(0)
        self.weights = np.array(
            [random.uniform(-1.0, 1.0) for _ in range(self.input_dim)], dtype=float
        )
        self.state = 0.0  # initial pheromone value

    def step(self, features: np.ndarray) -> float:
        """Perform one Euler update and return the new pheromone."""
        assert features.shape == (self.input_dim,)
        # Linear combination
        drive = float(np.dot(self.weights, features))
        # Euler update
        dp = -self.decay * self.state + drive
        self.state += self.dt * dp
        # Keep pheromone in a reasonable range
        self.state = max(min(self.state, 10.0), -10.0)
        return self.state


# ----------------------------------------------------------------------
# Count‑Min Sketch (Parent B)
# ----------------------------------------------------------------------
def _cms_hash(item: str, depth: int, width: int) -> list[int]:
    """Deterministic hash for Count‑Min Sketch rows."""
    return [
        int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16) % width
        for d in range(depth)
    ]


class CountMinSketch:
    """Simple Count‑Min Sketch for non‑negative scalar accumulation."""

    def __init__(self, width: int = 64, depth: int = 4):
        self.width = int(width)
        self.depth = int(depth)
        self.table = np.zeros((self.depth, self.width), dtype=float)

    def add(self, item: str, value: float) -> None:
        """Add ``value`` to the sketch entry for ``item``."""
        for row, col in enumerate(_cms_hash(item, self.depth, self.width)):
            self.table[row, col] += value

    def query(self, item: str) -> float:
        """Return the minimum count for ``item`` across all rows."""
        estimates = [
            self.table[row, col] for row, col in enumerate(_cms_hash(item, self.depth, self.width))
        ]
        return min(estimates)


# ----------------------------------------------------------------------
# Bandit policy (Parent A)
# ----------------------------------------------------------------------
_POLICY: dict[str, list[float]] = {}  # action_id -> [total_reward, count, total_propensity]


def reset_policy() -> None:
    """Clear the global bandit policy."""
    _POLICY.clear()


def _policy_stats(action_id: str) -> tuple[float, float, float]:
    """Return (total_reward, count, total_propensity) for ``action_id``."""
    return tuple(_POLICY.get(action_id, [0.0, 0.0, 0.0]))


def update_policy(updates: list[tuple[str, float, float]]) -> None:
    """Update the policy with a list of (action_id, reward, propensity)."""
    for action_id, reward, propensity in updates:
        total, cnt, total_prop = _policy_stats(action_id)
        _POLICY[action_id] = [
            total + float(reward),
            cnt + 1.0,
            total_prop + float(propensity),
        ]


def _expected_reward(action_id: str) -> float:
    total, cnt, _ = _policy_stats(action_id)
    return total / cnt if cnt > 0 else 0.0


def select_action(action_ids: list[str]) -> str:
    """UCB‑style selection using expected reward + confidence bound."""
    # Simple Upper Confidence Bound (UCB1) implementation
    total_counts = sum(_policy_stats(a)[1] for a in action_ids) + 1e-9
    best_score = -math.inf
    best_action = random.choice(action_ids)  # fallback
    for a in action_ids:
        exp = _expected_reward(a)
        _, cnt, _ = _policy_stats(a)
        confidence = math.sqrt(2 * math.log(total_counts) / (cnt + 1e-9))
        score = exp + confidence
        if score > best_score:
            best_score = score
            best_action = a
    return best_action


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def compute_pheromone(ltc: LTCNetwork, features: np.ndarray) -> float:
    """Generate a pheromone scalar from contextual features using the LTC network."""
    return ltc.step(features)


def fused_geometric_product(
    mv1: Multivector, mv2: Multivector, pheromone: float
) -> Multivector:
    """Geometric product of two multivectors scaled by the pheromone signal."""
    return geometric_product(mv1, mv2, pheromone=pheromone)


def sketch_based_bandit_step(
    context_id: str,
    mv: Multivector,
    sketch: CountMinSketch,
    action_space: list[str],
) -> str:
    """
    Reduce a multivector to a scalar, feed it to the sketch, and use the
    resulting estimate to update the bandit policy and select an action.

    Returns the chosen ``action_id``.
    """
    # Reduce multivector to scalar: sum of all coefficients (simple proxy)
    scalar_estimate = sum(mv.components.values())
    # Update sketch with the scalar (converted to string key)
    sketch_key = f"{context_id}:{scalar_estimate:.6f}"
    sketch.add(sketch_key, scalar_estimate)

    # Use sketch estimate as a pseudo‑reward for each candidate action
    updates = []
    for a in action_space:
        # Query the sketch for a key that mixes action and context
        reward_est = sketch.query(f"{context_id}:{a}")
        propensity = random.random()  # placeholder propensity
        updates.append((a, reward_est, propensity))

    update_policy(updates)
    return select_action(action_space)


# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Initialise components
    ltc = LTCNetwork(input_dim=3, decay=0.05, dt=0.1)
    sketch = CountMinSketch(width=128, depth=5)

    # 2. Create two example multivectors in a 3‑dimensional space
    mv_a = Multivector({(1,): 2.0, (2, 3): -1.5}, n=3)
    mv_b = Multivector({(2,): 0.5, (1, 3): 3.0}, n=3)

    # 3. Generate a pheromone value from dummy features (e.g., day of week, random)
    today = date.today()
    features = np.array(
        [
            today.weekday() / 6.0,               # normalized weekday
            random.random(),                    # external stochastic feature
            math.sin(today.toordinal()),         # periodic component
        ],
        dtype=float,
    )
    pheromone = compute_pheromone(ltc, features)

    # 4. Compute the fused geometric product
    mv_product = fused_geometric_product(mv_a, mv_b, pheromone)

    # 5. Perform a sketch‑based bandit step
    actions = ["codex", "groq", "cohere", "local_models"]
    chosen = sketch_based_bandit_step(
        context_id="demo_ctx",
        mv=mv_product,
        sketch=sketch,
        action_space=actions,
    )

    # 6. Report results
    print("Pheromone value:", _pct(pheromone))
    print("Multivector product:", mv_product)
    print("Chosen action:", chosen)
    print("Policy snapshot:", {k: [_pct(v[0]), v[1], _pct(v[2])] for k, v in _POLICY.items()})