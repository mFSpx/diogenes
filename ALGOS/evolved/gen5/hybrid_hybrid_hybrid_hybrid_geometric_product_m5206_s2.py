# DARWIN HAMMER — match 5206, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_hybrid_m1430_s3.py (gen4)
# parent_b: geometric_product.py (gen0)
# born: 2026-05-30T00:00:46Z

"""HybridPheromoneGeometricBandit
Combines:
- Parent A: HybridPheromoneBanditSystem (pheromone decay, bandit updates)
- Parent B: Clifford geometric algebra (Multivector with geometric product)

Mathematical bridge:
Pheromone signals are promoted from scalar values to full multivectors.
When a pheromone is updated we combine the existing multivector with the new
signal using the geometric product (a·b + a∧b).  Decay is performed by
geometric multiplication with a scalar decay factor encoded as a grade‑0
multivector.  Arm selection uses the magnitude (Euclidean norm) of each
arm's multivector value, providing a unified scalar that drives the bandit
policy while preserving the richer geometric information internally.
"""

import sys
import pathlib
import math
import random
import hashlib
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Dict, Any
import numpy as np
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Clifford geometric algebra core (Parent B)
# ---------------------------------------------------------------------------

def _blade_sign(indices):
    """Return (sorted_indices, sign) after bubble‑sorting index list.

    Swapping adjacent out‑of‑order indices flips the sign (anti‑commutativity).
    Duplicate indices cancel because e_i*e_i = 1.
    """
    lst = list(indices)
    sign = 1
    i = 0
    while i < len(lst) - 1:
        if lst[i] > lst[i + 1]:
            lst[i], lst[i + 1] = lst[i + 1], lst[i]
            sign *= -1
            i = max(i - 1, 0)
        elif lst[i] == lst[i + 1]:
            # Cancel the pair
            lst.pop(i)
            lst.pop(i)  # same index after removal
            sign *= 1
            i = max(i - 1, 0)
        else:
            i += 1
    return lst, sign

def _multiply_blades(blade_a: frozenset, blade_b: frozenset):
    """Geometric product of two basis blades.

    Returns (result_blade_frozenset, sign).
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

class Multivector:
    """Element of Cl(n,0) represented as a linear combination of basis blades."""
    def __init__(self, components: Dict[frozenset, float], n: int):
        self.n = int(n)
        # drop zero coefficients
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-15}

    @staticmethod
    def scalar(value: float, n: int):
        return Multivector({frozenset(): float(value)}, n)

    @staticmethod
    def basis_vector(idx: int, n: int):
        if idx < 0 or idx >= n:
            raise ValueError("basis index out of range")
        return Multivector({frozenset({idx}): 1.0}, n)

    def __add__(self, other: 'Multivector') -> 'Multivector':
        if self.n != other.n:
            raise ValueError("Dimension mismatch in addition")
        comps = self.components.copy()
        for k, v in other.components.items():
            comps[k] = comps.get(k, 0.0) + v
            if abs(comps[k]) < 1e-15:
                del comps[k]
        return Multivector(comps, self.n)

    def __sub__(self, other: 'Multivector') -> 'Multivector':
        return self + (-other)

    def __neg__(self) -> 'Multivector':
        return Multivector({k: -v for k, v in self.components.items()}, self.n)

    def __mul__(self, other: Any) -> 'Multivector':
        """Geometric product if other is Multivector, else scalar multiplication."""
        if isinstance(other, Multivector):
            if self.n != other.n:
                raise ValueError("Dimension mismatch in geometric product")
            result = {}
            for a_blade, a_coef in self.components.items():
                for b_blade, b_coef in other.components.items():
                    blade, sign = _multiply_blades(a_blade, b_blade)
                    coef = a_coef * b_coef * sign
                    result[blade] = result.get(blade, 0.0) + coef
            return Multivector(result, self.n)
        else:
            # scalar multiplication
            return Multivector({k: v * float(other) for k, v in self.components.items()}, self.n)

    __rmul__ = __mul__

    def norm(self) -> float:
        """Euclidean norm of the coefficient vector (grade‑agnostic)."""
        return math.sqrt(sum(v * v for v in self.components.values()))

    def __repr__(self):
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if not blade:
                term = f"{coef:.3g}"
            else:
                idxs = "*".join(f"e{i}" for i in sorted(blade))
                term = f"{coef:.3g}{idxs}"
            terms.append(term)
        return " + ".join(terms) if terms else "0"

# ---------------------------------------------------------------------------
# Hybrid pheromone‑bandit system with geometric algebra (Parent A + B)
# ---------------------------------------------------------------------------

class HybridGeoBanditSystem:
    """Bandit system where pheromone values are multivectors.

    Each arm stores a Multivector `value`.  Updates combine the stored
    multivector with the incoming signal via geometric product.
    Decay multiplies by a scalar decay factor encoded as a grade‑0 multivector.
    """
    def __init__(self, n_arms: int = 5, dim: int = 3,
                 alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        self.n_arms = n_arms
        self.dim = dim                      # dimension of the Clifford algebra
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pheromones: Dict[str, Dict[str, Any]] = {}
        self.counts = np.zeros(n_arms, dtype=int)
        # Initialise each arm's value as a zero multivector
        self.values: List[Multivector] = [Multivector.scalar(0.0, dim) for _ in range(n_arms)]
        self.total_pulls = 0

    # -----------------------------------------------------------------------
    # Time utilities (from Parent A)
    # -----------------------------------------------------------------------
    def _current_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _decayed_scalar(self, created: datetime, value: float, half_life: float) -> float:
        """Scalar exponential decay."""
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        elapsed = (self._current_utc() - created).total_seconds()
        decay_factor = 0.5 ** (elapsed / half_life)
        return value * decay_factor

    # -----------------------------------------------------------------------
    # Geometric‑algebra aware decay
    # -----------------------------------------------------------------------
    def geometric_decay(self, mv: Multivector, created: datetime, half_life: float) -> Multivector:
        """Decay every coefficient of `mv` by the same scalar factor."""
        factor = self._decayed_scalar(created, 1.0, half_life)
        decay_mv = Multivector.scalar(factor, self.dim)
        return mv * decay_mv

    # -----------------------------------------------------------------------
    # Pheromone update using geometric product
    # -----------------------------------------------------------------------
    def update_pheromone(self, surface_key: str, signal_kind: str,
                         signal_mv: Multivector, half_life_seconds: float) -> Multivector:
        """Create or update a pheromone entry.

        The new multivector is combined with the existing one via geometric
        product, then decayed according to its age.
        """
        now = self._current_utc()
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {
                'signal_kind': signal_kind,
                'mv': signal_mv,
                'created': now,
                'half_life': half_life_seconds
            }
            combined = signal_mv
        else:
            entry = self.pheromones[surface_key]
            # Decay the stored multivector to its current age
            aged = self.geometric_decay(entry['mv'], entry['created'], entry['half_life'])
            # Combine via geometric product
            combined = aged * signal_mv
            # Update entry
            entry.update({
                'signal_kind': signal_kind,
                'mv': combined,
                'created': now,
                'half_life': half_life_seconds
            })
        # Return the current (aged) multivector for external inspection
        return self.geometric_decay(combined, now, half_life_seconds)

    # -----------------------------------------------------------------------
    # Bandit arm selection using multivector magnitude
    # -----------------------------------------------------------------------
    def select_arm(self) -> int:
        """UCB‑like selection where the score is based on multivector norm."""
        total = self.total_pulls + 1e-9
        scores = np.empty(self.n_arms)
        for i in range(self.n_arms):
            exploitation = self.values[i].norm()
            exploration = self.alpha * math.sqrt(math.log(total) / (self.counts[i] + 1))
            scores[i] = exploitation + exploration
        chosen = int(np.argmax(scores))
        return chosen

    # -----------------------------------------------------------------------
    # Reward update – integrates geometric information into value estimate
    # -----------------------------------------------------------------------
    def update_reward(self, arm_index: int, reward: float,
                      signal_mv: Multivector = None):
        """Update counts, total pulls, and the multivector value for the arm.

        If a signal multivector is supplied it is combined with the current
        value via geometric product, otherwise a scalar reward is lifted to a
        grade‑0 multivector.
        """
        self.counts[arm_index] += 1
        self.total_pulls += 1

        reward_mv = Multivector.scalar(reward, self.dim)
        if signal_mv is not None:
            reward_mv = reward_mv * signal_mv

        # Simple exponential moving average in the geometric algebra space
        eta = self.beta
        old = self.values[arm_index]
        self.values[arm_index] = (1 - eta) * old + eta * reward_mv

    # -----------------------------------------------------------------------
    # Helper to generate a random multivector signal (for testing)
    # -----------------------------------------------------------------------
    def random_signal(self) -> Multivector:
        """Create a random multivector with coefficients in [-1,1]."""
        comps = {}
        # scalar part
        comps[frozenset()] = random.uniform(-1, 1)
        # random vector components
        for i in range(self.dim):
            comps[frozenset({i})] = random.uniform(-1, 1)
        # random bivector components (i<j)
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                comps[frozenset({i, j})] = random.uniform(-1, 1)
        return Multivector(comps, self.dim)

# ---------------------------------------------------------------------------
# Demonstration functions (require at least three)
# ---------------------------------------------------------------------------

def demo_geometric_product():
    """Showcase geometric product between two random multivectors."""
    mv1 = Multivector({frozenset(): 2.0, frozenset({0}): 1.0}, 3)
    mv2 = Multivector({frozenset(): -1.0, frozenset({1}): 0.5}, 3)
    prod = mv1 * mv2
    print("mv1 =", mv1)
    print("mv2 =", mv2)
    print("mv1 * mv2 =", prod)

def demo_pheromone_update():
    """Create a system, update a pheromone with two signals and print result."""
    sys = HybridGeoBanditSystem(n_arms=3, dim=3)
    sig1 = sys.random_signal()
    sig2 = sys.random_signal()
    print("Signal 1:", sig1)
    sys.update_pheromone("nodeA", "typeX", sig1, half_life_seconds=30.0)
    print("After first update:", sys.pheromones["nodeA"]["mv"])
    print("Signal 2:", sig2)
    sys.update_pheromone("nodeA", "typeX", sig2, half_life_seconds=30.0)
    print("After second update (geometric product + decay):", sys.pheromones["nodeA"]["mv"])

def demo_bandit_loop():
    """Run a tiny bandit loop using geometric signals as side information."""
    bandit = HybridGeoBanditSystem(n_arms=4, dim=3)
    for t in range(15):
        arm = bandit.select_arm()
        # Simulated reward: higher for lower arm index
        reward = 1.0 / (arm + 1)
        # Random side‑signal influencing the update
        side_signal = bandit.random_signal()
        bandit.update_reward(arm, reward, signal_mv=side_signal)
        print(f"Round {t:02d}: chose arm {arm}, reward {reward:.3f}, value norm {bandit.values[arm].norm():.3f}")

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Demo: Geometric Product ===")
    demo_geometric_product()
    print("\n=== Demo: Pheromone Update ===")
    demo_pheromone_update()
    print("\n=== Demo: Bandit Loop ===")
    demo_bandit_loop()