# DARWIN HAMMER — match 5795, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m871_s1.py (gen5)
# parent_b: hybrid_hybrid_perceptual_de_hybrid_hybrid_bandit_m255_s0.py (gen4)
# born: 2026-05-30T00:04:51Z

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


def _blade_sign(indices: List[int]) -> Tuple[frozenset, int]:
    """
    Return a canonical blade (as a frozenset of indices) and the sign
    resulting from re‑ordering the indices into ascending order.
    Duplicate indices cancel each other (Grassmann algebra rule).
    """
    # Remove pairs of duplicates first
    counts: Dict[int, int] = {}
    for i in indices:
        counts[i] = counts.get(i, 0) + 1
    cleaned = [i for i, c in counts.items() if c % 2 == 1]

    # No indices -> scalar (empty blade) with sign +1
    if not cleaned:
        return frozenset(), 1

    # Compute parity of the permutation that sorts the list
    # Use a simple bubble‑sort counting swaps
    sign = 1
    lst = cleaned[:]
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign = -sign
    return frozenset(lst), sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> Tuple[frozenset, int]:
    """Geometric product of two basis blades."""
    combined = list(blade_a) + list(blade_b)
    return _blade_sign(combined)


# ----------------------------------------------------------------------
# Multivector implementation (Cl(n,0))
# ----------------------------------------------------------------------


class Multivector:
    """Element of Cl(n,0) represented as a sparse sum of basis blades."""

    def __init__(self, components: Dict[frozenset, float], n: int):
        self.n = int(n)
        # store only non‑zero components
        self.components: Dict[frozenset, float] = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-12
        }

    # ------------------------------------------------------------------
    # Basic algebraic operations
    # ------------------------------------------------------------------

    def grade(self, k: int) -> "Multivector":
        """Return a new Multivector containing only grade‑k blades."""
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __mul__(self, other):
        if isinstance(other, Multivector):
            result: Dict[frozenset, float] = {}
            for blade_a, coef_a in self.components.items():
                for blade_b, coef_b in other.components.items():
                    blade, sign = _multiply_blades(blade_a, blade_b)
                    result[blade] = result.get(blade, 0.0) + sign * coef_a * coef_b
            return Multivector(result, self.n)
        else:  # scalar multiplication
            return Multivector({k: v * other for k, v in self.components.items()}, self.n)

    __rmul__ = __mul__

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
            if blade:
                basis = "^".join(str(i) for i in sorted(blade))
                terms.append(f"{_pct(coef)}*e{basis}")
            else:
                terms.append(f"{_pct(coef)}")
        return " + ".join(terms) if terms else "0"


# ----------------------------------------------------------------------
# Perceptual hash and clustering utilities
# ----------------------------------------------------------------------


def compute_phash(values: List[float]) -> int:
    """
    64‑bit perceptual hash of a numeric sequence.
    Bit i (from most‑significant) is 1 iff values[i] >= mean(values).
    If fewer than 64 values are supplied the hash is left‑aligned.
    """
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for i in range(64):
        bits <<= 1
        if i < len(values) and values[i] >= avg:
            bits |= 1
    return bits


def phash_to_cluster(phash: int, n_clusters: int) -> int:
    """Map a 64‑bit hash to a cluster identifier."""
    return phash % n_clusters


# ----------------------------------------------------------------------
# RBF surrogate model (one per cluster)
# ----------------------------------------------------------------------


def rbf_kernel(x: np.ndarray, center: np.ndarray, epsilon: float) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-epsilon * np.linalg.norm(x - center) ** 2)


# ----------------------------------------------------------------------
# Bandit (simple epsilon‑greedy) for action selection per cluster
# ----------------------------------------------------------------------


class EpsilonGreedyBandit:
    """Very small epsilon‑greedy bandit for a discrete action set."""

    def __init__(self, n_actions: int, epsilon: float = 0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.counts = [0] * n_actions
        self.values = [0.0] * n_actions

    def select(self) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        # exploit: choose action with highest estimated value
        max_val = max(self.values)
        candidates = [i for i, v in enumerate(self.values) if v == max_val]
        return random.choice(candidates)

    def update(self, action: int, reward: float):
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        # incremental mean update
        self.values[action] = value + (reward - value) / n


# ----------------------------------------------------------------------
# Store state (honeybee dance metaphor)
# ----------------------------------------------------------------------


@dataclass
class StoreState:
    dance: float = 0.0
    # optional: keep track of cumulative reward for diagnostics
    cumulative_reward: float = 0.0


def get_rbf_kernel_width(store: StoreState, epsilon0: float) -> float:
    """Kernel width grows with the current dance intensity."""
    return epsilon0 * (1.0 + store.dance)


# ----------------------------------------------------------------------
# Hybrid system – deeper integration of the two mathematical worlds
# ----------------------------------------------------------------------


class HybridSystem:
    """
    Manages the interaction between:
      * perceptual‑hash clustering,
      * per‑cluster RBF surrogates,
      * a tiny contextual bandit,
      * geometric algebra (Multivector) operations,
      * and the store’s dance dynamics.
    """

    def __init__(
        self,
        n_clusters: int,
        vector_dim: int,
        epsilon0: float = 1.0,
        n_actions: int = 3,
    ):
        self.n_clusters = n_clusters
        self.vector_dim = vector_dim
        self.epsilon0 = epsilon0

        # Initialise a random centre for each cluster’s RBF surrogate
        self.centers: List[np.ndarray] = [
            np.random.rand(vector_dim) for _ in range(n_clusters)
        ]

        # One bandit per cluster
        self.bandits: List[EpsilonGreedyBandit] = [
            EpsilonGreedyBandit(n_actions) for _ in range(n_clusters)
        ]

        # Store state shared across clusters
        self.store = StoreState()

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, values: List[float], x: np.ndarray) -> Multivector:
        """
        Perform a single hybrid iteration:
          1. Compute perceptual hash and obtain cluster id.
          2. Query the cluster’s RBF surrogate (using adaptive kernel width).
          3. Use the surrogate output as reward for the cluster’s bandit.
          4. Update the store’s dance based on the selected action.
          5. Build a multivector whose blade coefficients are driven by the hash bits
             and scaled by the surrogate output.
          6. Return the resulting multivector.
        """
        # 1. Hash → cluster
        phash = compute_phash(values)
        cluster_id = phash_to_cluster(phash, self.n_clusters)

        # 2. Adaptive kernel width & surrogate output
        epsilon_c = get_rbf_kernel_width(self.store, self.epsilon0)
        surrogate = rbf_kernel(x, self.centers[cluster_id], epsilon_c)

        # 3. Bandit update (reward = surrogate)
        bandit = self.bandits[cluster_id]
        action = bandit.select()
        bandit.update(action, surrogate)

        # 4. Dance dynamics – simple linear mapping from action to dance delta
        #    (action 0 → -0.1, 1 → 0.0, 2 → +0.1)
        delta = (action - 1) * 0.1
        self.store.dance = max(0.0, self.store.dance + delta)
        self.store.cumulative_reward += surrogate

        # 5. Construct multivector:
        #    - Use the first `vector_dim` bits of the hash to set coefficients.
        #    - Each bit i corresponds to the basis blade e_i (grade‑1).
        #    - Coefficient = surrogate * (+1 if bit==1 else -1)
        components: Dict[frozenset, float] = {}
        for i in range(self.vector_dim):
            bit = (phash >> (63 - i)) & 1  # most‑significant first
            coeff = surrogate * (1.0 if bit else -1.0)
            blade = frozenset({i})  # grade‑1 blade e_i
            components[blade] = coeff

        # Add a scalar part equal to the surrogate itself (grade‑0)
        components[frozenset()] = surrogate

        mv = Multivector(components, self.vector_dim)

        # 6. Optional geometric product with a fixed reference multivector
        #    (here we use the vector of all ones) to demonstrate deeper coupling.
        reference = Multivector({frozenset({i}): 1.0 for i in range(self.vector_dim)}, self.vector_dim)
        result = mv * reference

        return result

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def get_store_state(self) -> StoreState:
        return self.store

    def get_bandit_estimates(self, cluster_id: int) -> List[float]:
        return self.bandits[cluster_id].values.copy()


# ----------------------------------------------------------------------
# Example usage (executed when run as a script)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    system = HybridSystem(
        n_clusters=8,
        vector_dim=10,
        epsilon0=0.8,
        n_actions=3,
    )

    # Simulate a short trajectory
    for step in range(15):
        vals = [random.random() for _ in range(64)]
        x_vec = np.random.rand(10)
        mv = system.step(vals, x_vec)
        store = system.get_store_state()
        print(
            f"Step {step:02d} | Dance={store.dance:.2f} | "
            f"Scalar={mv.scalar_part():.4f} | Norm={np.linalg.norm(list(mv.components.values())):.4f}"
        )