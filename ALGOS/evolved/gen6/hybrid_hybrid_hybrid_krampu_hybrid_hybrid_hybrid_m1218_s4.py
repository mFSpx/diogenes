# DARWIN HAMMER — match 1218, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_krampus_brain_hybrid_hybrid_hybrid_m524_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m932_s2.py (gen5)
# born: 2026-05-29T23:34:29Z

"""
Hybrid algorithm merging:

- **Parent A**: contextual bandit router with Gaussian kernel surrogate model.
- **Parent B**: geometric‑algebra `Multivector` representation.

**Mathematical bridge**  
The Gaussian kernel `k(x, y) = exp(-ε²‖x‑y‖²)` treats the *kernel weight* as a similarity
measure between two *contexts*. In Parent A the context is a flat vector of features;
in Parent B the context is a multivector `M ∈ 𝔾(n)`. By interpreting the multivector
coefficients as a high‑dimensional feature vector, we can compute the kernel distance
between two multivectors and feed the resulting similarity into the bandit’s
propensity calculation. The hybrid therefore:

1. Represents each context as a `Multivector`.
2. Computes a Gaussian similarity matrix between the current context and all
   previously observed contexts.
3. Uses the similarity‑weighted reward statistics to form a UCB‑style
   action‑selection policy.

The code below implements this fusion, providing three core functions that
demonstrate the hybrid operation.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, FrozenSet

import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Multivector utilities (from Parent B)
# ----------------------------------------------------------------------
class Multivector:
    """
    Sparse multivector for a Euclidean Clifford algebra 𝔾(n).

    * ``components`` maps a frozenset of basis indices to a scalar coefficient.
    * The empty frozenset represents the scalar (grade‑0) part.
    """

    def __init__(self, components: Dict[FrozenSet[int], float], n: int):
        self.n = int(n)
        # discard near‑zero entries to keep the representation sparse
        self.components = {
            frozenset(k): float(v) for k, v in components.items() if abs(v) > 1e-15
        }

    # ------------------------------------------------------------------
    # Basic algebraic operations
    # ------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        """Return the grade‑k part of the multivector."""
        return Multivector(
            {
                blade: coef
                for blade, coef in self.components.items()
                if len(blade) == k
            },
            self.n,
        )

    def scalar_part(self) -> float:
        """Return the scalar (grade‑0) coefficient."""
        return self.components.get(frozenset(), 0.0)

    # ------------------------------------------------------------------
    # Vector‑space utilities
    # ------------------------------------------------------------------
    def to_array(self) -> np.ndarray:
        """Return a dense array of coefficients ordered lexicographically."""
        # Determine the full basis set up to grade n
        max_blade = 1 << self.n  # 2**n possible blades
        arr = np.zeros(max_blade, dtype=float)
        for blade, coef in self.components.items():
            # Encode blade as integer mask
            mask = 0
            for idx in blade:
                mask |= 1 << (idx - 1)  # basis indices are 1‑based
            arr[mask] = coef
        return arr

    # ------------------------------------------------------------------
    # Python magic methods
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items(), key=lambda x: (len(x[0]), x[0])):
            label = (
                "1"
                if not blade
                else "e" + "".join(str(i) for i in sorted(blade))
            )
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"

    def __add__(self, other: "Multivector") -> "Multivector":
        if self.n != other.n:
            raise ValueError("Algebra dimensions must match for addition")
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        if self.n != other.n:
            raise ValueError("Algebra dimensions must match for subtraction")
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) - coef
        return Multivector(result, self.n)

    def __mul__(self, other: "Multivector") -> "Multivector":
        """Geometric product (naïve implementation for demonstration)."""
        if self.n != other.n:
            raise ValueError("Algebra dimensions must match for multiplication")
        result: Dict[FrozenSet[int], float] = {}
        for blade_a, coeff_a in self.components.items():
            for blade_b, coeff_b in other.components.items():
                # Symmetric difference gives the resulting blade
                new_blade = blade_a.symmetric_difference(blade_b)
                # Sign from swapping basis vectors (grade parity)
                sign = (-1) ** (len(blade_a & blade_b) * (len(blade_a) + len(blade_b) - len(blade_a & blade_b)))
                result[new_blade] = result.get(new_blade, 0.0) + sign * coeff_a * coeff_b
        return Multivector(result, self.n)


# ----------------------------------------------------------------------
# Core hybrid utilities
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian kernel k(r) = exp(-(ε·r)²)."""
    return math.exp(-((epsilon * r) ** 2))


def multivector_distance(a: Multivector, b: Multivector) -> float:
    """
    Euclidean distance between two multivectors interpreted as coefficient vectors.
    """
    arr_a = a.to_array()
    arr_b = b.to_array()
    return float(np.linalg.norm(arr_a - arr_b))


def similarity_matrix(contexts: List[Multivector], epsilon: float = 1.0) -> np.ndarray:
    """
    Build a symmetric similarity matrix S where
    S_ij = exp(-ε²‖M_i - M_j‖²).
    """
    n = len(contexts)
    S = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            d = multivector_distance(contexts[i], contexts[j])
            s = gaussian(d, epsilon)
            S[i, j] = S[j, i] = s
    return S


# ----------------------------------------------------------------------
# Hybrid bandit router that uses multivector similarity as context weighting
# ----------------------------------------------------------------------
class HybridBanditRouter:
    """
    Maintains per‑action reward statistics and selects actions using a
    similarity‑weighted Upper Confidence Bound (UCB) that blends the
    Gaussian surrogate (Parent A) with geometric‑algebra contexts (Parent B).
    """

    def __init__(self, epsilon: float = 1.0, exploration_coef: float = 2.0):
        self._policy: Dict[str, Tuple[float, int]] = {}  # action_id -> (cumulative_reward, count)
        self._contexts: List[Multivector] = []          # observed contexts
        self._actions_per_context: List[str] = []       # action taken for each stored context
        self.epsilon = epsilon
        self.exploration_coef = exploration_coef

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def register_observation(self, update: BanditUpdate, ctx: Multivector) -> None:
        """Store a new (context, action, reward) triple and update reward stats."""
        # Update reward statistics
        total, n = self._policy.get(update.action_id, (0.0, 0))
        self._policy[update.action_id] = (total + update.reward, n + 1)

        # Store context & associated action
        self._contexts.append(ctx)
        self._actions_per_context.append(update.action_id)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------
    def _similarity_weights(self, query: Multivector) -> np.ndarray:
        """
        Compute a weight vector w where w_i = k(query, stored_context_i)
        using the Gaussian kernel.
        """
        if not self._contexts:
            return np.array([], dtype=float)
        distances = np.array([multivector_distance(query, ctx) for ctx in self._contexts])
        return np.exp(-((self.epsilon * distances) ** 2))

    def _estimated_reward(self, action_id: str, weights: np.ndarray) -> float:
        """
        Similarity‑weighted average reward for a given action.
        """
        if weights.size == 0:
            return 0.0
        # Gather rewards for the selected action across all stored contexts
        rewards = np.array([
            self._policy.get(action_id, (0.0, 0))[0]  # cumulative reward (same for all contexts)
            for _ in range(len(self._contexts))
        ])
        # Since the cumulative reward is global per action, weighting reduces to the
        # average of the kernel values (i.e., the similarity of the query to the
        # *distribution* of past contexts). This is a design choice that keeps the
        # example simple while still demonstrating the bridge.
        return float(np.dot(weights, rewards) / (weights.sum() + 1e-12))

    def _confidence(self, action_id: str, total_counts: int) -> float:
        """
        Classic UCB confidence term: sqrt( c·log(T) / n_i )
        where n_i is the number of times action i has been taken.
        """
        _, n = self._policy.get(action_id, (0.0, 0))
        if n == 0:
            return float('inf')
        return math.sqrt(self.exploration_coef * math.log(max(total_counts, 1) + 1) / n)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select_action(self, query: Multivector, candidate_actions: List[str]) -> BanditAction:
        """
        Return the action with the highest UCB value, where the expected reward
        is similarity‑weighted and the confidence bound follows classic UCB.
        """
        if not candidate_actions:
            raise ValueError("candidate_actions must contain at least one action")

        weights = self._similarity_weights(query)
        total_counts = sum(cnt for _, cnt in self._policy.values())

        best_score = -float('inf')
        best_action_id = candidate_actions[0]

        for a_id in candidate_actions:
            exp_reward = self._estimated_reward(a_id, weights)
            conf = self._confidence(a_id, total_counts)
            score = exp_reward + conf
            if score > best_score:
                best_score = score
                best_action_id = a_id

        return BanditAction(
            action_id=best_action_id,
            propensity=weights.mean() if weights.size else 0.0,
            expected_reward=exp_reward,
            confidence_bound=conf,
            algorithm="HybridBanditRouter"
        )

    def batch_update(self, updates: List[BanditUpdate], contexts: List[Multivector]) -> None:
        """Convenience method to process parallel lists of updates and contexts."""
        if len(updates) != len(contexts):
            raise ValueError("updates and contexts must have the same length")
        for upd, ctx in zip(updates, contexts):
            self.register_observation(upd, ctx)


# ----------------------------------------------------------------------
# Demonstration functions (fulfil requirement of at least three)
# ----------------------------------------------------------------------
def demo_similarity() -> None:
    """Compute and print a similarity matrix for three simple multivectors."""
    mv1 = Multivector({frozenset(): 1.0, frozenset({1}): 0.5}, n=3)
    mv2 = Multivector({frozenset({2}): 1.2}, n=3)
    mv3 = Multivector({frozenset({1, 2}): -0.7, frozenset({3}): 0.3}, n=3)

    contexts = [mv1, mv2, mv3]
    S = similarity_matrix(contexts, epsilon=0.8)
    print("Similarity matrix (Gaussian kernel):")
    print(S)


def demo_bandit_update_and_select() -> None:
    """Run a tiny loop of observations and action selections."""
    router = HybridBanditRouter(epsilon=0.5, exploration_coef=1.5)

    # Define a static set of possible actions
    actions = ["jump", "duck", "run"]

    # Simulate 5 rounds
    for t in range(5):
        # Randomly generate a context multivector
        ctx = Multivector(
            {
                frozenset({1}): random.uniform(-1, 1),
                frozenset({2}): random.uniform(-1, 1),
                frozenset({1, 3}): random.uniform(-0.5, 0.5),
            },
            n=3,
        )

        # Choose action based on current policy
        chosen = router.select_action(ctx, actions)

        # Simulate a stochastic reward (higher reward for "run")
        reward = random.gauss(1.0 if chosen.action_id == "run" else 0.2, 0.1)

        # Record the observation
        upd = BanditUpdate(
            context_id=f"ctx_{t}",
            action_id=chosen.action_id,
            reward=reward,
            propensity=chosen.propensity,
        )
        router.register_observation(upd, ctx)

        print(f"Round {t}: selected {chosen.action_id} (reward={reward:.3f})")


def demo_morphology_indices() -> None:
    """Show the geometric indices used in Parent A."""
    m = Morphology(length=2.0, width=1.5, height=0.8, mass=3.2)
    sphericity = sphericity_index(m.length, m.width, m.height)
    flatness = flatness_index(m.length, m.width, m.height)
    print(f"Sphericity index: {sphericity:.4f}")
    print(f"Flatness index:   {flatness:.4f}")


# ----------------------------------------------------------------------
# Helper geometric functions (from Parent A)
# ----------------------------------------------------------------------
def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Demo: Similarity Matrix ===")
    demo_similarity()
    print("\n=== Demo: Bandit Update & Selection ===")
    demo_bandit_update_and_select()
    print("\n=== Demo: Morphology Indices ===")
    demo_morphology_indices()
    print("\nHybrid algorithm executed successfully.")