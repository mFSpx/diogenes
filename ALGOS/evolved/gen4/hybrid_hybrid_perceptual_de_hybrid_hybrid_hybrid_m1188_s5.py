# DARWIN HAMMER — match 1188, survivor 5
# gen: 4
# parent_a: hybrid_perceptual_dedupe_hybrid_hybrid_rbf_su_m10_s3.py (gen3)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_ternar_m374_s4.py (gen3)
# born: 2026-05-29T23:33:26Z

import math
import time
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

Vector = Sequence[float]


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian radial basis function."""
    return math.exp(-((epsilon * r) ** 2))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_phash(values: List[float]) -> int:
    """Simple locality‑sensitive hash based on median threshold."""
    if not values:
        return 0
    median = np.median(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= median)
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integer bit‑patterns."""
    return (a ^ b).bit_count()


@dataclass
class PheromoneSignal:
    """Container for a single pheromone signal."""
    created: float
    value: float
    half_life: float

    def decayed(self, now: float) -> float:
        """Return the exponentially decayed value at time ``now``."""
        if self.half_life <= 0:
            raise ValueError("half_life must be positive")
        elapsed = now - self.created
        decay_factor = 0.5 ** (elapsed / self.half_life)
        return self.value * decay_factor


class HybridPheromoneRBFSystem:
    """
    Deep integration of a pheromone‑based decay model with a radial‑basis
    surrogate model for multi‑armed bandit decision making.

    * Pheromones decay according to a configurable half‑life.
    * Decayed pheromone values are used as *priors* that weight the
      contribution of each RBF centre, effectively biasing the surrogate
      towards regions recently reinforced by pheromones.
    * Bandit statistics are updated with the classic UCB1 confidence bound,
      but the expected reward term is replaced by the hybrid surrogate
      ``hybrid_score``.
    """

    def __init__(self, n_arms: int = 5, n_rbf: int = 10):
        self.n_arms = n_arms
        self.n_rbf = n_rbf

        # surface_key → signal_kind → PheromoneSignal
        self.pheromones: Dict[str, Dict[str, PheromoneSignal]] = {}

        # Randomly initialise RBF centres (each centre lives in the arm‑space)
        self.centers: np.ndarray = np.random.rand(n_rbf, n_arms)
        # Widths act as the epsilon parameter for each RBF
        self.widths: np.ndarray = np.full(n_rbf, 1.0)

        # Bandit statistics
        self.counts: np.ndarray = np.zeros(n_arms, dtype=int)   # pulls per arm
        self.values: np.ndarray = np.zeros(n_arms, dtype=float) # average reward per arm

    # --------------------------------------------------------------------- #
    # Time handling
    # --------------------------------------------------------------------- #
    @staticmethod
    def _now() -> float:
        """Current UTC timestamp in seconds."""
        return time.time()

    # --------------------------------------------------------------------- #
    # Pheromone handling
    # --------------------------------------------------------------------- #
    def update_pheromone(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        """
        Insert or update a pheromone signal for a given ``surface_key`` and
        ``signal_kind``. Returns the *current* decayed value.
        """
        now = self._now()
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {}
        self.pheromones[surface_key][signal_kind] = PheromoneSignal(
            created=now, value=signal_value, half_life=half_life_seconds
        )
        return self.pheromones[surface_key][signal_kind].decayed(now)

    def _aggregate_pheromone_prior(self, arm_vector: Vector) -> float:
        """
        Compute the pheromone prior for a given arm representation.
        Each pheromone contributes its decayed value multiplied by the
        RBF response of the arm vector.
        """
        now = self._now()
        prior = 0.0
        for surface_dict in self.pheromones.values():
            for signal in surface_dict.values():
                decayed = signal.decayed(now)
                prior += decayed * self._rbf_response(arm_vector)
        return prior

    # --------------------------------------------------------------------- #
    # Radial‑basis surrogate
    # --------------------------------------------------------------------- #
    def _rbf_response(self, x: Vector) -> float:
        """
        Evaluate the RBF surrogate at point ``x``.
        ``x`` must have dimensionality ``n_arms``.
        """
        if len(x) != self.n_arms:
            raise ValueError(f"Input vector must have length {self.n_arms}")
        # Vectorised distance computation for speed
        diffs = self.centers - np.asarray(x)
        dists = np.linalg.norm(diffs, axis=1)
        responses = np.vectorize(gaussian)(dists, self.widths)
        return float(responses.sum())

    # --------------------------------------------------------------------- #
    # Hybrid scoring & bandit policy
    # --------------------------------------------------------------------- #
    def hybrid_score(self, arm_index: int) -> float:
        """
        Hybrid expected reward for ``arm_index``.
        Combines the empirical average reward with a pheromone‑biased RBF prior.
        """
        if not (0 <= arm_index < self.n_arms):
            raise IndexError("arm_index out of range")
        arm_vector = np.zeros(self.n_arms)
        arm_vector[arm_index] = 1.0  # one‑hot representation of the arm
        pheromone_prior = self._aggregate_pheromone_prior(arm_vector)
        return self.values[arm_index] + pheromone_prior

    def select_arm(self, confidence: float = 2.0) -> int:
        """
        UCB1‑style selection where the exploitation term is ``hybrid_score``.
        ``confidence`` controls the exploration weight (higher → more exploration).
        """
        total_counts = self.counts.sum()
        if total_counts == 0:
            # No pulls yet – pick uniformly at random
            return random.randrange(self.n_arms)

        ucb_values = np.empty(self.n_arms)
        for a in range(self.n_arms):
            exploitation = self.hybrid_score(a)
            exploration = confidence * math.sqrt(
                math.log(total_counts) / (self.counts[a] + 1e-9)
            )
            ucb_values[a] = exploitation + exploration
        return int(np.argmax(ucb_values))

    def update_reward(self, arm_index: int, reward: float) -> None:
        """
        Incremental update of empirical statistics for ``arm_index``.
        """
        if not (0 <= arm_index < self.n_arms):
            raise IndexError("arm_index out of range")
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        # Incremental mean update
        self.values[arm_index] += (reward - self.values[arm_index]) / n

    # --------------------------------------------------------------------- #
    # Utility functions (unchanged but type‑annotated)
    # --------------------------------------------------------------------- #
    @staticmethod
    def cluster_by_phash(hashes: Dict[str, int], max_distance: int = 4) -> List[List[str]]:
        """
        Simple agglomerative clustering based on Hamming distance of phashes.
        """
        clusters: List[List[str]] = []
        for key, h in hashes.items():
            placed = False
            for cluster in clusters:
                if hamming_distance(h, hashes[cluster[0]]) <= max_distance:
                    cluster.append(key)
                    placed = True
                    break
            if not placed:
                clusters.append([key])
        return clusters

    @staticmethod
    def solve_linear(a: List[List[float]], b: List[float]) -> List[float]:
        """
        Solve a dense linear system ``Ax = b`` using Gaussian elimination
        with partial pivoting.
        """
        n = len(b)
        # Build augmented matrix
        m = [row[:] + [rhs] for row, rhs in zip(a, b)]

        for col in range(n):
            # Partial pivot
            pivot = max(range(col, n), key=lambda i: abs(m[i][col]))
            if abs(m[pivot][col]) < 1e-12:
                raise ValueError("Matrix is singular or near‑singular")
            m[col], m[pivot] = m[pivot], m[col]

            # Normalise pivot row
            pivot_val = m[col][col]
            m[col] = [x / pivot_val for x in m[col]]

            # Eliminate other rows
            for i in range(n):
                if i == col:
                    continue
                factor = m[i][col]
                m[i] = [mi - factor * mc for mi, mc in zip(m[i], m[col])]

        return [row[-1] for row in m]


if __name__ == "__main__":
    # Demonstration of the improved system
    system = HybridPheromoneRBFSystem(n_arms=3, n_rbf=6)

    # Simulate a few pheromone updates
    system.update_pheromone("surfaceA", "typeX", 1.2, half_life_seconds=30.0)
    system.update_pheromone("surfaceB", "typeY", 0.8, half_life_seconds=45.0)

    # Perform a few bandit rounds
    for _ in range(20):
        arm = system.select_arm(confidence=1.5)
        # Mock reward (e.g., sinusoidal with noise)
        reward = math.sin(arm) + random.gauss(0, 0.1)
        system.update_reward(arm, reward)

    # Show hybrid scores
    for a in range(system.n_arms):
        print(f"Arm {a}: hybrid_score = {system.hybrid_score(a):.4f}")

    # Phash clustering demo
    hashes = {
        "a": compute_phash([1.0, 2.0, 3.0]),
        "b": compute_phash([4.0, 5.0, 6.0]),
        "c": compute_phash([1.1, 2.1, 3.1]),
    }
    print("Clusters:", system.cluster_by_phash(hashes))

    # Linear solver demo
    A = [[2, 1], [5, 7]]
    b_vec = [11, 13]
    print("Solution:", system.solve_linear(A, b_vec))