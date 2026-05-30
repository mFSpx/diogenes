# DARWIN HAMMER — match 5670, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m2306_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m1347_s1.py (gen5)
# born: 2026-05-30T00:04:14Z

import math
import random
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Vector = Sequence[float]


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean distance between two equal‑length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gaussian(r: float, sigma: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-0.5 * (r / sigma) ** 2)


# ----------------------------------------------------------------------
# Count‑Min Sketch with independent hash functions
# ----------------------------------------------------------------------
def _hash_with_seed(value: int, seed: int, width: int) -> int:
    """Deterministic integer hash using SHA‑256 and a seed."""
    h = hashlib.sha256()
    h.update(seed.to_bytes(4, "little"))
    h.update(value.to_bytes(8, "little", signed=True))
    return int.from_bytes(h.digest()[:8], "little") % width


def count_min_sketch(
    rewards: Iterable[int], width: int, depth: int, seeds: Tuple[int, ...] | None = None
) -> np.ndarray:
    """
    Build a Count‑Min sketch of the reward stream.

    Parameters
    ----------
    rewards : iterable of int
        Observed reward values.
    width : int
        Number of columns (hash space size).
    depth : int
        Number of rows (independent hash functions).
    seeds : tuple[int, ...] | None
        Optional seeds for the hash functions; if omitted they are generated
        randomly.

    Returns
    -------
    np.ndarray
        A ``depth × width`` integer matrix.
    """
    if width <= 0 or depth <= 0:
        raise ValueError("width and depth must be positive")
    if seeds is None:
        seeds = tuple(random.randint(0, 2 ** 31 - 1) for _ in range(depth))
    elif len(seeds) != depth:
        raise ValueError("len(seeds) must equal depth")

    sketch = np.zeros((depth, width), dtype=int)
    for reward in rewards:
        for row, seed in enumerate(seeds):
            col = _hash_with_seed(int(reward), seed, width)
            sketch[row, col] += 1
    return sketch


def sketch_estimate_mean(sketch: np.ndarray) -> float:
    """Unbiased estimator of the mean count per cell."""
    return float(sketch.mean())


def sketch_estimate_variance(sketch: np.ndarray) -> float:
    """Unbiased estimator of the variance of the cell counts."""
    return float(sketch.var(ddof=0))


# ----------------------------------------------------------------------
# NLMS (Normalized Least Mean Squares) utilities
# ----------------------------------------------------------------------
def nlms_predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction w·x."""
    return float(weights @ x)


def nlms_update(
    weights: np.ndarray,
    x: np.ndarray,
    target: float,
    mu: float = 0.5,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, float]:
    """
    One step of the Normalized LMS algorithm.

    Returns the updated weight vector and the instantaneous error.
    """
    if x.ndim != 1:
        raise ValueError("input vector x must be one‑dimensional")
    error = target - nlms_predict(weights, x)
    norm = eps + float(x @ x)
    new_weights = weights + (mu * error / norm) * x
    return new_weights, error


# ----------------------------------------------------------------------
# Core hybrid system
# ----------------------------------------------------------------------
@dataclass
class HybridPheromoneBanditRBFSystem:
    """
    A deeper fusion of a semantic‑neighbor RBF layer, a pheromone‑biased
    Count‑Min sketch, and an NLMS adaptive linear predictor.

    The semantic similarity values are interpreted as *inverse* distances
    (higher similarity → smaller effective distance) and fed to a Gaussian
    RBF.  Each neighbor maintains its own sketch, allowing the pheromone
    signal to be neighbour‑specific.
    """

    semantic_neighbors: List[Tuple[str, float]]
    prior: float
    mu: float = 0.5
    sigma: float = 1.0
    sketch_width: int = 128
    sketch_depth: int = 4
    seeds: Tuple[int, ...] = field(default_factory=lambda: tuple(random.randint(0, 2**31-1) for _ in range(4)))

    def __post_init__(self) -> None:
        if not 0.0 <= self.prior <= 1.0:
            raise ValueError("prior must be in [0, 1]")
        if self.mu <= 0 or self.mu > 1:
            raise ValueError("mu must be in (0, 1]")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        self.n_neighbors = len(self.semantic_neighbors)
        if self.n_neighbors == 0:
            raise ValueError("at least one semantic neighbor required")
        # Initialise NLMS weights uniformly
        self.weights = np.full(self.n_neighbors, 1.0 / self.n_neighbors, dtype=float)
        # One sketch per neighbor
        self.sketches: List[np.ndarray] = [
            np.zeros((self.sketch_depth, self.sketch_width), dtype=int)
            for _ in range(self.n_neighbors)
        ]

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _rbf_vector(self) -> np.ndarray:
        """
        Convert semantic similarities to RBF activations.

        Similarity s∈[0,1] is first turned into a distance d = 1‑s,
        then passed through a Gaussian RBF.
        """
        distances = np.array([1.0 - sim for _, sim in self.semantic_neighbors], dtype=float)
        return np.vectorize(lambda d: gaussian(d, self.sigma))(distances)

    def _update_sketches(self, rewards: Iterable[int]) -> None:
        """
        Update each neighbor's sketch with the same reward stream.
        In a richer setting each neighbor could receive a different
        stream; here we keep it simple but still maintain per‑neighbor
        statistics.
        """
        for idx in range(self.n_neighbors):
            self.sketches[idx] = count_min_sketch(
                rewards,
                self.sketch_width,
                self.sketch_depth,
                seeds=self.seeds,
            )

    def _pheromone_vector(self) -> np.ndarray:
        """
        Convert sketch statistics to pheromone signals.

        The mean count of each sketch is interpreted as a raw pheromone
        intensity; a Gaussian RBF maps it to a bounded signal in (0,1].
        """
        means = np.array(
            [sketch_estimate_mean(sk) for sk in self.sketches], dtype=float
        )
        # Normalise by the maximum observed mean to keep values in a sensible range
        max_mean = max(1.0, means.max())
        normalized = means / max_mean
        return np.vectorize(lambda m: gaussian(m, self.sigma))(normalized)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, target: float, rewards: Iterable[int]) -> None:
        """
        Perform a single learning step.

        Parameters
        ----------
        target : float
            Desired output of the linear predictor.
        rewards : iterable of int
            Observed reward values (e.g., click counts, click‑through rates).
        """
        # 1️⃣ Update per‑neighbor sketches
        self._update_sketches(rewards)

        # 2️⃣ Build the combined input vector x = RBF ⊙ pheromone
        rbf = self._rbf_vector()
        pheromone = self._pheromone_vector()
        x = rbf * pheromone

        # 3️⃣ NLMS weight adaptation
        self.weights, _ = nlms_update(self.weights, x, target, self.mu)

    def predict(self) -> float:
        """
        Produce a prediction for the current state.

        The pheromone component is set to its maximal value (i.e. 1.0)
        because during inference we have no fresh reward signal.
        """
        rbf = self._rbf_vector()
        x = rbf  # pheromone = 1 for all neighbors
        return nlms_predict(self.weights, x)

    # ------------------------------------------------------------------
    # Optional Bayesian post‑processing (deeper integration)
    # ------------------------------------------------------------------
    def bayesian_adjustment(self, likelihood: float, false_positive: float) -> float:
        """
        Adjust the current prior using a single Bayesian update.

        This method demonstrates a tighter coupling between the probabilistic
        reasoning of the original parent algorithms and the adaptive linear
        predictor.
        """
        marginal = likelihood * self.prior + false_positive * (1.0 - self.prior)
        if marginal <= 0.0:
            raise ValueError("Marginal probability must be > 0")
        posterior = self.prior * likelihood / marginal
        self.prior = posterior
        return posterior


# ----------------------------------------------------------------------
# Simple demonstration (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example semantic neighbors: (identifier, similarity∈[0,1])
    neighbors = [("topic_A", 0.78), ("topic_B", 0.45), ("topic_C", 0.62)]

    system = HybridPheromoneBanditRBFSystem(
        semantic_neighbors=neighbors,
        prior=0.5,
        mu=0.3,
        sigma=0.4,
        sketch_width=64,
        sketch_depth=3,
    )

    # Simulated reward stream (e.g., user clicks)
    reward_stream = [1, 0, 2, 1, 3, 0, 1]

    # Perform several online updates
    for epoch in range(5):
        target_value = random.uniform(0.0, 1.0)  # synthetic target
        system.update(target=target_value, rewards=reward_stream)
        print(f"Epoch {epoch+1:02d} – prediction: {system.predict():.4f} – prior: {system.prior:.4f}")

    # Demonstrate Bayesian adjustment
    system.bayesian_adjustment(likelihood=0.7, false_positive=0.2)
    print(f"Adjusted prior after Bayesian update: {system.prior:.4f}")