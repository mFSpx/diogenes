# DARWIN HAMMER — match 4358, survivor 6
# gen: 6
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m2530_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m1347_s2.py (gen5)
# born: 2026-05-29T23:55:08Z

"""Hybrid Hypervector-Pheromone Bandit System
Parents:
- hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m2530_s2.py (hyper‑vector binding/unbinding,
  morphology feature graph and leader election)
- hybrid_hybrid_hybrid_percep_hybrid_hybrid_hybrid_m1347_s2.py (pheromone‑biased multi‑armed
  bandit with radial‑basis function surrogate and Count‑Min sketch)

Mathematical Bridge:
Both parents operate on high‑dimensional representations.  The first maps morphology
features to hyper‑vectors and uses Euclidean distances between bound vectors to weight
graph edges.  The second models expected rewards with Gaussian radial‑basis functions
(RBF) whose inputs are distances in a feature space and biases arm selection with a
pheromone field.  The bridge is therefore the *distance metric* on bound hyper‑vectors:
we bind each arm’s hyper‑vector with a context hyper‑vector, compute Euclidean distances
to a set of RBF centre hyper‑vectors, obtain a surrogate reward estimate via the Gaussian
kernel, and finally weight that estimate with a pheromone level before applying a
UCB‑style exploration term.  The Count‑Min sketch from the second parent provides a
lightweight estimate of how many times each arm has been sampled, supplying the
exploration denominator.
"""

import math
import random
import sys
import pathlib
from typing import Optional, Sequence, List, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Hyper‑vector utilities (from Parent A)
# ----------------------------------------------------------------------
def random_hv(d: int, kind: str = "real", seed: Optional[int] = None) -> np.ndarray:
    """Deterministic random hyper‑vector."""
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        hv = np.exp(1j * theta)
    elif kind == "bipolar":
        hv = rng.choice(np.array([-1.0, 1.0]), size=d)
    elif kind == "real":
        hv = rng.standard_normal(d)
        hv /= np.linalg.norm(hv) + 1e-30
    else:
        raise ValueError(f"Unsupported kind {kind!r}")
    return hv

def bind(x: np.ndarray, hv: np.ndarray) -> np.ndarray:
    """Circular convolution (binding) via FFT."""
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(hv))

def unbind(z: np.ndarray, hv: np.ndarray) -> np.ndarray:
    """Circular correlation (unbinding) via FFT."""
    return np.fft.ifft(np.fft.fft(z) * np.conj(np.fft.fft(hv)))

# ----------------------------------------------------------------------
# Radial‑basis and pheromone utilities (from Parent B)
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Gaussian RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))

def count_min_sketch(rewards: Sequence[int], width: int, depth: int) -> np.ndarray:
    """Simple Count‑Min sketch for integer rewards."""
    sketch = np.zeros((depth, width), dtype=int)
    for reward in rewards:
        for i in range(depth):
            hash_value = hash((i, reward)) % width
            sketch[i, hash_value] += 1
    return sketch

def estimate_counts(sketch: np.ndarray) -> float:
    """Mean count across the sketch – proxy for total pulls."""
    return float(np.mean(sketch))

# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def rbf_estimate(
    bound_vec: np.ndarray,
    centres: List[np.ndarray],
    epsilon: float = 1.0,
) -> float:
    """
    Compute the surrogate reward estimate for a bound vector.
    The estimate is the average Gaussian RBF value over all centres.
    """
    if not centres:
        return 0.0
    rbf_vals = [gaussian(euclidean(bound_vec, c), epsilon) for c in centres]
    return sum(rbf_vals) / len(rbf_vals)


def update_pheromone(
    pheromones: np.ndarray,
    arm_idx: int,
    reward: float,
    decay: float = 0.9,
    boost: float = 1.0,
) -> None:
    """
    Exponential decay of all pheromones and reinforcement of the selected arm.
    """
    pheromones *= decay
    pheromones[arm_idx] += boost * reward


def select_arm_ucb(
    estimates: np.ndarray,
    pheromones: np.ndarray,
    pulls_sketch: np.ndarray,
    total_pulls: float,
    c: float = 2.0,
) -> int:
    """
    UCB‑style selection where the exploitation term is the pheromone‑weighted
    RBF estimate and the exploration term uses the Count‑Min sketch estimate of
    arm pulls.
    """
    # Estimate pulls per arm from the sketch (mean count as proxy)
    arm_pull_est = estimate_counts(pulls_sketch)
    exploration = c * math.sqrt(math.log(max(total_pulls, 1.0)) / max(arm_pull_est, 1e-6))
    scores = pheromones * estimates + exploration
    return int(np.argmax(scores))


# ----------------------------------------------------------------------
# Hybrid system class
# ----------------------------------------------------------------------
class HybridHyperbanditSystem:
    """
    Unified system that:
    1. Represents each arm as a hyper‑vector.
    2. Binds the arm vector with a context hyper‑vector.
    3. Evaluates a surrogate reward via Gaussian RBFs on Euclidean distances.
    4. Biases the surrogate with a pheromone field.
    5. Chooses arms using a UCB rule whose exploration term is derived from a
       Count‑Min sketch of past rewards.
    """

    def __init__(
        self,
        n_arms: int = 5,
        dim: int = 256,
        n_rbf: int = 12,
        epsilon: float = 0.5,
        decay: float = 0.95,
        width: int = 20,
        depth: int = 5,
        seed: Optional[int] = None,
    ):
        self.n_arms = n_arms
        self.dim = dim
        self.epsilon = epsilon
        self.decay = decay
        self.width = width
        self.depth = depth
        self.rng = np.random.default_rng(seed)

        # Hyper‑vectors for arms and RBF centres
        self.arm_hvs = [random_hv(dim, seed=self.rng.integers(1_000_000)) for _ in range(n_arms)]
        self.rbf_centres = [random_hv(dim, seed=self.rng.integers(1_000_000)) for _ in range(n_rbf)]

        # Pheromone levels (initially uniform)
        self.pheromones = np.ones(n_arms, dtype=float)

        # Count‑Min sketch for reward history (starts empty)
        self.reward_sketch = np.zeros((depth, width), dtype=int)

        # Pull counters for UCB denominator
        self.total_pulls = 0.0
        self.arm_pull_counts = np.zeros(n_arms, dtype=int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def propose_context(self) -> np.ndarray:
        """Generate a fresh random context hyper‑vector."""
        return random_hv(self.dim, seed=self.rng.integers(1_000_000))

    def evaluate_estimates(self, context_hv: np.ndarray) -> np.ndarray:
        """
        For each arm, bind with the context, compute the RBF surrogate,
        and return the vector of estimates.
        """
        estimates = np.empty(self.n_arms, dtype=float)
        for i, arm_hv in enumerate(self.arm_hvs):
            bound = bind(arm_hv, context_hv)
            estimates[i] = rbf_estimate(bound.real, self.rbf_centres, self.epsilon)
        return estimates

    def choose_arm(self, context_hv: np.ndarray) -> int:
        """Select an arm using the hybrid UCB rule."""
        estimates = self.evaluate_estimates(context_hv)
        arm = select_arm_ucb(
            estimates,
            self.pheromones,
            self.reward_sketch,
            self.total_pulls,
        )
        return arm

    def receive_reward(self, arm_idx: int, reward: float, context_hv: np.ndarray) -> None:
        """Update internal state after observing a reward."""
        # Update pheromones
        update_pheromone(self.pheromones, arm_idx, reward, self.decay)

        # Update Count‑Min sketch with discretised reward (int)
        reward_int = int(round(reward * 100))  # scale to preserve precision
        self.reward_sketch = count_min_sketch(
            list(self.reward_sketch.ravel()) + [reward_int],
            self.width,
            self.depth,
        )

        # Update pull statistics
        self.total_pulls += 1.0
        self.arm_pull_counts[arm_idx] += 1

    # ------------------------------------------------------------------
    # Demonstration helpers (three required functions)
    # ------------------------------------------------------------------
    def step(self) -> Tuple[int, float]:
        """
        Perform a full interaction step:
        1. Sample a context.
        2. Choose an arm.
        3. Simulate a stochastic reward (here: noisy RBF estimate).
        4. Update the system.
        Returns the chosen arm index and the observed reward.
        """
        ctx = self.propose_context()
        arm = self.choose_arm(ctx)
        # Simulated reward: true RBF value plus Gaussian noise
        true_val = self.evaluate_estimates(ctx)[arm]
        noise = self.rng.normal(0.0, 0.05)
        reward = max(0.0, min(1.0, true_val + noise))
        self.receive_reward(arm, reward, ctx)
        return arm, reward

    def get_state_snapshot(self) -> dict:
        """Return a dictionary summarising the internal state (for debugging)."""
        return {
            "pheromones": self.pheromones.copy(),
            "total_pulls": self.total_pulls,
            "arm_pull_counts": self.arm_pull_counts.copy(),
            "reward_sketch_mean": float(np.mean(self.reward_sketch)),
        }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    system = HybridHyperbanditSystem(n_arms=4, dim=128, seed=42)
    for t in range(15):
        arm, rew = system.step()
        print(f"Step {t+1:2d}: arm={arm}, reward={rew:.3f}")
    print("Final state snapshot:", system.get_state_snapshot())