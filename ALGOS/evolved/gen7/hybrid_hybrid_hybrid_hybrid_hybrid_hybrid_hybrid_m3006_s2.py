# DARWIN HAMMER — match 3006, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_model__hybrid_hybrid_hybrid_m1206_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_percep_hybrid_hybrid_vorono_m2504_s0.py (gen6)
# born: 2026-05-29T23:47:11Z

"""Hybrid Model Management with SSIM‑Guided RBF Bandit

This module fuses two parent algorithms:

* **Parent A** – uses Structural Similarity Index (SSIM) to compare GPU memory
  signals with periodic signals and drives model‑pool load/eviction decisions.
* **Parent B** – builds a radial‑basis‑function (RBF) surrogate for reward
  estimation and clusters the RBF centres using a Voronoi partition, applying
  a bandit policy for selection.

**Mathematical bridge**

The SSIM value computed for a candidate model is treated as a scalar *reward*
signal.  That reward is modelled by the RBF surrogate of the bandit policy:
for a set of RBF centres \(c_i\) the predicted reward is
\[
\hat r = \sum_i w_i \exp\!\bigl(-(\epsilon \|s - c_i\|)^2\bigr),
\]
where \(s\) is the SSIM score and \(\epsilon\) controls the kernel width.
The Voronoi seeds are the same RBF centres, so assigning a model to the nearest
seed (a Voronoi cell) provides the index of the bandit arm to pull.  The
resulting predicted reward guides the model‑pool’s load/eviction policy.

The implementation below integrates the SSIM computation, the RBF‑based
bandit, and a model‑pool that respects RAM ceilings and tier exclusivity.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import defaultdict

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[2] if pathlib.Path(__file__).exists() else pathlib.Path.cwd()

def ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Simplified SSIM for 1‑D signals."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(numerator / denominator)

# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """RBF kernel."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("vectors must have same shape")
    return float(np.sqrt(np.sum((a - b) ** 2)))

def distance(a: tuple, b: tuple) -> float:
    """2‑D Euclidean distance for Voronoi seeds."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: tuple, seeds: list) -> int:
    """Return index of the nearest seed (Voronoi cell)."""
    if not seeds:
        raise ValueError("seeds required")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
class ModelTier:
    """Simple descriptor for a model."""
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier  # e.g. "T1", "T2", "T3"

class ModelPool:
    """Manages loaded models under RAM ceiling and tier exclusivity."""
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: dict[str, ModelTier] = {}
        self.sensitive_records = []

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        """Evict arbitrary models until there is space, then load."""
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            evicted_name = next(iter(self.loaded))
            del self.loaded[evicted_name]
        self.load(model)

# ----------------------------------------------------------------------
# Sheaf (kept for completeness – not used directly in the hybrid logic)
# ----------------------------------------------------------------------
class Sheaf:
    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = dict(node_dims)
        self.edges = list(edges)
        self._restrictions = {}
        self._sections = {}

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError
        self._restrictions[edge] = (src_map, dst_map)

# ----------------------------------------------------------------------
# Bandit policy driven by RBF surrogate
# ----------------------------------------------------------------------
class BanditPolicy:
    """
    A simple contextual bandit where each arm corresponds to a Voronoi seed /
    RBF centre.  The policy maintains average observed reward per arm and uses
    an epsilon‑greedy selection strategy.
    """
    def __init__(self, rbf_centers: np.ndarray, epsilon: float = 1.0, explore_prob: float = 0.1):
        self.centers = rbf_centers          # shape (n_arms, dim)
        self.epsilon = epsilon
        self.explore_prob = explore_prob
        self.counts = np.zeros(len(rbf_centers), dtype=int)
        self.values = np.zeros(len(rbf_centers), dtype=float)

    def predict(self, context: np.ndarray) -> np.ndarray:
        """
        Predict reward for each arm using RBF kernel mixture.
        context is a 1‑D array (e.g., SSIM score reshaped to (1,)).
        """
        diffs = self.centers - context   # broadcasting
        dists = np.linalg.norm(diffs, axis=1)
        kernels = np.vectorize(gaussian)(dists, self.epsilon)
        # Weighted sum where weights are current value estimates (avoid zero division)
        weight_sum = np.sum(kernels * self.values)
        norm = np.sum(kernels) + 1e-12
        return weight_sum / norm + np.random.randn() * 0.01  # tiny noise

    def select_arm(self, context: np.ndarray) -> int:
        """Epsilon‑greedy arm selection."""
        if random.random() < self.explore_prob:
            return random.randrange(len(self.centers))
        # Exploit: choose arm whose centre is nearest to the context (Voronoi)
        return nearest(tuple(context.tolist()), [tuple(c.tolist()) for c in self.centers])

    def update(self, arm_idx: int, reward: float) -> None:
        """Incremental average update."""
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        self.values[arm_idx] += (reward - self.values[arm_idx]) / n

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def compute_similarity_reward(mem_signal: np.ndarray, periodic_signal: np.ndarray,
                              rbf_centers: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Compute SSIM between two signals and map the scalar similarity to a reward
    using the RBF surrogate defined by ``rbf_centers``.
    """
    s = ssim(mem_signal, periodic_signal)          # scalar in [-1, 1]
    # Normalise to [0, 1] for stability
    s_norm = (s + 1) / 2.0
    context = np.array([s_norm])
    diffs = rbf_centers - context
    dists = np.linalg.norm(diffs, axis=1)
    kernels = np.vectorize(gaussian)(dists, epsilon)
    reward = float(np.sum(kernels) / (np.sum(kernels) + 1e-12))
    return reward

def hybrid_load_decision(pool: ModelPool, candidates: list[ModelTier],
                         periodic_signal: np.ndarray,
                         bandit: BanditPolicy) -> None:
    """
    For each candidate model:
      1. Simulate a GPU memory signal (random for demo).
      2. Compute SSIM‑derived reward.
      3. Use the bandit to select an arm (Voronoi cell) based on the reward.
      4. Update the bandit with the observed reward.
      5. Attempt to load the model; if insufficient RAM, evict least‑recent.
    """
    for model in candidates:
        # Simulated memory trace for this model (length 256, values 0‑255)
        mem_signal = np.random.randint(0, 256, size=256).astype(float)
        reward = compute_similarity_reward(mem_signal, periodic_signal,
                                            bandit.centers, bandit.epsilon)

        # Context for bandit is the normalised SSIM score (1‑D)
        context = np.array([(ssim(mem_signal, periodic_signal) + 1) / 2.0])
        arm = bandit.select_arm(context)
        bandit.update(arm, reward)

        # Try to load; on failure, evict until possible
        try:
            pool.load(model)
        except RuntimeError:
            pool.load_with_eviction(model)

def generate_periodic_signal(length: int = 256) -> np.ndarray:
    """Create a simple sinusoidal periodic signal."""
    t = np.linspace(0, 2 * math.pi, length)
    return 127.5 * (1 + np.sin(t))  # scaled to [0,255]

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise model pool
    pool = ModelPool(ram_ceiling_mb=2000)

    # Create a handful of candidate models with varying RAM and tiers
    candidates = [
        ModelTier("model_A", ram_mb=400, tier="T1"),
        ModelTier("model_B", ram_mb=800, tier="T2"),
        ModelTier("model_C", ram_mb=600, tier="T3"),
        ModelTier("model_D", ram_mb=300, tier="T1"),
        ModelTier("model_E", ram_mb=500, tier="T2"),
    ]

    # Periodic reference signal
    periodic = generate_periodic_signal()

    # Initialise bandit with 3 random RBF centres (1‑D)
    centres = np.random.rand(3, 1)  # values in [0,1] matching normalised SSIM
    bandit = BanditPolicy(rbf_centers=centres, epsilon=2.0, explore_prob=0.2)

    # Run hybrid loading decisions
    hybrid_load_decision(pool, candidates, periodic, bandit)

    # Report loaded models
    print("Loaded models after hybrid decision:")
    for name, mdl in pool.loaded.items():
        print(f" - {name}: {mdl.ram_mb} MB, tier {mdl.tier}")

    # Show bandit statistics
    print("\nBandit arm statistics:")
    for i, (cnt, val) in enumerate(zip(bandit.counts, bandit.values)):
        print(f" Arm {i}: pulls={cnt}, avg_reward={val:.4f}")