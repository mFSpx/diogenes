# DARWIN HAMMER — match 1529, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_voronoi_parti_hybrid_hybrid_hybrid_m104_s6.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m1120_s2.py (gen5)
# born: 2026-05-29T23:37:15Z

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Basic geometric types
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Blade = frozenset[int]          # basis blade represented by a set of indices
Multivector = Dict[Blade, float]  # mapping blade → coefficient

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    """Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Voronoi helpers
# ----------------------------------------------------------------------
def compute_voronoi_regions(points: List[Point],
                            sites: List[Point]) -> Dict[int, List[Point]]:
    """
    Assign each point to the nearest site.
    Returns a dict ``site_index → list[points]``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions


def build_adjacency_from_regions(regions: Dict[int, List[Point]],
                                 sites: List[Point],
                                 radius_factor: float = 1.5) -> np.ndarray:
    """
    Construct a simple weighted adjacency matrix for the Voronoi graph.
    Two sites are connected if the Euclidean distance between them is
    smaller than ``radius_factor`` times the median inter‑site distance.
    Edge weight = 1 / distance (higher weight for closer neighbours).
    """
    n = len(sites)
    adj = np.zeros((n, n), dtype=float)

    # median inter‑site distance as a scale
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(euclidean_distance(sites[i], sites[j]))
    median_dist = np.median(dists) if dists else 1.0

    threshold = radius_factor * median_dist
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean_distance(sites[i], sites[j])
            if d <= threshold:
                w = 1.0 / (d + 1e-9)          # avoid division by zero
                adj[i, j] = w
                adj[j, i] = w
    return adj


# ----------------------------------------------------------------------
# Endpoint Circuit Breaker
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """
    Simple failure counter. The breaker opens after ``failure_threshold``
    consecutive failures. ``allow`` returns ``True`` when the breaker is closed.
    """
    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self.consecutive_failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.open = False
        self.last_event_at = self._now_z()

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            self.open = True
        self.last_event_at = self._now_z()

    def allow(self) -> bool:
        return not self.open

    @staticmethod
    def _now_z() -> str:
        # Placeholder for a proper timestamp; kept lightweight.
        return ""


# ----------------------------------------------------------------------
# Schoolfield temperature model
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0               # rate at 25 °C (298.15 K)
    delta_H: float = 50_000.0         # activation energy (J/mol)
    R: float = 8.314                  # gas constant (J/(mol·K))

    def rate(self, T: float) -> float:
        """Compute the temperature‑dependent rate (T in °C)."""
        T_K = T + 273.15
        return self.rho_25 * math.exp((self.delta_H / self.R) * (1.0 / 298.15 - 1.0 / T_K))


# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature (simple approximation)
# ----------------------------------------------------------------------
def compute_ollivier_ricci_curvature(adj: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Approximate Ollivier‑Ricci curvature for each node.
    For node i we compare the probability measures of i and each neighbour j:
        μ_i = (1‑α)·δ_i + α·(uniform over neighbours)
    The curvature κ(i,j) = 1 - W1(μ_i, μ_j) / d(i,j)
    We return the average curvature per node.
    """
    n = adj.shape[0]
    curv = np.zeros(n, dtype=float)

    # degree and neighbour lists
    degrees = np.maximum(adj.sum(axis=1), 1e-9)
    for i in range(n):
        neigh = np.where(adj[i] > 0)[0]
        if len(neigh) == 0:
            curv[i] = 0.0
            continue
        # construct μ_i
        mu_i = np.full(n, 0.0)
        mu_i[i] = 1.0 - alpha
        mu_i[neigh] = alpha / len(neigh)

        κ_sum = 0.0
        for j in neigh:
            # μ_j
            neigh_j = np.where(adj[j] > 0)[0]
            mu_j = np.full(n, 0.0)
            mu_j[j] = 1.0 - alpha
            if len(neigh_j):
                mu_j[neigh_j] = alpha / len(neigh_j)

            # Earth mover distance on the line metric = sum |CDF_i - CDF_j|
            # For a discrete space we can use L1 distance as a cheap proxy.
            w1 = np.sum(np.abs(mu_i - mu_j))
            d_ij = 1.0 / (adj[i, j] + 1e-9)   # inverse of weight ≈ Euclidean distance
            κ = 1.0 - w1 / d_ij
            κ_sum += κ
        curv[i] = κ_sum / len(neigh)
    return curv


# ----------------------------------------------------------------------
# Bandit structures
# ----------------------------------------------------------------------
@dataclass
class BanditNodeState:
    """Tracks statistics for a node used by the UCB bandit."""
    pulls: int = 0
    reward_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / self.pulls if self.pulls > 0 else 0.0


# ----------------------------------------------------------------------
# Hybrid algorithm (deepened integration)
# ----------------------------------------------------------------------
def hybrid_algorithm(points: List[Point],
                     sites: List[Point],
                     T: float,
                     steps: int = 30,
                     failure_threshold: int = 3) -> Dict[str, Any]:
    """
    Execute the hybrid Voronoi‑Bandit routine.

    Returns a dictionary with final adjacency, curvature, and bandit statistics.
    """
    # 1. Voronoi partition
    regions = compute_voronoi_regions(points, sites)

    # 2. Build adjacency graph from the partition
    adj = build_adjacency_from_regions(regions, sites)

    # 3. Compute Ollivier‑Ricci curvature (used as a prior reward)
    curvature = compute_ollivier_ricci_curvature(adj)

    # 4. Initialise bandit state per node
    bandit_state: List[BanditNodeState] = [BanditNodeState() for _ in sites]

    # 5. Temperature‑dependent learning rate
    temp_model = SchoolfieldParams()
    learning_rate = temp_model.rate(T)          # scales how strongly we update edges

    # 6. Circuit breaker to guard against pathological updates
    breaker = EndpointCircuitBreaker(failure_threshold=failure_threshold)

    for t in range(1, steps + 1):
        # ---- Select node using Upper Confidence Bound (UCB) ----
        ucb_values = []
        total_pulls = sum(s.pulls for s in bandit_state) + 1e-9
        for i, state in enumerate(bandit_state):
            if state.pulls == 0:
                # force exploration on never‑tried nodes
                ucb = float('inf')
            else:
                bonus = math.sqrt(2 * math.log(total_pulls) / state.pulls)
                ucb = state.mean_reward + bonus + curvature[i]   # curvature as prior boost
            ucb_values.append(ucb)

        chosen = int(np.argmax(ucb_values))

        # ---- Simulate a stochastic reward ----
        # Reward model: base = curvature, plus Gaussian noise scaled by temperature
        noise = np.random.normal(loc=0.0, scale=0.1 / (learning_rate + 1e-9))
        reward = max(0.0, curvature[chosen] + noise)

        # ---- Update bandit statistics ----
        bandit_state[chosen].pulls += 1
        bandit_state[chosen].reward_sum += reward

        # ---- Update graph edges incident to the chosen node ----
        # Edge weights are increased proportionally to the observed reward
        adj[chosen, :] *= (1.0 + learning_rate * reward * 0.05)
        adj[:, chosen] = adj[chosen, :]   # keep symmetry

        # ---- Circuit breaker logic ----
        # Define a failure as a reward below the 10‑th percentile of curvature
        curvature_threshold = np.percentile(curvature, 10)
        if reward < curvature_threshold:
            breaker.record_failure()
        else:
            breaker.record_success()

        # If the breaker is open, abort further updates to protect stability
        if not breaker.allow():
            print(f"Breaker opened at step {t}; aborting further updates.", file=sys.stderr)
            break

    # Re‑compute curvature after all updates (optional deeper coupling)
    final_curvature = compute_ollivier_ricci_curvature(adj)

    # Package results
    result = {
        "adjacency": adj,
        "initial_curvature": curvature,
        "final_curvature": final_curvature,
        "bandit_state": [asdict(s) for s in bandit_state],
        "circuit_breaker_open": breaker.open,
        "steps_executed": t,
    }
    return result


# ----------------------------------------------------------------------
# Example execution (can be removed or guarded by __main__)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    points = [(random.random(), random.random()) for _ in range(200)]
    sites = [(random.random(), random.random()) for _ in range(15)]
    temperature_c = 30.0

    outcome = hybrid_algorithm(points, sites, temperature_c, steps=50)
    print("Executed steps:", outcome["steps_executed"])
    print("Circuit breaker open:", outcome["circuit_breaker_open"])
    print("Final curvature (first 5):", outcome["final_curvature"][:5])