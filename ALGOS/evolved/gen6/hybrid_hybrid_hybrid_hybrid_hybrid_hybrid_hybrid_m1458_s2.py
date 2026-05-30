# DARWIN HAMMER — match 1458, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m449_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_voronoi_parti_m586_s0.py (gen5)
# born: 2026-05-29T23:36:30Z

"""
Hybrid Thompson‑Bandit / Caputo‑Fractional Voronoi Algorithm.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m449_s0.py (Thompson‑sampling bandit enriched
  with Ollivier‑Ricci curvature on the action space).
- hybrid_hybrid_hybrid_hybrid_hybrid_voronoi_parti_m586_s0.py (Caputo fractional derivative
  modelling pheromone decay, with Voronoi partitioning of the feature space).

Mathematical bridge:
Both parents rely on a Voronoi tessellation of a point set.  In this fusion the Voronoi
cells become *action regions* for the bandit.  For each cell we compute a curvature‑
weighted fractional pheromone signal:
    φ_i(t) =  κ_i · 𝔻_C^{α}[f_i](t)
where κ_i is the Ollivier‑Ricci curvature associated with the cell’s centroid,
𝔻_C^{α} denotes the Caputo fractional derivative of order α, and f_i(t) is a
baseline reward‑signal (here a simple exponential decay).  The resulting φ_i(t) is used
as a data‑driven prior update for the Thompson sampler, thus fusing the two topologies
into a single probabilistic decision engine.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable

# ----------------------------------------------------------------------
# Fractional‑calculus utilities (from Parent B)
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def _gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function for real z > 0."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * _gamma_lanczos(1 - z))
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def caputo_derivative(f: Callable[[np.ndarray], np.ndarray],
                     alpha: float,
                     t: np.ndarray,
                     dt: float = 0.01) -> np.ndarray:
    """
    Discrete Caputo fractional derivative of order ``alpha`` for a vector of times ``t``.
    The integral is approximated with the trapezoidal rule.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1) for this implementation")
    deriv = np.empty_like(t)
    for idx, ti in enumerate(t):
        tau = np.arange(0, ti, dt)
        if tau.size == 0:
            deriv[idx] = 0.0
            continue
        f_tau = f(tau)
        kernel = (ti - tau) ** (-alpha)
        integral = np.trapz(f_tau * kernel, tau)
        deriv[idx] = integral / _gamma_lanczos(1 - alpha)
    return deriv

def fractional_decay_kernel(alpha: float, t: np.ndarray) -> np.ndarray:
    """Kernel t^{alpha-1} / Gamma(alpha) used for fractional decay."""
    return t ** (alpha - 1) / _gamma_lanczos(alpha)

# ----------------------------------------------------------------------
# Voronoi utilities (common to both parents)
# ----------------------------------------------------------------------
def voronoi_partition(points: List[Tuple[float, float]],
                      seeds: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """
    Assign each point to the nearest seed (Euclidean distance).
    Returns a dict mapping seed index -> list of points in that cell.
    """
    regions: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(seeds))}
    for p in points:
        nearest = min(
            range(len(seeds)),
            key=lambda i: (math.hypot(p[0] - seeds[i][0], p[1] - seeds[i][1]), i)
        )
        regions[nearest].append(p)
    return regions

def cell_centroid(region: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Simple arithmetic mean of points in a region; returns (0,0) for empty region."""
    if not region:
        return (0.0, 0.0)
    xs, ys = zip(*region)
    return (float(np.mean(xs)), float(np.mean(ys)))

# ----------------------------------------------------------------------
# Curvature (Ollivier‑Ricci) placeholder (from Parent A)
# ----------------------------------------------------------------------
def ollivier_ricci_curvature(p1: Tuple[float, float],
                             p2: Tuple[float, float],
                             epsilon: float = 1e-3) -> float:
    """
    Very coarse surrogate for Ollivier‑Ricci curvature between two points.
    For a true implementation one would need optimal transport; here we use
    an inverse‑distance heuristic.
    """
    d = math.hypot(p1[0] - p2[0], p1[1] - p2[1]) + epsilon
    return 1.0 / d  # larger curvature for closer points

def region_curvature(centroid: Tuple[float, float],
                     all_centroids: List[Tuple[float, float]]) -> float:
    """
    Aggregate curvature of a region with respect to all other region centroids.
    """
    if not all_centroids:
        return 0.0
    curv_sum = sum(ollivier_ricci_curvature(centroid, other) for other in all_centroids if other != centroid)
    return curv_sum / (len(all_centroids) - 1) if len(all_centroids) > 1 else 0.0

# ----------------------------------------------------------------------
# Thompson‑sampling bandit (from Parent A)
# ----------------------------------------------------------------------
@dataclass
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "thompson_sampling"

class ThompsonBandit:
    """
    Simple Thompson‑sampling bandit for continuous rewards.
    Each action maintains a Gaussian posterior (mean, variance) approximated
    via Beta‑like α,β parameters for convenience.
    """
    def __init__(self, actions: List[str], prior_mu: float = 0.0, prior_sigma: float = 1.0):
        self._mu: Dict[str, float] = {a: prior_mu for a in actions}
        self._sigma: Dict[str, float] = {a: prior_sigma for a in actions}
        self._actions = actions

    def sample(self) -> BanditAction:
        """Draw a sample from each posterior and return the best action."""
        samples = {a: random.gauss(self._mu[a], max(self._sigma[a], 1e-6)) for a in self._actions}
        best = max(samples, key=samples.get)
        # Propensity approximated as softmax over means
        exp_means = np.exp([self._mu[a] for a in self._actions])
        probs = exp_means / exp_means.sum()
        propensity = probs[self._actions.index(best)]
        # Confidence bound (simple std dev)
        confidence = self._sigma[best]
        return BanditAction(
            action_id=best,
            propensity=propensity,
            expected_reward=self._mu[best],
            confidence_bound=confidence
        )

    def update(self, action_id: str, reward: float, weight: float = 1.0):
        """
        Bayesian‑like update: treat reward as observation with variance inversely
        proportional to ``weight`` (larger weight → more confidence).
        """
        sigma2 = self._sigma[action_id] ** 2
        # Precision update
        new_precision = 1.0 / sigma2 + weight
        new_mu = (self._mu[action_id] / sigma2 + reward * weight) / new_precision
        new_sigma = math.sqrt(1.0 / new_precision)
        self._mu[action_id] = new_mu
        self._sigma[action_id] = new_sigma

# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def fractional_pheromone_signal(base_value: float,
                                half_life: float,
                                alpha: float,
                                t_grid: np.ndarray) -> np.ndarray:
    """
    Compute a pheromone signal that decays according to a fractional kernel.
    The baseline decay is an exponential with the given half‑life; the
    Caputo derivative of that exponential is taken to inject memory effects.
    """
    # Baseline exponential decay
    lam = math.log(2) / half_life
    baseline = lambda tau: base_value * np.exp(-lam * tau)
    # Apply Caputo fractional derivative
    return caputo_derivative(baseline, alpha, t_grid)

def curvature_weighted_signal(region_points: List[Tuple[float, float]],
                              all_regions: List[List[Tuple[float, float]]],
                              base_value: float,
                              half_life: float,
                              alpha: float,
                              t_grid: np.ndarray) -> float:
    """
    For a given Voronoi cell, compute the fractional pheromone signal and weight it
    by the region's curvature relative to the whole partition.
    Returns the *average* signal over the time grid (scalar) to be used as a prior boost.
    """
    centroid = cell_centroid(region_points)
    other_centroids = [cell_centroid(r) for r in all_regions if r is not region_points]
    curv = region_curvature(centroid, other_centroids)
    raw_signal = fractional_pheromone_signal(base_value, half_life, alpha, t_grid)
    # Weight by curvature (normalize curvature to [0,1] via tanh)
    weight = math.tanh(curv)
    return float(np.mean(raw_signal) * (1.0 + weight))

def hybrid_select_action(bandit: ThompsonBandit,
                         points: List[Tuple[float, float]],
                         seeds: List[Tuple[float, float]],
                         base_value: float = 1.0,
                         half_life: float = 10.0,
                         alpha: float = 0.7,
                         t_end: float = 5.0,
                         dt: float = 0.05) -> BanditAction:
    """
    1. Partition ``points`` into Voronoi cells defined by ``seeds``.
    2. For each cell compute a curvature‑weighted fractional pheromone boost.
    3. Inject the boost into the corresponding bandit action's posterior mean.
    4. Sample the Thompson bandit and return the chosen action.
    """
    regions = voronoi_partition(points, seeds)
    t_grid = np.arange(0, t_end, dt)

    # Assume each seed corresponds to an action with the same identifier
    for idx, seed in enumerate(seeds):
        action_id = str(idx)
        region_pts = regions.get(idx, [])
        boost = curvature_weighted_signal(
            region_pts,
            list(regions.values()),
            base_value,
            half_life,
            alpha,
            t_grid
        )
        # Apply boost as a pseudo‑observation with moderate weight
        bandit.update(action_id, reward=boost, weight=0.5)

    return bandit.sample()

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic 2‑D points
    random.seed(42)
    np.random.seed(42)
    points = [(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(200)]

    # Choose 5 seeds (actions)
    seeds = [(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(5)]
    action_ids = [str(i) for i in range(len(seeds))]

    # Initialise bandit
    bandit = ThompsonBandit(actions=action_ids, prior_mu=0.0, prior_sigma=1.0)

    # Run a few hybrid selections
    for step in range(3):
        chosen = hybrid_select_action(
            bandit,
            points,
            seeds,
            base_value=1.0,
            half_life=8.0,
            alpha=0.6,
            t_end=4.0,
            dt=0.1
        )
        print(f"Step {step+1}: chosen action {chosen.action_id} "
              f"(propensity={chosen.propensity:.3f}, "
              f"exp_reward={chosen.expected_reward:.3f})")
        # Simulate a stochastic reward
        reward = random.gauss(0.5, 0.2)
        bandit.update(chosen.action_id, reward, weight=1.0)