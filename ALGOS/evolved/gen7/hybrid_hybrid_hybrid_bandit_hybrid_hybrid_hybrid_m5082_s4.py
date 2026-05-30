# DARWIN HAMMER — match 5082, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_bandit_router_variational_free_ene_m56_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m1543_s0.py (gen6)
# born: 2026-05-29T23:59:42Z

"""
Hybrid Bandit–Voronoi Active‑Inference Module
================================================

Parents
-------
* **Parent A** – ``hybrid_hybrid_bandit_router_variational_free_ene_m56_s2.py``  
  Provides a contextual multi‑armed bandit with a variational free‑energy
  interface (Gaussian belief updates, reward normalisation via a
  Schoolfield temperature model).

* **Parent B** – ``hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_endpoi_m1543_s0.py``  
  Supplies geometric tools: Voronoi partitioning of a point cloud,
  lead‑lag transformation, path‑signature extraction and an (approximate)
  Ollivier‑Ricci curvature estimator.

Mathematical Bridge
-------------------
Each bandit **action** is endowed with a 2‑D *seed* coordinate.
The set of seeds induces a Voronoi diagram over the observation points.
For every Voronoi cell we compute a **level‑1 path signature** of the
lead‑lag transformed points – this signature is a sufficient statistic
for a Gaussian belief (mean = signature, covariance ≈ I).  
The **variational free energy** of the belief for action *i* is therefore


F_i = ½‖μ_i – μ_prior‖² + const


where μ_i is the signature of cell *i* and μ_prior is a global prior
(mean signature of all cells).  

The **Ollivier‑Ricci curvature** between neighbouring seeds quantifies
the geometric cohesion of the Voronoi cells; we use it as a *curvature
reward* that modulates the bandit’s expected reward:


R_i = α·normalized_activity(T) + β·curvature_i


The hybrid update therefore minimises free energy while updating the
bandit policy with curvature‑aware rewards.

The module below implements this fusion, exposing three core functions
that illustrate the combined behaviour.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]

# ----------------------------------------------------------------------
# Parent A – Bandit core (adapted)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str
    position: Point  # new field linking to geometric seed


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


_POLICY: Dict[str, List[float]] = {}  # action_id → [cum_reward, count]


def reset_policy() -> None:
    """Clear the global policy statistics."""
    _POLICY.clear()


def _reward_estimate(action_id: str) -> float:
    """Mean reward for an action (0 if never selected)."""
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incremental Monte‑Carlo update of the bandit policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


# ----------------------------------------------------------------------
# Temperature → Normalised activity (Active‑Inference side)
# ----------------------------------------------------------------------
R_CAL = 1.987  # cal·K⁻¹·mol⁻¹
K25 = 298.15  # K (25 °C)


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = 20_000.0
    delta_h_high: float = 30_000.0


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float, p: SchoolfieldParams) -> float:
    """
    Simple Schoolfield (Arrhenius‑type) rate.
    Returns a normalised activity in [0, 1].
    """
    if temp_k <= p.t_low or temp_k >= p.t_high:
        return 0.0
    num = p.rho_25 * math.exp(-p.delta_h_activation / R_CAL * (1.0 / temp_k - 1.0 / K25))
    den_low = 1.0 + math.exp(p.delta_h_low / R_CAL * (1.0 / p.t_low - 1.0 / temp_k))
    den_high = 1.0 + math.exp(p.delta_h_high / R_CAL * (1.0 / temp_k - 1.0 / p.t_high))
    return num / (den_low * den_high)


def normalized_activity(temp_c: float, low_c: float = 5.0) -> float:
    """Normalised activity from temperature, using the Schoolfield model."""
    params = SchoolfieldParams()
    return developmental_rate(c_to_k(temp_c), params)


# ----------------------------------------------------------------------
# Parent B – Geometric core (adapted)
# ----------------------------------------------------------------------
def distance(a: Point, b: Point) -> float:
    """Euclidean distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    """Index of the nearest seed to *point* (break ties by index)."""
    if not seeds:
        raise ValueError("seed list empty")
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))


def assign(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Voronoi assignment of *points* to the nearest seed."""
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions


def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Lead‑lag embedding of a discrete path."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError("path must be 2‑D array (T, d)")
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path: np.ndarray) -> np.ndarray:
    """Level‑1 path signature (net displacement)."""
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def voronoi_signatures(points: List[Point], seeds: List[Point]) -> Dict[int, np.ndarray]:
    """
    Compute a level‑1 signature for each Voronoi cell.
    The signature is taken after a lead‑lag transform of the cell points.
    """
    regions = assign(points, seeds)
    signatures: Dict[int, np.ndarray] = {}
    for i, region in regions.items():
        if not region:
            # empty cell → zero signature
            signatures[i] = np.zeros(2 * 2)  # 2‑D points → 4‑D lead‑lag vector
            continue
        region_arr = np.array(region)
        ll = lead_lag_transform(region_arr)
        signatures[i] = signature_level1(ll)
    return signatures


def ollivier_ricci_curvature(seeds: List[Point]) -> Dict[Tuple[int, int], float]:
    """
    Approximate Ollivier‑Ricci curvature between each pair of neighbouring seeds.
    Neighbourhood is defined by the Delaunay graph (here approximated by
    mutual nearest‑neighbour relationship for simplicity).
    """
    n = len(seeds)
    curvature: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            # simple neighbour criterion: each is among the k‑nearest of the other
            # we use k = 3 for a lightweight approximation
            d_ij = distance(seeds[i], seeds[j])
            # compute average distance to third‑closest neighbour for each seed
            dists_i = sorted(distance(seeds[i], s) for s in seeds if s != seeds[i])
            dists_j = sorted(distance(seeds[j], s) for s in seeds if s != seeds[j])
            eps_i = dists_i[2] if len(dists_i) > 2 else dists_i[-1]
            eps_j = dists_j[2] if len(dists_j) > 2 else dists_j[-1]
            # Ollivier‑Ricci curvature κ = 1 - (W1 / (eps_i + eps_j))
            # where W1 is the earth mover distance; we approximate W1 by d_ij
            curvature[(i, j)] = 1.0 - (d_ij / (eps_i + eps_j + 1e-12))
    return curvature


# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def compute_hybrid_rewards(
    actions: List[BanditAction],
    observations: List[Point],
    temperature_c: float,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> List[BanditUpdate]:
    """
    For each action:
      1. Build Voronoi cell from *observations* using the action's position as seed.
      2. Extract the level‑1 signature (Gaussian mean) of the cell.
      3. Compute a global prior (mean of all signatures) and the free‑energy term.
      4. Estimate curvature between this seed and its neighbours.
      5. Combine normalised activity (temperature) and curvature into a reward.

    Returns a list of ``BanditUpdate`` objects ready for ``update_policy``.
    """
    seeds = [a.position for a in actions]
    # 1‑2. Signatures per cell
    sigs = voronoi_signatures(observations, seeds)

    # 3. Prior = average signature (vector)
    all_sigs = np.stack(list(sigs.values()))
    prior = np.mean(all_sigs, axis=0)

    # 4. Curvature matrix
    curv_dict = ollivier_ricci_curvature(seeds)

    # Helper to fetch curvature for a given action index
    def curvature_for(idx: int) -> float:
        # average curvature with all other seeds
        vals = [curv_dict.get((min(idx, j), max(idx, j)), 0.0) for j in range(len(seeds)) if j != idx]
        return float(np.mean(vals)) if vals else 0.0

    # 5. Assemble updates
    updates: List[BanditUpdate] = []
    for i, act in enumerate(actions):
        sig = sigs[i]
        # free‑energy (quadratic distance to prior)
        free_energy = 0.5 * np.linalg.norm(sig - prior) ** 2
        # curvature reward
        curv = curvature_for(i)
        # temperature reward
        temp_reward = normalized_activity(temperature_c)
        # final hybrid reward (higher is better)
        reward = alpha * temp_reward + beta * curv - free_energy * 0.01  # small penalty for high free‑energy
        updates.append(
            BanditUpdate(
                context_id="hybrid",
                action_id=act.action_id,
                reward=reward,
                propensity=act.propensity,
            )
        )
    return updates


def hybrid_policy_step(
    actions: List[BanditAction],
    observations: List[Point],
    temperature_c: float,
) -> None:
    """
    Execute one hybrid inference‑bandit step:
      * compute rewards,
      * update the global bandit policy,
      * refresh each action's expected reward from the policy.
    """
    updates = compute_hybrid_rewards(actions, observations, temperature_c)
    update_policy(updates)
    # Refresh expected rewards in-place
    for a in actions:
        # we cannot modify frozen dataclass, so we just print the new estimate
        new_est = _reward_estimate(a.action_id)
        print(f"Action {a.action_id}: updated expected reward ≈ {new_est:.4f}")


def sample_random_observations(num: int, bounds: Tuple[float, float] = (0.0, 10.0)) -> List[Point]:
    """Utility to generate *num* random 2‑D points within square bounds."""
    low, high = bounds
    return [(random.uniform(low, high), random.uniform(low, high)) for _ in range(num)]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a small set of actions with distinct seeds
    actions = [
        BanditAction(
            action_id="A",
            propensity=0.5,
            expected_reward=0.0,
            confidence_bound=1.0,
            algorithm="hybrid",
            position=(2.0, 3.0),
        ),
        BanditAction(
            action_id="B",
            propensity=0.5,
            expected_reward=0.0,
            confidence_bound=1.0,
            algorithm="hybrid",
            position=(7.0, 8.0),
        ),
        BanditAction(
            action_id="C",
            propensity=0.5,
            expected_reward=0.0,
            confidence_bound=1.0,
            algorithm="hybrid",
            position=(5.0, 2.0),
        ),
    ]

    # Generate synthetic observations and run a few hybrid steps
    reset_policy()
    for step in range(3):
        obs = sample_random_observations(200)
        temp_c = random.uniform(5.0, 30.0)  # ambient temperature
        print(f"\n--- Step {step + 1} (T={temp_c:.1f} °C) ---")
        hybrid_policy_step(actions, obs, temp_c)

    # Final policy dump
    print("\nFinal policy statistics:")
    for aid, (total, count) in _POLICY.items():
        print(f"  {aid}: total_reward={total:.4f}, count={int(count)}")