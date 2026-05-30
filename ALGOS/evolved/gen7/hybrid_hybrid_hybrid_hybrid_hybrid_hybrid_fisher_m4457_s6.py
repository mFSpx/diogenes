# DARWIN HAMMER — match 4457, survivor 6
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m1513_s0.py (gen6)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_vorono_m1224_s3.py (gen5)
# born: 2026-05-29T23:55:59Z

"""Hybrid algorithm combining:
- Parent A: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m1513_s0 (bandit routing with pheromone modulation and temperature‑dependent developmental rate).
- Parent B: hybrid_hybrid_fisher_locali_hybrid_hybrid_vorono_m1224_s3 (Fisher‑information scoring, Voronoi partitioning and a circuit‑breaker).

Mathematical bridge:
Bandit actions are embedded into a 2‑D feature space (propensity, expected_reward).  
Voronoi partitioning of this space yields regions of “similar” actions.  
Within each region the Fisher‑information score of a Gaussian beam, whose width is
the temperature‑dependent developmental rate from the Schoolfield model, quantifies
the local sensitivity of the action distribution.  
The Fisher score is used as a modulation factor for the pheromone signal that
updates the bandit policy.  A circuit‑breaker monitors the average reward of each
Voronoi region and disables updates when a region repeatedly under‑performs.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from typing import List, Tuple, Dict

# -------------------- Parent A components --------------------

R_CAL = 1.987
K25 = 298.15


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float          # probability of being selected (0‑1)
    expected_reward: float     # current estimate of reward
    confidence_bound: float    # UCB‑style confidence term
    algorithm: str             # identifier of the underlying bandit algorithm


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


_POLICY: Dict[str, List[float]] = {}  # action_id → [cumulative_reward, count]


def reset_policy() -> None:
    """Clear the global policy store."""
    _POLICY.clear()


def _reward(a: str) -> float:
    """Return the mean reward for action ``a`` (0 if unseen)."""
    total, n = _POLICY.get(a, [0.0, 0.0])
    return total / n if n else 0.0


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incrementally update the global policy with a batch of bandit observations."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15


def developmental_rate(temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield temperature‑dependence model.
    Returns a dimensionless scaling factor ρ(T) = exp[ΔHₐ (1/T₍₂₅₎ – 1/T)].
    """
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = params.delta_h_activation * (1.0 / K25 - 1.0 / temp_k)
    return math.exp(numerator)


def pheromone_modulation(base_signal: float,
                         modulation_factor: float) -> float:
    """
    Simple multiplicative modulation of a pheromone signal.
    """
    return base_signal * modulation_factor


# -------------------- Parent B components --------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity profile."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float,
                 center: float,
                 width: float,
                 eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian beam evaluated at ``theta``.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def euclidean_distance(a: Tuple[float, float],
                       b: Tuple[float, float]) -> float:
    """Standard Euclidean distance in ℝ²."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_voronoi_regions(points: List[Tuple[float, float]],
                            sites: List[Tuple[float, float]]) -> Dict[int, List[Tuple[float, float]]]:
    """
    Assign each point to the index of the nearest site.
    Returns a dict ``site_index → list[points]``.
    """
    regions: Dict[int, List[Tuple[float, float]]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions


class EndpointCircuitBreaker:
    """
    Simple failure counter that opens (disables) after ``failure_threshold``
    consecutive failures in a given region.
    """

    def __init__(self, failure_threshold: int = 3):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        self.failure_threshold = failure_threshold
        self._fail_counts: Dict[int, int] = {}   # region_index → consecutive failures
        self._open: Dict[int, bool] = {}        # region_index → circuit state

    def record_failure(self, region_idx: int) -> None:
        """Record a failure for ``region_idx`` and possibly open the circuit."""
        cnt = self._fail_counts.get(region_idx, 0) + 1
        self._fail_counts[region_idx] = cnt
        if cnt >= self.failure_threshold:
            self._open[region_idx] = True

    def record_success(self, region_idx: int) -> None:
        """Reset failure count on success."""
        self._fail_counts[region_idx] = 0
        self._open[region_idx] = False

    def can_proceed(self, region_idx: int) -> bool:
        """Return ``False`` if the circuit for ``region_idx`` is open."""
        return not self._open.get(region_idx, False)


# -------------------- Hybrid core functions --------------------


def actions_to_points(actions: List[BanditAction]) -> List[Tuple[float, float]]:
    """
    Embed each BanditAction into a 2‑D point:
    (propensity, expected_reward).  This representation is the interface
    between the bandit world and the geometric Voronoi/Fisher machinery.
    """
    return [(a.propensity, a.expected_reward) for a in actions]


def select_voronoi_sites(actions: List[BanditAction],
                         n_sites: int = 3) -> List[Tuple[float, float]]:
    """
    Choose ``n_sites`` representative points from the action cloud.
    For simplicity we pick the actions with the highest confidence bounds.
    """
    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    sorted_actions = sorted(actions,
                            key=lambda a: a.confidence_bound,
                            reverse=True)
    chosen = sorted_actions[:n_sites]
    return [(a.propensity, a.expected_reward) for a in chosen]


def hybrid_fisher_voronoi_modulation(actions: List[BanditAction],
                                    temp_k: float,
                                    params: SchoolfieldParams,
                                    n_sites: int = 3) -> Dict[str, float]:
    """
    Compute a modulation factor for each action by:
    1. Building Voronoi regions from a set of ``n_sites`` representative actions.
    2. Within each region, evaluating the Fisher information of a Gaussian beam
       whose width equals the temperature‑dependent developmental rate.
    3. Scaling the resulting Fisher score by the action's confidence bound
       (higher confidence ⇒ stronger influence).

    Returns a mapping ``action_id → modulation_factor``.
    """
    points = actions_to_points(actions)
    sites = select_voronoi_sites(actions, n_sites=n_sites)

    # Partition the action points into Voronoi cells.
    regions = compute_voronoi_regions(points, sites)

    # Pre‑compute the Gaussian width from the temperature model.
    width = developmental_rate(temp_k, params)

    # Invert the region dict to map point → region index for quick lookup.
    point_to_region: Dict[Tuple[float, float], int] = {}
    for idx, pts in regions.items():
        for pt in pts:
            point_to_region[pt] = idx

    modulation: Dict[str, float] = {}
    for action, pt in zip(actions, points):
        region_idx = point_to_region[pt]
        site_center = sites[region_idx]  # (center_propensity, center_reward)
        # Use only the propensity dimension for Fisher scoring (theta axis).
        theta = action.propensity
        center = site_center[0]
        fisher = fisher_score(theta, center, width)
        # Confidence bound acts as a linear weight.
        factor = fisher * (1.0 + action.confidence_bound)
        modulation[action.action_id] = factor
    return modulation


def hybrid_update_policy(updates: List[BanditUpdate],
                         actions: List[BanditAction],
                         temp_k: float,
                         params: SchoolfieldParams,
                         breaker: EndpointCircuitBreaker,
                         n_sites: int = 3) -> None:
    """
    Perform a policy update that respects the hybrid modulation and the circuit‑breaker.

    Steps:
    1. Build a dictionary of current actions for fast lookup.
    2. Compute modulation factors via ``hybrid_fisher_voronoi_modulation``.
    3. For each update, scale the reported reward by the corresponding modulation.
    4. Aggregate rewards per Voronoi region to decide whether the circuit opens.
    5. Apply the (possibly filtered) updates to the global policy.
    """
    action_dict = {a.action_id: a for a in actions}
    modulation = hybrid_fisher_voronoi_modulation(actions, temp_k, params, n_sites=n_sites)

    # Determine Voronoi assignment for each action (reuse logic from modulation).
    points = actions_to_points(actions)
    sites = select_voronoi_sites(actions, n_sites=n_sites)
    regions = compute_voronoi_regions(points, sites)
    point_to_region: Dict[Tuple[float, float], int] = {}
    for idx, pts in regions.items():
        for pt in pts:
            point_to_region[pt] = idx

    # Prepare filtered updates.
    filtered_updates: List[BanditUpdate] = []
    region_rewards: Dict[int, List[float]] = {}

    for upd in updates:
        act = action_dict.get(upd.action_id)
        if act is None:
            continue  # unknown action, ignore

        pt = (act.propensity, act.expected_reward)
        region_idx = point_to_region[pt]

        if not breaker.can_proceed(region_idx):
            # Skip updates from a tripped region.
            continue

        mod_factor = modulation.get(upd.action_id, 1.0)
        mod_reward = pheromone_modulation(upd.reward, mod_factor)

        # Record for region‑level statistics.
        region_rewards.setdefault(region_idx, []).append(mod_reward)

        filtered_updates.append(
            BanditUpdate(
                context_id=upd.context_id,
                action_id=upd.action_id,
                reward=mod_reward,
                propensity=upd.propensity,
            )
        )

    # Evaluate region performance and possibly trip the breaker.
    for region_idx, rewards in region_rewards.items():
        avg = sum(rewards) / len(rewards)
        if avg < 0.0:                     # heuristic: negative average reward → failure
            breaker.record_failure(region_idx)
        else:
            breaker.record_success(region_idx)

    # Apply the filtered, modulated updates to the global policy.
    update_policy(filtered_updates)


def sample_hybrid_run() -> None:
    """
    Demonstration routine that creates synthetic actions, performs a few update
    cycles, and prints the resulting policy.
    """
    random.seed(42)
    np.random.seed(42)

    # Create a synthetic set of BanditAction objects.
    actions = [
        BanditAction(
            action_id=f"a{i}",
            propensity=random.random(),
            expected_reward=random.uniform(-1.0, 1.0),
            confidence_bound=random.random() * 0.5,
            algorithm="UCB1"
        )
        for i in range(7)
    ]

    # Simulate a batch of observations.
    updates = []
    for a in actions:
        # Simulated reward drawn from a normal distribution centred at the
        # current expected reward.
        reward = np.random.normal(loc=a.expected_reward, scale=0.3)
        updates.append(
            BanditUpdate(
                context_id="ctx",
                action_id=a.action_id,
                reward=reward,
                propensity=a.propensity,
            )
        )

    # Temperature (Kelvin) and Schoolfield parameters.
    temp_k = c_to_k(25.0)  # 25 °C → 298.15 K
    params = SchoolfieldParams()

    breaker = EndpointCircuitBreaker(failure_threshold=2)

    # Run the hybrid update.
    hybrid_update_policy(updates, actions, temp_k, params, breaker, n_sites=3)

    # Print the resulting policy statistics.
    print("=== Policy after hybrid update ===")
    for aid, (total, cnt) in _POLICY.items():
        print(f"Action {aid}: mean reward = {total/cnt:.4f} over {int(cnt)} samples")


if __name__ == "__main__":
    # Smoke test – should run without raising any exception.
    reset_policy()
    sample_hybrid_run()