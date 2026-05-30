# DARWIN HAMMER — match 5169, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_voronoi_parti_m1327_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_vorono_hybrid_hybrid_hybrid_m1529_s1.py (gen6)
# born: 2026-05-30T00:00:15Z

"""Hybrid Voronoi‑Bandit‑Signature Fusion

This module combines the core mathematics of the two parent algorithms:

* **Parent A** – provides Min‑Hash style *signature* generation, *similarity*,
  a Gaussian kernel and a Voronoi assignment helper.
* **Parent B** – supplies a Voronoi region builder, a *Schoolfield* temperature‑
  dependent rate function and the conceptual “bandit” context.

The mathematical bridge is the Voronoi partition itself: each Voronoi cell
becomes a *bandit arm*.  For every arm we compute a Min‑Hash signature from
the tokens attached to the points that fall inside the cell.  The expected
reward of an arm is defined as the similarity between its signature and a
global target signature, scaled by the temperature‑dependent Schoolfield rate.
Thus geometry (Voronoi), set similarity (Min‑Hash) and thermally‑modulated
learning (Schoolfield) are fused into a single unified system.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, FrozenSet, Iterable
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Min‑Hash signature utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """Min‑Hash signature of length *k* for a token list."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity between two Min‑Hash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian kernel."""
    return math.exp(-((epsilon * r) ** 2))


# ----------------------------------------------------------------------
# Parent B – Voronoi + Schoolfield temperature model
# ----------------------------------------------------------------------
def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_voronoi_regions(points: List[Point], sites: List[Point]) -> Dict[int, List[Point]]:
    """
    Assign each point to the nearest site.
    Returns a mapping ``site_index → list[points]``.
    """
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(sites))}
    for pt in points:
        distances = [euclidean_distance(pt, s) for s in sites]
        nearest = int(np.argmin(distances))
        regions[nearest].append(pt)
    return regions


@dataclass(frozen=True)
class SchoolfieldParams:
    """Parameters of the Schoolfield temperature model."""
    rho_25: float = 1.0          # rate at 25 °C (298.15 K)
    delta_h_a: float = 0.0       # activation energy (eV)
    delta_h_d: float = 0.0       # deactivation energy (eV)
    t_l: float = 0.0             # lower temperature limit (K)
    t_u: float = 0.0             # upper temperature limit (K)


def schoolfield_rate(t: float, params: SchoolfieldParams) -> float:
    """
    Temperature‑dependent rate according to the Schoolfield model.

    Args:
        t: Temperature in Kelvin.
        params: Model parameters.

    Returns:
        A positive scaling factor.
    """
    k_b = 8.617333262145e-5  # Boltzmann constant in eV·K⁻¹
    if t <= 0:
        raise ValueError("temperature must be positive Kelvin")
    # Guard against division by zero in denominator
    denom = 1.0 + math.exp((params.delta_h_d / k_b) * (1.0 / params.t_u - 1.0 / t)) \
                if params.t_u > 0 else 1.0
    num = math.exp((params.delta_h_a / k_b) * (1.0 / 298.15 - 1.0 / t))
    rate = params.rho_25 * num / denom
    return max(rate, 0.0)


# ----------------------------------------------------------------------
# Hybrid core – Voronoi cells as bandit arms with thermally‑scaled reward
# ----------------------------------------------------------------------
class HybridBandit:
    """
    A multi‑armed bandit where each arm corresponds to a Voronoi region.
    Expected reward = similarity(region_signature, target_signature) *
                     schoolfield_rate(temperature).
    """
    def __init__(self,
                 sites: List[Point],
                 target_tokens: List[str],
                 temperature: float,
                 sf_params: SchoolfieldParams,
                 sig_k: int = 128):
        self.sites = sites
        self.k = sig_k
        self.target_sig = signature(target_tokens, k=self.k)
        self.temperature = temperature
        self.sf_params = sf_params

        # Bandit bookkeeping
        self.counts: List[int] = [0] * len(sites)          # pulls per arm
        self.values: List[float] = [0.0] * len(sites)      # average reward per arm

    def _region_signature(self, region_points: List[Point],
                          token_map: Dict[Point, List[str]]) -> List[int]:
        """Aggregate tokens of all points in a region and compute its signature."""
        agg_tokens: List[str] = []
        for pt in region_points:
            agg_tokens.extend(token_map.get(pt, []))
        return signature(agg_tokens, k=self.k)

    def evaluate(self,
                 points: List[Point],
                 token_map: Dict[Point, List[str]]) -> Dict[int, float]:
        """
        Compute the reward for each arm given the current point cloud.

        Returns:
            Mapping ``arm_index → reward``.
        """
        regions = compute_voronoi_regions(points, self.sites)
        temp_factor = schoolfield_rate(self.temperature, self.sf_params)
        rewards: Dict[int, float] = {}

        for arm, pts in regions.items():
            reg_sig = self._region_signature(pts, token_map)
            sim = similarity(reg_sig, self.target_sig)
            reward = sim * temp_factor
            rewards[arm] = reward
        return rewards

    def select_arm(self, rewards: Dict[int, float]) -> int:
        """
        Simple Upper‑Confidence‑Bound (UCB1) selection using the computed rewards
        as the exploitation term.
        """
        total_counts = sum(self.counts) + 1e-9
        ucb_values = []
        for i in range(len(self.sites)):
            avg = self.values[i]
            bonus = math.sqrt(2 * math.log(total_counts) / (self.counts[i] + 1e-9))
            ucb = avg + bonus + rewards.get(i, 0.0)  # incorporate current reward estimate
            ucb_values.append(ucb)
        return int(np.argmax(ucb_values))

    def update(self, chosen_arm: int, reward: float) -> None:
        """Incremental update of the chosen arm's statistics."""
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Incremental mean update
        self.values[chosen_arm] = ((n - 1) * value + reward) / n

    # ------------------------------------------------------------------
    # Demonstrative hybrid operations
    # ------------------------------------------------------------------
    def step(self,
             points: List[Point],
             token_map: Dict[Point, List[str]]) -> Tuple[int, float]:
        """
        Perform a full bandit step:
        1. Evaluate rewards for all arms.
        2. Select an arm via UCB.
        3. Observe the reward of the selected arm.
        4. Update internal statistics.

        Returns:
            (chosen_arm, observed_reward)
        """
        rewards = self.evaluate(points, token_map)
        arm = self.select_arm(rewards)
        observed = rewards[arm]
        self.update(arm, observed)
        return arm, observed

    def gaussian_weighted_similarity(self,
                                     points: List[Point],
                                     token_map: Dict[Point, List[str]],
                                     epsilon: float = 1.0) -> float:
        """
        Compute a global similarity measure where each region's contribution
        is weighted by a Gaussian of its centroid distance to the global centroid.
        """
        regions = compute_voronoi_regions(points, self.sites)
        all_pts = [pt for sublist in regions.values() for pt in sublist]
        if not all_pts:
            return 0.0
        global_cent = centroid(all_pts)

        weighted_sum = 0.0
        weight_total = 0.0
        for pts in regions.values():
            if not pts:
                continue
            cent = centroid(pts)
            r = euclidean_distance(cent, global_cent)
            w = gaussian(r, epsilon=epsilon)
            reg_sig = self._region_signature(pts, token_map)
            sim = similarity(reg_sig, self.target_sig)
            weighted_sum += w * sim
            weight_total += w
        return weighted_sum / weight_total if weight_total > 0 else 0.0


def centroid(points: List[Point]) -> Point:
    """Geometric centroid of a point set."""
    if not points:
        raise ValueError("cannot compute centroid of empty set")
    xs, ys = zip(*points)
    return (sum(xs) / len(xs), sum(ys) / len(ys))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic data
    random.seed(42)
    np.random.seed(42)

    # Create 10 random sites (Voronoi seeds)
    sites = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(5)]

    # Generate 200 random points with associated token bags
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(200)]
    token_pool = ["alpha", "beta", "gamma", "delta", "epsilon"]
    token_map: Dict[Point, List[str]] = {}
    for pt in points:
        # each point gets 1‑3 random tokens
        token_map[pt] = random.sample(token_pool, random.randint(1, 3))

    # Target signature (e.g., desired topic)
    target_tokens = ["alpha", "gamma", "epsilon"]

    # Initialise hybrid bandit
    sf_params = SchoolfieldParams(rho_25=1.2, delta_h_a=0.65, delta_h_d=0.3, t_l=280.0, t_u=320.0)
    hybrid = HybridBandit(sites=sites,
                          target_tokens=target_tokens,
                          temperature=298.15,
                          sf_params=sf_params,
                          sig_k=64)

    # Run a few steps
    for _ in range(10):
        arm, rew = hybrid.step(points, token_map)
        print(f"Chosen arm {arm}, observed reward {rew:.4f}")

    # Compute a global Gaussian‑weighted similarity for inspection
    gws = hybrid.gaussian_weighted_similarity(points, token_map, epsilon=0.2)
    print(f"Gaussian‑weighted global similarity: {gws:.4f}")