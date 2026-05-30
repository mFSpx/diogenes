# DARWIN HAMMER — match 3395, survivor 1
# gen: 6
# parent_a: hybrid_voronoi_partition_percyphon_m779_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m1662_s1.py (gen5)
# born: 2026-05-29T23:49:49Z

"""Hybrid Voronoi‑Doomsday‑Bayesian Algorithm

Parents
-------
* **Parent A** – ``hybrid_voronoi_partition_percyphon_m779_s0.py``  
  Provides seed‑point generation from a hash string, Voronoi assignment and a
  procedural‑slot description.

* **Parent B** – ``hybrid_hybrid_hybrid_doomsd_hybrid_hybrid_hybrid_m1662_s1.py``  
  Supplies a Doomsday‑calendar weekday‑count context, a Gini‑based uniformity
  reward, a SSIM similarity measure and a Beta‑Bernoulli Bayesian update that
  is later used inside a LinUCB‑style action selector.

Mathematical Bridge
-------------------
The bridge is the *distribution* of points produced by Parent A.  After the
Voronoi assignment we obtain a region‑size vector **v** (counts per seed).  This
vector is normalised to a probability distribution and then compared with the
first ``len(v)`` components of the Doomsday context vector **x** (the flattened,
L2‑normalised weighted‑difference matrix).  The similarity ``s = SSIM(v, x[:N])``
feeds the Bayesian reward together with the Gini‑based uniformity term from the
weekday counts.  The resulting reward ``R`` updates a Beta posterior that
drives a LinUCB‑style action selector whose confidence term is scaled by the
norm ``‖x‖`` of the Doomsday context.

The module therefore fuses spatial partitioning, calendar statistics and
probabilistic learning into a single coherent system."""
import math
import hashlib
import random
import sys
from pathlib import Path
from datetime import datetime, date
from typing import List, Tuple, Iterable, Dict

import numpy as np

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
Point = Tuple[float, float]

# ----------------------------------------------------------------------
# Parent A – Voronoi / Procedural utilities
# ----------------------------------------------------------------------
def _hash_hex(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()


def generate_seed_points(seed_string: str, num_points: int) -> List[Point]:
    """Map a textual seed to *num_points* deterministic 2‑D points."""
    pts: List[Point] = []
    for i in range(num_points):
        h = _hash_hex(f"{seed_string}:{i}")
        # Use first 16 hex digits → 64 bits → split into two 32‑bit floats in [0,1)
        x = int(h[0:8], 16) / 0xFFFFFFFF
        y = int(h[8:16], 16) / 0xFFFFFFFF
        pts.append((x, y))
    return pts


def _euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest(point: Point, seeds: List[Point]) -> int:
    """Return the index of the nearest seed to *point* (ties broken by index)."""
    if not seeds:
        raise ValueError("seed list empty")
    return min(range(len(seeds)), key=lambda i: (_euclidean(point, seeds[i]), i))


def assign_voronoi(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each *point* to the nearest *seed* and return region dictionaries."""
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for p in points:
        idx = nearest(p, seeds)
        regions[idx].append(p)
    return regions


def region_counts(seeds: List[Point], points: List[Point]) -> np.ndarray:
    """Return a 1‑D array of counts per Voronoi region (order matches *seeds*)."""
    regions = assign_voronoi(points, seeds)
    return np.array([len(regions[i]) for i in range(len(seeds))], dtype=np.float64)


# ----------------------------------------------------------------------
# Parent B – Doomsday calendar / Bayesian utilities
# ----------------------------------------------------------------------
def doomsday_weekday_numbers(
    years: np.ndarray, months: np.ndarray, days: np.ndarray
) -> np.ndarray:
    """Vectorised conversion of Y‑M‑D to weekday numbers where Sunday==0."""
    dates = np.stack([years, months, days], axis=-1).astype("datetime64[D]")
    flat = dates.ravel()
    py_weekday = np.fromiter(
        (
            datetime.utcfromtimestamp(int(d.astype("datetime64[s]"))).weekday()
            for d in flat
        ),
        dtype=np.int8,
        count=flat.size,
    )
    py_weekday = py_weekday.reshape(dates.shape[:-1])
    return (py_weekday + 1) % 7  # shift Mon=0 → Sun=0


def weekday_counts(dates: Iterable[date]) -> np.ndarray:
    """Return a length‑7 vector with the count of each weekday (Sun..Sat)."""
    counts = np.zeros(7, dtype=np.int64)
    for d in dates:
        counts[d.weekday() % 7] += 1
    return counts


def gini_coefficient(x: np.ndarray) -> float:
    """Gini coefficient of a non‑negative 1‑D array."""
    if x.size == 0:
        return 0.0
    sorted_x = np.sort(x.astype(float))
    n = x.size
    cumulative = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


def weighted_difference_matrix(counts: np.ndarray) -> np.ndarray:
    """Build W = outer(c,c) * |i-j| as described in Parent B."""
    c = counts.astype(float)
    diff = np.abs(np.arange(len(c))[:, None] - np.arange(len(c))[None, :])
    return np.outer(c, c) * diff


def flatten_normalise(mat: np.ndarray) -> np.ndarray:
    """Flatten *mat* to a vector and L2‑normalise it."""
    vec = mat.ravel().astype(float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def ssim_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Structural Similarity Index for 1‑D signals (range [-1,1])."""
    C1 = 1e-4
    C2 = 9e-4
    mu_x, mu_y = x.mean(), y.mean()
    sigma_x2, sigma_y2 = x.var(), y.var()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    numerator = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return numerator / denominator if denominator != 0 else 0.0


def beta_bernoulli_update(alpha: float, beta: float, reward: float) -> Tuple[float, float]:
    """Conjugate update for a Bernoulli likelihood with reward ∈ [0,1]."""
    # Treat reward as a pseudo‑observation: success with prob=reward
    return alpha + reward, beta + (1 - reward)


# ----------------------------------------------------------------------
# Hybrid core – mathematical fusion of both parents
# ----------------------------------------------------------------------
def compute_hybrid_reward(
    seed_string: str,
    dates: List[date],
    prototype: np.ndarray,
    num_voronoi_seeds: int = 5,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the hybrid reward R = r1 * r2.

    * r1 = 1 - Gini(weekday_counts)
    * r2 = (1 + s)/2 where s = SSIM(v_norm, x[:N])
      - v_norm : normalised Voronoi region counts (size N)
      - x      : flattened‑L2‑normalised weighted‑difference matrix from dates
      - N      = min(len(v_norm), len(x), len(prototype))

    Returns the reward and a diagnostic dictionary.
    """
    # ---- Part A : Voronoi region statistics --------------------------------
    seeds = generate_seed_points(seed_string, num_voronoi_seeds)
    # generate a modest number of random points in the unit square
    random_pts = [(random.random(), random.random()) for _ in range(200)]
    v_counts = region_counts(seeds, random_pts)
    if v_counts.sum() == 0:
        v_norm = v_counts
    else:
        v_norm = v_counts / v_counts.sum()

    # ---- Part B : Doomsday calendar context ---------------------------------
    years = np.array([d.year for d in dates], dtype=np.int32)
    months = np.array([d.month for d in dates], dtype=np.int32)
    days = np.array([d.day for d in dates], dtype=np.int32)
    weekday_vec = weekday_counts(dates)
    w_gini = gini_coefficient(weekday_vec)
    r1 = 1.0 - w_gini

    W = weighted_difference_matrix(weekday_vec)
    x = flatten_normalise(W)

    # ---- Part C : Similarity & reward ---------------------------------------
    N = min(len(v_norm), len(x), len(prototype))
    s = ssim_1d(v_norm[:N], x[:N] * prototype[:N])  # scale x by prototype for a richer match
    r2 = (1.0 + s) / 2.0
    R = r1 * r2

    diagnostics = {
        "gini": w_gini,
        "r1": r1,
        "ssim": s,
        "r2": r2,
        "reward": R,
    }
    return R, diagnostics


def linucb_action_selector(
    context: np.ndarray,
    alpha: float,
    beta: float,
    action_features: np.ndarray,
    counts: np.ndarray,
    total_counts: int,
    confidence_scale: float = 1.0,
) -> int:
    """
    Simple LinUCB‑style selector.

    * Expected reward for action i is the posterior mean α/(α+β) (same for all actions).
    * Confidence term = confidence_scale * sqrt( (2 * log(total_counts+1)) / (counts[i]+1) ) * ||context||
    * Choose action maximising mean + confidence.
    """
    if action_features.shape[0] != counts.shape[0]:
        raise ValueError("action_features and counts length mismatch")
    mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.0
    norm_ctx = np.linalg.norm(context)
    log_term = math.log(total_counts + 1 + 1e-9)
    confidences = confidence_scale * np.sqrt((2 * log_term) / (counts + 1)) * norm_ctx
    scores = mean + confidences
    return int(np.argmax(scores))


def hybrid_step(
    seed_string: str,
    dates: List[date],
    prototype: np.ndarray,
    alpha: float,
    beta: float,
    action_features: np.ndarray,
    action_counts: np.ndarray,
    total_actions: int,
) -> Tuple[int, float, Dict[str, float], float, float]:
    """
    Perform one iteration of the hybrid learning loop.

    Returns:
        chosen_action, reward, diagnostics, new_alpha, new_beta
    """
    reward, diagnostics = compute_hybrid_reward(seed_string, dates, prototype)

    # Update Bayesian posterior
    new_alpha, new_beta = beta_bernoulli_update(alpha, beta, reward)

    # Build context vector for LinUCB (use the Doomsday context x)
    years = np.array([d.year for d in dates], dtype=np.int32)
    months = np.array([d.month for d in dates], dtype=np.int32)
    days = np.array([d.day for d in dates], dtype=np.int32)
    W = weighted_difference_matrix(weekday_counts(dates))
    context = flatten_normalise(W)

    # Select action
    chosen = linucb_action_selector(
        context,
        new_alpha,
        new_beta,
        action_features,
        action_counts,
        total_actions,
    )
    return chosen, reward, diagnostics, new_alpha, new_beta


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # deterministic seed for reproducibility
    random.seed(42)

    # Example seed string and prototype
    seed_str = "HybridSeed-2026"
    prototype_vec = np.array([0.2, 0.5, 0.1, 0.15, 0.05], dtype=np.float64)

    # Generate a list of random dates within a year
    base = date(2024, 1, 1)
    dates_list = [base.replace(day=((i % 28) + 1), month=((i % 12) + 1)) for i in range(50)]

    # Initialise Bayesian parameters and a dummy action set
    a, b = 1.0, 1.0  # uniform prior
    num_actions = 4
    action_feats = np.eye(num_actions)  # identity as placeholder features
    action_cnts = np.zeros(num_actions, dtype=np.int64)
    total = 0

    # Run a few hybrid steps
    for step in range(5):
        chosen, rew, diag, a, b = hybrid_step(
            seed_string=seed_str,
            dates=dates_list,
            prototype=prototype_vec,
            alpha=a,
            beta=b,
            action_features=action_feats,
            action_counts=action_cnts,
            total_actions=total,
        )
        action_cnts[chosen] += 1
        total += 1
        print(
            f"Step {step+1}: action={chosen}, reward={rew:.4f}, "
            f"alpha={a:.2f}, beta={b:.2f}, diagnostics={diag}"
        )
    sys.exit(0)