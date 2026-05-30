# DARWIN HAMMER — match 4222, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_ternary_route_hybrid_workshare_all_m171_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1043_s0.py (gen4)
# born: 2026-05-29T23:54:17Z

"""
Hybrid Algorithm integrating:
- Parent A: `compute_ssim` and deterministic workshare allocation.
- Parent B: `HybridPheromoneKrampusSystem` with doomsday‑based weighting and pseudo‑feature extraction.

Mathematical bridge:
The SSIM similarity score (a scalar in [−1,1]) is interpreted as a reward signal that
updates a pheromone matrix `P[g]` for each group *g*.  The day‑of‑week factor from the
doomsday calendar (`d ∈ {0,…,6}`) scales the reward, and a deterministic pseudo‑feature
vector `f(text)` (derived from a hash‑seeded RNG) provides a per‑group modulation factor.
The final allocation weight for a group is

    w_g = P[g] · (1 + α·ssim) · (1 + β·d/6) · (1 + γ·f_g)

where α,β,γ are tunable scalars.  The workshare allocator then distributes the
non‑deterministic portion of the total units proportionally to `w_g`.  This fuses the
similarity‑routing logic of Parent A with the pheromone‑bandit dynamics of Parent B
into a single unified system.
"""

import sys
import math
import random
import hashlib
import numpy as np
from datetime import date, datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------
# Shared constants
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

# ----------------------------------------------------------------------
# Parent A – SSIM computation
# ----------------------------------------------------------------------
def _pct(value: float) -> float:
    """Round to six decimal places (shared helper)."""
    return round(float(value), 6)


def compute_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two 1‑D vectors.
    The formula is the classic luminance‑contrast‑structure decomposition.
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    ssim = numerator / denominator
    return _pct(ssim)


# ----------------------------------------------------------------------
# Parent B – Pheromone / Krampus system
# ----------------------------------------------------------------------
class HybridPheromoneKrampusSystem:
    """
    Maintains a pheromone level per group and provides deterministic
    pseudo‑features extracted from an arbitrary text string.
    """

    def __init__(self):
        # Initialise a pheromone level of 1.0 for each group (neutral)
        self.pheromones: dict[str, float] = {g: 1.0 for g in GROUPS}
        # Feature cache to avoid recomputation for identical texts
        self._feature_cache: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Utility helpers (mirroring Parent B)
    # ------------------------------------------------------------------
    def doomsday(self, year: int, month: int, day: int) -> int:
        """Return a 0‑based day‑of‑week index (Monday=0 … Sunday=6)."""
        return date(year, month, day).weekday()

    def _rng_from_text(self, text: str) -> random.Random:
        """Deterministic RNG seeded from a SHA‑256 hash of the input text."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "big")
        return random.Random(seed)

    def extract_full_features(self, text: str) -> dict[str, float]:
        """
        Produce a deterministic pseudo‑feature vector for the supplied text.
        The feature set mirrors Parent B's symbolic operators but is reduced to
        a numeric value per group for simplicity.
        """
        if text in self._feature_cache:
            return self._feature_cache[text]

        rng = self._rng_from_text(text)
        # Generate a small float in [0,1) for each group; these act as modulation factors.
        features = {g: rng.random() for g in GROUPS}
        self._feature_cache[text] = features
        return features

    # ------------------------------------------------------------------
    # Core pheromone update – the mathematical bridge
    # ------------------------------------------------------------------
    def update_pheromones(
        self,
        ssim_score: float,
        day_factor: float,
        feature_factors: dict[str, float],
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
    ) -> None:
        """
        Update each group's pheromone level using the fused equation:

            P[g] ← P[g] * (1 + α·ssim) * (1 + β·day_factor) * (1 + γ·f_g)

        where:
        - `ssim` ∈ [−1,1] is the similarity reward,
        - `day_factor` = d/6 ∈ [0,1] (d = weekday index),
        - `f_g` ∈ [0,1) is the deterministic pseudo‑feature for group g.
        """
        for g in GROUPS:
            f_g = feature_factors.get(g, 0.0)
            multiplier = (1 + alpha * ssim_score) * (1 + beta * day_factor) * (1 + gamma * f_g)
            self.pheromones[g] = _pct(self.pheromones[g] * multiplier)

    def get_normalised_weights(self) -> dict[str, float]:
        """Return pheromone levels normalised to sum to 1."""
        total = sum(self.pheromones.values())
        if total == 0:
            # Avoid division by zero – fall back to uniform distribution
            return {g: 1.0 / len(GROUPS) for g in GROUPS}
        return {g: self.pheromones[g] / total for g in GROUPS}


# ----------------------------------------------------------------------
# Hybrid functions (demonstrating the combined operation)
# ----------------------------------------------------------------------
def hybrid_allocate(
    total_units: float,
    text: str,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    deterministic_target_pct: float = 90.0,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> dict[str, float]:
    """
    Allocate `total_units` among groups using:
    1. SSIM(x_vec, y_vec) as a reward.
    2. Day‑of‑week weighting from the doomsday calendar.
    3. Deterministic pseudo‑features extracted from `text`.
    4. Pheromone‑driven proportional distribution for the non‑deterministic portion.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")

    # 1. Compute similarity reward
    ssim_score = compute_ssim(x_vec, y_vec)

    # 2. Day factor (today in UTC)
    today = datetime.now(timezone.utc).date()
    day_idx = HybridPheromoneKrampusSystem().doomsday(today.year, today.month, today.day)
    day_factor = day_idx / 6.0  # normalise to [0,1]

    # 3. Extract deterministic features from the provided text
    system = HybridPheromoneKrampusSystem()
    features = system.extract_full_features(text)

    # 4. Update pheromones with the fused equation
    system.update_pheromones(
        ssim_score=ssim_score,
        day_factor=day_factor,
        feature_factors=features,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    pheromone_weights = system.get_normalised_weights()

    # Deterministic allocation (as in Parent A)
    deterministic_units = total_units * deterministic_target_pct / 100.0
    remaining_units = total_units - deterministic_units

    # Proportional allocation according to pheromone weights
    allocation: dict[str, float] = {
        "total_units": _pct(total_units),
        "deterministic_units": _pct(deterministic_units),
        "remaining_units": _pct(remaining_units),
    }

    for g in GROUPS:
        group_share = remaining_units * pheromone_weights[g]
        allocation[g] = _pct(group_share)

    return allocation


def route_packet(
    packet: np.ndarray,
    prototypes: dict[str, np.ndarray],
    text: str,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> str:
    """
    Route a packet to the best matching group.

    For each group we compute:
        score_g = SSIM(packet, prototype_g) * pheromone_weight_g

    The pheromone weights are refreshed using the same bridge as in
    `hybrid_allocate`, ensuring that routing decisions are influenced by the
    current day and deterministic text features.
    """
    system = HybridPheromoneKrampusSystem()
    # Day factor
    today = datetime.now(timezone.utc).date()
    day_idx = system.doomsday(today.year, today.month, today.day)
    day_factor = day_idx / 6.0

    # Features from text
    features = system.extract_full_features(text)

    # Compute a temporary SSIM for each prototype
    ssim_scores = {
        g: compute_ssim(packet, proto) for g, proto in prototypes.items() if proto.shape == packet.shape
    }

    # Update pheromones using the *average* SSIM across all prototypes as a global reward
    avg_ssim = float(np.mean(list(ssim_scores.values()))) if ssim_scores else 0.0
    system.update_pheromones(
        ssim_score=avg_ssim,
        day_factor=day_factor,
        feature_factors=features,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    pher_weights = system.get_normalised_weights()

    # Final routing score per group
    routing_scores = {
        g: ssim_scores.get(g, -1.0) * pher_weights.get(g, 0.0) for g in GROUPS
    }

    # Choose the group with the highest score
    chosen_group = max(routing_scores, key=routing_scores.get)
    return chosen_group


def simulate_step(
    total_units: float,
    text: str,
    x_vec: np.ndarray,
    y_vec: np.ndarray,
    prototypes: dict[str, np.ndarray],
) -> dict[str, any]:
    """
    Perform a full simulation step:
    1. Allocate workshare using `hybrid_allocate`.
    2. Generate a synthetic packet (average of x and y) and route it.
    3. Return a summary dictionary.
    """
    allocation = hybrid_allocate(total_units, text, x_vec, y_vec)

    # Synthetic packet: element‑wise mean
    packet = (x_vec + y_vec) / 2.0

    routed_group = route_packet(packet, prototypes, text)

    summary = {
        "allocation": allocation,
        "routed_group": routed_group,
        "ssim": compute_ssim(x_vec, y_vec),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return summary


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple deterministic vectors
    rng = np.random.default_rng(42)
    x = rng.random(100)
    y = rng.random(100)

    # Prototypes for each group (slightly perturbed versions of x)
    prototypes = {
        g: x + rng.normal(scale=0.05, size=x.shape) for g in GROUPS
    }

    result = simulate_step(
        total_units=1000.0,
        text="The quick brown fox jumps over the lazy dog.",
        x_vec=x,
        y_vec=y,
        prototypes=prototypes,
    )

    print("Simulation summary:")
    for key, value in result.items():
        print(f"{key}: {value}")