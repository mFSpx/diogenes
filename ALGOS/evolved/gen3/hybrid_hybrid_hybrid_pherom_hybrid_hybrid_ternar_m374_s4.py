# DARWIN HAMMER — match 374, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_pheromone_inf_privacy_m54_s1.py (gen2)
# parent_b: hybrid_hybrid_ternary_route_hybrid_bandit_router_m56_s2.py (gen2)
# born: 2026-05-29T23:28:35Z

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np


class HybridPheromoneBanditSystem:
    """
    A tighter integration of a pheromone‑based decay model with a
    multi‑armed bandit (UCB1) algorithm.

    * Pheromone signals decay exponentially according to a half‑life.
    * The decayed pheromone value is used as a *prior* for the
      expected reward of each arm, biasing exploration toward arms that
      have recently received strong pheromone cues.
    * Rewards are updated online, and the UCB confidence term is
      combined with the pheromone prior to compute a hybrid score.
    * The system also provides a principled entropy estimator for
      privacy‑risk calculations.
    """

    def __init__(self, n_arms: int = 5):
        self.n_arms = n_arms

        # Pheromone store: surface_key → dict with decay parameters
        self.pheromones: Dict[str, Dict[str, Any]] = {}

        # Bandit statistics
        self.counts = np.zeros(n_arms, dtype=int)          # pulls per arm
        self.values = np.zeros(n_arms, dtype=float)        # average reward per arm

        # Global statistics for scaling
        self.total_pulls = 0
        self.store = 0.0                                     # cumulative pheromone mass

    # --------------------------------------------------------------------- #
    # Pheromone handling
    # --------------------------------------------------------------------- #
    def _current_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _decayed_signal(self, created: datetime, value: float, half_life: float) -> float:
        """
        Return the exponentially decayed signal value.
        """
        if half_life <= 0:
            raise ValueError("half_life_seconds must be positive")
        elapsed = (self._current_utc() - created).total_seconds()
        decay_factor = 0.5 ** (elapsed / half_life)
        return value * decay_factor

    def update_pheromone(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        """
        Insert or update a pheromone entry.
        Returns the *current* decayed value after the update.
        """
        now = self._current_utc()
        if surface_key in self.pheromones:
            entry = self.pheromones[surface_key]
            # decay the previous value before overwriting
            decayed = self._decayed_signal(entry["created_time"], entry["signal_value"], entry["half_life_seconds"])
            self.store -= entry["signal_value"]  # remove old raw mass
            entry.update(
                {
                    "signal_kind": signal_kind,
                    "signal_value": signal_value,
                    "half_life_seconds": half_life_seconds,
                    "created_time": now,
                }
            )
        else:
            entry = {
                "signal_kind": signal_kind,
                "signal_value": signal_value,
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            }
            self.pheromones[surface_key] = entry

        self.store += signal_value
        return signal_value

    def get_pheromone(self, surface_key: str) -> float:
        """
        Retrieve the decayed pheromone value for a given key.
        If the key does not exist, returns 0.0.
        """
        entry = self.pheromones.get(surface_key)
        if not entry:
            return 0.0
        return self._decayed_signal(
            entry["created_time"], entry["signal_value"], entry["half_life_seconds"]
        )

    # --------------------------------------------------------------------- #
    # Information‑theoretic utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def calculate_entropy(probabilities: List[float], eps: float = 1e-12) -> float:
        """
        Shannon entropy of a discrete distribution.
        """
        total = sum(probabilities)
        if total <= 0:
            raise ValueError("positive probability mass required")
        probs = [p / total for p in probabilities if p > 0]
        return -sum(p * math.log(max(p, eps)) for p in probs)

    @staticmethod
    def expected_entropy(p_hit: float, hit_state: List[float], miss_state: List[float]) -> float:
        """
        Expected entropy after a binary observation.
        """
        if not 0.0 <= p_hit <= 1.0:
            raise ValueError("p_hit must be in [0, 1]")
        return (
            p_hit * HybridPheromoneBanditSystem.calculate_entropy(hit_state)
            + (1.0 - p_hit) * HybridPheromoneBanditSystem.calculate_entropy(miss_state)
        )

    # --------------------------------------------------------------------- #
    # Structural similarity (SSIM) – corrected formulation
    # --------------------------------------------------------------------- #
    @staticmethod
    def compute_ssim(
        x: List[float],
        y: List[float],
        dynamic_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> float:
        """
        Compute the Structural Similarity Index (SSIM) for 1‑D signals.
        The implementation follows the standard definition with
        constants C1 = (k1*L)^2 and C2 = (k2*L)^2.
        """
        if len(x) != len(y):
            raise ValueError("samples must have equal length")
        if not x:
            raise ValueError("samples must not be empty")

        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        mu_x = np.mean(x_arr)
        mu_y = np.mean(y_arr)
        sigma_x = np.var(x_arr, ddof=0)
        sigma_y = np.var(y_arr, ddof=0)
        sigma_xy = np.mean((x_arr - mu_x) * (y_arr - mu_y))

        c1 = (k1 * dynamic_range) ** 2
        c2 = (k2 * dynamic_range) ** 2

        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)

        return float(numerator / denominator)

    # --------------------------------------------------------------------- #
    # Bandit logic with pheromone‑driven prior
    # --------------------------------------------------------------------- #
    def _ucb_score(self, arm: int, total_pulls: int, pheromone_prior: float) -> float:
        """
        Upper Confidence Bound (UCB1) score augmented with a pheromone prior.
        The prior is added linearly to the exploitation term.
        """
        if self.counts[arm] == 0:
            # Force initial exploration
            return float("inf")
        exploitation = self.values[arm] + pheromone_prior
        confidence = math.sqrt((2 * math.log(total_pulls)) / self.counts[arm])
        return exploitation + confidence

    def select_arm(self, surface_key: str, x: List[float], y: List[float]) -> int:
        """
        Choose an arm using the hybrid UCB‑pheromone criterion.
        """
        pheromone = self.get_pheromone(surface_key)
        ssim = self.compute_ssim(x, y)

        # Map SSIM ([-1, 1]) to a non‑negative scaling factor.
        # Values close to 1 increase the influence of pheromone.
        ssim_factor = 0.5 + 0.5 * max(ssim, 0.0)

        # The final prior blends raw pheromone mass with similarity.
        prior = pheromone * ssim_factor

        total = max(1, self.total_pulls)  # avoid division by zero
        scores = [
            self._ucb_score(arm, total, prior) for arm in range(self.n_arms)
        ]
        chosen = int(np.argmax(scores))
        return chosen

    def update(self, arm: int, reward: float) -> None:
        """
        Incremental update of the bandit statistics after pulling an arm.
        """
        self.counts[arm] += 1
        self.total_pulls += 1
        n = self.counts[arm]
        # incremental mean update
        self.values[arm] += (reward - self.values[arm]) / n

    # --------------------------------------------------------------------- #
    # Public API – one step of the hybrid system
    # --------------------------------------------------------------------- #
    def hybrid_step(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
        x: List[float],
        y: List[float],
    ) -> Tuple[int, float]:
        """
        Perform a single iteration:
        1. Update pheromone store.
        2. Choose an arm using the hybrid criterion.
        3. Simulate (or receive) a reward.
        4. Update bandit statistics.

        Returns the selected arm and the observed reward.
        """
        # 1. Pheromone update (the returned value is the raw, not decayed, signal)
        self.update_pheromone(surface_key, signal_kind, signal_value, half_life_seconds)

        # 2. Arm selection
        arm = self.select_arm(surface_key, x, y)

        # 3. Reward generation – in a real system this would come from the environment.
        #    Here we keep a stochastic placeholder that could be replaced.
        reward = random.random()

        # 4. Statistics update
        self.update(arm, reward)

        return arm, reward


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid pheromone‑bandit demo.")
    parser.add_argument("--steps", type=int, default=10, help="Number of iterations")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    system = HybridPheromoneBanditSystem(n_arms=5)

    # Example static inputs – in practice these would be dynamic per request.
    surface_key = "example_key"
    signal_kind = "example_kind"
    signal_value = 0.5
    half_life_seconds = 3600
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 4, 5, 6]

    for _ in range(args.steps):
        arm, reward = system.hybrid_step(
            surface_key,
            signal_kind,
            signal_value,
            half_life_seconds,
            x,
            y,
        )
        print(f"Selected arm: {arm}, reward: {reward:.4f}")


if __name__ == "__main__":
    main()