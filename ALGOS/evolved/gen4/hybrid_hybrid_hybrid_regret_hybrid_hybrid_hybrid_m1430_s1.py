# DARWIN HAMMER — match 1430, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s4.py (gen3)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_ternar_m374_s4.py (gen3)
# born: 2026-05-29T23:37:48Z

import sys
import pathlib
import math
import random
import hashlib
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Dict
import numpy as np

# ----------------------------------------------------------------------
# Data structures (union of both parents)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class MathAction:
    """Action definition used by the regret‑weighted component."""
    id: str
    tokens: Tuple[str, ...]          # token set for MinHash
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection performed by the bandit."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretBandit"


@dataclass(frozen=True)
class BanditUpdate:
    """Observation used to update the policy (not used directly in the demo)."""
    context_id: str
    action_id: str
    reward: float

# ----------------------------------------------------------------------
# HybridPheromoneBanditSystem with MinHash similarity
# ----------------------------------------------------------------------

class HybridPheromoneBanditSystem:
    """
    A tighter integration of a pheromone‑based decay model with a
    multi‑armed bandit (UCB1) algorithm and a regret‑weighted strategy
    with MinHash.

    * Pheromone signals decay exponentially according to a half‑life.
    * The decayed pheromone value is used as a *prior* for the
      expected reward of each arm, biasing exploration toward arms that
      have recently received strong pheromone cues.
    * Rewards are updated online, and the UCB confidence term is
      combined with the pheromone prior to compute a hybrid score.
    * The system also provides a principled entropy estimator for
      privacy‑risk calculations and a similarity metric based on MinHash.
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

        # MinHash similarity
        self.minhash_similarities = {}

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
        Insert or update a pheromone signal.
        """
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'created': self._current_utc(), 'half_life': half_life_seconds}
        else:
            self.pheromones[surface_key]['created'] = self._current_utc()
            self.pheromones[surface_key]['signal_value'] = signal_value
            self.pheromones[surface_key]['half_life'] = half_life_seconds
        return self._decayed_signal(self.pheromones[surface_key]['created'], self.pheromones[surface_key]['signal_value'], self.pheromones[surface_key]['half_life'])

    # --------------------------------------------------------------------- #
    # Bandit handling
    # --------------------------------------------------------------------- #
    def _ucb_score(self, arm_index: int) -> float:
        """
        Compute the UCB score for the given arm.
        """
        if self.counts[arm_index] == 0:
            return float('-inf')
        return self.values[arm_index] + np.sqrt(2 * np.log(self.total_pulls) / self.counts[arm_index])

    def _pheromone_score(self, arm_index: int) -> float:
        """
        Compute the pheromone score for the given arm.
        """
        if arm_index not in self.pheromones:
            return 0.0
        return self.pheromones[arm_index]['signal_value']

    def _minhash_score(self, action: MathAction) -> float:
        """
        Compute the MinHash similarity score for the given action.
        """
        # Create a hash object
        hash_object = hashlib.sha256()

        # Update the hash object with the action's tokens
        for token in action.tokens:
            hash_object.update(token.encode('utf-8'))

        # Get the hash value
        hash_value = int(hash_object.hexdigest(), 16)

        # Check if the action's hash value is in the MinHash similarities dictionary
        if hash_value in self.minhash_similarities:
            # If it is, return the corresponding similarity score
            return self.minhash_similarities[hash_value]
        else:
            # If it's not, return a default value (e.g., 0.0)
            return 0.0

    def _hybrid_score(self, arm_index: int, action: MathAction) -> float:
        """
        Compute the hybrid score for the given arm and action.
        """
        # Compute the UCB score for the given arm
        ucb_score = self._ucb_score(arm_index)

        # Compute the pheromone score for the given arm
        pheromone_score = self._pheromone_score(arm_index)

        # Compute the MinHash similarity score for the given action
        minhash_score = self._minhash_score(action)

        # Combine the scores using a weighted average
        hybrid_score = 0.5 * ucb_score + 0.3 * pheromone_score + 0.2 * minhash_score

        return hybrid_score

    def select_arm(self, action: MathAction) -> Tuple[int, float]:
        """
        Select the arm with the highest hybrid score.
        """
        # Compute the hybrid score for each arm
        hybrid_scores = [self._hybrid_score(i, action) for i in range(self.n_arms)]

        # Get the arm index with the highest hybrid score
        best_arm_index = np.argmax(hybrid_scores)

        # Return the best arm index and its corresponding hybrid score
        return best_arm_index, hybrid_scores[best_arm_index]

    def update_policy(self, reward: float, arm_index: int):
        """
        Update the policy using the reward and arm index.
        """
        # Increment the count for the selected arm
        self.counts[arm_index] += 1

        # Update the average reward for the selected arm
        self.values[arm_index] = (self.values[arm_index] * (self.counts[arm_index] - 1) + reward) / self.counts[arm_index]

        # Update the total pulls
        self.total_pulls += 1

        # Update the store
        self.store += reward

    def add_minhash_similarity(self, action: MathAction, similarity: float):
        """
        Add a MinHash similarity to the dictionary.
        """
        # Create a hash object
        hash_object = hashlib.sha256()

        # Update the hash object with the action's tokens
        for token in action.tokens:
            hash_object.update(token.encode('utf-8'))

        # Get the hash value
        hash_value = int(hash_object.hexdigest(), 16)

        # Add the similarity to the dictionary
        self.minhash_similarities[hash_value] = similarity

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

def main():
    # Create an instance of the HybridPheromoneBanditSystem
    system = HybridPheromoneBanditSystem(n_arms=5)

    # Define some actions
    actions = [
        MathAction(id='action1', tokens=('token1', 'token2', 'token3'), expected_value=1.0),
        MathAction(id='action2', tokens=('token4', 'token5', 'token6'), expected_value=2.0),
        MathAction(id='action3', tokens=('token7', 'token8', 'token9'), expected_value=3.0),
    ]

    # Select an arm for each action
    for action in actions:
        arm_index, hybrid_score = system.select_arm(action)
        print(f"Selected arm for action {action.id}: {arm_index} (hybrid score: {hybrid_score})")

    # Update the policy
    system.update_policy(reward=1.0, arm_index=0)

    # Add a MinHash similarity
    system.add_minhash_similarity(actions[0], similarity=0.5)

if __name__ == "__main__":
    main()