# DARWIN HAMMER — match 1430, survivor 3
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
from datetime import datetime, timezone

@dataclass(frozen=True)
class MathAction:
    id: str
    tokens: Tuple[str, ...]
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridRegretBandit"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float

class HybridPheromoneBanditSystem:
    def __init__(self, n_arms: int = 5, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        self.n_arms = n_arms
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pheromones = {}
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)
        self.total_pulls = 0
        self.store = 0.0
        self.minhash_similarities = {}

    def _current_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def _decayed_signal(self, created: datetime, value: float, half_life: float) -> float:
        if half_life <= 0:
            raise ValueError("half_life_seconds must be positive")
        elapsed = (self._current_utc() - created).total_seconds()
        decay_factor = 0.5 ** (elapsed / half_life)
        return value * decay_factor

    def update_pheromone(self, surface_key: str, signal_kind: str, signal_value: float, half_life_seconds: float) -> float:
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'created': self._current_utc(), 'half_life': half_life_seconds}
        else:
            self.pheromones[surface_key]['created'] = self._current_utc()
            self.pheromones[surface_key]['signal_value'] = signal_value
            self.pheromones[surface_key]['half_life'] = half_life_seconds
        return self._decayed_signal(self.pheromones[surface_key]['created'], self.pheromones[surface_key]['signal_value'], self.pheromones[surface_key]['half_life'])

    def _ucb_score(self, arm_index: int) -> float:
        if self.counts[arm_index] == 0:
            return float('-inf')
        return self.values[arm_index] + np.sqrt(2 * np.log(self.total_pulls) / self.counts[arm_index])

    def _pheromone_score(self, arm_index: int) -> float:
        if arm_index not in self.pheromones:
            return 0.0
        return self.pheromones[arm_index]['signal_value']

    def _minhash_score(self, action: MathAction) -> float:
        hash_object = hashlib.sha256()
        for token in action.tokens:
            hash_object.update(token.encode('utf-8'))
        hash_value = int(hash_object.hexdigest(), 16)
        return self.minhash_similarities.get(hash_value, 0.0)

    def _hybrid_score(self, arm_index: int, action: MathAction) -> float:
        ucb_score = self._ucb_score(arm_index)
        pheromone_score = self._pheromone_score(arm_index)
        minhash_score = self._minhash_score(action)
        return self.alpha * ucb_score + self.beta * pheromone_score + self.gamma * minhash_score

    def select_arm(self, action: MathAction) -> Tuple[int, float]:
        hybrid_scores = [self._hybrid_score(i, action) for i in range(self.n_arms)]
        best_arm_index = np.argmax(hybrid_scores)
        return best_arm_index, hybrid_scores[best_arm_index]

    def update_policy(self, reward: float, arm_index: int):
        self.counts[arm_index] += 1
        self.values[arm_index] = (self.values[arm_index] * (self.counts[arm_index] - 1) + reward) / self.counts[arm_index]
        self.total_pulls += 1
        self.store += reward

    def add_minhash_similarity(self, action: MathAction, similarity: float):
        hash_object = hashlib.sha256()
        for token in action.tokens:
            hash_object.update(token.encode('utf-8'))
        hash_value = int(hash_object.hexdigest(), 16)
        self.minhash_similarities[hash_value] = similarity

    def get_arm_pheromone(self, arm_index: int) -> float:
        if arm_index not in self.pheromones:
            return 0.0
        return self.pheromones[arm_index]['signal_value']

    def get_arm_ucb(self, arm_index: int) -> float:
        return self._ucb_score(arm_index)

    def get_arm_minhash(self, arm_index: int, action: MathAction) -> float:
        return self._minhash_score(action)