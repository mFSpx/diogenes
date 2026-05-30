# DARWIN HAMMER — match 626, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_regret_m236_s1.py (gen4)
# parent_b: hybrid_sparse_wta_hybrid_privacy_model_m29_s2.py (gen2)
# born: 2026-05-29T23:30:13Z

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

PROTOTYPE_VECTOR = np.array([0.2, 0.5, 0.3, 0.7, 0.1], dtype=np.float64)

def compute_ssim(
    x: List[float],
    y: List[float],
    dynamic_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    if len(x) != len(y):
        raise ValueError("samples must have equal length")
    if not x:
        raise ValueError("samples must not be empty")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    mx = x_arr.mean()
    my = y_arr.mean()
    vx = ((x_arr - mx) ** 2).mean()
    vy = ((y_arr - my) ** 2).mean()
    cov = ((x_arr - mx) * (y_arr - my)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(numerator / denominator)


def expand(values: List[float], m: int, salt: str = "") -> List[float]:
    if m <= 0:
        raise ValueError("m must be positive")
    out = [0.0] * m
    for i, v in enumerate(values):
        for r in range(3):  
            h = hashlib.sha256(f"{salt}:{i}:{r}".encode()).digest()
            j = int.from_bytes(h[:8], "big") % m
            sign = 1.0 if h[8] & 1 else -1.0
            out[j] += sign * v
    return out


def top_k_mask(values: List[float], k: int) -> List[int]:
    k = max(0, min(k, len(values)))
    winners = {
        i for i, _ in sorted(enumerate(values), key=lambda x: (-x[1], x[0]))[:k]
    }
    return [1 if i in winners else 0 for i in range(len(values))]


def hamming(a: List[int], b: List[int]) -> int:
    if len(a) != len(b):
        raise ValueError("vectors must be the same length")
    return sum(ai != bi for ai, bi in zip(a, b))


def add_laplace_noise(value: float, scale: float) -> float:
    if scale <= 0:
        return value  
    noise = np.random.laplace(loc=0.0, scale=scale)
    return float(value + noise)


def compute_risk(unique_quasi_identifiers: int, total_records: int) -> float:
    if total_records <= 0:
        raise ValueError("total_records must be positive")
    return float(unique_quasi_identifiers) / float(total_records)


def hybrid_expand_ssim(
    payload: List[float],
    m: int = 128,
    salt: str = "",
) -> float:
    expanded_payload = expand(payload, m, salt)
    expanded_proto = expand(PROTOTYPE_VECTOR.tolist(), m, salt)
    return compute_ssim(expanded_payload, expanded_proto)


class RegretMatcher:
    def __init__(self, actions: List[str], epsilon: float = 1.0):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.actions = actions
        self.epsilon = epsilon
        self.regrets: Dict[str, float] = {a: 0.0 for a in actions}
        self._last_utilities: Dict[str, float] = {a: 0.0 for a in actions}

    def _positive_regrets(self) -> List[float]:
        return [max(r, 0.0) for r in self.regrets.values()]

    def get_strategy(self) -> Dict[str, float]:
        pos = self._positive_regrets()
        total = sum(pos)
        if total == 0.0:
            prob = 1.0 / len(self.actions)
            return {a: prob for a in self.actions}
        return {a: r / total for a, r in zip(self.actions, pos)}

    def sample_action(self) -> str:
        strategy = self.get_strategy()
        rnd = random.random()
        cumulative = 0.0
        for a, p in strategy.items():
            cumulative += p
            if rnd <= cumulative:
                return a
        return self.actions[-1]

    def update_regrets(
        self,
        chosen_action: str,
        utility: float,
        risk_scale: float,
    ) -> None:
        for a in self.actions:
            self._last_utilities[a] = utility

        avg_utility = sum(self._last_utilities.values()) / len(self.actions)

        for a in self.actions:
            regret = self._last_utilities[a] - avg_utility
            scale = risk_scale / self.epsilon
            noisy_regret = add_laplace_noise(regret, scale)
            self.regrets[a] += noisy_regret


def improved_hybrid_expand_ssim(
    payload: List[float],
    m: int = 128,
    salt: str = "",
    k: int = 3,
) -> float:
    expanded_payload = expand(payload, m, salt)
    expanded_proto = expand(PROTOTYPE_VECTOR.tolist(), m, salt)
    mask = top_k_mask(expanded_payload, k)
    masked_payload = [x * y for x, y in zip(expanded_payload, mask)]
    masked_proto = [x * y for x, y in zip(expanded_proto, mask)]
    return compute_ssim(masked_payload, masked_proto)


class ImprovedRegretMatcher(RegretMatcher):
    def __init__(self, actions: List[str], epsilon: float = 1.0, k: int = 3):
        super().__init__(actions, epsilon)
        self.k = k

    def update_regrets(
        self,
        chosen_action: str,
        payload: List[float],
        risk_scale: float,
    ) -> None:
        utility = improved_hybrid_expand_ssim(payload, k=self.k)
        super().update_regrets(chosen_action, utility, risk_scale)