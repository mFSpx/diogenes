# DARWIN HAMMER — match 5127, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_rete_bandit_g_hybrid_hybrid_hybrid_m1966_s1.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1078_s0.py (gen5)
# born: 2026-05-30T00:00:07Z

import sys
import math
import random
from datetime import date, datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np

@dataclass
class StoreState:
    """Dynamic store representing honey‑bee “dance” pheromone signal."""
    level: float = 0.0
    alpha: float = 1.0   # inflow coefficient
    beta: float = 1.0    # outflow coefficient
    dt: float = 1.0
    base: float = 1.0   # baseline dance intensity
    gain: float = 1.0   # gain applied to delta
    limit: float = 10.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._store_last_delta(delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        delta = getattr(self, "_last_delta", 0.0)
        return max(0.0, min(self.limit, self.base + self.gain * delta))

    def _store_last_delta(self, delta: float) -> None:
        self._last_delta = delta


class Multivector:
    def __init__(self, components: Dict[str, float], n: int):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k: int) -> 'Multivector':
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def vector_magnitude(self) -> float:
        vec = self.grade(1).components
        return math.sqrt(sum(v * v for v in vec.values()))

    def __repr__(self) -> str:
        return f"Multivector({self.components}, n={self.n})"


def _pct(value: float) -> float:
    return round(float(value), 6)


def day_of_week_likelihood(day: int) -> float:
    angle = (day - 2) * (2 * math.pi / 7)
    return 1.0 + 0.5 * math.cos(angle)


def beta_posterior(alpha: float, beta: float, successes: int, failures: int) -> Tuple[float, float]:
    return alpha + successes, beta + failures


def compute_pheromone_weights(store: StoreState, groups: Tuple[str, ...]) -> Dict[str, float]:
    components = {"s": store.level}
    for i, grp in enumerate(groups, start=1):
        components[f"e{i}"] = store.dance * (i / len(groups))

    mv = Multivector(components, n=len(groups) + 1)
    magnitude = mv.vector_magnitude()
    if magnitude == 0:
        return {g: 1.0 / len(groups) for g in groups}

    raw_weights = {}
    for i, grp in enumerate(groups, start=1):
        coeff = mv.components.get(f"e{i}", 0.0)
        raw_weights[grp] = max(0.0, coeff)  

    total = sum(raw_weights.values())
    if total == 0:
        return {g: 1.0 / len(groups) for g in groups}
    return {g: _pct(w / total) for g, w in raw_weights.items()}


def bayesian_llm_share(total_units: float,
                       deterministic_target_pct: float,
                       day: int,
                       prior_alpha: float = 2.0,
                       prior_beta: float = 2.0) -> Tuple[float, float]:
    prior_mean = prior_alpha / (prior_alpha + prior_beta)

    likelihood = day_of_week_likelihood(day)  
    pseudo_success = int(np.round(likelihood * 10))  
    pseudo_failure = int(np.round((1.0 - (likelihood - 0.5) / 1.0) * 10))

    post_alpha, post_beta = beta_posterior(prior_alpha, prior_beta,
                                            pseudo_success,
                                            pseudo_failure)
    posterior_mean = post_alpha / (post_alpha + post_beta)

    llm_share_pct = (100.0 - deterministic_target_pct) * posterior_mean / prior_mean
    llm_share_pct = max(0.0, min(100.0, llm_share_pct))

    deterministic_units = _pct(total_units * deterministic_target_pct / 100.0)
    llm_units = _pct(total_units * llm_share_pct / 100.0)
    return deterministic_units, llm_units


def hybrid_allocate_workshare(total_units: float,
                              deterministic_target_pct: float = 90.0,
                              groups: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models"),
                              day: int = 0) -> Dict[str, Tuple[float, float]]:
    deterministic_units, llm_units = bayesian_llm_share(total_units, deterministic_target_pct, day)
    pheromone_weights = compute_pheromone_weights(StoreState(), groups)

    allocation = {}
    for group in groups:
        group_deterministic_units = _pct(deterministic_units * pheromone_weights[group])
        group_llm_units = _pct(llm_units * pheromone_weights[group])
        allocation[group] = (group_deterministic_units, group_llm_units)

    return allocation


if __name__ == "__main__":
    total_units = 100.0
    deterministic_target_pct = 90.0
    groups = ("codex", "groq", "cohere", "local_models")
    day = 0
    allocation = hybrid_allocate_workshare(total_units, deterministic_target_pct, groups, day)
    for group, units in allocation.items():
        print(f"Group: {group}, Deterministic Units: {units[0]}, LLM Units: {units[1]}")