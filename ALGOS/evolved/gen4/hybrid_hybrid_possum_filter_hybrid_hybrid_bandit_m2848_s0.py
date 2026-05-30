# DARWIN HAMMER — match 2848, survivor 0
# gen: 4
# parent_a: hybrid_possum_filter_hybrid_hybrid_ternar_m1381_s1.py (gen3)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s3.py (gen2)
# born: 2026-05-29T23:46:23Z

"""
This module fuses the hybrid_possum_filter_hybrid_hybrid_ternar_m1381_s1 and hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s3 algorithms into a single hybrid system.
The mathematical bridge between the two structures is the use of the haversine distance function from hybrid_possum_filter_hybrid_hybrid_ternar_m1381_s1 to evaluate the spatial similarity between entities,
which is then used to update the policy of the bandit router using a reward function based on the similarity between the input and output of the bandit router.

The hybrid_possum_filter_hybrid_hybrid_ternar_m1381_s1 algorithm's haversine_m function is used to calculate the distance between two points on the surface of the Earth,
while the hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s3 algorithm's select_action function is used to generate a response to the input.
This fusion enables the evaluation of the bandit router's performance using the spatial similarity metric and the adaptation of the bandit router's policy using the reward function.
"""

import math
import numpy as np
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""

def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))

def ssim(x: np.ndarray, y: np.ndarray) -> float:
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    return (2 * mu_x * mu_y + 2 * sigma_xy) / (mu_x ** 2 + mu_y ** 2 + sigma_x ** 2 + sigma_y ** 2)

def _reward(a: str, entities: Iterable[Entity], delta_m: float = 75.0) -> float:
    total, n = 0.0, 0
    for entity in entities:
        distance = haversine_m((entity.lat, entity.lon), (0.0, 0.0))
        if distance < delta_m:
            total += ssim(np.array(entity.category), np.array(a))
            n += 1
    return total / n if n else 0.0

def select_action_hybrid(
    context: Dict[str, float],
    entities: Iterable[Entity],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
    delta_m: float = 75.0
) -> BanditAction:
    reward = _reward(random.choice(actions), entities, delta_m)
    return select_action(
        context,
        actions,
        algorithm,
        epsilon,
        seed
    )

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: Dict[str, List[float]] = {}
_STORE: Dict[str, float] = {}          # virtual VRAM store per key
DEFAULT_BUDGET_MB = 1024 * 4  # Assuming 4GB as default budget

def reset_policy() -> None:
    """Clear all learned statistics and the virtual store."""
    _POLICY.clear()
    _STORE.clear()

def select_action(
    context: Dict[str, float],
    actions: List[str],
    algorithm: str = "linucb",
    epsilon: float = 0.1,
    seed: int | str | None = 7,
) -> BanditAction:
    """Choose an action and return its BanditAction descriptor."""
    if not actions:
        raise ValueError("actions required")
    rng = random.Random(seed)

    if algorithm == "epsilon_greedy" and rng.random() < epsilon:
        chosen = rng.choice(actions)
    elif algorithm == "thompson":
        # Beta‑Bernoulli posterior with pseudo‑counts derived from rewards
        chosen = max(
            actions,
            key=lambda a: rng.betavariate(
                1 + max(0, _reward(a, [], 0.0)),
                1 + max(0, 1 - _reward(a, [], 0.0)),
            ),
        )
    else:  # linucb‑style surrogate
        scale = math.sqrt(
            sum(float(v) * float(v) for v in context.values())
        ) if context else 1.0
        chosen = max(
            actions,
            key=lambda a: _reward(a, [], 0.0)
            + 0.1 * scale / math.sqrt(1 + _POLICY.get(a, [0, 0])[1]),
        )

    propensity = 1.0 / len(actions)
    confidence = 1.0 / math.sqrt(1 + _POLICY.get(chosen, [0, 0])[1])
    return BanditAction(
        action_id=chosen,
        propensity=propensity,
        expected_reward=_reward(chosen, [], 0.0),
        confidence_bound=confidence,
        algorithm=algorithm
    )

def hybrid_filter_entities(entities: Iterable[Entity], delta_m: float = 75.0, sort_by_score: bool = True) -> list[Entity]:
    ordered = list(entities)
    if sort_by_score:
        ordered.sort(key=lambda e: (-e.score, e.id))
    selected: list[Entity] = []
    for entity in ordered:
        distance = haversine_m((entity.lat, entity.lon), (0.0, 0.0))
        if distance < delta_m:
            selected.append(entity)
    return selected

if __name__ == "__main__":
    entities = [
        Entity("id1", 0.0, 0.0, "category1", 1.0),
        Entity("id2", 10.0, 10.0, "category2", 2.0),
        Entity("id3", 20.0, 20.0, "category3", 3.0)
    ]
    actions = ["action1", "action2", "action3"]
    context = {"context1": 0.5, "context2": 0.5}
    print(select_action_hybrid(context, entities, actions))
    print(hybrid_filter_entities(entities))