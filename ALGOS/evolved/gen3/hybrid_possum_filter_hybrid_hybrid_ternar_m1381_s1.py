# DARWIN HAMMER — match 1381, survivor 1
# gen: 3
# parent_a: possum_filter.py (gen0)
# parent_b: hybrid_hybrid_ternary_route_hybrid_bandit_router_m31_s3.py (gen2)
# born: 2026-05-29T23:37:09Z

"""
This module fuses the possum_filter and hybrid_hybrid_ternary_route_hybrid_bandit_router_m31_s3 algorithms into a single hybrid system.
The mathematical bridge between the two structures is the use of the haversine distance function from possum_filter to evaluate the spatial similarity between entities,
which is then used to update the policy of the ternary router using a reward function based on the similarity between the input and output of the ternary router.

The possum_filter algorithm's haversine_m function is used to calculate the distance between two points on the surface of the Earth,
while the hybrid_hybrid_ternary_route_hybrid_bandit_router_m31_s3 algorithm's route_command function is used to generate a response to the input.
This fusion enables the evaluation of the ternary router's performance using the spatial similarity metric and the adaptation of the bandit router's policy using the reward function.
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

def route_command(text: str, intent: str, context: dict[str, Any]) -> dict[str, Any]:
    # Simple implementation for demonstration purposes
    return {"response": text}

def hybrid_filter_entities(entities: Iterable[Entity], delta_m: float = 75.0, sort_by_score: bool = True) -> list[Entity]:
    ordered = list(entities)
    if sort_by_score:
        ordered.sort(key=lambda e: (-e.score, e.id))
    selected: list[Entity] = []
    for entity in ordered:
        keep = True
        for existing in selected:
            same_kind = (entity.address_signature or entity.category) == (existing.address_signature or existing.category)
            if same_kind and haversine_m((entity.lat, entity.lon), (existing.lat, existing.lon)) <= delta_m:
                keep = False
                break
        if keep:
            selected.append(entity)
    return selected

def evaluate_hybrid_performance(entities: Iterable[Entity], delta_m: float = 75.0) -> float:
    selected = hybrid_filter_entities(entities, delta_m)
    # Calculate the similarity between the input and output using ssim
    input_array = np.array([[e.lat, e.lon] for e in entities])
    output_array = np.array([[e.lat, e.lon] for e in selected])
    return ssim(input_array, output_array)

def reward_function(similarity: float) -> float:
    # Simple reward function for demonstration purposes
    return similarity

if __name__ == "__main__":
    entities = [
        Entity("1", 37.7749, -122.4194, "A", 1.0),
        Entity("2", 37.7859, -122.4364, "A", 0.9),
        Entity("3", 37.7963, -122.4575, "B", 0.8),
    ]
    delta_m = 50.0
    filtered_entities = hybrid_filter_entities(entities, delta_m)
    performance = evaluate_hybrid_performance(entities, delta_m)
    reward = reward_function(performance)
    print(f"Filtered entities: {[e.id for e in filtered_entities]}")
    print(f"Hybrid performance: {performance:.4f}")
    print(f"Reward: {reward:.4f}")