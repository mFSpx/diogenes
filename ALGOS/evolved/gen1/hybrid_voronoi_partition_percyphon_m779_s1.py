# DARWIN HAMMER — match 779, survivor 1
# gen: 1
# parent_a: voronoi_partition.py (gen0)
# parent_b: percyphon.py (gen0)
# born: 2026-05-29T23:30:49Z

#!/usr/bin/env python3
"""Hybrid algorithm combining Voronoi partitioning and procedural entity generation.
The mathematical bridge between the two parent algorithms, voronoi_partition.py and percyphon.py,
lies in the use of spatial partitioning to generate procedural entities with unique properties.
The Voronoi diagram is used to partition the space, and the procedural entity generator is used
to assign unique properties to each partition."""

import math
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence
import numpy as np
import random
import sys
import pathlib

Point = tuple[float, float]

@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest(point: Point, seeds: list[Point]) -> int:
    if not seeds:
        raise ValueError('seeds required')
    return min(range(len(seeds)), key=lambda i: (distance(point, seeds[i]), i))

def assign(points: list[Point], seeds: list[Point]) -> dict[int, list[Point]]:
    regions = {i: [] for i in range(len(seeds))}
    for p in points:
        regions[nearest(p, seeds)].append(p)
    return regions

def _uuid_from_sha256(seed: str) -> str:
    h = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

def _slot_name(seed: str, idx: int) -> tuple[str, str, str]:
    h = hashlib.sha256(f"{seed}:{idx}".encode()).hexdigest()
    name = f"Villager-{int(h[:6], 16) % 5000:04d}"
    alias = f"Alias-{h[6:10]}"
    persona = ["ledger", "runner", "witness", "archivist", "carrier", "scribe"][int(h[10:12], 16) % 6]
    return name, alias, persona

def procedural_entity_generator(
    villagers: Sequence[str] | None = None,
    *,
    psyche_wrath_velocity: float = 0.0,
    psyche_forensic_shield_ratio: float = 0.0,
    fluid_slots: int = 88,
) -> dict[str, Any]:
    villagers = list(villagers or [])
    seed = "|".join(villagers[:5000]) or "lucidota-villager-baseline"
    ternary_offset = 0
    if psyche_wrath_velocity > psyche_forensic_shield_ratio:
        ternary_offset = 1
    elif psyche_forensic_shield_ratio > psyche_wrath_velocity:
        ternary_offset = -1

    slots: list[ProceduralSlot] = []
    for idx in range(12):
        name, alias, persona = _slot_name(seed, idx)
        slots.append(
            ProceduralSlot(
                slot_index=idx,
                name=name,
                alias=alias,
                persona=persona,
                uuid=_uuid_from_sha256(f"{seed}:{idx}"),
                ternary_offset=ternary_offset,
            )
        )

    fluid: list[dict[str, Any]] = []
    for idx in range(int(fluid_slots)):
        fluid.append(
            {
                "fluid_slot": idx,
                "villager_ref": villagers[idx % len(villagers)] if villagers else f"baseline-{idx:04d}",
                "slot_uuid": _uuid_from_sha256(f"{seed}:fluid:{idx}"),
                "offset": ternary_offset,
            }
        )

    return {
        "schema": "lucidota.percyphon.procedural_entity_generator.v1",
        "source_count": min(5000, len(villagers) or 5000),
        "slot_count": 12,
        "fluid_slots": fluid,
        "slots": [slot.as_dict() for slot in slots],
    }

def hybrid_voronoi_procedural_entity_generator(
    points: list[Point],
    seeds: list[Point],
    villagers: Sequence[str] | None = None,
    *,
    psyche_wrath_velocity: float = 0.0,
    psyche_forensic_shield_ratio: float = 0.0,
    fluid_slots: int = 88,
) -> dict[str, Any]:
    regions = assign(points, seeds)
    village_names = []
    for region in regions.values():
        seed = "|".join([f"{point[0]}:{point[1]}" for point in region])
        village_name = _slot_name(seed, 0)[0]
        village_names.append(village_name)

    return procedural_entity_generator(
        villagers=village_names,
        psyche_wrath_velocity=psyche_wrath_velocity,
        psyche_forensic_shield_ratio=psyche_forensic_shield_ratio,
        fluid_slots=fluid_slots,
    )

def generate_random_points(num_points: int, bounds: tuple[float, float, float, float]) -> list[Point]:
    points = []
    for _ in range(num_points):
        x = random.uniform(bounds[0], bounds[1])
        y = random.uniform(bounds[2], bounds[3])
        points.append((x, y))
    return points

def generate_random_seeds(num_seeds: int, bounds: tuple[float, float, float, float]) -> list[Point]:
    seeds = []
    for _ in range(num_seeds):
        x = random.uniform(bounds[0], bounds[1])
        y = random.uniform(bounds[2], bounds[3])
        seeds.append((x, y))
    return seeds

if __name__ == "__main__":
    points = generate_random_points(100, (-10, 10, -10, 10))
    seeds = generate_random_seeds(5, (-10, 10, -10, 10))
    result = hybrid_voronoi_procedural_entity_generator(points, seeds)
    print(json.dumps(result, indent=4))