# DARWIN HAMMER — match 1346, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hoeffding_tre_hybrid_hybrid_pherom_m330_s0.py (gen4)
# parent_b: hybrid_voronoi_partition_percyphon_m779_s2.py (gen1)
# born: 2026-05-29T23:35:25Z

"""Hybrid Hoeffding‑Voronoi‑Pheromone Algorithm
Parents:
- hybrid_hybrid_hoeffding_tre_hybrid_hybrid_pherom_m330_s0.py (Hoeffding‑Gini tree with pheromone decay)
- hybrid_voronoi_partition_percyphon_m779_s2.py (Voronoi partition + procedural entity generation)

Mathematical Bridge:
The Voronoi diagram supplies a spatial partition of the feature space. Each Voronoi cell is treated as a
candidate split region for a Hoeffding tree. For a given cell we compute the Gini impurity of the
labels of the samples that fall inside it and obtain a gain ΔG. The Hoeffding bound  
ε = sqrt( (R² * ln(1/δ)) / (2n) )  (R = 1 for Gini) tells us when the observed gain gap is statistically
significant. A pheromone value τ_i is associated with each cell i; τ_i is decayed exponentially with a
half‑life and reinforced proportionally to the observed gain. The final split decision combines the
statistical certainty (ε), the impurity gain (ΔG) and the pheromone strength (τ_i) into a unified score
S_i = ΔG / (ε + 1e-9) * τ_i. The cell with the highest S_i triggers a tree split, while low‑score cells are
pruned via a probability derived from τ_i.

The module implements this hybrid topology and also provides a lightweight procedural‑entity generator
that uses the same Voronoi seeds to create unique slots."""
import math
import random
import sys
import pathlib
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
Point = Tuple[float, float]


@dataclass(frozen=True)
class SplitDecision:
    """Result of a Hoeffding‑Gini split test for a Voronoi cell."""
    should_split: bool
    cell_index: int
    epsilon: float
    gain_gap: float
    pheromone: float
    reason: str


@dataclass(frozen=True)
class ProceduralSlot:
    slot_index: int
    name: str
    alias: str
    persona: str
    uuid: str
    ternary_offset: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Pheromone system (adapted from parent A)
# ----------------------------------------------------------------------
class HybridPheromoneSystem:
    """Manages pheromone values per Voronoi cell with exponential decay."""

    def __init__(self, half_life_seconds: float = 30.0):
        self.half_life_seconds = half_life_seconds
        self._values: Dict[int, float] = {}
        self._timestamps: Dict[int, datetime.datetime] = {}

    def _now(self) -> datetime.datetime:
        return datetime.datetime.now(datetime.timezone.utc)

    def get(self, cell_id: int) -> float:
        """Return current pheromone value after decay."""
        if cell_id not in self._values:
            return 0.0
        elapsed = (self._now() - self._timestamps[cell_id]).total_seconds()
        decayed = self._values[cell_id] * 0.5 ** (elapsed / self.half_life_seconds)
        self._values[cell_id] = decayed
        self._timestamps[cell_id] = self._now()
        return decayed

    def reinforce(self, cell_id: int, amount: float) -> None:
        """Increase pheromone for a cell (after decay)."""
        current = self.get(cell_id)
        self._values[cell_id] = current + amount
        self._timestamps[cell_id] = self._now()


# ----------------------------------------------------------------------
# Voronoi utilities (adapted from parent B)
# ----------------------------------------------------------------------
def _distance_matrix(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Return a (n_points, n_seeds) matrix of Euclidean distances."""
    diff = points[:, None, :] - seeds[None, :, :]
    return np.linalg.norm(diff, axis=2)


def assign_points_to_cells(points: List[Point], seeds: List[Point]) -> Dict[int, List[Point]]:
    """Assign each point to its nearest seed, returning a dict cell_id → points."""
    if not seeds:
        raise ValueError("At least one seed required")
    pts = np.asarray(points, dtype=float)
    sds = np.asarray(seeds, dtype=float)
    dmat = _distance_matrix(pts, sds)
    nearest = np.argmin(dmat, axis=1)
    regions: Dict[int, List[Point]] = {i: [] for i in range(len(seeds))}
    for idx, cell_id in enumerate(nearest):
        regions[cell_id].append(tuple(pts[idx]))
    return regions


def generate_random_seeds(k: int, bounds: Tuple[float, float, float, float]) -> List[Point]:
    """Generate k random seeds inside the rectangular bounds (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = bounds
    return [(random.uniform(xmin, xmax), random.uniform(ymin, ymax)) for _ in range(k)]


# ----------------------------------------------------------------------
# Statistical utilities (parent A)
# ----------------------------------------------------------------------
def gini_impurity(labels: List[Any]) -> float:
    """Compute Gini impurity for a list of categorical labels."""
    if not labels:
        return 0.0
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    n = len(labels)
    impurity = 1.0 - sum((c / n) ** 2 for c in counts.values())
    return impurity


def hoeffding_bound(R: float, n: int, delta: float) -> float:
    """Hoeffding bound ε = sqrt( (R^2 * ln(1/δ)) / (2n) )."""
    if n == 0:
        return float('inf')
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2.0 * n))


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def evaluate_cells_gain(
    points: List[Point],
    labels: List[Any],
    seeds: List[Point],
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    For each Voronoi cell compute:
        - Gini impurity of the labels inside the cell.
        - Number of samples inside the cell.
    Returns two dicts: cell_id → impurity, cell_id → count.
    """
    regions = assign_points_to_cells(points, seeds)
    impurity_per_cell: Dict[int, float] = {}
    count_per_cell: Dict[int, int] = {}
    for cell_id, pts in regions.items():
        # collect corresponding labels
        indices = [points.index(p) for p in pts]  # O(N^2) but fine for demo
        cell_labels = [labels[i] for i in indices]
        impurity_per_cell[cell_id] = gini_impurity(cell_labels)
        count_per_cell[cell_id] = len(cell_labels)
    return impurity_per_cell, count_per_cell


def hybrid_split_decision(
    points: List[Point],
    labels: List[Any],
    seeds: List[Point],
    pheromone_system: HybridPheromoneSystem,
    delta: float = 0.05,
) -> SplitDecision:
    """
    Decide whether to split the tree using Voronoi cells as candidate splits.
    The decision combines:
        - Hoeffding bound ε (R=1 for Gini)
        - Gain gap ΔG = parent_impurity - weighted_child_impurity
        - Pheromone τ_i for the best cell
    """
    if len(points) != len(labels):
        raise ValueError("points and labels must have the same length")
    # overall impurity before split
    parent_impurity = gini_impurity(labels)
    # per‑cell impurity and counts
    cell_impurity, cell_counts = evaluate_cells_gain(points, labels, seeds)

    # weighted impurity after split
    total_n = len(labels)
    weighted_child_impurity = sum(
        (cell_counts[i] / total_n) * cell_impurity[i] for i in cell_impurity
    )
    gain_gap = parent_impurity - weighted_child_impurity

    # Hoeffding bound (R=1 for Gini)
    epsilon = hoeffding_bound(R=1.0, n=total_n, delta=delta)

    # choose best cell (lowest impurity -> highest gain contribution)
    best_cell = min(cell_impurity, key=cell_impurity.get, default=-1)

    # pheromone handling
    tau = pheromone_system.get(best_cell)
    # reinforce proportionally to observed gain (positive gain only)
    if gain_gap > 0:
        pheromone_system.reinforce(best_cell, amount=gain_gap)

    # composite score
    score = (gain_gap / (epsilon + 1e-9)) * (tau + 1.0)  # +1 to avoid zero

    should_split = (gain_gap > epsilon) and (score > 1.0)

    reason = (
        f"gain_gap={gain_gap:.4f}, epsilon={epsilon:.4f}, "
        f"tau={tau:.4f}, score={score:.4f}"
    )
    return SplitDecision(
        should_split=should_split,
        cell_index=best_cell,
        epsilon=epsilon,
        gain_gap=gain_gap,
        pheromone=tau,
        reason=reason,
    )


def generate_procedural_slots(seeds: List[Point], base_seed: str = "hybrid") -> List[ProceduralSlot]:
    """
    Produce a ProceduralSlot for each Voronoi seed using deterministic hashing.
    Mirrors the Percyphon side‑effect while re‑using the same seed geometry.
    """
    slots: List[ProceduralSlot] = []
    for idx, pt in enumerate(seeds):
        seed_str = f"{base_seed}:{idx}:{pt[0]:.3f}:{pt[1]:.3f}"
        h = hashlib.sha256(seed_str.encode()).hexdigest()
        name = f"Entity-{int(h[:6], 16) % 10000:04d}"
        alias = f"Alias-{h[6:10]}"
        persona = random.choice(
            ["ledger", "runner", "witness", "archivist", "carrier", "sentry"]
        )
        uuid = f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
        ternary_offset = int(h[32:34], 16) % 3
        slots.append(
            ProceduralSlot(
                slot_index=idx,
                name=name,
                alias=alias,
                persona=persona,
                uuid=uuid,
                ternary_offset=ternary_offset,
            )
        )
    return slots


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. generate synthetic 2‑D points and binary labels
    random.seed(42)
    np.random.seed(42)
    N = 200
    points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(N)]
    labels = [0 if p[0] + p[1] < 10 else 1 for p in points]  # simple linear decision boundary

    # 2. create Voronoi seeds
    seeds = generate_random_seeds(k=5, bounds=(0.0, 10.0, 0.0, 10.0))

    # 3. instantiate pheromone system
    pheromones = HybridPheromoneSystem(half_life_seconds=20.0)

    # 4. evaluate split decision
    decision = hybrid_split_decision(points, labels, seeds, pheromones, delta=0.05)
    print("Split decision:", decision)

    # 5. generate procedural slots from the same seeds
    slots = generate_procedural_slots(seeds, base_seed="demo")
    for s in slots[:3]:
        print("Procedural slot:", s.as_dict())

    # 6. simple sanity check – ensure pheromone for best cell increased
    tau_after = pheromones.get(decision.cell_index)
    print(f"Pheromone for cell {decision.cell_index} after reinforcement: {tau_after:.4f}")