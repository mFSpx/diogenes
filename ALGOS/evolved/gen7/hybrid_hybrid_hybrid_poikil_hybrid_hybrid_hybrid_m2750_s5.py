# DARWIN HAMMER — match 2750, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_poikilotherm__hybrid_regret_engine_m1595_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m2473_s2.py (gen6)
# born: 2026-05-29T23:45:48Z

"""Hybrid Algorithm integrating Schoolfield poikilotherm rate, regret‑weighted pheromone dynamics,
and spatial tree‑cost evaluation.

Parents:
- hybrid_hybrid_poikilotherm__hybrid_regret_engine_m1595_s1.py (temperature‑dependent rate,
  pheromone system, Gini‑modulated regret weighting)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m2473_s2.py (pheromone entry model,
  tree cost computation, bandit‑style decision scores)

Mathematical bridge:
The pheromone signal λ_i is updated by a temperature‑scaled Schoolfield rate r(T) and
a regret weight w_regret = exp(−Δ/τ) where Δ is the regret magnitude and τ a temperature
parameter. The Gini coefficient G of an external distribution modulates the exploration
intensity by scaling λ_i → λ_i·(1−G). The spatial structure of decisions is captured by a
tree‑cost C_tree computed over a graph of contexts; this cost regularizes the final
action score S_i = (E_i−C_i)·r(T)·(1−w_regret) + λ_i·(1−G) − α·C_tree, where α is a
tunable regularization constant.
"""

import math
import random
import sys
import pathlib
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Sequence, Any

import numpy as np

# ----------------------------------------------------------------------
# Core data structures (merged from both parents)
# ----------------------------------------------------------------------


R_CAL = 1.987  # cal·mol⁻¹·K⁻¹


@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL


@dataclass(frozen=True)
class MathAction:
    """Action description used in the regret‑weighted part."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass
class PheromoneEntry:
    """Compact pheromone record (merged from parent B)."""
    uuid: str
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: float
    created_at: float
    last_decay: float


# ----------------------------------------------------------------------
# Utility functions (shared)
# ----------------------------------------------------------------------


def doomsday(year: int, month: int, day: int) -> int:
    """Weekday index used for external Gini calculation (parent A)."""
    return (dt.date(year, month, day).weekday() + 1) % 7


def gini_coefficient(values: Iterable[float]) -> float:
    """Gini coefficient, identical to parent A implementation."""
    xs = sorted(float(x) for x in values)
    if not xs or sum(xs) == 0:
        return 0.0
    if xs[0] < 0:
        raise ValueError("values must be non-negative")
    n = len(xs)
    return sum((2 * i - n - 1) * x for i, x in enumerate(xs, 1)) / (n * sum(xs))


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points (parent B)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_cost(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    path_weight: float = 0.2,
) -> float:
    """
    Spatial regularization cost (parent B).
    Material cost = sum of edge lengths.
    Path cost  = weighted sum of distances from root to every node.
    """
    if root not in nodes:
        raise ValueError("Root must be one of the node keys.")

    material = 0.0
    adjacency: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
        material += length(nodes[a], nodes[b])

    # BFS to compute distance from root
    dist: Dict[str, float] = {root: 0.0}
    frontier: List[str] = [root]
    visited: set = {root}
    while frontier:
        cur = frontier.pop()
        for nxt in adjacency[cur]:
            if nxt not in visited:
                visited.add(nxt)
                dist[nxt] = dist[cur] + length(nodes[cur], nodes[nxt])
                frontier.append(nxt)

    path = sum(dist.values())
    return material + path_weight * path


# ----------------------------------------------------------------------
# Core mathematical models
# ----------------------------------------------------------------------


def schoolfield_rate(T: float, params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Temperature‑dependent enzymatic rate (parent A).
    Implements the full Schoolfield‐Rollinson equation.
    """
    if T <= 0:
        raise ValueError("Temperature must be positive Kelvin.")
    # High‑temperature deactivation term
    term_high = 1 + math.exp(
        (params.delta_h_high / params.r_cal) * (1 / params.t_high - 1 / T)
    )
    # Low‑temperature deactivation term
    term_low = 1 + math.exp(
        (params.delta_h_low / params.r_cal) * (1 / params.t_low - 1 / T)
    )
    # Main Arrhenius term
    arrhenius = math.exp(
        (params.delta_h_activation / params.r_cal) * (1 / params.t_low - 1 / T)
    )
    rate = params.rho_25 * arrhenius / (term_low * term_high)
    return rate


def update_pheromone(
    pheromone: PheromoneEntry,
    action: MathAction,
    T: float,
    regret: float,
    external_gini: float,
    decay_factor: float = 0.9,
) -> None:
    """
    Regret‑weighted pheromone update (fusion of parents A & B).

    λ_new = λ_old * decay_factor
            + r(T) * (1 - exp(-regret / T)) * (1 - external_gini)

    The Gini term reduces exploration when the external distribution is highly unequal.
    """
    rate = schoolfield_rate(T)
    regret_weight = 1 - math.exp(-regret / max(T, 1e-6))
    increment = rate * regret_weight * (1.0 - external_gini)
    pheromone.signal_value = pheromone.signal_value * decay_factor + increment


def hybrid_action_score(
    action: MathAction,
    pheromone: PheromoneEntry,
    T: float,
    regret: float,
    tree_c: float,
    alpha: float = 0.05,
) -> float:
    """
    Unified scoring function.

    S = (E - C) * r(T) * (1 - w_regret) + λ * (1 - G) - α * C_tree

    where:
    - E, C are expected value and cost from MathAction,
    - r(T) is the Schoolfield rate,
    - w_regret = exp(-regret / T),
    - λ is the current pheromone signal,
    - G is the Gini coefficient embedded in λ (pre‑scaled by update_pheromone),
    - C_tree is the spatial regularization cost,
    - α controls the influence of the tree cost.
    """
    rate = schoolfield_rate(T)
    w_regret = math.exp(-regret / max(T, 1e-6))
    base = (action.expected_value - action.cost) * rate * (1 - w_regret)
    pheromone_term = pheromone.signal_value  # already contains (1 - G) scaling
    score = base + pheromone_term - alpha * tree_c
    return score


def select_best_action(
    actions: List[MathAction],
    pheromones: Dict[str, PheromoneEntry],
    T: float,
    regrets: Dict[str, float],
    tree_nodes: Dict[str, Tuple[float, float]],
    tree_edges: List[Tuple[str, str]],
    root_node: str,
) -> MathAction:
    """
    End‑to‑end hybrid decision routine.
    1. Compute spatial tree cost (shared for all actions).
    2. Update pheromones with current regrets and external Gini.
    3. Score each action with `hybrid_action_score`.
    4. Return the action with maximal score.
    """
    tree_c = tree_cost(tree_nodes, tree_edges, root_node)

    # External Gini derived from a simple synthetic distribution (could be any real data)
    external_distribution = [len(v) for v in tree_nodes.values()]
    external_gini = gini_coefficient(external_distribution)

    # Update pheromones
    for act in actions:
        ph = pheromones.get(act.id)
        if ph is None:
            # initialise a fresh pheromone entry if missing
            ph = PheromoneEntry(
                uuid=str(random.getrandbits(128)),
                surface_key=act.id,
                signal_kind="hybrid",
                signal_value=0.0,
                half_life_seconds=3600.0,
                created_at=dt.datetime.utcnow().timestamp(),
                last_decay=dt.datetime.utcnow().timestamp(),
            )
            pheromones[act.id] = ph
        update_pheromone(
            pheromone=ph,
            action=act,
            T=T,
            regret=regrets.get(act.id, 0.0),
            external_gini=external_gini,
        )

    # Score actions
    best_act = None
    best_score = -math.inf
    for act in actions:
        ph = pheromones[act.id]
        score = hybrid_action_score(
            action=act,
            pheromone=ph,
            T=T,
            regret=regrets.get(act.id, 0.0),
            tree_c=tree_c,
        )
        if score > best_score:
            best_score = score
            best_act = act
    return best_act  # type: ignore


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny synthetic graph
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (1.0, 1.0),
    }
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    root = "A"

    # Define actions
    actions = [
        MathAction(id="act1", expected_value=10.0, cost=2.0),
        MathAction(id="act2", expected_value=8.0, cost=1.0),
        MathAction(id="act3", expected_value=12.0, cost=5.0),
    ]

    # Initial pheromone store (empty)
    pheromones: Dict[str, PheromoneEntry] = {}

    # Regret estimates (could be from counterfactual analysis)
    regrets = {"act1": 1.5, "act2": 0.5, "act3": 2.0}

    # Choose a temperature (Kelvin)
    temperature = 298.15  # ~25 °C

    best = select_best_action(
        actions=actions,
        pheromones=pheromones,
        T=temperature,
        regrets=regrets,
        tree_nodes=nodes,
        tree_edges=edges,
        root_node=root,
    )
    print(f"Selected action: {best.id} with expected value {best.expected_value}")
    # Verify that pheromone values have been populated
    for aid, ph in pheromones.items():
        print(f"Pheromone[{aid}] = {ph.signal_value:.4f}")