# DARWIN HAMMER — match 2947, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m520_s2.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_possum_filter_m2032_s1.py (gen6)
# born: 2026-05-29T23:46:51Z

"""Hybrid Physarum‑Bandit‑Semantic Fusion

Parents:
- hybrid_hybrid_physarum_netw_hybrid_hybrid_m520_s2.py (Physarum flux network with weekday weight modulation)
- hybrid_hybrid_hybrid_hybrid_hybrid_possum_filter_m2032_s1.py (NLMS‑velocity‑field, diffusion‑forcing sheaf, diversity filter & semantic‑morphology)

Mathematical bridge:
The weekday weight vector (a sinusoidal, normalized vector) is interpreted as the NLMS weight/velocity field that
scales the diversity‑filter ranking (bandit propensities).  The semantic hybrid score h(i,j) – derived from
geodesic distance and sphericity – modulates the diffusion‑forcing schedule, i.e. the conductance update of the
Physarum network.  Consequently, flux magnitudes influence bandit propensities while hybrid scores influence
conductance growth, yielding a tightly coupled dynamical system.
"""

import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Constants & simple utilities (from Parent A)
# ----------------------------------------------------------------------
GROUPS: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
MAX64 = (1 << 64) - 1
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


def _pct(value: float) -> float:
    return round(float(value), 6)


def doomsday(year: int, month: int, day: int) -> int:
    """Return day‑of‑week index used by the weekday weight vector (0=Mon,…,6=Sun)."""
    return (datetime(year, month, day).weekday() + 1) % 7


def weekday_weight_vector(groups: Sequence[str], dow: int) -> np.ndarray:
    """Sinusoidal weight vector (NLMS‑like velocity field) for a given day of week."""
    n = len(groups)
    if n == 0:
        raise ValueError("groups must contain at least one element")
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)


def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    """Physarum flux on a single edge."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, gain: float, decay: float, dt: float) -> float:
    """Standard Physarum conductance update."""
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


# ----------------------------------------------------------------------
# Semantic‑Morphology & Diversity Filter utilities (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "hybrid"


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    morphology: Morphology
    score: float = 0.0
    address_signature: str = ""


def haversine_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def sphericity_index(length: float, width: float, height: float) -> float:
    """Dimension‑based shape factor, 0 < s ≤ 1 (1 → perfect sphere)."""
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


# ----------------------------------------------------------------------
# Hybrid core functions (the mathematical fusion)
# ----------------------------------------------------------------------
def hybrid_score(e1: Entity, e2: Entity) -> float:
    """
    Semantic‑morphology hybrid score h(i,j).
    Combines geodesic proximity and shape compatibility.
    """
    dist = haversine_distance((e1.lat, e1.lon), (e2.lat, e2.lon))
    # Normalise distance to [0,1] assuming a 10 km relevance radius
    d_norm = max(0.0, min(1.0, 1.0 - dist / 10_000.0))
    sph1 = sphericity_index(e1.morphology.length,
                            e1.morphology.width,
                            e1.morphology.height)
    sph2 = sphericity_index(e2.morphology.length,
                            e2.morphology.width,
                            e2.morphology.height)
    sph_sim = 1.0 - abs(sph1 - sph2)  # 1 → identical sphericity
    return 0.6 * d_norm + 0.4 * sph_sim


def network_step(nodes: List[str],
                 edges: List[Tuple[int, int, float]],
                 conductances: np.ndarray,
                 pressures: np.ndarray,
                 dow: int,
                 gain: float = 1.0,
                 decay: float = 0.1,
                 dt: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    One Physarum iteration where the weekday weight vector acts as a
    velocity field scaling the pressure source term.

    Parameters
    ----------
    nodes : list of node identifiers
    edges : list of (src_idx, dst_idx, length)
    conductances : array of shape (E,) – current conductances
    pressures : array of shape (N,) – node pressures
    dow : day‑of‑week index (0‑6)

    Returns
    -------
    new_conductances, new_pressures
    """
    # 1. Build NLMS‑like velocity field from weekday weights
    vel = weekday_weight_vector(GROUPS, dow)  # length = len(GROUPS)
    # Map velocities onto nodes (simple modulo mapping)
    node_vel = np.array([vel[i % len(vel)] for i in range(len(nodes))])

    # 2. Adjust pressures by adding a small velocity‑driven term
    pressures_adj = pressures + 0.05 * node_vel

    # 3. Compute fluxes on each edge
    fluxes = np.empty_like(conductances)
    for idx, (src, dst, length) in enumerate(edges):
        q = flux(conductances[idx], length, pressures_adj[src], pressures_adj[dst])
        fluxes[idx] = q

    # 4. Update conductances using hybrid_score as a diffusion‑forcing modifier.
    # For demonstration we pick a random pair of entities to supply a score.
    # In a real system this would be systematic.
    dummy_entity_a = Entity(
        id="A", lat=0.0, lon=0.0, category="x",
        morphology=Morphology(1.0, 1.0, 1.0, 1.0), score=0.0
    )
    dummy_entity_b = Entity(
        id="B", lat=0.0, lon=0.0, category="y",
        morphology=Morphology(1.2, 0.9, 1.1, 1.0), score=0.0
    )
    h_score = hybrid_score(dummy_entity_a, dummy_entity_b)

    new_conductances = np.empty_like(conductances)
    for i, q in enumerate(fluxes):
        # extra diffusion forcing term proportional to h_score
        new_conductances[i] = update_conductance(conductances[i], q, gain, decay, dt) \
                              + dt * 0.2 * h_score * abs(q)

    # 5. Re‑solve pressures (simple linear relaxation for demo purposes)
    #   p_new = p_old + epsilon * (divergence of fluxes)
    divergence = np.zeros(len(nodes))
    for idx, (src, dst, _) in enumerate(edges):
        divergence[src] -= fluxes[idx]
        divergence[dst] += fluxes[idx]
    pressures_new = pressures_adj + 0.1 * divergence

    return new_conductances, pressures_new


def update_bandit_actions(actions: List[BanditAction],
                          fluxes: np.ndarray,
                          hybrid_scores: np.ndarray,
                          alpha: float = 0.3,
                          beta: float = 0.2) -> List[BanditAction]:
    """
    Adjust bandit propensities using Physarum flux magnitudes (diversity filter)
    and semantic hybrid scores (diffusion forcing).

    Each action is paired with a flux and a hybrid score by index.
    """
    updated = []
    for act, q, h in zip(actions, fluxes, hybrid_scores):
        # Scale propensity up with flux (more flow → higher interest)
        prop = act.propensity * (1.0 + alpha * abs(q))
        # Further modulate by hybrid semantic compatibility
        prop = prop * (1.0 + beta * h)
        # Clamp to a reasonable range
        prop = max(0.0, min(1.0, prop))
        updated.append(BanditAction(
            action_id=act.action_id,
            propensity=prop,
            expected_reward=act.expected_reward,
            confidence_bound=act.confidence_bound,
            algorithm=act.algorithm
        ))
    return updated


def hybrid_simulation_step(state: StoreState,
                           nodes: List[str],
                           edges: List[Tuple[int, int, float]],
                           conductances: np.ndarray,
                           pressures: np.ndarray,
                           bandit_actions: List[BanditAction],
                           dow: int) -> Tuple[StoreState, np.ndarray, np.ndarray, List[BanditAction]]:
    """
    Executes a full hybrid iteration:
      1. Physarum network update (flux, conductance)
      2. Compute hybrid scores for each edge (using dummy entities)
      3. Update bandit propensities
    Returns the possibly updated StoreState and the new network variables.
    """
    # 1. Network dynamics
    new_cond, new_press = network_step(
        nodes, edges, conductances, pressures, dow,
        gain=state.base, decay=state.base * 0.1, dt=state.dt
    )

    # 2. Generate a hybrid score per edge (here we reuse dummy entities)
    dummy_a = Entity(
        id="A", lat=0.0, lon=0.0, category="x",
        morphology=Morphology(1.0, 1.0, 1.0, 1.0)
    )
    dummy_b = Entity(
        id="B", lat=0.0, lon=0.0, category="y",
        morphology=Morphology(1.2, 0.9, 1.1, 1.0)
    )
    h_per_edge = np.full(len(edges), hybrid_score(dummy_a, dummy_b))

    # 3. Update bandit actions
    # For demonstration we map each action to the flux of the corresponding edge
    # (if fewer actions than edges we repeat, otherwise truncate)
    fluxes = np.array([flux(c, l, new_press[s], new_press[t]) for (s, t, l), c in zip(edges, new_cond)])
    actions_updated = update_bandit_actions(bandit_actions, fluxes, h_per_edge)

    # 4. Optionally evolve StoreState (e.g., decay alpha/beta)
    new_state = StoreState(
        level=state.level + state.dt,
        alpha=max(0.0, state.alpha - 0.01 * state.dt),
        beta=max(0.0, state.beta - 0.005 * state.dt),
        dt=state.dt,
        base=state.base
    )
    return new_state, new_cond, new_press, actions_updated


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal deterministic setup
    random.seed(0)
    np.set_printoptions(precision=4, suppress=True)

    # Nodes and edges for a tiny triangular network
    nodes = ["N0", "N1", "N2"]
    edges = [
        (0, 1, 1.0),  # edge 0
        (1, 2, 1.2),  # edge 1
        (2, 0, 0.9)   # edge 2
    ]
    conductances = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    pressures = np.array([1.0, 0.5, 0.0], dtype=np.float64)

    # Dummy bandit actions (one per edge)
    bandit_actions = [
        BanditAction(action_id="act0", propensity=0.2, expected_reward=0.0, confidence_bound=0.1),
        BanditAction(action_id="act1", propensity=0.5, expected_reward=0.0, confidence_bound=0.1),
        BanditAction(action_id="act2", propensity=0.8, expected_reward=0.0, confidence_bound=0.1)
    ]

    # Store state
    state = StoreState(level=0.0, alpha=1.0, beta=1.0, dt=0.5, base=1.0)

    # Day of week (e.g., today)
    today = datetime.now(timezone.utc)
    dow = doomsday(today.year, today.month, today.day)

    # Run a single hybrid step
    new_state, new_cond, new_press, new_actions = hybrid_simulation_step(
        state, nodes, edges, conductances, pressures, bandit_actions, dow
    )

    # Print results (just to verify execution