# DARWIN HAMMER — match 3070, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1913_s3.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2323_s0.py (gen4)
# born: 2026-05-29T23:47:44Z

"""Hybrid Algorithm integrating:
- Parent A: LSM text vectors and geometric edge costs for a Minimum‑Cost Spanning Tree.
- Parent B: Temperature‑scaled bandit propensities with pheromone entropy bonus and a linear state‑space model.

Mathematical Bridge
-------------------
Each action (graph edge) carries a textual identifier.  
1. From the identifier we obtain an LSM scalar weight `w_lsm` (L1 norm of term‑frequency).  
2. A bandit policy yields a raw propensity `π_raw = expected_reward + b_pheromone`.  
   The developmental rate `r(t)` (temperature‑dependent) scales this propensity:
   `π_scaled = r(t) * π_raw`.  
3. Soft‑max converts the scaled propensities to a probability distribution
   `p_regret`.  
4. The hybrid probability fuses the two modalities multiplicatively:
   `p_hybrid = p_regret * w_lsm`.  
5. The hybrid cost for an edge `(i,j)` with Euclidean length `ℓ(i,j)` is
   `c_hybrid(i,j) = ℓ(i,j) / p_hybrid`.  
   The MST built on `c_hybrid` therefore respects geometry, text‑driven importance,
   and temperature‑modulated decision confidence.

The same developmental rate `r(t)` also scales the state‑transition matrix of a
latent linear dynamical system `x_{t+1}=A_t x_t + B u_t` with `A_t = r(t)·A₀`,
closing the loop between belief‑driven action selection and physiological dynamics.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent A (LSM and geometry)
# ----------------------------------------------------------------------
def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def words(text: str) -> List[str]:
    """Lower‑case word tokenisation (alphabetic + optional apostrophe)."""
    import re

    return [w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())]


def lsm_vector(text: str) -> Dict[str, float]:
    """Term‑frequency LSM vector (relative frequencies)."""
    ws = words(text)
    total = max(1, len(ws))
    cnt = {}
    for w in ws:
        cnt[w] = cnt.get(w, 0) + 1
    return {word: cnt[word] / total for word in cnt}


def lsm_scalar_weight(text: str) -> float:
    """Collapse LSM vector to a positive scalar (L1 norm = 1 for non‑empty text)."""
    vec = lsm_vector(text)
    return sum(abs(v) for v in vec.values()) or 1e-8  # avoid zero


# ----------------------------------------------------------------------
# Utilities from Parent B (Bandit‑State‑Space duality)
# ----------------------------------------------------------------------
def developmental_rate(temperature: float) -> float:
    """
    Simple temperature‑to‑rate mapping.
    At 20 °C the rate is 1.0; it grows linearly with temperature.
    """
    return max(0.1, 0.05 * (temperature - 20.0) + 1.0)


def pheromone_entropy_bonus(p: float, eta: float = 0.1) -> float:
    """Entropy‑based pheromone bonus: -η·p·log(p)."""
    if p <= 0.0:
        return 0.0
    return -eta * p * math.log(p)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable soft‑max."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    """Represents an edge/action with a textual label and bandit statistics."""
    action_id: str                # textual identifier (e.g., "north‑west")
    expected_reward: float       # estimated reward from bandit learning
    pheromone: float              # raw pheromone concentration (non‑negative)


@dataclass(frozen=True)
class Edge:
    i: int
    j: int
    action: Action


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def hybrid_propensities(
    actions: List[Action],
    temperature: float,
    pheromone_norm: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute hybrid probabilities for a list of actions.

    Returns
    -------
    p_regret : np.ndarray
        Soft‑max probabilities after temperature scaling.
    p_hybrid : np.ndarray
        p_regret multiplied by the LSM scalar weight of each action.
    """
    # 1. Normalise pheromone concentrations if requested
    pher_vals = np.array([a.pheromone for a in actions], dtype=float)
    if pheromone_norm:
        total = pher_vals.sum()
        pher_norm = pher_vals / total if total > 0 else np.full_like(pher_vals, 1.0 / len(pher_vals))
    else:
        pher_norm = pher_vals

    # 2. Entropy‑based pheromone bonus
    bonuses = np.array([pheromone_entropy_bonus(p) for p in pher_norm])

    # 3. Raw propensities (expected reward + bonus)
    raw = np.array([a.expected_reward for a in actions]) + bonuses

    # 4. Temperature scaling via developmental rate
    r = developmental_rate(temperature)
    scaled = r * raw

    # 5. Soft‑max to obtain a probability distribution (p_regret)
    p_regret = softmax(scaled)

    # 6. LSM scalar weights
    w_lsm = np.array([lsm_scalar_weight(a.action_id) for a in actions])

    # 7. Hybrid probability (element‑wise product)
    p_hybrid = p_regret * w_lsm
    # Normalise hybrid probabilities to keep them a distribution (optional)
    p_hybrid /= p_hybrid.sum() if p_hybrid.sum() > 0 else 1.0

    return p_regret, p_hybrid


def hybrid_mst(
    points: List[Tuple[float, float]],
    edges: List[Edge],
    temperature: float,
) -> List[Edge]:
    """
    Build a Minimum‑Cost Spanning Tree where edge cost is
        cost = Euclidean length / p_hybrid

    The hybrid probability p_hybrid is obtained from `hybrid_propensities`.
    """
    # Extract actions in the order of edges
    actions = [e.action for e in edges]

    # Compute hybrid probabilities for all actions
    _, p_hybrid = hybrid_propensities(actions, temperature)

    # Build cost list
    eps = 1e-8
    edge_costs = []
    for edge, ph in zip(edges, p_hybrid):
        ℓ = length(points[edge.i], points[edge.j])
        cost = ℓ / (ph + eps)
        edge_costs.append((cost, edge))

    # Kruskal's algorithm
    parent = list(range(len(points)))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru
            return True
        return False

    mst_edges: List[Edge] = []
    for cost, edge in sorted(edge_costs, key=lambda x: x[0]):
        if union(edge.i, edge.j):
            mst_edges.append(edge)
        if len(mst_edges) == len(points) - 1:
            break

    return mst_edges


def hybrid_state_step(
    x: np.ndarray,
    A0: np.ndarray,
    B: np.ndarray,
    temperature: float,
    selected_action: Action,
) -> np.ndarray:
    """
    Perform one step of the latent linear dynamical system using the
    temperature‑scaled transition matrix and a control input derived from
    the selected action.

    The control vector `u` is taken as the LSM vector (flattened) of the
    action identifier, padded/truncated to match `B.shape[1]`.
    """
    r = developmental_rate(temperature)
    A_t = r * A0  # temperature‑scaled transition

    # Build control vector from LSM representation
    lsm = lsm_vector(selected_action.action_id)
    # Convert dict to fixed‑size vector
    vocab = sorted(lsm.keys())
    u_raw = np.array([lsm[w] for w in vocab], dtype=float)

    # Pad or truncate to required dimension
    dim = B.shape[1]
    if u_raw.size < dim:
        u = np.pad(u_raw, (0, dim - u_raw.size))
    else:
        u = u_raw[:dim]

    # State update
    x_next = A_t @ x + B @ u
    return x_next


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny geometric graph (4 points forming a square)
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    # Define actions/edges with textual identifiers, rewards, pheromones
    raw_edges = [
        (0, 1, "east"),
        (1, 2, "north"),
        (2, 3, "west"),
        (3, 0, "south"),
        (0, 2, "diagonal"),
        (1, 3, "anti‑diagonal"),
    ]

    edges: List[Edge] = []
    for i, j, txt in raw_edges:
        act = Action(
            action_id=txt,
            expected_reward=random.uniform(0.0, 1.0),
            pheromone=random.uniform(0.1, 1.0),
        )
        edges.append(Edge(i=i, j=j, action=act))

    # Choose a temperature (e.g., 30 °C)
    temp = 30.0

    # Compute hybrid MST
    mst = hybrid_mst(pts, edges, temperature=temp)
    print("Hybrid MST edges (i, j, label):")
    for e in mst:
        print(f"  ({e.i}, {e.j}) – {e.action.action_id}")

    # Initialise a latent state and matrices for the state‑space model
    state_dim = 3
    x0 = np.random.randn(state_dim)
    A0 = np.eye(state_dim) * 0.9
    B = np.random.randn(state_dim, 5)  # allow up to 5 LSM dimensions

    # Simulate a few steps using actions selected from the MST (cyclic)
    x = x0.copy()
    for step in range(5):
        act = mst[step % len(mst)].action
        x = hybrid_state_step(x, A0, B, temperature=temp, selected_action=act)
        print(f"Step {step+1}, selected '{act.action_id}', state = {x}")

    sys.exit(0)