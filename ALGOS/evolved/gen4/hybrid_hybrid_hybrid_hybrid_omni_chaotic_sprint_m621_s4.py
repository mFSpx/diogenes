# DARWIN HAMMER — match 621, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s0.py (gen3)
# parent_b: omni_chaotic_sprint.py (gen0)
# born: 2026-05-29T23:30:15Z

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# Fractional‑memory utilities (Caputo kernel)
# ----------------------------------------------------------------------
def caputo_kernel(alpha: float, delta: int) -> float:
    """
    Caputo weighting κₐ(Δ) for integer lag Δ ≥ 0 and 0 < α < 1.
    """
    if delta < 0:
        raise ValueError("Delta must be non‑negative")
    if not (0.0 < alpha < 1.0):
        raise ValueError("Alpha must be in (0,1)")
    # Γ(2‑α) = (1‑α)·Γ(1‑α)
    return ((delta + 1) ** (1.0 - alpha) - delta ** (1.0 - alpha)) / math.gamma(2.0 - alpha)


def fractional_memory_sum(alpha: float, values: List[float]) -> float:
    """
    Compute Σ_{k=0}^{t} κₐ(t‑k)·values[k] for the current day t,
    where ``values`` holds the history up to and including today.
    """
    total = 0.0
    t = len(values) - 1
    for k, v in enumerate(values):
        total += caputo_kernel(alpha, t - k) * v
    return total


# ----------------------------------------------------------------------
# Liquid‑Time‑Constant (LTC) utilities
# ----------------------------------------------------------------------
def init_ltc_parameters(base_tau: float = 1.0, amplitude: float = 0.3) -> Dict[str, float]:
    """
    Initialise parameters for the LTC module.
    """
    return {
        "base_tau": float(base_tau),
        "amplitude": float(amplitude),
        "gamma": 2 * math.pi / 7.0,  # weekly period
    }


def effective_time_constant(day: int, params: Dict[str, float]) -> float:
    """
    τₛᵧₛ(day) = base_tau · (1 + amplitude·sin(gamma·day)).
    ``day`` is an integer (e.g. 0‑6 for a week).
    """
    return params["base_tau"] * (1.0 + params["amplitude"] * math.sin(params["gamma"] * day))


def hybrid_allocate_by_dates(
    days: List[int],
    groups: List[str],
    ltc_params: Dict[str, float],
    total_daily_budget: float = 100.0,
    seed: int = 0,
) -> Dict[int, Dict[str, float]]:
    """
    Allocate a daily budget among groups using the LTC‑modulated share.
    The share for a group g on day d is proportional to a random gating
    factor multiplied by τₛᵧₛ(d).  The function returns:

        allocations[day][group] = amount
    """
    allocations: Dict[int, Dict[str, float]] = {}
    rng = random.Random(seed)  # deterministic RNG per run
    for d in days:
        tau = effective_time_constant(d, ltc_params)
        # Random gating per group (same seed each day for reproducibility)
        gates = {g: rng.random() for g in groups}
        # Weight each group by gate * τ
        weighted = {g: gates[g] * tau for g in groups}
        total_weight = sum(weighted.values())
        day_alloc = {
            g: (weighted[g] / total_weight) * total_daily_budget for g in groups
        }
        allocations[d] = day_alloc
    return allocations


# ----------------------------------------------------------------------
# Graph utilities (Chaotic Omni‑Engine side)
# ----------------------------------------------------------------------
def build_random_graph(
    num_nodes: int,
    edge_prob: float = 0.4,
    base_weight: float = 1.0,
    seed: int = 42,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Create an undirected random graph represented as an adjacency list.
    All edges start with the same ``base_weight``.
    """
    rng = random.Random(seed)
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(num_nodes)}
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < edge_prob:
                adj[i].append((j, base_weight))
                adj[j].append((i, base_weight))
    return adj


def hybrid_edge_cost(
    base_weight: float,
    day: int,
    ltc_params: Dict[str, float],
    alpha: float,
    edge_history: List[float],
) -> float:
    """
    Compute the hybrid cost Cₑ(t) for a single edge on a given day.

    - Instantaneous term: τₛᵧₛ(t)·w₀
    - Memory term: fractional sum of past *edge‑specific* allocations.
    """
    tau = effective_time_constant(day, ltc_params)
    instant = tau * base_weight
    memory = fractional_memory_sum(alpha, edge_history) if edge_history else 0.0
    return instant + memory


def hybrid_seismic_trace(
    start_node: int,
    day: int,
    allocations_by_day: Dict[int, Dict[str, float]],
    graph: Dict[int, List[Tuple[int, float]]],
    ltc_params: Dict[str, float],
    alpha: float = 0.5,
    edge_history_map: Dict[Tuple[int, int], List[float]] = None,
) -> List[int]:
    """
    Deterministic breadth‑first traversal where expansion order is
    governed by the hybrid edge cost Cₑ(t).  ``edge_history_map`` stores,
    for each unordered edge (u, v), a list of past allocations that have
    traversed that edge (most recent last).  If omitted, a zero‑history
    is assumed, which reduces the cost to the pure LTC term.
    """
    if edge_history_map is None:
        edge_history_map = {}

    visited = set()
    order: List[int] = []
    queue: List[int] = [start_node]

    # Pre‑compute cumulative total allocation per day up to ``day``.
    # This scalar is used for edges that have no dedicated history.
    cumulative_totals = [
        sum(g.values())
        for d, g in sorted(allocations_by_day.items())
        if d <= day
    ]

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        order.append(node)

        # Gather neighbor costs
        neighbor_costs: List[Tuple[float, int]] = []
        for nbr, w0 in graph.get(node, []):
            if nbr in visited:
                continue
            # Retrieve edge‑specific history (order‑independent key)
            edge_key = tuple(sorted((node, nbr)))
            history = edge_history_map.get(edge_key, cumulative_totals)
            cost = hybrid_edge_cost(w0, day, ltc_params, alpha, history)
            neighbor_costs.append((cost, nbr))

        # Sort by ascending cost (lower cost ⇒ higher priority)
        neighbor_costs.sort(key=lambda x: x[0])
        queue.extend([nbr for _, nbr in neighbor_costs])

    return order


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Initialise LTC
    ltc_params = init_ltc_parameters(base_tau=1.2, amplitude=0.4)

    # 2. Temporal horizon and groups
    days = list(range(7))                     # Monday … Sunday
    groups = ["research", "devops", "qa"]

    # 3. Compute daily allocations (LTC side)
    allocations = hybrid_allocate_by_dates(
        days, groups, ltc_params, total_daily_budget=120.0, seed=123
    )

    # 4. Build a small random graph (Chaotic Omni side)
    graph = build_random_graph(num_nodes=6, edge_prob=0.5, seed=7)

    # 5. Optional: fabricate per‑edge allocation histories for deeper fusion
    #    Here we simply reuse the total daily allocations as a proxy.
    edge_histories: Dict[Tuple[int, int], List[float]] = {}
    for u in graph:
        for v, _ in graph[u]:
            if u < v:  # store each undirected edge once
                # Example: treat each day's total allocation as the amount
                # that would have traversed this edge.
                edge_histories[(u, v)] = [
                    sum(allocations[d].values()) for d in sorted(allocations) if d <= 3
                ]

    # 6. Run a hybrid trace on day 3 (Thursday)
    visit_order = hybrid_seismic_trace(
        start_node=0,
        day=3,
        allocations_by_day=allocations,
        graph=graph,
        ltc_params=ltc_params,
        alpha=0.6,
        edge_history_map=edge_histories,
    )

    print("Visit order on day 3:", visit_order)