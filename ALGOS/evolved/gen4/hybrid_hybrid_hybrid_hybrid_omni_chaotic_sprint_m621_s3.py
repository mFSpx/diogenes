# DARWIN HAMMER — match 621, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s0.py (gen3)
# parent_b: omni_chaotic_sprint.py (gen0)
# born: 2026-05-29T23:30:15Z

import math
import random
import sys
from pathlib import Path
import numpy as np

def caputo_kernel(alpha: float, delta: int) -> float:
    if delta < 0:
        raise ValueError("Delta must be non-negative")
    if not (0.0 < alpha < 1.0):
        raise ValueError("Alpha must be in (0,1)")
    gamma_term = math.gamma(2.0 - alpha)
    return ((delta + 1) ** (1.0 - alpha) - delta ** (1.0 - alpha)) / gamma_term

def fractional_memory_sum(alpha: float, allocations: list[float]) -> float:
    total = 0.0
    t = len(allocations) - 1
    for k, a in enumerate(allocations):
        delta = t - k
        total += caputo_kernel(alpha, delta) * a
    return total

def init_ltc_parameters(base_tau: float = 1.0, amplitude: float = 0.3) -> dict:
    return {
        "base_tau": float(base_tau),
        "amplitude": float(amplitude),
        "gamma": 2 * math.pi / 7.0,  
    }

def effective_time_constant(day: int, params: dict) -> float:
    base = params["base_tau"]
    amp = params["amplitude"]
    gamma = params["gamma"]
    return base * (1.0 + amp * math.sin(gamma * day))

def hybrid_allocate_by_dates(
    days: list[int],
    groups: list[str],
    ltc_params: dict,
    total_daily_budget: float = 100.0,
) -> dict:
    allocations: dict[int, dict[str, float]] = {}
    random.seed(0)  
    for d in days:
        tau = effective_time_constant(d, ltc_params)
        gates = {g: random.random() for g in groups}
        total_gate = sum(gates.values())
        day_alloc = {}
        for g in groups:
            share = (gates[g] / total_gate) * tau
            day_alloc[g] = share * total_daily_budget / sum(
                effective_time_constant(d, ltc_params) for _ in groups
            )
        allocations[d] = day_alloc
    return allocations

def build_random_graph(num_nodes: int, edge_prob: float = 0.4, seed: int = 42) -> dict[int, list[tuple[int, float]]]:
    random.seed(seed)
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(num_nodes)}
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                weight = 1.0
                adj[i].append((j, weight))
                adj[j].append((i, weight))
    return adj

def hybrid_edge_cost(
    base_weight: float,
    day: int,
    ltc_params: dict,
    alpha: float,
    past_allocations: list[float],
) -> float:
    tau = effective_time_constant(day, ltc_params)
    instant = tau * base_weight
    memory = fractional_memory_sum(alpha, past_allocations)
    return instant + memory

def hybrid_seismic_trace(
    start_node: int,
    day: int,
    allocations_by_day: dict[int, dict[str, float]],
    graph: dict[int, list[tuple[int, float]]],
    ltc_params: dict,
    alpha: float = 0.5,
) -> list[int]:
    visited = set()
    order = []
    queue = [start_node]
    past_alloc_totals = [
        sum(group_alloc.values()) for d, group_alloc in sorted(allocations_by_day.items())
        if d <= day
    ]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        neighbors = graph.get(node, [])
        costs = []
        for nbr, w0 in neighbors:
            if nbr in visited:
                continue
            cost = hybrid_edge_cost(w0, day, ltc_params, alpha, past_alloc_totals)
            costs.append((cost, nbr))
        costs.sort(key=lambda x: x[0])
        queue.extend([nbr for _, nbr in costs])
    return order

def improved_hybrid_seismic_trace(
    start_node: int,
    day: int,
    allocations_by_day: dict[int, dict[str, float]],
    graph: dict[int, list[tuple[int, float]]],
    ltc_params: dict,
    alpha: float = 0.5,
) -> list[int]:
    visited = set()
    order = []
    queue = [(0, start_node)]
    past_alloc_totals = [
        sum(group_alloc.values()) for d, group_alloc in sorted(allocations_by_day.items())
        if d <= day
    ]
    while queue:
        _, node = min(queue)
        queue.remove((_, node))
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        neighbors = graph.get(node, [])
        for nbr, w0 in neighbors:
            if nbr in visited:
                continue
            cost = hybrid_edge_cost(w0, day, ltc_params, alpha, past_alloc_totals)
            queue.append((cost, nbr))
    return order

if __name__ == "__main__":
    ltc_params = init_ltc_parameters(base_tau=1.2, amplitude=0.4)
    days = list(range(7))                     
    groups = ["research", "devops", "qa"]
    allocations = hybrid_allocate_by_dates(days, groups, ltc_params, total_daily_budget=120.0)
    graph = build_random_graph(num_nodes=6, edge_prob=0.5, seed=7)
    visit_order = improved_hybrid_seismic_trace(0, 3, allocations, graph, ltc_params)
    print(visit_order)