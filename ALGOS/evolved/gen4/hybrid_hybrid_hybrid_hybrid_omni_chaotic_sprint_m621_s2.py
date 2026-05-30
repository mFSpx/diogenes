# DARWIN HAMMER — match 621, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s0.py (gen3)
# parent_b: omni_chaotic_sprint.py (gen0)
# born: 2026-05-29T23:30:15Z

"""Hybrid Allocation‑LTC & Chaotic Omni‑Graph Engine

Parents:
- **Hybrid Allocation‑LTC & Fractional‑Memory Tree Cost** (hybrid_hybrid_hybrid_worksh_hybrid_caputo_fracti_m46_s0.py)
- **Chaotic Omni‑Front Synthesis Core** (omni_chaotic_sprint.py)

Mathematical Bridge
-------------------
Both parents manipulate a *temporal* scalar that modulates a discrete
structure:

1. **LTC (Liquid‑Time‑Constant)** supplies an effective time‑constant
   τₛᵧₛ(t) – a scalar that varies with the day‑of‑week and gates the
   allocation of a resource (LLM share).

2. **Chaotic Omni‑Engine** traverses a graph (the “seismic ray‑trace”) where
   each edge carries a base weight *w₀*.  The original engine treats the
   weight as static.

The fusion replaces the static edge weight by a *fractionally‑weighted*
temporal term that uses the same τₛᵧₛ(t) from the LTC module and the
Caputo‑type memory kernel from the fractional‑memory tree cost.  For a
given day *t* the hybrid edge cost is

    Cₑ(t) = τₛᵧₛ(t) · w₀ₑ  +  Σ_{k=0}^{t}  κₐ(t‑k) · A(k)

where

- κₐ(Δ) = ( (Δ+1)^{1‑α} – Δ^{1‑α} ) / Γ(2‑α)   (Caputo kernel of order α)
- A(k) is the total allocation produced by the LTC module on day *k*.

Thus the LTC scalar directly scales the instantaneous graph metric,
while the fractional memory term injects a history‑dependent “inertia”
into the traversal cost.  The three public functions below demonstrate
initialisation, allocation, and a τ‑modulated graph trace that jointly
exercise both parent topologies."""

import math
import random
import sys
from pathlib import Path
import numpy as np

# ----------------------------------------------------------------------
# Fractional‑memory utilities (from Parent A)
# ----------------------------------------------------------------------
def caputo_kernel(alpha: float, delta: int) -> float:
    """Caputo weighting κₐ(Δ) for integer lag Δ ≥ 0 and 0<α<1."""
    if delta < 0:
        raise ValueError("Delta must be non‑negative")
    if not (0.0 < alpha < 1.0):
        raise ValueError("Alpha must be in (0,1)")
    # Γ(2‑α) = (1‑α)·Γ(1‑α)
    gamma_term = math.gamma(2.0 - alpha)
    return ((delta + 1) ** (1.0 - alpha) - delta ** (1.0 - alpha)) / gamma_term

def fractional_memory_sum(alpha: float, allocations: list[float]) -> float:
    """
    Compute Σ_{k=0}^{t} κₐ(t‑k)·A(k) for the current day t,
    where allocations[0] … allocations[t] are the daily total allocations.
    """
    total = 0.0
    t = len(allocations) - 1
    for k, a in enumerate(allocations):
        delta = t - k
        total += caputo_kernel(alpha, delta) * a
    return total

# ----------------------------------------------------------------------
# LTC (Liquid‑Time‑Constant) utilities (from Parent A)
# ----------------------------------------------------------------------
def init_ltc_parameters(base_tau: float = 1.0, amplitude: float = 0.3) -> dict:
    """
    Initialise parameters for the LTC module.
    - base_tau : baseline time‑constant.
    - amplitude: sinusoidal modulation amplitude (0‑1).
    Returns a dict with the parameters and a simple gating function.
    """
    return {
        "base_tau": float(base_tau),
        "amplitude": float(amplitude),
        "gamma": 2 * math.pi / 7.0,  # weekly period
    }

def effective_time_constant(day: int, params: dict) -> float:
    """
    Compute τₛᵧₛ(day) = base_tau · (1 + amplitude·sin(gamma·day)).
    Day is an integer 0‑6 (Monday‑Sunday).
    """
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
    """
    Allocate a daily budget among groups using the LTC‑modulated share.
    The share for a group g on day d is proportional to a random gating
    factor multiplied by τₛᵧₛ(d).  The function returns a mapping:
        allocations[day][group] = amount
    """
    allocations: dict[int, dict[str, float]] = {}
    random.seed(0)  # deterministic for the smoke test
    for d in days:
        tau = effective_time_constant(d, ltc_params)
        # Random gating per group, same seed each day for reproducibility
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

# ----------------------------------------------------------------------
# Graph utilities (inspired by Parent B)
# ----------------------------------------------------------------------
def build_random_graph(num_nodes: int, edge_prob: float = 0.4, seed: int = 42) -> dict[int, list[tuple[int, float]]]:
    """
    Create an undirected random graph represented as an adjacency list.
    Each edge gets a base weight w₀ = 1.0 (the same as Parent B's default).
    """
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
    """
    Compute the hybrid cost Cₑ(t) for a single edge on a given day.
    """
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
    """
    Perform a deterministic breadth‑first traversal where the expansion
    order is governed by the hybrid edge cost Cₑ(t).  Nodes are visited
    once; the function returns the visitation order.
    """
    visited = set()
    order = []
    queue = [start_node]
    # Build a list of total allocations per past day (including current)
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
        # Compute hybrid costs for all outgoing edges
        neighbors = graph.get(node, [])
        costs = []
        for nbr, w0 in neighbors:
            if nbr in visited:
                continue
            cost = hybrid_edge_cost(w0, day, ltc_params, alpha, past_alloc_totals)
            costs.append((cost, nbr))
        # Sort by ascending cost (lower cost = higher priority)
        costs.sort(key=lambda x: x[0])
        queue.extend([nbr for _, nbr in costs])
    return order

# ----------------------------------------------------------------------
# Demonstration / Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Initialise LTC
    ltc_params = init_ltc_parameters(base_tau=1.2, amplitude=0.4)

    # 2. Define temporal horizon and groups
    days = list(range(7))                     # Monday … Sunday
    groups = ["research", "devops", "qa"]

    # 3. Compute allocations (LTC side)
    allocations = hybrid_allocate_by_dates(days, groups, ltc_params, total_daily_budget=120.0)

    # 4. Build a small random graph (Chaotic Omni side)
    graph = build_random_graph(num_nodes=6, edge_prob=0.5, seed=7)

    # 5. Run a hybrid trace on day 3 (Thursday)
    visit_order = hybrid_seismic_trace(
        start_node=0,
        day=3,
        allocations_by_day=allocations,
        graph=graph,
        ltc_params=ltc_params,
        alpha=0.6,
    )

    # 6. Output results
    print("LTC Parameters:", ltc_params)
    print("\nDaily Allocations (total per day):")
    for d in days:
        total = sum(allocations[d].values())
        print(f" Day {d}: {total:.3f}")

    print("\nGraph adjacency list (base weight = 1.0):")
    for n, nbrs in graph.items():
        print(f" {n}: {[nbr for nbr, _ in nbrs]}")

    print("\nHybrid seismic trace visitation order on day 3:", visit_order)