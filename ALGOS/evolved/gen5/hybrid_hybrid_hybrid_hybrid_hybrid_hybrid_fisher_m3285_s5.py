# DARWIN HAMMER — match 3285, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m1238_s5.py (gen4)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_regret_m1900_s1.py (gen4)
# born: 2026-05-29T23:48:59Z

"""Hybrid Privacy‑VRAM & Fisher‑Regret Scheduler

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_model__m1238_s5.py (privacy‑DP,
  VRAM‑aware Ollivier‑Ricci curvature, circuit‑breaker)
- hybrid_hybrid_fisher_locali_hybrid_hybrid_regret_m1900_s1.py (Fisher‑information
  scoring, ternary‑lens regret engine)

Mathematical bridge:
Each artefact (ModelTier) is a node *i* in an undirected graph G with mass weight  

    w_i = estimated_mb_i / Σ_j estimated_mb_j .

The lazy random‑walk measure μ_i(v) uses a laziness α scaled by w_i.  
Ollivier‑Ricci curvature κ_i is approximated from μ_i and the graph metric.

The Fisher information F(θ) of a Gaussian beam (parent B) is then used to
modulate the privacy budget ε allocated to node *i*:

    ε_i = ε_0 · (1 + κ_i) / (1 + F(θ))

Thus curvature (VRAM distribution) expands the privacy allowance when the
memory layout is “flat”, while high Fisher information (sharp beam) contracts
it.  The resulting ε_i feeds a ternary‑lens regret decision where each action
a receives a regret‑weighted score

    S_a = (EV_a – cost_a·ε_i) · σ(τ·v_a)

with σ the sigmoid, τ a scaling derived from the ternary vector, and v_a the
corresponding ternary component.  The action with maximal S_a is selected.

The module implements this fused pipeline with three core functions:
    1. compute_curvature(...)
    2. compute_combined_epsilon(...)
    3. select_regret_action(...)
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (both parents)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model (parent A)."""
    name: str
    ram_mb: int
    tier: str


@dataclass(frozen=True)
class MathAction:
    """Action used by the regret engine (parent B)."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


# ----------------------------------------------------------------------
# Helper functions from parent B
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ)."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for the Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------


def compute_node_weights(tiers: List[ModelTier]) -> Dict[str, float]:
    """
    Compute mass weight w_i for each tier based on its RAM usage.
    """
    total = sum(t.ram_mb for t in tiers)
    if total == 0:
        raise ValueError("Total RAM must be positive")
    return {t.name: t.ram_mb / total for t in tiers}


def compute_curvature(
    adjacency: Dict[str, List[str]],
    weights: Dict[str, float],
    alpha: float = 0.5,
) -> Dict[str, float]:
    """
    Approximate Ollivier‑Ricci curvature κ_i for each node.

    For node i:
        μ_i = α·w_i·δ_i + (1‑α)·w_i·(1/deg(i))·∑_{u∈N(i)} δ_u
    The curvature is approximated as the excess mass that stays at i after a
    lazy step, i.e. κ_i = 1 - (1‑α)·(1/deg(i))·∑_{u∈N(i)} w_u / w_i.
    This simple proxy captures that well‑connected, balanced mass yields higher
    curvature.
    """
    curvature: Dict[str, float] = {}
    for node, neighbors in adjacency.items():
        deg = len(neighbors)
        if deg == 0:
            curvature[node] = 0.0
            continue
        w_i = weights[node]
        neighbor_mass = sum(weights.get(nb, 0.0) for nb in neighbors)
        transport_factor = (1.0 - alpha) * (neighbor_mass / deg) / w_i
        curvature[node] = max(0.0, 1.0 - transport_factor)  # clamp to [0,1]
    return curvature


def compute_combined_epsilon(
    base_epsilon: float,
    curvature: Dict[str, float],
    fisher: float,
    epsilon_cap: float = 10.0,
) -> Dict[str, float]:
    """
    Fuse curvature κ_i and Fisher information F into a node‑specific privacy
    budget ε_i.

        ε_i = min( ε_cap , ε_0 · (1 + κ_i) / (1 + F) )
    """
    combined: Dict[str, float] = {}
    for node, kappa in curvature.items():
        eps = base_epsilon * (1.0 + kappa) / (1.0 + fisher)
        combined[node] = min(eps, epsilon_cap)
    return combined


def select_regret_action(
    actions: List[MathAction],
    ternary_vector: np.ndarray,
    epsilon_i: float,
    tau: float = 1.0,
) -> Tuple[MathAction, float]:
    """
    Regret‑weighted ternary decision.

    For each action a:
        S_a = (EV_a – cost_a·ε_i) · σ( τ·v_a )
    where v_a is the ternary component associated with a (cyclicly taken).
    The action with maximal S_a is returned together with its score.
    """
    if ternary_vector.ndim != 1:
        raise ValueError("ternary_vector must be 1‑D")
    dim = ternary_vector.shape[0]
    scores = []
    for idx, act in enumerate(actions):
        v = ternary_vector[idx % dim]
        weight = sigmoid(np.array([tau * v]))[0]
        score = (act.expected_value - act.cost * epsilon_i) * weight
        scores.append(score)
    best_idx = int(np.argmax(scores))
    return actions[best_idx], float(scores[best_idx])


# ----------------------------------------------------------------------
# Example usage (smoke test)
# ----------------------------------------------------------------------


def _build_example_graph(tiers: List[ModelTier]) -> Dict[str, List[str]]:
    """
    Build a simple undirected graph where each tier is connected to the next
    tier in the list (ring topology) – enough to exercise the curvature code.
    """
    names = [t.name for t in tiers]
    adjacency: Dict[str, List[str]] = {n: [] for n in names}
    n = len(names)
    for i, name in enumerate(names):
        left = names[(i - 1) % n]
        right = names[(i + 1) % n]
        adjacency[name].extend([left, right])
    return adjacency


def _demo() -> None:
    # ---- Define model tiers (VRAM landscape) ----
    tiers = [
        ModelTier("qwen-0.5b", 512, "T1"),
        ModelTier("reasoning-t2", 3000, "T2"),
        ModelTier("tool-t2", 2600, "T2"),
        ModelTier("qwen-7b", 7000, "T3"),
    ]

    # ---- Compute VRAM‑derived weights and curvature ----
    weights = compute_node_weights(tiers)
    adjacency = _build_example_graph(tiers)
    curvature = compute_curvature(adjacency, weights, alpha=0.4)

    # ---- Fisher information (Gaussian beam) ----
    theta, center, width = 0.3, 0.0, 1.0
    fisher = fisher_score(theta, center, width)

    # ---- Fuse into node‑specific privacy budgets ----
    base_eps = 1.0
    epsilons = compute_combined_epsilon(base_eps, curvature, fisher)

    # ---- Define actions for regret engine ----
    actions = [
        MathAction("a1", expected_value=5.0, cost=1.2),
        MathAction("a2", expected_value=4.5, cost=0.9),
        MathAction("a3", expected_value=6.0, cost=1.8),
    ]

    # ---- Ternary lens (randomly generated) ----
    rng = np.random.default_rng(42)
    ternary_vec = rng.uniform(-1.0, 1.0, size=12)  # TERNARY_DIMS = 12

    # ---- Choose action per node (demonstrate hybrid) ----
    for node, eps in epsilons.items():
        chosen, score = select_regret_action(actions, ternary_vec, eps, tau=2.0)
        print(
            f"Node {node:12s} | κ={curvature[node]:.3f} | ε={eps:.3f} | "
            f"Chosen Action={chosen.id} (score={score:.3f})"
        )


if __name__ == "__main__":
    _demo()