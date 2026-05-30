# DARWIN HAMMER — match 863, survivor 0
# gen: 5
# parent_a: hybrid_xgboost_objective_hybrid_ternary_lens__m33_s0.py (gen2)
# parent_b: hybrid_hybrid_distributed_l_hybrid_physarum_netw_m110_s0.py (gen4)
# born: 2026-05-29T23:32:48Z

"""
Hybrid XGBoost–Physarum Leader Algorithm

This module fuses the core mathematics of:

* **Parent A** – XGBoost objective utilities (gradient, hessian, optimal leaf weight,
  split gain) and its decreasing‑rate pruning schedule.
* **Parent B** – Distributed leader election with a Physarum‑style conductance
  network where broadcast probability acts as a pressure driving flux.

**Mathematical bridge**

We interpret the XGBoost *margin* (the raw model output) as a **pressure** on each
node of a leader‑election graph.  The pressure difference between two connected
nodes drives a **flux** through the edge proportional to its current conductance.
The absolute flux updates the conductance (Physarum dynamics).  Conductance,
in turn, modulates the XGBoost regularisation term `λ` used in leaf‑weight and
split‑gain calculations, thus coupling the two systems.

The hybrid temperature `T_h = T_cool * p_broadcast` (Parent B) is used as a
global scaling factor for the regularisation strength, providing an annealed
control over tree growth while the conductance network continuously reshapes
the effective penalty on each split.

The implementation below provides:
* `binary_logistic_grad_hess` – gradient/hessian (Parent A).
* `flux_update_conductance` – Physarum flux and conductance update (Parent B).
* `hybrid_split_gain` – split gain that incorporates conductance‑modulated
  regularisation and the hybrid temperature.
* `HybridXGBoostPhysarum` – a lightweight orchestrator exposing the three core
  operations.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A utilities (XGBoost core)
# ----------------------------------------------------------------------
def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-margin))


def binary_logistic_grad_hess(
    y_true: np.ndarray, margin: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient and hessian for binary logistic loss.

    Parameters
    ----------
    y_true : np.ndarray
        Ground‑truth labels (0 or 1).
    margin : np.ndarray
        Raw model predictions (logits).

    Returns
    -------
    g, h : tuple of np.ndarray
        Gradient and Hessian vectors.
    """
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h


def optimal_leaf_weight(
    gradient_sum: float, hessian_sum: float, reg_lambda: float = 1.0
) -> float:
    """
    Closed‑form optimal leaf weight for XGBoost with L2 regularisation.
    """
    return -float(gradient_sum) / (float(hessian_sum) + float(reg_lambda))


def split_gain(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
) -> float:
    """
    Classic XGBoost split gain.
    """
    gl, hl = float(left_gradient), float(left_hessian)
    gr, hr = float(right_gradient), float(right_hessian)
    parent = (gl + gr) ** 2 / (hl + hr + reg_lambda)
    children = gl ** 2 / (hl + reg_lambda) + gr ** 2 / (hr + reg_lambda)
    return 0.5 * (children - parent) - gamma


# ----------------------------------------------------------------------
# Parent B utilities (Physarum leader election)
# ----------------------------------------------------------------------
Node = str
Graph = Dict[Node, List[Node]]
Edge = Tuple[Node, Node]


@dataclass
class HybridLeaderElection:
    graph: Graph
    phases: int
    phase: int
    t0: float = 1.0
    alpha: float = 0.95
    edge_length: float = 1.0
    eps: float = 1e-12
    conductance: Dict[Edge, float] = field(default_factory=dict)

    def __post_init__(self):
        # Initialise unit conductance for every directed edge if not supplied
        if not self.conductance:
            self.conductance = {
                (u, v): 1.0
                for u, nbrs in self.graph.items()
                for v in nbrs
            }

    # ----- broadcast probability (Parent A's decreasing schedule) -----
    def broadcast_probability(self) -> float:
        """p = 1 / 2^{phases‑phase}, clamped to [0,1]."""
        if self.phases < 1 or self.phase < 1:
            raise ValueError("phases and phase must be positive")
        return min(1.0, 1.0 / (2 ** max(0, self.phases - self.phase)))

    # ----- cooling schedule (Parent B) -----
    def cooling_temperature(self) -> float:
        """Exponential cooling."""
        if self.phase < 0 or self.t0 < 0 or not (0 <= self.alpha <= 1):
            raise ValueError("invalid cooling parameters")
        return self.t0 * (self.alpha ** (self.phase - 1))

    # ----- hybrid temperature -----
    def hybrid_temperature(self) -> float:
        """Product of broadcast probability and cooling temperature."""
        return self.broadcast_probability() * self.cooling_temperature()

    # ----- flux through an edge -----
    def flux(
        self, edge: Edge, pressure: Dict[Node, float]
    ) -> float:
        """
        Physarum flux: ϕ_{ab} = g_{ab} * (p_a - p_b) / L,
        where g is conductance, L is edge length (constant here).
        """
        a, b = edge
        g = self.conductance.get(edge, self.eps)
        delta_p = pressure.get(a, 0.0) - pressure.get(b, 0.0)
        return g * delta_p / self.edge_length

    # ----- conductance update (Physarum dynamics) -----
    def update_conductance(
        self, pressure: Dict[Node, float]
    ) -> None:
        """
        Conductance evolves as:
            g_{ab}^{new} = (|ϕ_{ab}| + eps) / (1 + T_h)
        where T_h is the hybrid temperature (global annealing factor).
        """
        T_h = self.hybrid_temperature()
        for edge in list(self.conductance.keys()):
            phi = self.flux(edge, pressure)
            self.conductance[edge] = (abs(phi) + self.eps) / (1.0 + T_h)


# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_split_gain(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    *,
    base_lambda: float = 1.0,
    gamma: float = 0.0,
    edge_conductance: float,
    hybrid_temp: float,
) -> float:
    """
    Split gain where the regularisation λ is scaled by both the edge conductance
    and the hybrid temperature:

        λ_eff = base_lambda * (1 + edge_conductance) * hybrid_temp

    This creates a smooth coupling: high conductance (strong flux) relaxes the
    penalty, while a low hybrid temperature (early phases) tightens it.
    """
    λ_eff = base_lambda * (1.0 + edge_conductance) * hybrid_temp
    return split_gain(
        left_gradient,
        left_hessian,
        right_gradient,
        right_hessian,
        reg_lambda=λ_eff,
        gamma=gamma,
    )


def flux_update_conductance(
    election: HybridLeaderElection,
    margins: Dict[Node, float],
) -> None:
    """
    Wrapper that computes pressures from XGBoost margins and triggers the
    Physarum conductance update.
    """
    # Use margins directly as pressures
    election.update_conductance(pressure=margins)


def compute_hybrid_leaf_weight(
    gradient_sum: float,
    hessian_sum: float,
    *,
    base_lambda: float = 1.0,
    edge_conductance: float,
    hybrid_temp: float,
) -> float:
    """
    Leaf weight using conductance‑modulated regularisation.
    """
    λ_eff = base_lambda * (1.0 + edge_conductance) * hybrid_temp
    return optimal_leaf_weight(gradient_sum, hessian_sum, reg_lambda=λ_eff)


# ----------------------------------------------------------------------
# Orchestrator class
# ----------------------------------------------------------------------
class HybridXGBoostPhysarum:
    """
    High‑level façade that ties together XGBoost gradient/hessian computation,
    Physarum conductance dynamics, and hybrid split‑gain evaluation.
    """

    def __init__(
        self,
        graph: Graph,
        phases: int,
        t0: float = 1.0,
        alpha: float = 0.95,
        base_lambda: float = 1.0,
        gamma: float = 0.0,
    ):
        self.election = HybridLeaderElection(
            graph=graph,
            phases=phases,
            phase=1,
            t0=t0,
            alpha=alpha,
        )
        self.base_lambda = base_lambda
        self.gamma = gamma

    def step(
        self,
        y_true: np.ndarray,
        margin: np.ndarray,
        node_mapping: List[Node],
    ) -> Tuple[float, Dict[Edge, float]]:
        """
        Perform one hybrid iteration:

        1. Compute gradient/hessian per sample.
        2. Aggregate per graph node (via `node_mapping` list of same length as data).
        3. Update conductance using margins as pressures.
        4. Return a representative split gain for the first edge (demo purpose)
           and the updated conductance map.
        """
        if len(y_true) != len(margin) or len(y_true) != len(node_mapping):
            raise ValueError("Input arrays must share the same length")

        # 1. XGBoost gradients
        g, h = binary_logistic_grad_hess(y_true, margin)

        # 2. Aggregate per node
        grad_sum: Dict[Node, float] = {}
        hess_sum: Dict[Node, float] = {}
        for node, gi, hi in zip(node_mapping, g, h):
            grad_sum[node] = grad_sum.get(node, 0.0) + float(gi)
            hess_sum[node] = hess_sum.get(node, 0.0) + float(hi)

        # 3. Conductance update (Physarum)
        # Build pressure dict from raw margins (average per node)
        pressure: Dict[Node, float] = {}
        count: Dict[Node, int] = {}
        for node, m in zip(node_mapping, margin):
            pressure[node] = pressure.get(node, 0.0) + float(m)
            count[node] = count.get(node, 0) + 1
        for node in pressure:
            pressure[node] /= max(1, count[node])

        flux_update_conductance(self.election, pressure)

        # 4. Demonstrate hybrid split gain on the first edge
        first_edge = next(iter(self.election.conductance))
        a, b = first_edge
        # Use aggregated statistics of the two incident nodes as left/right stats
        left_g = grad_sum.get(a, 0.0)
        left_h = hess_sum.get(a, 0.0)
        right_g = grad_sum.get(b, 0.0)
        right_h = hess_sum.get(b, 0.0)

        split_gain_val = hybrid_split_gain(
            left_g,
            left_h,
            right_g,
            right_h,
            base_lambda=self.base_lambda,
            gamma=self.gamma,
            edge_conductance=self.election.conductance[first_edge],
            hybrid_temp=self.election.hybrid_temperature(),
        )

        # Advance phase for next iteration
        self.election.phase = min(self.election.phases, self.election.phase + 1)

        return split_gain_val, dict(self.election.conductance)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic data
    np.random.seed(42)
    y = np.random.randint(0, 2, size=20).astype(float)
    logits = np.random.randn(20) * 0.5

    # Build a tiny undirected graph (represented as directed adjacency)
    graph = {
        "A": ["B", "C"],
        "B": ["A", "C"],
        "C": ["A", "B"],
    }

    # Randomly assign each sample to a graph node
    nodes = np.random.choice(list(graph.keys()), size=20)

    hybrid = HybridXGBoostPhysarum(
        graph=graph,
        phases=5,
        t0=1.0,
        alpha=0.9,
        base_lambda=1.0,
        gamma=0.1,
    )

    # Run a few hybrid steps
    for step in range(3):
        gain, conduct = hybrid.step(y_true=y, margin=logits, node_mapping=nodes)
        print(f"Step {step+1}: split_gain={gain:.6f}")
        # Show a couple of conductance values for inspection
        sample_edges = list(conduct.items())[:3]
        for e, g in sample_edges:
            print(f"  edge {e} -> conductance {g:.4f}")
        print("-" * 40)