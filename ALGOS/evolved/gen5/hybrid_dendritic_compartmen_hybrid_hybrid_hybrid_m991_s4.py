# DARWIN HAMMER — match 991, survivor 4
# gen: 5
# parent_a: dendritic_compartment.py (gen0)
# parent_b: hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s2.py (gen4)
# born: 2026-05-29T23:32:07Z

"""Hybrid Dendritic‑Regret Ternary Analyzer (HDRTA)

This module fuses two distinct parents:

* **Parent A – dendritic_compartment.py**  
  Provides Hodgkin‑Huxley multi‑compartment ODEs for a dendritic tree.

* **Parent B – hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s2.py**  
  Supplies regret‑weighted probabilities, a mapping to a ternary alphabet and a
  path‑signature‑pruning mechanism.

**Mathematical bridge** – The membrane potentials **V_i** of each compartment are
interpreted as “expected values” of abstract actions.  Regret‑weighted
probabilities are computed from these values, then mapped to a ternary symbol
(‑1, 0, 1).  The resulting symbolic sequence is fed to a *path‑signature* that
accumulates the product of ternary symbols along every root‑to‑leaf path.
If the absolute signature of a branch falls below a user‑defined threshold the
axial conductance *g_ij* of that branch is set to zero, effectively pruning the
dendritic sub‑tree.  Thus the biophysical dynamics (Parent A) and the
information‑theoretic pruning (Parent B) are tightly coupled in a single
update step.

The public API offers three illustrative hybrid functions and a simple
simulation driver.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (from Parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


# ----------------------------------------------------------------------
# Hodgkin‑Huxley utilities (from Parent A)
# ----------------------------------------------------------------------
def alpha_m(V: float) -> float:
    """Alpha for Na activation."""
    return 0.1 * (V + 40.0) / (1.0 - math.exp(-(V + 40.0) / 10.0))


def beta_m(V: float) -> float:
    """Beta for Na activation."""
    return 4.0 * math.exp(-(V + 65.0) / 18.0)


def alpha_h(V: float) -> float:
    """Alpha for Na inactivation."""
    return 0.07 * math.exp(-(V + 65.0) / 20.0)


def beta_h(V: float) -> float:
    """Beta for Na inactivation."""
    return 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))


def alpha_n(V: float) -> float:
    """Alpha for K activation."""
    return 0.01 * (V + 55.0) / (1.0 - math.exp(-(V + 55.0) / 10.0))


def beta_n(V: float) -> float:
    """Beta for K activation."""
    return 0.125 * math.exp(-(V + 65) / 80.0)


def update_gate(x: float, alpha: float, beta: float, dt: float) -> float:
    """Euler update for a gating variable."""
    return x + dt * (alpha * (1.0 - x) - beta * x)


def sodium_current(V: np.ndarray, m: np.ndarray, h: np.ndarray,
                   g_Na: float = 120.0, E_Na: float = 50.0) -> np.ndarray:
    return g_Na * (m ** 3) * h * (V - E_Na)


def potassium_current(V: np.ndarray, n: np.ndarray,
                     g_K: float = 36.0, E_K: float = -77.0) -> np.ndarray:
    return g_K * (n ** 4) * (V - E_K)


def leak_current(V: np.ndarray, g_L: float = 0.3, E_L: float = -54.387) -> np.ndarray:
    return g_L * (V - E_L)


# ----------------------------------------------------------------------
# Tree data structure
# ----------------------------------------------------------------------
class DendriteTree:
    """
    Simple undirected tree where each node is a compartment.
    Attributes
    ----------
    N : int
        Number of compartments.
    adj : List[List[int]]
        Adjacency list.
    g_axial : np.ndarray
        Symmetric matrix of axial conductances (size N×N).  Zero where no edge.
    C_m : float
        Membrane capacitance (same for all compartments).
    """
    def __init__(self, N: int):
        self.N = N
        self.adj: List[List[int]] = [[] for _ in range(N)]
        self.g_axial = np.zeros((N, N), dtype=float)
        self.C_m = 1.0  # uF/cm^2 (typical)

    def add_edge(self, i: int, j: int, g_ij: float = 0.1):
        """Create an undirected axial connection."""
        self.adj[i].append(j)
        self.adj[j].append(i)
        self.g_axial[i, j] = self.g_axial[j, i] = g_ij

    def prune_edge(self, i: int, j: int):
        """Set conductance to zero – logical pruning."""
        if j in self.adj[i]:
            self.adj[i].remove(j)
        if i in self.adj[j]:
            self.adj[j].remove(i)
        self.g_axial[i, j] = self.g_axial[j, i] = 0.0


def build_linear_dendrite(num_compartments: int, g_axial: float = 0.1) -> DendriteTree:
    """Convenient builder for a straight chain of compartments."""
    tree = DendriteTree(num_compartments)
    for i in range(num_compartments - 1):
        tree.add_edge(i, i + 1, g_axial)
    return tree


# ----------------------------------------------------------------------
# Regret‑weighted ternary utilities (from Parent B)
# ----------------------------------------------------------------------
def calculate_regret_weighted_probabilities(actions: List[MathAction]) -> np.ndarray:
    """
    Regret‑weighted probability: p_i ∝ expected_value_i.
    Normalised to sum to 1.
    """
    ev = np.array([a.expected_value for a in actions], dtype=float)
    total = ev.sum()
    if total == 0:
        # avoid division by zero – uniform distribution
        return np.full_like(ev, 1.0 / len(ev))
    return ev / total


def map_probabilities_to_ternary(probs: np.ndarray) -> np.ndarray:
    """
    Map probabilities onto a ternary alphabet:
        p > 2/3      →  1
        1/3 ≤ p ≤ 2/3 → 0
        p < 1/3      → -1
    Returns an integer array of -1, 0, 1.
    """
    ternary = np.where(probs > 2.0 / 3.0, 1,
               np.where(probs < 1.0 / 3.0, -1, 0))
    return ternary.astype(int)


def path_signature(tree: DendriteTree, ternary_seq: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    Compute a simple multiplicative signature for every axial edge (i, j):
    signature(i,j) = product of ternary symbols on the unique path from the leaf
    node *j* (assumed deeper) up to *i* (assumed parent).  For a linear chain the
    signature reduces to the product of symbols between the two nodes.
    Returns a dictionary keyed by ordered edge tuples (parent, child).
    """
    sig: Dict[Tuple[int, int], float] = {}
    # Perform a DFS from node 0 (root) gathering path products.
    stack = [(0, -1, 1.0)]  # node, parent, accumulated product
    while stack:
        node, parent, acc = stack.pop()
        for child in tree.adj[node]:
            if child == parent:
                continue
            new_acc = acc * ternary_seq[child]
            sig[(node, child)] = new_acc
            stack.append((child, node, new_acc))
    return sig


def prune_tree_by_signature(tree: DendriteTree,
                            ternary_seq: np.ndarray,
                            threshold: float = 0.2) -> None:
    """
    Prune edges whose absolute signature falls below *threshold*.
    The pruning is performed in‑place.
    """
    signatures = path_signature(tree, ternary_seq)
    for (i, j), value in signatures.items():
        if abs(value) < threshold:
            tree.prune_edge(i, j)


# ----------------------------------------------------------------------
# Hybrid operation – three demonstrative functions
# ----------------------------------------------------------------------
def hybrid_regret_ternary_from_potentials(V: np.ndarray) -> np.ndarray:
    """
    Treat each compartment voltage as an expected value, compute regret‑weighted
    probabilities and map them to the ternary alphabet.
    """
    actions = [MathAction(id=str(idx), expected_value=float(v))
               for idx, v in enumerate(V)]
    probs = calculate_regret_weighted_probabilities(actions)
    return map_probabilities_to_ternary(probs)


def hybrid_compartment_step(tree: DendriteTree,
                            V: np.ndarray,
                            m: np.ndarray,
                            h: np.ndarray,
                            n: np.ndarray,
                            dt: float = 0.01,
                            prune_threshold: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    One hybrid integration step:
    1. Standard HH update for each compartment.
    2. Convert resulting voltages to ternary symbols.
    3. Prune the dendritic tree according to the path‑signature rule.
    Returns updated (V, m, h, n).
    """
    # ---- Hodgkin‑Huxley dynamics ------------------------------------
    I_Na = sodium_current(V, m, h)
    I_K = potassium_current(V, n)
    I_L = leak_current(V)

    # Axial coupling term: sum_j g_ij * (V_j - V_i)
    coupling = np.zeros_like(V)
    for i in range(tree.N):
        for j in tree.adj[i]:
            coupling[i] += tree.g_axial[i, j] * (V[j] - V[i])

    dV = ( - I_Na - I_K - I_L + coupling ) / tree.C_m
    V_new = V + dt * dV

    # ---- Gating updates --------------------------------------------
    m_new = np.empty_like(m)
    h_new = np.empty_like(h)
    n_new = np.empty_like(n)
    for idx, Vc in enumerate(V):
        a_m, b_m = alpha_m(Vc), beta_m(Vc)
        a_h, b_h = alpha_h(Vc), beta_h(Vc)
        a_n, b_n = alpha_n(Vc), beta_n(Vc)
        m_new[idx] = update_gate(m[idx], a_m, b_m, dt)
        h_new[idx] = update_gate(h[idx], a_h, b_h, dt)
        n_new[idx] = update_gate(n[idx], a_n, b_n, dt)

    # ---- Regret‑ternary mapping & pruning ---------------------------
    ternary_seq = hybrid_regret_ternary_from_potentials(V_new)
    prune_tree_by_signature(tree, ternary_seq, threshold=prune_threshold)

    return V_new, m_new, h_new, n_new


def simulate_hybrid_tree(duration: float = 5.0,
                         dt: float = 0.01,
                         num_compartments: int = 5) -> Tuple[np.ndarray, DendriteTree]:
    """
    Run a short hybrid simulation.
    Returns the time‑series of soma potentials (compartment 0) and the final tree.
    """
    steps = int(duration / dt)
    tree = build_linear_dendrite(num_compartments)

    # Initialise state variables
    V = -65.0 * np.ones(num_compartments)          # mV
    m = 0.05 * np.ones(num_compartments)
    h = 0.6  * np.ones(num_compartments)
    n = 0.32 * np.ones(num_compartments)

    soma_trace = np.empty(steps)

    for t in range(steps):
        V, m, h, n = hybrid_compartment_step(tree, V, m, h, n, dt)
        soma_trace[t] = V[0]  # record soma (index 0)

    return soma_trace, tree


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Run a quick simulation and print a few soma values.
    trace, final_tree = simulate_hybrid_tree(duration=0.2, dt=0.01, num_compartments=4)
    print("Soma voltage trace (mV):")
    print(trace[:10])
    print("\nFinal adjacency list after pruning:")
    for idx, neighbors in enumerate(final_tree.adj):
        print(f"  Node {idx}: {neighbors}")
    sys.exit(0)