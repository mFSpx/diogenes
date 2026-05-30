# DARWIN HAMMER — match 2686, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s3.py (gen3)
# parent_b: dendritic_compartment.py (gen0)
# born: 2026-05-29T23:43:40Z

"""
Hybrid Module: Sheaf‑Associative‑Dendrite (SAD)  
Parents:
- hybrid_hybrid_hybrid_sketch_dense_associative_me_m32_s3.py (Sheaf + Dense Associative Memory)
- dendritic_compartment.py (Hodgkin‑Huxley multi‑compartment ODEs)

Mathematical Bridge
-------------------
A Sheaf assigns a vector (section) to each node of a directed graph.  
In a dendritic tree each compartment *i* has a membrane potential V_i that
obeys the Hodgkin‑Huxley ODE and is coupled to neighbours j by axial
conductances g_ij.  By interpreting the restriction maps of a Sheaf edge
(u→v) as the *linear coupling* between the sections of the two nodes, we
obtain exactly the conductance term  

    g_ij (V_j - V_i)   ↔   src_map·s_u – dst_map·s_v .

Thus the sheaf’s restriction matrices become the *connectivity matrix* of the
dendritic tree.  The Dense Associative Memory (DAM) supplies an energy
landscape  

    E_DAM(s) = - (1/β) log Σ_k exp(β·p_k·s)  

with patterns p_k.  Adding this scalar to the total membrane‑energy yields a
single unified energy whose gradient drives the dynamics of both the
membrane potentials (via HH currents) and the global associative term
(via a synaptic‑like current I_syn = -∂E_DAM/∂s).

The hybrid system therefore evolves the sections s_i (≈ V_i) according to

    C_m·dV_i/dt = -I_Li - I_Nai - I_Ki
                  + Σ_j g_ij (V_j - V_i)        # sheaf restriction coupling
                  - ∂E_DAM/∂V_i                 # associative drive

The code below implements this fusion, provides three high‑level hybrid
functions, and a minimal smoke test.
"""

import numpy as np
import random
import math
import sys
import pathlib

__all__ = [
    "Sheaf",
    "DenseAssociativeMemory",
    "HybridDendriticNetwork",
    "hybrid_energy",
    "hybrid_step",
    "hybrid_retrieve",
]

# ---------------------------------------------------------------------------
# 1. Sheaf (graph + linear restrictions)
# ---------------------------------------------------------------------------

class Sheaf:
    """
    Cellular sheaf on a directed graph.

    * node_dims: dict mapping node identifier → dimension (int)
    * edges: list of (u, v) directed edges
    * Restrictions are stored as (src_map, dst_map) where each map projects the
      node vector onto a common edge space.
    """
    def __init__(self, node_dims: dict, edges: list):
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions = {}
        self._sections = {}

    def set_restriction(self, edge: tuple, src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        src_map = np.asarray(src_map, dtype=float)
        dst_map = np.asarray(dst_map, dtype=float)
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float)
        if value.shape[0] != self.node_dims[node]:
            raise ValueError("Section dimension must match node dimension")
        self._sections[node] = value

    def get_section(self, node):
        return self._sections.get(node, np.zeros(self.node_dims[node]))

    def get_restriction(self, edge):
        return self._restrictions.get(edge, (None, None))

# ---------------------------------------------------------------------------
# 2. Dense Associative Memory (modern Hopfield network)
# ---------------------------------------------------------------------------

class DenseAssociativeMemory:
    """
    Classical dense associative memory with a softmax‑based energy.
    Patterns are rows of shape (P, N) where N equals the dimensionality of the
    global state vector.
    """
    def __init__(self, patterns: np.ndarray, beta: float = 1.0):
        self.patterns = np.asarray(patterns, dtype=float)  # shape (P, N)
        self.beta = float(beta)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        z = np.asarray(z, dtype=float)
        z_max = np.max(z)
        e = np.exp(self.beta * (z - z_max))
        return e / np.sum(e)

    def energy(self, state: np.ndarray) -> float:
        """
        Energy E(state) = -(1/β) log Σ_k exp(β·p_k·state)
        """
        state = np.asarray(state, dtype=float)
        inner = self.patterns @ state  # shape (P,)
        # Use log‑sum‑exp trick
        max_inner = np.max(self.beta * inner)
        sum_exp = np.sum(np.exp(self.beta * inner - max_inner))
        return -(1.0 / self.beta) * (max_inner + np.log(sum_exp))

    def gradient(self, state: np.ndarray) -> np.ndarray:
        """
        ∂E/∂state = - Σ_k w_k p_k   where w_k = softmax(β·p_k·state)
        """
        state = np.asarray(state, dtype=float)
        inner = self.patterns @ state
        w = self._softmax(inner)[:, None]          # shape (P,1)
        return -np.sum(w * self.patterns, axis=0)   # shape (N,)

    def retrieve(self, init_state: np.ndarray, dt: float = 0.1, steps: int = 100) -> np.ndarray:
        """
        Gradient descent on the energy (continuous Hopfield dynamics).
        """
        s = np.asarray(init_state, dtype=float).copy()
        for _ in range(steps):
            grad = self.gradient(s)
            s -= dt * grad
        return s

# ---------------------------------------------------------------------------
# 3. Hodgkin‑Huxley utilities (from dendritic_compartment.py)
# ---------------------------------------------------------------------------

def alpha_m(V): return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
def beta_m(V):  return 4.0 * np.exp(-(V + 65.0) / 18.0)

def alpha_h(V): return 0.07 * np.exp(-(V + 65.0) / 20.0)
def beta_h(V):  return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

def alpha_n(V): return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
def beta_n(V):  return 0.125 * np.exp(-(V + 65) / 80.0)

def sodium_current(V, m, h, g_Na=120.0, E_Na=50.0):
    return g_Na * (m ** 3) * h * (V - E_Na)

def potassium_current(V, n, g_K=36.0, E_K=-77.0):
    return g_K * (n ** 4) * (V - E_K)

def leak_current(V, g_L=0.3, E_L=-54.387):
    return g_L * (V - E_L)

# ---------------------------------------------------------------------------
# 4. Hybrid Dendritic Network (Sheaf + HH + DAM)
# ---------------------------------------------------------------------------

class HybridDendriticNetwork:
    """
    Combines:
    * Sheaf → graph topology and linear coupling (g_ij)
    * Hodgkin‑Huxley dynamics per node (membrane potentials, gates)
    * DenseAssociativeMemory → global associative drive
    """
    def __init__(self,
                 sheaf: Sheaf,
                 dam: DenseAssociativeMemory,
                 C_m: float = 1.0,
                 dt: float = 0.01):
        self.sheaf = sheaf
        self.dam = dam
        self.C_m = float(C_m)          # membrane capacitance (common for all nodes)
        self.dt = float(dt)            # integration step

        # Initialise HH gating variables for each node (same dimension as node)
        self.gates = {}  # node → dict with keys 'm','h','n'
        for node, dim in sheaf.node_dims.items():
            # For scalar compartments dim == 1; for vector sections we broadcast
            self.gates[node] = {
                'm': np.full(dim, 0.05),
                'h': np.full(dim, 0.6),
                'n': np.full(dim, 0.32)
            }
            # Initialise sections to resting potential -65 mV
            sheaf.set_section(node, np.full(dim, -65.0))

    def _coupling_current(self, node):
        """
        Compute Σ_j g_ij (V_j - V_i) using sheaf restriction matrices.
        For each outgoing edge (node → v) we interpret src_map as g_ij.
        For each incoming edge (u → node) we interpret dst_map as g_ij.
        """
        V_i = self.sheaf.get_section(node)
        I = np.zeros_like(V_i)

        # Outgoing edges
        for (u, v) in self.sheaf.edges:
            if u == node:
                src_map, dst_map = self.sheaf.get_restriction((u, v))
                V_j = self.sheaf.get_section(v)
                # Conductance matrix is src_map (maps V_i to edge space)
                g_mat = src_map  # shape (e_dim, dim_i)
                # Project V_i and V_j onto edge space
                proj_i = g_mat @ V_i
                proj_j = dst_map @ V_j
                I += (proj_j - proj_i)  # edge contribution summed in node space
        # Incoming edges
        for (u, v) in self.sheaf.edges:
            if v == node:
                src_map, dst_map = self.sheaf.get_restriction((u, v))
                V_u = self.sheaf.get_section(u)
                g_mat = dst_map  # now acts on V_i
                proj_i = g_mat @ V_i
                proj_u = src_map @ V_u
                I += (proj_u - proj_i)

        return I

    def _associative_current(self, node):
        """
        Compute -∂E_DAM/∂V_i for the variables belonging to `node`.
        The global state vector is the concatenation of all node sections in a
        deterministic order (sorted keys).
        """
        # Build global state
        ordered_nodes = sorted(self.sheaf.node_dims.keys())
        state_parts = [self.sheaf.get_section(n) for n in ordered_nodes]
        state = np.concatenate(state_parts)
        grad = self.dam.gradient(state)  # shape (total_dim,)

        # Slice gradient belonging to this node
        start = sum(self.sheaf.node_dims[n] for n in ordered_nodes if n < node)
        end = start + self.sheaf.node_dims[node]
        return -grad[start:end]  # negative because we add I_syn = -∂E/∂V

    def step(self):
        """
        Perform a single integration step for all nodes using explicit Euler.
        """
        # Cache new values before writing (to keep update synchronous)
        new_sections = {}
        new_gates = {}

        for node, dim in self.sheaf.node_dims.items():
            V = self.sheaf.get_section(node)
            m = self.gates[node]['m']
            h = self.gates[node]['h']
            n = self.gates[node]['n']

            # Hodgkin‑Huxley ionic currents
            I_Na = sodium_current(V, m, h)
            I_K = potassium_current(V, n)
            I_L = leak_current(V)

            # Linear coupling via sheaf restrictions
            I_coup = self._coupling_current(node)

            # Global associative drive
            I_assoc = self._associative_current(node)

            # Membrane equation: C_m dV/dt = - (I_Na + I_K + I_L) + I_coup + I_assoc
            dV = ( -I_Na - I_K - I_L + I_coup + I_assoc ) / self.C_m
            V_new = V + self.dt * dV

            # Update gating variables (Euler)
            a_m = alpha_m(V)
            b_m = beta_m(V)
            a_h = alpha_h(V)
            b_h = beta_h(V)
            a_n = alpha_n(V)
            b_n = beta_n(V)

            dm = a_m * (1 - m) - b_m * m
            dh = a_h * (1 - h) - b_h * h
            dn = a_n * (1 - n) - b_n * n

            m_new = m + self.dt * dm
            h_new = h + self.dt * dh
            n_new = n + self.dt * dn

            # Store
            new_sections[node] = V_new
            new_gates[node] = {'m': m_new, 'h': h_new, 'n': n_new}

        # Commit updates
        for node, V_new in new_sections.items():
            self.sheaf.set_section(node, V_new)
            self.gates[node] = new_gates[node]

    def run(self, steps: int = 1000):
        """Integrate the hybrid system for `steps` timesteps."""
        for _ in range(steps):
            self.step()

    def current_state(self):
        """Return the concatenated global state vector."""
        ordered_nodes = sorted(self.sheaf.node_dims.keys())
        return np.concatenate([self.sheaf.get_section(n) for n in ordered_nodes])

# ---------------------------------------------------------------------------
# 5. High‑level hybrid utilities
# ---------------------------------------------------------------------------

def hybrid_energy(network: HybridDendriticNetwork) -> float:
    """
    Composite energy = Σ_i (∫ I_ion dV)  +  E_DAM(state)

    For simplicity we approximate the ionic contribution by the standard HH
    energy density ½ C_m V² (ignoring the detailed integral of ionic currents).
    """
    state = network.current_state()
    ionic_energy = 0.5 * network.C_m * np.sum(state ** 2)
    associative_energy = network.dam.energy(state)
    return ionic_energy + associative_energy

def hybrid_step(network: HybridDendriticNetwork) -> None:
    """One explicit Euler step plus a diagnostic print of the hybrid energy."""
    prev_E = hybrid_energy(network)
    network.step()
    new_E = hybrid_energy(network)
    # Simple monotonicity check (energy should not increase dramatically)
    if new_E > prev_E + 1e-3:
        print(f"Warning: Energy increased from {prev_E:.4f} to {new_E:.4f}")

def hybrid_retrieve(network: HybridDendriticNetwork,
                    max_iter: int = 2000,
                    tol: float = 1e-4) -> np.ndarray:
    """
    Run the hybrid dynamics until the global state stabilises (Δstate < tol)
    or `max_iter` steps are performed.  Returns the converged state.
    """
    prev_state = network.current_state()
    for i in range(max_iter):
        hybrid_step(network)
        cur_state = network.current_state()
        if np.linalg.norm(cur_state - prev_state) < tol:
            print(f"Converged after {i+1} iterations.")
            break
        prev_state = cur_state
    return cur_state

# ---------------------------------------------------------------------------
# 6. Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Construct a tiny directed graph (3 nodes) with 1‑dimensional sections
    node_dims = {'A': 1, 'B': 1, 'C': 1}
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]

    sheaf = Sheaf(node_dims, edges)

    # Simple symmetric conductance matrices (scalar conductance g=0.1)
    g = 0.1
    for (u, v) in edges:
        src = np.array([[g]])   # maps V_u (1,) → edge space (1,)
        dst = np.array([[g]])   # maps V_v → edge space
        sheaf.set_restriction((u, v), src, dst)

    # Define two random patterns for associative memory (dimension = 3)
    patterns = np.array([
        [ 1.0, -1.0,  0.5],
        [-0.5,  0.8, -1.2]
    ])

    dam = DenseAssociativeMemory(patterns, beta=2.0)

    #