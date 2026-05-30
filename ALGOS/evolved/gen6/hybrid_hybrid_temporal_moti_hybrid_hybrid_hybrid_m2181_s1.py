# DARWIN HAMMER — match 2181, survivor 1
# gen: 6
# parent_a: hybrid_temporal_motifs_hybrid_hybrid_hybrid_m867_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_dendritic_com_m862_s0.py (gen4)
# born: 2026-05-29T23:41:11Z

"""Hybrid Temporal‑Motif / Dendritic‑Compartment Model
Parents:
- hybrid_temporal_motifs_hybrid_hybrid_hybrid_m867_s1.py (sheaf‑based temporal motifs & burst detection)
- hybrid_hybrid_hybrid_hybrid_hybrid_dendritic_com_m862_s0.py (stylometry‑driven Hodgkin‑Huxley & Ollivier‑Ricci curvature)

Mathematical bridge:
Temporal motifs are encoded as sections of a cellular sheaf on a directed graph.
Each node’s section vector is interpreted as a membrane‑potential state vector.
Stylometry feature frequencies extracted from a corpus are used to scale the
ionic conductances (g_Na, g_K) of a Hodgkin‑Huxley‑type ODE defined on each node.
Ollivier‑Ricci curvature computed on the underlying graph provides a
regularisation factor that modulates the gating‑variable dynamics.
Thus the hybrid system simultaneously:
* propagates sheaf sections via linear restriction maps (parent A),
* evolves them with biophysical dynamics modulated by linguistic features
  and curvature (parent B),
* and detects high‑activity bursts via z‑score statistics on the resulting
  potentials.

The code below implements this fusion and provides three public functions that
demonstrate the hybrid operation.
"""

import numpy as np
import math
import random
import sys
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

# ----------------------------------------------------------------------
# Data structures shared by both parents
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BurstSignal:
    key: str
    count: int
    z_score: float

@dataclass(frozen=True)
class TemporalMotif:
    pattern: Tuple[str, ...]
    support: int

# ----------------------------------------------------------------------
# Sheaf definition (from Parent A)
# ----------------------------------------------------------------------
class Sheaf:
    """
    Cellular sheaf on a directed graph.

    * node_dims: mapping node → dimension of its vector space.
    * edges: list of directed edges (u, v).
    * _restrictions[(u,v)] = (src_map, dst_map) where
        src_map : ℝ^{dim(u)} → ℝ^{k}
        dst_map : ℝ^{dim(v)} → ℝ^{k}
    * _sections[node] = vector ∈ ℝ^{dim(node)}
    """

    def __init__(self, node_dims: Dict[Any, int], edges: List[Tuple[Any, Any]]):
        self.node_dims = node_dims
        self.edges = edges
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}

    def set_restriction(self, edge: Tuple[Any, Any],
                        src_map: np.ndarray, dst_map: np.ndarray) -> None:
        u, v = edge
        if src_map.shape[1] != self.node_dims[u]:
            raise ValueError("src_map column dimension must match dim(node u)")
        if dst_map.shape[1] != self.node_dims[v]:
            raise ValueError("dst_map column dimension must match dim(node v)")
        if src_map.shape[0] != dst_map.shape[0]:
            raise ValueError("src_map and dst_map must have the same row dimension")
        self._restrictions[(u, v)] = (np.asarray(src_map, dtype=float),
                                      np.asarray(dst_map, dtype=float))

    def set_section(self, node: Any, value: np.ndarray) -> None:
        if value.shape[0] != self.node_dims[node]:
            raise ValueError("section vector length must match node dimension")
        self._sections[node] = np.asarray(value, dtype=float)

    def get_section(self, node: Any) -> np.ndarray:
        return self._sections.get(node, np.zeros(self.node_dims[node]))

    def propagate(self) -> None:
        """
        One propagation step: for each edge (u,v) enforce consistency
        by projecting the source and destination sections onto the shared
        restriction space and averaging.
        """
        for (u, v), (src_map, dst_map) in self._restrictions.items():
            su = self.get_section(u)
            sv = self.get_section(v)
            proj_u = src_map @ su
            proj_v = dst_map @ sv
            # simple average in the restriction space
            avg = 0.5 * (proj_u + proj_v)
            # lift back to node spaces (least‑squares)
            su_new = np.linalg.lstsq(src_map, avg, rcond=None)[0]
            sv_new = np.linalg.lstsq(dst_map, avg, rcond=None)[0]
            self.set_section(u, su_new)
            self.set_section(v, sv_new)

# ----------------------------------------------------------------------
# Stylometry extraction (from Parent B)
# ----------------------------------------------------------------------
def compute_stylometry_features(text: str) -> Dict[str, float]:
    """
    Very lightweight stylometry: frequency of function‑category words.
    Returns a normalized frequency vector (sum to 1) over the categories.
    """
    words = [w.lower().strip('.,;!?"\'') for w in text.split()]
    cat_counts = Counter()
    total = 0
    for cat, vocab in FUNCTION_CATS.items():
        cnt = sum(1 for w in words if w in vocab)
        cat_counts[cat] = cnt
        total += cnt
    if total == 0:
        return {cat: 0.0 for cat in FUNCTION_CATS}
    return {cat: cat_counts[cat] / total for cat in FUNCTION_CATS}

# ----------------------------------------------------------------------
# Ollivier‑Ricci curvature approximation (from Parent B)
# ----------------------------------------------------------------------
def approx_ollivier_ricci_curvature(edges: List[Tuple[Any, Any]],
                                   node_degrees: Dict[Any, int]) -> Dict[Tuple[Any, Any], float]:
    """
    Simple surrogate: curvature(e) = 1 - |deg(u)-deg(v)| / max_deg
    """
    max_deg = max(node_degrees.values()) if node_degrees else 1
    curvature = {}
    for u, v in edges:
        curv = 1.0 - abs(node_degrees.get(u, 0) - node_degrees.get(v, 0)) / max_deg
        curvature[(u, v)] = curv
    return curvature

# ----------------------------------------------------------------------
# Hodgkin‑Huxley dynamics with stylometry‑modulated conductances
# ----------------------------------------------------------------------
def hh_step(V: float, m: float, h: float, n: float,
            I_ext: float, dt: float,
            gNa_scale: float, gK_scale: float,
            curvature_factor: float) -> Tuple[float, float, float, float]:
    """
    Perform a single Euler step of the HH equations.
    Conductances are scaled by stylometry factors (gNa_scale, gK_scale).
    curvature_factor ∈ [0,1] dampens gating dynamics (higher curvature → slower change).
    Returns updated (V, m, h, n).
    """
    # Constants (typical values)
    C_m = 1.0
    gNa = 120.0 * gNa_scale
    gK  = 36.0  * gK_scale
    gL  = 0.3
    E_Na = 50.0
    E_K  = -77.0
    E_L  = -54.387

    # Alpha/Beta functions
    alpha_m = (0.1*(V+40.0)) / (1.0 - math.exp(-(V+40.0)/10.0))
    beta_m  = 4.0 * math.exp(-(V+65.0)/18.0)
    alpha_h = 0.07 * math.exp(-(V+65.0)/20.0)
    beta_h  = 1.0 / (1.0 + math.exp(-(V+35.0)/10.0))
    alpha_n = (0.01*(V+55.0)) / (1.0 - math.exp(-(V+55.0)/10.0))
    beta_n  = 0.125 * math.exp(-(V+65)/80.0)

    # Gating updates with curvature damping
    dm = curvature_factor * (alpha_m * (1.0 - m) - beta_m * m) * dt
    dh = curvature_factor * (alpha_h * (1.0 - h) - beta_h * h) * dt
    dn = curvature_factor * (alpha_n * (1.0 - n) - beta_n * n) * dt

    m += dm
    h += dh
    n += dn

    # Membrane potential update
    I_Na = gNa * (m**3) * h * (V - E_Na)
    I_K  = gK  * (n**4)      * (V - E_K)
    I_L  = gL               * (V - E_L)

    dV = (I_ext - I_Na - I_K - I_L) / C_m * dt
    V += dV

    return V, m, h, n

# ----------------------------------------------------------------------
# Hybrid functions (demonstrating the fused system)
# ----------------------------------------------------------------------
def initialise_hybrid_sheaf(num_nodes: int) -> Sheaf:
    """
    Create a fully‑connected directed graph of `num_nodes` nodes.
    Each node carries a 4‑dimensional vector representing (V, m, h, n).
    Random linear restrictions are generated for each edge.
    Sections are initialised to typical HH resting values.
    """
    node_dims = {i: 4 for i in range(num_nodes)}
    edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    sheaf = Sheaf(node_dims, edges)

    # Random restrictions (same output dimension for all edges)
    k = 3  # restriction space dimension
    rng = np.random.default_rng(42)
    for (u, v) in edges:
        src_map = rng.normal(size=(k, node_dims[u]))
        dst_map = rng.normal(size=(k, node_dims[v]))
        sheaf.set_restriction((u, v), src_map, dst_map)

    # Initialise sections with HH resting state (V≈-65 mV, gating≈steady)
    V0 = -65.0
    m0 = 0.05
    h0 = 0.6
    n0 = 0.32
    for node in range(num_nodes):
        sheaf.set_section(node, np.array([V0, m0, h0, n0]))
    return sheaf

def hybrid_evolve(sheaf: Sheaf,
                  text_corpus: str,
                  steps: int = 10,
                  dt: float = 0.01,
                  I_ext: float = 10.0) -> List[Dict[int, np.ndarray]]:
    """
    Evolve the sheaf over `steps` Euler steps.
    * Stylometry from `text_corpus` yields scaling factors for gNa and gK.
    * Ollivier‑Ricci curvature provides per‑edge damping factors.
    Returns a history list where each entry maps node → current section vector.
    """
    # 1) Stylometry → conductance scaling
    feats = compute_stylometry_features(text_corpus)
    # Map two categories to conductance scales (arbitrary choice)
    gNa_scale = 0.5 + feats.get('pronoun', 0.0)   # between 0.5 and 1.5
    gK_scale  = 0.5 + feats.get('preposition', 0.0)

    # 2) Curvature per edge
    node_degrees = {n: 0 for n in sheaf.node_dims}
    for u, v in sheaf.edges:
        node_degrees[u] += 1
        node_degrees[v] += 1
    curvature = approx_ollivier_ricci_curvature(sheaf.edges, node_degrees)

    history = []
    for _ in range(steps):
        # Update each node independently using HH step
        new_sections = {}
        for node in sheaf.node_dims:
            V, m, h, n = sheaf.get_section(node)
            # Average curvature of incident edges as damping factor
            incident = [curvature[e] for e in sheaf.edges if e[0] == node or e[1] == node]
            curv_factor = np.mean(incident) if incident else 1.0
            V, m, h, n = hh_step(V, m, h, n, I_ext, dt,
                                 gNa_scale, gK_scale, curv_factor)
            new_sections[node] = np.array([V, m, h, n])
        # Install updated sections
        for node, vec in new_sections.items():
            sheaf.set_section(node, vec)

        # Propagate sheaf consistency (Parent A)
        sheaf.propagate()

        # Record snapshot
        snapshot = {node: sheaf.get_section(node).copy() for node in sheaf.node_dims}
        history.append(snapshot)

    return history

def detect_bursts(history: List[Dict[int, np.ndarray]],
                  z_thresh: float = 2.0) -> List[BurstSignal]:
    """
    Analyse the voltage component (index 0) across all nodes and time.
    Returns BurstSignal objects for nodes whose voltage exceeds mean by
    `z_thresh` standard deviations at any time step.
    """
    # Gather all voltages per node
    node_voltages: Dict[int, List[float]] = {node: [] for node in history[0]}
    for snapshot in history:
        for node, vec in snapshot.items():
            node_voltages[node].append(vec[0])

    bursts: List[BurstSignal] = []
    for node, vals in node_voltages.items():
        arr = np.array(vals)
        mu, sigma = arr.mean(), arr.std(ddof=1)
        if sigma == 0:
            continue
        # count how many times voltage exceeds threshold
        exceed = np.where((arr - mu) / sigma > z_thresh)[0]
        if exceed.size:
            bursts.append(BurstSignal(key=str(node),
                                      count=int(exceed.size),
                                      z_score=float((arr[exceed] - mu).max() / sigma)))
    return bursts

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal example with 5 nodes and a tiny text corpus
    num_nodes = 5
    sheaf = initialise_hybrid_sheaf(num_nodes)

    sample_text = ("I think that the quick brown fox jumps over the lazy dog. "
                   "You can see how it moves across the field and why it is fast.")
    history = hybrid_evolve(sheaf, sample_text, steps=20, dt=0.025, I_ext=7.0)

    bursts = detect_bursts(history, z_thresh=1.5)
    print("Detected bursts:")
    for b in bursts:
        print(f"Node {b.key}: count={b.count}, max_z={b.z_score:.2f}")

    # Verify that history length matches steps
    assert len(history) == 20
    # Verify that each snapshot contains all nodes
    for snap in history:
        assert set(snap.keys()) == set(range(num_nodes))
    print("Smoke test completed successfully.")