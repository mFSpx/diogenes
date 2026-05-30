# DARWIN HAMMER — match 64, survivor 0
# gen: 4
# parent_a: physarum_network.py (gen0)
# parent_b: hybrid_hybrid_hybrid_sketch_epistemic_certainty_m2_s3.py (gen3)
# born: 2026-05-29T23:26:38Z

"""Hybrid Physarum‑Sheaf Certainty Module.

Parents:
- physarum_network.py (flux‑based conductance dynamics)
- hybrid_hybrid_hybrid_sketch_epistemic_certainty_m2_s3.py (sheaf sections with epistemic certainty)

Mathematical bridge:
Each edge (u,v) carries a *conductance* g_uv (physarum) and a *restriction matrix* R_uv
(mapping the sheaf section at u into the edge space).  
Node u possesses a pressure p_u, a sheaf section s_u∈ℝ^{d_u} and a certainty scalar
c_u∈[0,1] derived from a `CertaintyFlag`.  

For an edge we define two coupled quantities:

1. **Flux** (physarum)  
   q_uv = g_uv / ℓ_uv * (p_u − p_v)

2. **Weighted discrepancy** (sheaf‑certainty)  
   δ_uv = R_uv·s_u − s_v′          (pull‑back of s_v to the edge space)  
   d_uv = √(c_u·c_v) * ‖δ_uv‖₂

The hybrid conductance update blends the absolute flux |q_uv| (information
transport) with the discrepancy magnitude d_uv (information loss) :

   g_uv ← max(0, g_uv + Δt·(α·|q_uv| + β·d_uv − γ·g_uv))

where α,β are gains and γ is a decay rate.  This fuses the physarum
self‑reinforcement mechanism with a confidence‑weighted sheaf cohomology
penalty, yielding a unified dynamical system.

The module implements the above equations and provides a small smoke test.
"""

from __future__ import annotations

import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Certainty infrastructure (from epistemic_certainty.py)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=confidence_bps,
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(evidence_refs),
    )


# ----------------------------------------------------------------------
# Core physarum‑sheaf primitives
# ----------------------------------------------------------------------
def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    """Flux on an edge given conductance, length and endpoint pressures."""
    if edge_length <= 0:
        raise ValueError("edge_length must be positive")
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def weighted_discrepancy(
    R: np.ndarray,
    s_u: np.ndarray,
    s_v: np.ndarray,
    certainty_u: float,
    certainty_v: float,
) -> float:
    """
    Confidence‑weighted ℓ₂ discrepancy for a sheaf edge.

    Parameters
    ----------
    R : np.ndarray
        Restriction matrix from node u to the edge space.
    s_u : np.ndarray
        Section at node u (shape compatible with R).
    s_v : np.ndarray
        Section at node v already expressed in the edge space.
    certainty_u, certainty_v : float
        Scalars in [0,1] derived from CertaintyFlag confidence.

    Returns
    -------
    d : float
        Weighted discrepancy magnitude.
    """
    delta = R @ s_u - s_v
    norm = np.linalg.norm(delta)  # ℓ₂ norm
    weight = math.sqrt(max(0.0, certainty_u * certainty_v))
    return weight * norm


def hybrid_update_conductance(
    conductance: float,
    q: float,
    d: float,
    dt: float = 1.0,
    gain_flux: float = 1.0,
    gain_disc: float = 0.5,
    decay: float = 0.05,
) -> float:
    """
    Conductance update that blends flux magnitude and sheaf discrepancy.

    g ← max(0, g + dt·(gain_flux·|q| + gain_disc·d − decay·g))
    """
    if dt < 0 or decay < 0:
        raise ValueError("dt and decay must be non‑negative")
    increment = gain_flux * abs(q) + gain_disc * d - decay * conductance
    return max(0.0, conductance + dt * increment)


# ----------------------------------------------------------------------
# Graph‑level helpers
# ----------------------------------------------------------------------
NodeId = int
EdgeId = Tuple[NodeId, NodeId]


class HybridGraph:
    """
    Simple undirected graph storing:
    - pressures (scalar)
    - sheaf sections (numpy vectors)
    - certainty flags (scalar confidence)
    - edge lengths
    - conductances (physarum)
    - restriction matrices per directed edge
    """

    def __init__(self) -> None:
        self.nodes: Dict[NodeId, Dict[str, Any]] = {}
        self.edges: Dict[EdgeId, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Node manipulation
    # ------------------------------------------------------------------
    def add_node(
        self,
        node: NodeId,
        pressure: float,
        section: np.ndarray,
        certainty_flag: CertaintyFlag,
    ) -> None:
        self.nodes[node] = {
            "pressure": float(pressure),
            "section": np.asarray(section, dtype=float),
            "certainty": certainty_flag.confidence_bps / 10000.0,
        }

    # ------------------------------------------------------------------
    # Edge manipulation
    # ------------------------------------------------------------------
    def add_edge(
        self,
        u: NodeId,
        v: NodeId,
        length: float,
        conductance: float,
        restriction_uv: np.ndarray,
        restriction_vu: np.ndarray,
    ) -> None:
        if length <= 0:
            raise ValueError("edge length must be positive")
        key = (min(u, v), max(u, v))
        self.edges[key] = {
            "length": float(length),
            "conductance": float(conductance),
            "R_uv": np.asarray(restriction_uv, dtype=float),
            "R_vu": np.asarray(restriction_vu, dtype=float),
        }

    # ------------------------------------------------------------------
    # Hybrid dynamics
    # ------------------------------------------------------------------
    def step(
        self,
        dt: float = 1.0,
        gain_flux: float = 1.0,
        gain_disc: float = 0.5,
        decay: float = 0.05,
    ) -> None:
        """
        Perform one hybrid update for all edges.
        Conductances are updated in‑place.
        """
        # Compute updates first to avoid order‑dependence
        updates: List[Tuple[EdgeId, float]] = []

        for (a, b), data in self.edges.items():
            # Retrieve node data (order respects stored direction)
            if a < b:
                u, v = a, b
                R_uv, R_vu = data["R_uv"], data["R_vu"]
            else:
                u, v = b, a
                R_uv, R_vu = data["R_vu"], data["R_uv"]

            p_u = self.nodes[u]["pressure"]
            p_v = self.nodes[v]["pressure"]
            s_u = self.nodes[u]["section"]
            s_v = self.nodes[v]["section"]
            c_u = self.nodes[u]["certainty"]
            c_v = self.nodes[v]["certainty"]
            length = data["length"]
            g = data["conductance"]

            # 1) Physarum flux
            q = flux(g, length, p_u, p_v)

            # 2) Sheaf weighted discrepancy (use restriction from u→v)
            # Pull‑back s_v into the edge space using the transpose of R_uv
            # (a simple symmetric choice for this demo)
            s_v_edge = R_uv.T @ s_v if R_uv.shape[0] == s_v.shape[0] else s_v
            d = weighted_discrepancy(R_uv, s_u, s_v_edge, c_u, c_v)

            # 3) Hybrid conductance update
            g_new = hybrid_update_conductance(g, q, d, dt, gain_flux, gain_disc, decay)
            updates.append(((a, b), g_new))

        # Apply updates
        for key, g_new in updates:
            self.edges[key]["conductance"] = g_new

    # ------------------------------------------------------------------
    # Utility for inspection
    # ------------------------------------------------------------------
    def conductances(self) -> Dict[EdgeId, float]:
        return {k: v["conductance"] for k, v in self.edges.items()}


# ----------------------------------------------------------------------
# Demonstration functions (required at least three)
# ----------------------------------------------------------------------
def create_demo_graph() -> HybridGraph:
    """
    Build a tiny graph with two nodes and a single edge.
    Node sections are 2‑vectors, restriction matrices are 2×2 identity.
    Certainty flags are arbitrary but valid.
    """
    g = HybridGraph()
    # Node 0
    flag0 = certainty(
        "FACT",
        confidence_bps=8000,
        authority_class="A",
        rationale="experimental measurement",
    )
    g.add_node(0, pressure=1.0, section=np.array([1.0, 0.0]), certainty_flag=flag0)

    # Node 1
    flag1 = certainty(
        "PROBABLE",
        confidence_bps=5000,
        authority_class="B",
        rationale="simulation estimate",
    )
    g.add_node(1, pressure=0.0, section=np.array([0.0, 1.0]), certainty_flag=flag1)

    # Edge 0‑1
    length = 1.0
    conductance = 0.1
    R = np.eye(2)  # identity restriction in both directions
    g.add_edge(0, 1, length, conductance, restriction_uv=R, restriction_vu=R)
    return g


def run_hybrid_simulation(steps: int = 10) -> List[Dict[EdgeId, float]]:
    """
    Execute `steps` iterations of the hybrid dynamics on the demo graph.
    Returns a list of conductance snapshots.
    """
    graph = create_demo_graph()
    snapshots = [graph.conductances()]
    for _ in range(steps):
        graph.step(dt=0.5, gain_flux=1.2, gain_disc=0.3, decay=0.04)
        snapshots.append(graph.conductances())
    return snapshots


def print_simulation_results(snapshots: List[Dict[EdgeId, float]]) -> None:
    """Pretty‑print conductance evolution."""
    for i, state in enumerate(snapshots):
        g_val = list(state.values())[0] if state else float("nan")
        print(f"Step {i:2d}: conductance = {g_val:.6f}")


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    results = run_hybrid_simulation(steps=15)
    print_simulation_results(results)