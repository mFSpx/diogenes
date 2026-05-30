# DARWIN HAMMER — match 2, survivor 4
# gen: 3
# parent_a: hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s0.py (gen2)
# parent_b: epistemic_certainty.py (gen0)
# born: 2026-05-29T23:25:08Z

"""Hybrid Sheaf‑Certainty Cohomology (HSCC)

This module fuses two distinct parent algorithms:

* **HybridSheaf** – a sheaf‑theoretic representation of Count‑Min sketches
  and MinHash LSH, using linear restriction maps between node‑sections and
  edge‑sections.  Its core operation is the coboundary operator
  `δ(section) = R_u·s_u – R_v·s_v` for each edge (u,v).

* **CertaintyFlag** – a deterministic epistemic‑certainty container that
  attaches a confidence weight (basis points 0‥10000) to any piece of
  information.

The mathematical bridge is the **confidence weight**.  By interpreting a
`CertaintyFlag` as a scalar factor `w = confidence_bps / 10000`,
we can scale the linear maps and sections before applying the coboundary
operator.  The resulting **certainty‑weighted coboundary** measures
information loss while respecting epistemic certainty, and its norm can be
used as a surrogate for the Real Log Canonical Threshold (RLCT) in the
original hybrid sketch algorithm.

The implementation provides a `HybridSheafCertainty` class that stores
sheaf data together with per‑node and per‑edge certainty flags, and three
utility functions that demonstrate the hybrid workflow.
"""

import numpy as np
import hashlib
from collections import defaultdict
import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Tuple, Dict, List

# ----------------------------------------------------------------------
# Parent B – Epistemic certainty helpers (adapted)
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
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


def filesystem_observation(*, sha256: str, path: str, mtime_utc: str | None = None) -> CertaintyFlag:
    refs = [f"sha256:{sha256}", f"path:{path}"]
    if mtime_utc:
        refs.append(f"mtime:{mtime_utc}")
    return certainty(
        "FACT",
        confidence_bps=10000,
        authority_class="filesystem_observation",
        rationale="Local file bytes were hashed and copied into CAS; this proves byte custody, not semantic truth.",
        evidence_refs=refs,
    )


def parser_extraction(*, sha256: str, extract_method: str, injection_detected: bool = False) -> CertaintyFlag:
    if injection_detected:
        return certainty(
            "BULLSHIT",
            confidence_bps=9000,
            authority_class="prompt_injection_signature",
            rationale="Untrusted source text matched instruction‑injection signature",
            evidence_refs=[f"sha256:{sha256}", f"method:{extract_method}"],
        )
    return certainty(
        "FACT",
        confidence_bps=9500,
        authority_class="parser_extraction",
        rationale="Parser successfully extracted data",
        evidence_refs=[f"sha256:{sha256}", f"method:{extract_method}"],
    )


# ----------------------------------------------------------------------
# Parent A – HybridSheaf (trimmed & extended)
# ----------------------------------------------------------------------
class HybridSheaf:
    """
    Minimal sheaf structure supporting:
      * node dimensions (size of each local vector)
      * edges with linear restriction maps (src → edge, dst → edge)
      * sections (vectors) attached to nodes
    """

    def __init__(self, node_dims: Dict[Any, int], edge_list: List[Tuple[Any, Any]], width: int = 64, depth: int = 4):
        self.node_dims: Dict[Any, int] = dict(node_dims)
        self.edges: List[Tuple[Any, Any]] = list(edge_list)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}
        self.width = width
        self.depth = depth

    # ------------------------------------------------------------------
    # API for building the sheaf
    # ------------------------------------------------------------------
    def set_restriction(self, edge: Tuple[Any, Any], src_map: Iterable[float], dst_map: Iterable[float]) -> None:
        """Register linear maps for edge (u,v).  Both maps are 2‑D arrays with
        shape (edge_dim, node_dim).  For simplicity we accept flat iterables and
        reshape automatically based on node dimensions."""
        u, v = edge
        src_dim = self.node_dims[u]
        dst_dim = self.node_dims[v]

        src_arr = np.array(src_map, dtype=float).reshape(-1, src_dim)
        dst_arr = np.array(dst_map, dtype=float).reshape(-1, dst_dim)

        if src_arr.shape[0] != dst_arr.shape[0]:
            raise ValueError("src and dst restriction maps must have the same edge dimension")
        self._restrictions[(u, v)] = (src_arr, dst_arr)

    def set_section(self, node: Any, value: Iterable[float]) -> None:
        """Attach a local vector (section) to a node."""
        dim = self.node_dims[node]
        arr = np.array(value, dtype=float).reshape(dim)
        self._sections[node] = arr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _edge_dim(self, u: Any, v: Any) -> int:
        if (u, v) in self._restrictions:
            return self._restrictions[(u, v)][0].shape[0]
        if (v, u) in self._restrictions:
            return self._restrictions[(v, u)][1].shape[0]
        raise KeyError(f"No restriction map for edge ({u}, {v})")

    def _c0_layout(self):
        """Return (nodes, offsets, total_dim) for the 0‑cochain space."""
        nodes = list(self.node_dims.keys())
        offsets: Dict[Any, int] = {}
        pos = 0
        for n in nodes:
            offsets[n] = pos
            pos += self.node_dims[n]
        return nodes, offsets, pos

    def _c1_layout(self):
        """Return (offsets, total_dim) for the 1‑cochain space (edges)."""
        offsets: Dict[Tuple[Any, Any], Tuple[int, int]] = {}
        pos = 0
        for e in self.edges:
            u, v = e
            d = self._edge_dim(u, v)
            offsets[e] = (pos, d)
            pos += d
        return offsets, pos

    # ------------------------------------------------------------------
    # Coboundary operator (δ)
    # ------------------------------------------------------------------
    def coboundary(self) -> np.ndarray:
        """Compute the raw coboundary vector δ(s) ∈ ⨁_edges ℝ^{edge_dim}."""
        edge_offsets, total_dim = self._c1_layout()
        result = np.zeros(total_dim, dtype=float)

        for (u, v) in self.edges:
            src_map, dst_map = self._restrictions[(u, v)]
            s_u = self._sections[u]
            s_v = self._sections[v]

            # δ_e = R_u·s_u - R_v·s_v
            delta_e = src_map @ s_u - dst_map @ s_v

            off, dim = edge_offsets[(u, v)]
            result[off : off + dim] = delta_e

        return result


# ----------------------------------------------------------------------
# HybridSheafCertainty – the fused structure
# ----------------------------------------------------------------------
class HybridSheafCertainty(HybridSheaf):
    """
    Extends HybridSheaf by attaching a CertaintyFlag to every node and edge.
    The confidence weight w = confidence_bps / 10000 scales the corresponding
    linear maps and sections before the coboundary is evaluated.
    """

    def __init__(self, node_dims, edge_list, width=64, depth=4):
        super().__init__(node_dims, edge_list, width, depth)
        self.node_cert: Dict[Any, CertaintyFlag] = {}
        self.edge_cert: Dict[Tuple[Any, Any], CertaintyFlag] = {}

    # ------------------------------------------------------------------
    # Certainty setters
    # ------------------------------------------------------------------
    def set_node_certainty(self, node: Any, flag: CertaintyFlag) -> None:
        if node not in self.node_dims:
            raise KeyError(f"unknown node {node!r}")
        self.node_cert[node] = flag

    def set_edge_certainty(self, edge: Tuple[Any, Any], flag: CertaintyFlag) -> None:
        if edge not in self.edges:
            raise KeyError(f"unknown edge {edge!r}")
        self.edge_cert[edge] = flag

    # ------------------------------------------------------------------
    # Weighted coboundary
    # ------------------------------------------------------------------
    def weighted_coboundary(self) -> np.ndarray:
        """
        Compute δ_w(s) where each restriction map and each node section is
        multiplied by sqrt(w) (the square‑root keeps the overall norm scaling
        linear in confidence).  Edge‑wise confidence further scales the final
        edge contribution.
        """
        edge_offsets, total_dim = self._c1_layout()
        result = np.zeros(total_dim, dtype=float)

        for (u, v) in self.edges:
            # Retrieve raw maps
            src_map, dst_map = self._restrictions[(u, v)]
            s_u = self._sections[u]
            s_v = self._sections[v]

            # Node confidence weights (default to 1.0 if missing)
            w_u = self.node_cert.get(u, certainty("PROBABLE", confidence_bps=5000,
                                                 authority_class="default", rationale="missing")).confidence_bps / 10000.0
            w_v = self.node_cert.get(v, certainty("PROBABLE", confidence_bps=5000,
                                                 authority_class="default", rationale="missing")).confidence_bps / 10000.0

            # Edge confidence weight
            w_e = self.edge_cert.get((u, v), certainty("PROBABLE", confidence_bps=5000,
                                                       authority_class="default", rationale="missing")).confidence_bps / 10000.0

            # Scale maps and sections
            src_scaled = math.sqrt(w_u) * src_map
            dst_scaled = math.sqrt(w_v) * dst_map
            s_u_scaled = math.sqrt(w_u) * s_u
            s_v_scaled = math.sqrt(w_v) * s_v

            delta_e = src_scaled @ s_u_scaled - dst_scaled @ s_v_scaled
            delta_e *= math.sqrt(w_e)  # final edge scaling

            off, dim = edge_offsets[(u, v)]
            result[off : off + dim] = delta_e

        return result

    # ------------------------------------------------------------------
    # Global inconsistency measure (RLCT surrogate)
    # ------------------------------------------------------------------
    def global_inconsistency(self) -> float:
        """
        Return the L2 norm of the weighted coboundary.  A larger value indicates
        higher information loss / lower global coherence, analogous to a higher
        RLCT.
        """
        vec = self.weighted_coboundary()
        return float(np.linalg.norm(vec, 2))


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def build_example_sheaf() -> HybridSheafCertainty:
    """
    Construct a tiny sheaf with two nodes and one edge, attach random
    sections, linear maps, and certainty flags.
    """
    node_dims = {"A": 3, "B": 2}
    edges = [("A", "B")]
    sheaf = HybridSheafCertainty(node_dims, edges)

    # Random sections
    rng = np.random.default_rng(42)
    sheaf.set_section("A", rng.normal(size=3))
    sheaf.set_section("B", rng.normal(size=2))

    # Restriction maps: edge_dim = 2 (chosen arbitrarily)
    src_map = rng.normal(size=(2, 3))
    dst_map = rng.normal(size=(2, 2))
    sheaf.set_restriction(("A", "B"), src_map, dst_map)

    # Certainty flags
    sheaf.set_node_certainty("A", filesystem_observation(sha256="a"*64, path="/tmp/a"))
    sheaf.set_node_certainty("B", parser_extraction(sha256="b"*64, extract_method="json"))
    sheaf.set_edge_certainty(("A", "B"), certainty(
        "PROBABLE",
        confidence_bps=7500,
        authority_class="edge_estimate",
        rationale="heuristic similarity",
        evidence_refs=["sim:0.73"]
    ))
    return sheaf


def compute_and_report(sheaf: HybridSheafCertainty) -> None:
    """
    Compute the weighted coboundary, its norm, and emit a summary
    CertaintyFlag that reflects the overall epistemic state.
    """
    vec = sheaf.weighted_coboundary()
    norm = np.linalg.norm(vec, 2)

    # Derive a synthetic overall confidence (higher norm → lower confidence)
    # Map norm in [0, ∞) to confidence in [10000, 2000] (arbitrary heuristic)
    confidence = max(2000, int(10000 - 2000 * math.tanh(norm)))
    overall_flag = certainty(
        "PROBABLE" if confidence > 5000 else "POSSIBLE",
        confidence_bps=confidence,
        authority_class="hybrid_sheaf_certainty",
        rationale=f"Weighted coboundary norm={norm:.3f}",
        evidence_refs=[f"norm:{norm:.3f}"]
    )
    print("Weighted coboundary vector:", vec)
    print("L2 norm:", norm)
    print("Overall epistemic flag:", overall_flag.as_dict())


def rlct_estimate(sheaf: HybridSheafCertainty) -> float:
    """
    Provide a simple RLCT‑like estimate using the global inconsistency
    divided by the sum of all confidence weights.
    """
    inc = sheaf.global_inconsistency()
    total_conf = sum(f.confidence_bps for f in sheaf.node_cert.values()) + \
                 sum(f.confidence_bps for f in sheaf.edge_cert.values())
    if total_conf == 0:
        return float("inf")
    return inc / (total_conf / 10000.0)


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    example = build_example_sheaf()
    compute_and_report(example)
    rlct_val = rlct_estimate(example)
    print(f"RLCT‑like estimate: {rlct_val:.4f}")