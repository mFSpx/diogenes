# DARWIN HAMMER — match 2, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s0.py (gen2)
# parent_b: epistemic_certainty.py (gen0)
# born: 2026-05-29T23:25:08Z

"""Hybrid Sheaf‑Certainty Module
Combines:
- `HybridSheaf` from *hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s0.py* (matrix‑based sheaf cohomology)
- `CertaintyFlag` utilities from *epistemic_certainty.py* (scalar confidence metadata)

Mathematical bridge:
Each sheaf section  s(v) ∈ ℝ^{d_v} is now paired with a certainty scalar  c(v) ∈ [0,1] derived from
`CertaintyFlag.confidence_bps / 10000`.  The coboundary discrepancy on an edge (u,v)

    δ_{u→v} = R_{u→v}·s(u) – s(v)′

(where R_{u→v} is the restriction linear map and s(v)′ is the pull‑back of s(v) to the
edge space) is weighted by the geometric mean of the endpoint certainties.
Thus the global inconsistency metric becomes a confidence‑weighted ℓ₂‑norm,
providing a unified measure of information loss (RLCT‑style) and epistemic certainty.
"""

import numpy as np
import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Tuple, Dict, List

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
        rationale=(
            "Local file bytes were hashed and copied into CAS; this proves byte custody, "
            "not semantic truth."
        ),
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
        rationale="Parser extracted data without detecting injection",
        evidence_refs=[f"sha256:{sha256}", f"method:{extract_method}"],
    )


# ----------------------------------------------------------------------
# Sheaf infrastructure (from hybrid_hybrid_sketches_rlct_sheaf_cohomology_m11_s0.py)
# ----------------------------------------------------------------------
class HybridSheaf:
    def __init__(self, node_dims: Dict[Any, int], edge_list: List[Tuple[Any, Any]], width: int = 64, depth: int = 4):
        self.node_dims = dict(node_dims)               # node → dimension
        self.edges = list(edge_list)                  # list of (u, v)
        self._restrictions: Dict[Tuple[Any, Any], Tuple[np.ndarray, np.ndarray]] = {}
        self._sections: Dict[Any, np.ndarray] = {}
        self.width = width
        self.depth = depth

    def set_restriction(self, edge: Tuple[Any, Any], src_map: Iterable[float], dst_map: Iterable[float]) -> None:
        u, v = edge
        src_map = np.array(src_map, dtype=float)
        dst_map = np.array(dst_map, dtype=float)
        self._restrictions[(u, v)] = (src_map, dst_map)

    def set_section(self, node: Any, value: Iterable[float]) -> None:
        self._sections[node] = np.array(value, dtype=float)

    def _edge_dim(self, u: Any, v: Any) -> int:
        if (u, v) in self._restrictions:
            return self._restrictions[(u, v)][0].shape[0]
        if (v, u) in self._restrictions:
            return self._restrictions[(v, u)][1].shape[0]
        raise KeyError(f"No restriction map for edge ({u}, {v})")

    def _c0_layout(self):
        nodes = list(self.node_dims.keys())
        offsets = {}
        pos = 0
        for n in nodes:
            offsets[n] = pos
            pos += self.node_dims[n]
        return nodes, offsets, pos

    def _c1_layout(self):
        offsets = {}
        pos = 0
        for e in self.edges:
            u, v = e
            d = self._edge_dim(u, v)
            offsets[e] = (pos, d)
            pos += d
        return offsets, pos


# ----------------------------------------------------------------------
# Hybrid sheaf‑certainty class
# ----------------------------------------------------------------------
class HybridSheafCertainty(HybridSheaf):
    """
    Extends HybridSheaf by attaching a CertaintyFlag to each node section.
    The confidence weight w(v) = flag.confidence_bps / 10000 ∈ [0,1] scales the
    contribution of node v in global inconsistency calculations.
    """

    def __init__(self, node_dims, edge_list, width=64, depth=4):
        super().__init__(node_dims, edge_list, width, depth)
        self._certainties: Dict[Any, CertaintyFlag] = {}

    def set_section_with_certainty(
        self,
        node: Any,
        value: Iterable[float],
        flag: CertaintyFlag,
    ) -> None:
        self.set_section(node, value)
        self._certainties[node] = flag

    def _confidence(self, node: Any) -> float:
        flag = self._certainties.get(node)
        if flag is None:
            return 0.0
        return flag.confidence_bps / 10000.0

    def weighted_edge_discrepancy(self, edge: Tuple[Any, Any]) -> float:
        """
        Compute || R_{u→v}·s(u) – s(v)′ ||₂² weighted by sqrt(conf_u * conf_v).
        """
        u, v = edge
        # Retrieve restriction maps
        if (u, v) in self._restrictions:
            src_map, dst_map = self._restrictions[(u, v)]
            # src_map : ℝ^{dim_u} → ℝ^{edge_dim}
            # dst_map : ℝ^{dim_v} → ℝ^{edge_dim}
        elif (v, u) in self._restrictions:
            # use the opposite orientation
            dst_map, src_map = self._restrictions[(v, u)]
        else:
            raise KeyError(f"No restriction for edge {edge}")

        su = self._sections[u]
        sv = self._sections[v]

        # Project sections onto the edge space
        proj_u = src_map @ su
        proj_v = dst_map @ sv

        diff = proj_u - proj_v
        norm_sq = float(np.linalg.norm(diff) ** 2)

        # Confidence weighting (geometric mean)
        cw = math.sqrt(self._confidence(u) * self._confidence(v))
        return cw * norm_sq

    def total_weighted_inconsistency(self) -> float:
        """Sum of weighted edge discrepancies over the entire sheaf."""
        total = 0.0
        for e in self.edges:
            total += self.weighted_edge_discrepancy(e)
        return total

    def rlct_estimate(self) -> float:
        """
        A toy Real Log Canonical Threshold estimator:
        RLCT ≈ 0.5 * log(1 + total_weighted_inconsistency)
        """
        inc = self.total_weighted_inconsistency()
        return 0.5 * math.log1p(inc)


# ----------------------------------------------------------------------
# Public API – three demonstration functions
# ----------------------------------------------------------------------
def make_hybrid_sheaf(
    node_dims: Dict[Any, int],
    edges: List[Tuple[Any, Any]],
) -> HybridSheafCertainty:
    """
    Construct a HybridSheafCertainty with identity restrictions on every edge.
    """
    sheaf = HybridSheafCertainty(node_dims, edges)
    for (u, v) in edges:
        du = node_dims[u]
        dv = node_dims[v]
        edge_dim = min(du, dv)  # simple choice
        # Restriction maps: truncate / pad with zeros to match edge_dim
        src_map = np.zeros((edge_dim, du))
        dst_map = np.zeros((edge_dim, dv))
        for i in range(edge_dim):
            src_map[i, i] = 1.0
            dst_map[i, i] = 1.0
        sheaf.set_restriction((u, v), src_map, dst_map)
    return sheaf


def add_section_with_certainty(
    sheaf: HybridSheafCertainty,
    node: Any,
    values: Iterable[float],
    flag: CertaintyFlag,
) -> None:
    """
    Wrapper that forwards to HybridSheafCertainty.set_section_with_certainty.
    """
    sheaf.set_section_with_certainty(node, values, flag)


def compute_global_metrics(sheaf: HybridSheafCertainty) -> Tuple[float, float]:
    """
    Returns a tuple (total_weighted_inconsistency, rlct_estimate).
    """
    inc = sheaf.total_weighted_inconsistency()
    rlct = sheaf.rlct_estimate()
    return inc, rlct


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple graph: two nodes connected by one edge
    node_dims = {"A": 3, "B": 3}
    edges = [("A", "B")]

    # Build hybrid sheaf
    hs = make_hybrid_sheaf(node_dims, edges)

    # Random sections
    rng = np.random.default_rng(42)
    sec_A = rng.random(3)
    sec_B = rng.random(3)

    # Certainty flags
    flag_A = filesystem_observation(sha256="deadbeef"*8, path="/tmp/a")
    flag_B = parser_extraction(sha256="feedface"*8, extract_method="json", injection_detected=False)

    # Attach sections with certainty
    add_section_with_certainty(hs, "A", sec_A, flag_A)
    add_section_with_certainty(hs, "B", sec_B, flag_B)

    # Compute metrics
    inc, rlct = compute_global_metrics(hs)

    print(f"Weighted inconsistency: {inc:.6f}")
    print(f"RLCT estimate: {rlct:.6f}")

    # Ensure the code runs without raising
    sys.exit(0)