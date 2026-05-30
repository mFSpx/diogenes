# DARWIN HAMMER — match 4133, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2557_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m2543_s2.py (gen6)
# born: 2026-05-29T23:53:48Z

import numpy as np
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

# ----------------------------------------------------------------------
# Regex‑based evidence extraction (Algorithm A)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|"
    r"receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|"
    r"check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|"
    r"prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|"
    r"test|smoke)\b",
    re.I,
)

# ----------------------------------------------------------------------
# Stylometry categories (Algorithm B)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers "
        "they them their theirs we us our ours".split()
    ),
    "article": {"a", "an", "the"},
    "preposition": set(
        "about above across against along among around at before behind below beside "
        "between by down during from in inside into near of off on onto out "
        "outside over past since till under underneath until up upon with within "
        "without".split()
    ),
}

# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class PheromoneEntry:
    """A single pheromone signal attached to a textual feature."""
    feature: str
    count: float
    half_life: float
    signal: float

    @staticmethod
    def from_count(feature: str, count: float) -> "PheromoneEntry":
        """Create a pheromone entry; signal at time 0 equals the raw count."""
        # Guard against zero/negative counts (should not happen but be safe)
        count = max(count, 0.0)
        # Half‑life grows with count → high‑frequency features decay slower
        half_life = 1.0 + np.log1p(count)
        signal = count  # initial signal (t = 0)
        return PheromoneEntry(feature, count, half_life, signal)


class HybridSheaf:
    """
    A minimal sheaf implementation on a graph.
    Nodes store sections (vectors) and edges store linear restriction maps.
    Cohomology is obtained from the coboundary matrix between 0‑ and 1‑cochains.
    """

    def __init__(self, node_dims: Dict[str, np.ndarray], edge_list: Dict[str, List[str]]):
        self.node_dims = {k: np.asarray(v, dtype=float) for k, v in node_dims.items()}
        self.edge_list = {k: list(v) for k, v in edge_list.items()}
        self.sections: Dict[str, np.ndarray] = {}          # 0‑cochain values
        self.restrictions: Dict[Tuple[str, str], np.ndarray] = {}  # edge linear maps

    # ------------------------------------------------------------------
    # Building the sheaf
    # ------------------------------------------------------------------
    def set_section(self, node: str, vector: np.ndarray) -> None:
        self.sections[node] = np.asarray(vector, dtype=float)

    def set_restriction(self, src: str, dst: str, matrix: np.ndarray) -> None:
        """Linear map from src‑section to dst‑section."""
        self.restrictions[(src, dst)] = np.asarray(matrix, dtype=float)

    # ------------------------------------------------------------------
    # Cohomology utilities
    # ------------------------------------------------------------------
    def _build_coboundary(self) -> np.ndarray:
        """
        Construct the (|E| × |V|) coboundary matrix B where each row corresponds
        to an oriented edge (u→v) and encodes (restriction_uv) – I.
        """
        nodes = list(self.node_dims.keys())
        node_index = {n: i for i, n in enumerate(nodes)}
        rows: List[np.ndarray] = []

        for u, neighbors in self.edge_list.items():
            for v in neighbors:
                # Identity on source side
                dim = self.node_dims[u].shape[0]
                I = np.eye(dim)

                # Restriction map; if not supplied, use identity (no distortion)
                R = self.restrictions.get((u, v), I)

                # Row = [ … -I_u … +R_uv … ] placed in a block‑diagonal fashion
                row = np.zeros((dim, dim * len(nodes)))
                row[:, node_index[u] * dim : (node_index[u] + 1) * dim] = -I
                row[:, node_index[v] * dim : (node_index[v] + 1) * dim] = R
                rows.append(row)

        if not rows:
            return np.zeros((0, len(nodes) * self._dim()))
        return np.vstack(rows)

    def _dim(self) -> int:
        """Assume all node vectors share the same dimension."""
        sample = next(iter(self.node_dims.values()))
        return int(sample.shape[0])

    def betti_0(self) -> int:
        """
        Compute the 0‑th Betti number = dim(ker B) = #connected components
        when restrictions are identities. With arbitrary restrictions it
        measures the space of global sections.
        """
        B = self._build_coboundary()
        if B.size == 0:
            return len(self.node_dims)  # no edges → each node isolated
        # Use SVD to obtain rank robustly
        _, s, _ = np.linalg.svd(B, full_matrices=False)
        rank = np.sum(s > 1e-10)
        total_dim = self._dim() * len(self.node_dims)
        return int(total_dim - rank)

    # ------------------------------------------------------------------
    # Utility to combine sections into a single vector for curvature weighting
    # ------------------------------------------------------------------
    def aggregate_section_norm(self) -> float:
        """Return the Frobenius norm of the concatenated sections."""
        if not self.sections:
            return 0.0
        vecs = [self.sections[n] for n in sorted(self.sections.keys())]
        return np.linalg.norm(np.concatenate(vecs))


# ----------------------------------------------------------------------
# Core mathematical utilities
# ----------------------------------------------------------------------
def compute_ollivier_ricci_curvature(
    node_dims: Dict[str, np.ndarray],
    edge_list: Dict[str, List[str]],
    weight_factor: float = 1.0,
) -> Dict[str, float]:
    """
    Simple Ollivier‑Ricci estimator.
    The curvature of node u is summed over neighbours v:
        κ(u) = Σ 1 / (1 + ‖dim_u – dim_v‖)  × weight_factor
    """
    curvature: Dict[str, float] = {}
    for u, dim_u in node_dims.items():
        cur = 0.0
        for v in edge_list.get(u, []):
            dim_v = node_dims[v]
            cur += 1.0 / (1.0 + np.linalg.norm(dim_u - dim_v))
        curvature[u] = cur * weight_factor
    return curvature


def compute_ternary_lens_audit_report(text: str) -> Dict[Tuple[str, str], float]:
    """
    Produce a richer edge map:
        – edges between evidence tokens and planning tokens,
        – self‑edges for repeated category tokens.
    The weight is the co‑occurrence count within the same sentence.
    """
    report: Dict[Tuple[str, str], float] = {}
    sentences = re.split(r"[.!?]\s*", text)

    for sent in sentences:
        ev_tokens = [m.group().lower() for m in EVIDENCE_RE.finditer(sent)]
        pl_tokens = [m.group().lower() for m in PLANNING_RE.finditer(sent)]

        # cross‑category edges
        for ev in ev_tokens:
            for pl in pl_tokens:
                edge = (ev, pl)
                report[edge] = report.get(edge, 0.0) + 1.0

        # intra‑category self‑edges (captures frequency)
        for token in ev_tokens + pl_tokens:
            edge = (token, token)
            report[edge] = report.get(edge, 0.0) + 1.0

    return report


# ----------------------------------------------------------------------
# Hybrid algorithm
# ----------------------------------------------------------------------
def hybrid_algorithm(
    text: str,
    node_dims: Dict[str, np.ndarray],
    edge_list: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    1. Build a ternary‑lens audit report.
    2. Translate report edges into pheromone entries.
    3. Populate a sheaf whose restrictions encode half‑life decay.
    4. Compute 0‑th Betti number (global section space size).
    5. Use Betti number to modulate Ollivier‑Ricci curvature.
    """
    # 1. Audit report
    report = compute_ternary_lens_audit_report(text)

    # 2. Pheromone entries
    pheromones: List[PheromoneEntry] = []
    for (src, dst), cnt in report.items():
        feature = f"{src}|{dst}"
        pheromones.append(PheromoneEntry.from_count(feature, cnt))

    # 3. Sheaf construction
    sheaf = HybridSheaf(node_dims, edge_list)

    # Sections: each node gets a vector whose length equals the sheaf dimension.
    dim = sheaf._dim()
    for node in node_dims:
        # Initialise with zeros; later we will add pheromone contributions.
        sheaf.set_section(node, np.zeros(dim))

    # Populate sections and edge restrictions
    for ph in pheromones:
        src, dst = ph.feature.split("|")
        # If the feature matches a node name, add its signal to that node's section.
        if src in sheaf.sections:
            sheaf.sections[src] += ph.signal * np.ones(dim) / ph.half_life
        if dst in sheaf.sections:
            sheaf.sections[dst] += ph.signal * np.ones(dim) / ph.half_life

        # Define a simple restriction map for the oriented edge src→dst.
        # The map scales the source vector by exp(-1/half_life) to model decay.
        if src in node_dims and dst in node_dims:
            scale = math.exp(-1.0 / ph.half_life)
            R = scale * np.eye(dim)
            sheaf.set_restriction(src, dst, R)

    # 4. Betti‑0 (size of global section space)
    beta0 = sheaf.betti_0()

    # 5. Curvature weighting: more global sections → stronger smoothing
    weight = 1.0 + 0.1 * beta0
    curvature = compute_ollivier_ricci_curvature(node_dims, edge_list, weight_factor=weight)

    # Optional: attach a diagnostic field
    curvature["_beta0"] = float(beta0)
    curvature["_aggregate_section_norm"] = float(sheaf.aggregate_section_norm())

    return curvature


# ----------------------------------------------------------------------
# Example usage (executed only when run as script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The audit revealed evidence of a breach. "
        "We must plan a remediation checklist and schedule a follow‑up. "
        "Further evidence was logged and documented."
    )
    node_dims_example = {
        "evidence": np.array([1.0, 0.5, 0.2]),
        "plan": np.array([0.9, 0.4, 0.3]),
        "schedule": np.array([0.8, 0.6, 0.1]),
        "audit": np.array([1.2, 0.7, 0.4]),
    }
    edge_list_example = {
        "evidence": ["plan", "audit"],
        "plan": ["schedule"],
        "schedule": ["plan"],
        "audit": ["evidence"],
    }

    result = hybrid_algorithm(sample_text, node_dims_example, edge_list_example)
    for k, v in result.items():
        print(f"{k}: {v:.4f}")