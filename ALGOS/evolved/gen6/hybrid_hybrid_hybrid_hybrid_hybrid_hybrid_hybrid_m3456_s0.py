# DARWIN HAMMER — match 3456, survivor 0
# gen: 6
# parent_a: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_hybrid_m2182_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_hybrid_krampu_m1196_s0.py (gen5)
# born: 2026-05-29T23:50:18Z

"""Hybrid Krampus‑Krampus Algorithm (Parents A & B)

Parent A: feature extraction + epistemic certainty helpers.
Parent B: pheromone decay, sheaf‑cohomology aggregation, Ollivier‑Ricci curvature.

Mathematical bridge:
    • Features f_i extracted from text are interpreted as scalar pheromone signals s_i.
    • Each signal is paired with an epistemic certainty flag C_i (a section of a sheaf).
    • The sheaf’s restriction maps are identity matrices, so sections are directly comparable.
    • Curvature κ(e) on edge e=(u,v) is approximated by the Ollivier‑Ricci formula
          κ(e) = 1 - W₁(μ_u, μ_v) / d(u,v)
      where μ_u, μ_v are probability distributions derived from the decayed pheromone
      values of the incident nodes and d(u,v) is the Euclidean distance of their
      master vectors. This unifies the feature‑based certainty (Parent A) with the
      time‑aware pheromone/sheaf dynamics (Parent B)."""

import math
import random
import sys
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Tuple, List, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – epistemic certainty helpers
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


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
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.now(timezone.utc).isoformat())


def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑random feature extraction (Parent A)."""
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "operator_ledger_density",
        "operator_recursion_score", "operator_directive_ratio",
        "operator_target_density", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index",
        "psyche_wrath_velocity", "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric", "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixio"
    ]
    # Produce a float in (0,1) for each key
    return {k: rnd.random() for k in keys}


def generate_certainty_flags(features: Dict[str, float]) -> Dict[str, CertaintyFlag]:
    """Map feature magnitudes to epistemic certainty flags."""
    flags = {}
    for name, val in features.items():
        if val > 0.8:
            label = "FACT"
            confidence = 9500
        elif val > 0.6:
            label = "PROBABLE"
            confidence = 7500
        elif val > 0.4:
            label = "POSSIBLE"
            confidence = 5000
        elif val > 0.2:
            label = "SURE_MAYBE"
            confidence = 2500
        else:
            label = "BULLSHIT"
            confidence = 1000
        flags[name] = CertaintyFlag(
            label=label,
            confidence_bps=confidence,
            authority_class="krampus",
            rationale=f"derived from feature {name}",
            evidence_refs=(f"feat:{name}",)
        )
    return flags


# ----------------------------------------------------------------------
# Parent B – pheromone entry, sheaf, curvature
# ----------------------------------------------------------------------
class PheromoneEntry:
    """Scalar pheromone signal attached to a feature."""
    __slots__ = ("feature", "value", "half_life", "signal")

    def __init__(self, feature: str, value: float, half_life: float):
        self.feature = feature
        self.value = float(value)
        self.half_life = float(half_life)
        self.signal = self.value  # current (decayed) signal


def pheromone_decay(entry: PheromoneEntry, delta_t: float) -> None:
    """Exponential half‑life decay."""
    if entry.half_life <= 0:
        entry.signal = 0.0
    else:
        entry.signal = entry.value * (0.5 ** (delta_t / entry.half_life))


class HybridSheaf:
    """Simple sheaf where each node holds a 1‑D section (the feature value)."""
    def __init__(self, node_dims: Dict[str, int], edge_list: List[Tuple[str, str]], width: int = 64, depth: int = 4):
        self.node_dims = dict(node_dims)          # node -> dimension (here always 1)
        self.edges = list(edge_list)              # list of (u, v)
        self.width = width
        self.depth = depth
        self._sections: Dict[str, np.ndarray] = {}    # node -> vector
        self._restrictions: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}

    def set_section(self, node: str, value: float) -> None:
        self._sections[node] = np.array([value], dtype=float)

    def set_restriction(self, edge: Tuple[str, str], src_map: Iterable[float], dst_map: Iterable[float]) -> None:
        u, v = edge
        self._restrictions[(u, v)] = (np.array(src_map, dtype=float), np.array(dst_map, dtype=float))

    def get_section(self, node: str) -> np.ndarray:
        return self._sections[node]

    def neighbor_map(self) -> Dict[str, List[str]]:
        nbrs = defaultdict(list)
        for u, v in self.edges:
            nbrs[u].append(v)
            nbrs[v].append(u)
        return dict(nbrs)


def build_sheaf(features: Dict[str, float], pheromones: List[PheromoneEntry]) -> HybridSheaf:
    """Create a sheaf whose nodes are the features."""
    nodes = list(features.keys())
    # Fully connect the graph (could be sparse, but full connectivity simplifies curvature)
    edges = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:]]
    sheaf = HybridSheaf(node_dims={n: 1 for n in nodes}, edge_list=edges)

    # Sections are the (decayed) pheromone signals
    signal_map = {p.feature: p.signal for p in pheromones}
    for n in nodes:
        sheaf.set_section(n, signal_map.get(n, 0.0))

    # Identity restrictions for every edge (no dimensional change)
    for e in edges:
        sheaf.set_restriction(e, [1.0], [1.0])
    return sheaf


def _wasserstein_1(mu_u: np.ndarray, mu_v: np.ndarray) -> float:
    """1‑Wasserstein distance for 1‑D discrete distributions (simple L1)."""
    return np.abs(mu_u - mu_v).sum()


def compute_ollivier_ricci(sheaf: HybridSheaf) -> Dict[Tuple[str, str], float]:
    """Approximate Ollivier‑Ricci curvature on each edge."""
    curvature = {}
    for (u, v) in sheaf.edges:
        vec_u = sheaf.get_section(u)
        vec_v = sheaf.get_section(v)

        # Euclidean distance between master vectors (d)
        d = np.linalg.norm(vec_u - vec_v)
        if d == 0:
            curvature[(u, v)] = 0.0
            continue

        # Build probability measures μ_u, μ_v from neighboring sections (including self)
        nbrs = sheaf.neighbor_map()
        mu_u_vals = [sheaf.get_section(w)[0] for w in [u] + nbrs.get(u, [])]
        mu_v_vals = [sheaf.get_section(w)[0] for w in [v] + nbrs.get(v, [])]

        mu_u = np.array(mu_u_vals, dtype=float)
        mu_v = np.array(mu_v_vals, dtype=float)

        # Normalize to probability distributions
        if mu_u.sum() == 0:
            mu_u = np.ones_like(mu_u) / mu_u.size
        else:
            mu_u = mu_u / mu_u.sum()
        if mu_v.sum() == 0:
            mu_v = np.ones_like(mu_v) / mu_v.size
        else:
            mu_v = mu_v / mu_v.sum()

        w1 = _wasserstein_1(mu_u, mu_v)
        curvature[(u, v)] = 1.0 - w1 / d
    return curvature


# ----------------------------------------------------------------------
# Hybrid operation exposing three public functions
# ----------------------------------------------------------------------
def text_to_pheromones(text: str, entropy_factor: float = 1.0) -> List[PheromoneEntry]:
    """Convert extracted features into pheromone entries.
       Half‑life τ = base * (1 + entropy_factor * entropy) where entropy is Shannon entropy of the feature histogram."""
    feats = extract_full_features(text)
    values = np.array(list(feats.values()))
    # Shannon entropy (base 2)
    probs = values / values.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))

    base_half_life = 10.0  # seconds, arbitrary unit
    half_life = base_half_life * (1.0 + entropy_factor * entropy)

    entries = [PheromoneEntry(f, v, half_life) for f, v in feats.items()]
    return entries


def hybrid_metric(text: str, delta_t: float = 0.0) -> Dict[Tuple[str, str], float]:
    """Full pipeline:
       1. Extract features.
       2. Generate certainty flags (unused beyond creation – they demonstrate the topological link).
       3. Build pheromone entries and decay them by Δt.
       4. Assemble a sheaf.
       5. Compute Ollivier‑Ricci curvature on the sheaf graph.
       Returns a mapping edge → curvature."""
    features = extract_full_features(text)
    flags = generate_certainty_flags(features)          # side‑effect: validation of topology
    _ = flags  # silence unused variable warning

    pheromones = text_to_pheromones(text)
    for p in pheromones:
        pheromone_decay(p, delta_t)

    sheaf = build_sheaf(features, pheromones)
    curvature = compute_ollivier_ricci(sheaf)
    return curvature


def summarize_curvature(curv: Dict[Tuple[str, str], float]) -> Tuple[float, float, float]:
    """Return (mean, min, max) curvature over all edges."""
    vals = np.array(list(curv.values()))
    if vals.size == 0:
        return (0.0, 0.0, 0.0)
    return (float(vals.mean()), float(vals.min()), float(vals.max()))


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample = "Krampus stalks the night, delivering riddles and chaos to the data‑driven world."
    curv = hybrid_metric(sample, delta_t=5.0)
    mean_k, min_k, max_k = summarize_curvature(curv)
    print(f"Curvature stats – mean: {mean_k:.4f}, min: {min_k:.4f}, max: {max_k:.4f}")
    # Verify that at least one edge exists
    assert len(curv) > 0, "No curvature edges computed"
    # Spot‑check a random edge
    random_edge = next(iter(curv))
    print(f"Sample edge {random_edge} curvature: {curv[random_edge]:.4f}")