# DARWIN HAMMER — match 1316, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_sketch_hybrid_hybrid_minimu_m662_s2.py (gen3)
# parent_b: hybrid_hybrid_physarum_netw_hybrid_infotaxis_min_m967_s1.py (gen5)
# born: 2026-05-29T23:35:11Z

"""Hybrid Physarum‑Sheaf + Count‑Min‑Sketch / MinHash Epistemic Network
===================================================================

Parent algorithms:
- **hybrid_hybrid_hybrid_sketch_hybrid_hybrid_minimu_m662_s2.py** – provides
  Count‑Min sketch, MinHash LSH, Bayesian epistemic‑certainty handling and a
  simple Euclidean distance metric.
- **hybrid_hybrid_physarum_netw_hybrid_infotaxis_min_m967_s1.py** – provides
  Physarum‑Sheaf flux dynamics, MinHash‑based infotaxis similarity and a
  `CertaintyFlag` dataclass.

Mathematical bridge
-------------------
The fusion rests on the *information‑theoretic* quantities that appear in both
parents:

1. **MinHash signatures** are used in the Physarum‑Sheaf model (parent B) to
   compute a Jaccard‑like similarity `J_uv`.  The same signatures are the
   output of the MinHash component of parent A.
2. **Count‑Min sketch tables** give an inexpensive approximation of token
   frequencies per node.  The *sketch discrepancy*  
   `Δ_uv = Σ|CM_u[d][i] - CM_v[d][i]|` supplies a data‑driven edge weight.
3. **Epistemic certainty flags** (parent A) are attached to edges.  Their
   confidence (`confidence_bps`) is converted into a prior probability and
   updated with the Bayesian formulas from parent A, using the similarity
   `J_uv` as likelihood.

The hybrid edge weight `w_uv` therefore combines a data‑driven term
`Δ_uv` with an epistemic term derived from the posterior flag confidence.
The Physarum flux equation is then modulated by the *effective gain*
`α_uv = J_uv / (1 + w_uv)`.  This yields a unified update rule that respects
both topological (sheaf) and probabilistic (epistemic) structures.

The module implements three core functions that realise this fusion:
`build_features`, `compute_edge_properties`, and `update_sheaf`.
"""

import math
import random
import sys
import pathlib
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Shared constants and utilities
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Bayesian marginal from parent A."""
    if not all(0.0 <= x <= 1.0 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)


def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Bayesian posterior from parent A."""
    if marginal <= 0.0:
        raise ValueError("marginal must be > 0")
    return (likelihood * prior) / marginal


# ----------------------------------------------------------------------
# Certainty infrastructure (from parent B)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat() + "Z")

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")


# ----------------------------------------------------------------------
# Sketch / MinHash utilities (from parent A)
# ----------------------------------------------------------------------
def count_min_sketch(items: Iterable[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Return a Count‑Min sketch table for the given items."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = hashlib.sha256(f"{d}:{item}".encode()).hexdigest()
            idx = int(h, 16) % width
            table[d][idx] += 1
    return table


def minhash_signature(items: Iterable[str], num_hashes: int = 8) -> List[int]:
    """Simple MinHash signature: for each seed compute the minimum hash value."""
    signature = []
    for seed in range(num_hashes):
        min_val = None
        for token in items:
            h = hashlib.blake2b(seed.to_bytes(4, "big") + token.encode(), digest_size=8).digest()
            val = int.from_bytes(h, "big")
            if (min_val is None) or (val < min_val):
                min_val = val
        signature.append(min_val if min_val is not None else 0)
    return signature


def jaccard_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def build_features(
    tokens_by_node: Dict[str, Iterable[str]],
    sketch_width: int = 64,
    sketch_depth: int = 4,
    mh_hashes: int = 8,
) -> Tuple[
    Dict[str, List[List[int]]],  # Count‑Min sketches per node
    Dict[str, List[int]],        # MinHash signatures per node
]:
    """
    Build the data‑reduction structures for each node.

    Returns
    -------
    sketches : dict node → Count‑Min table (depth × width)
    signatures : dict node → MinHash signature (list of ints)
    """
    sketches: Dict[str, List[List[int]]] = {}
    signatures: Dict[str, List[int]] = {}
    for node, tokens in tokens_by_node.items():
        token_list = list(tokens)
        sketches[node] = count_min_sketch(token_list, width=sketch_width, depth=sketch_depth)
        signatures[node] = minhash_signature(token_list, num_hashes=mh_hashes)
    return sketches, signatures


def compute_edge_properties(
    edges: List[Tuple[str, str]],
    sketches: Dict[str, List[List[int]]],
    signatures: Dict[str, List[int]],
    flags_by_edge: Dict[Tuple[str, str], CertaintyFlag],
    false_positive: float = 0.01,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    For each edge compute:
    * sketch discrepancy Δ_uv,
    * Jaccard similarity J_uv,
    * epistemic‑adjusted weight w_uv,
    * flux gain α_uv,
    * discrepancy term d_uv (used in sheaf dynamics).

    Returns a dict edge → property dict.
    """
    edge_props: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for u, v in edges:
        # ----- Sketch discrepancy ------------------------------------------------
        cm_u, cm_v = sketches[u], sketches[v]
        delta = sum(
            abs(cm_u[d][i] - cm_v[d][i])
            for d in range(len(cm_u))
            for i in range(len(cm_u[0]))
        )

        # ----- MinHash Jaccard ----------------------------------------------------
        J = jaccard_similarity(signatures[u], signatures[v])

        # ----- Epistemic weight ---------------------------------------------------
        flag = flags_by_edge.get((u, v)) or flags_by_edge.get((v, u))
        if flag is None:
            # default to a neutral flag
            flag = CertaintyFlag(
                label="PROBABLE",
                confidence_bps=5000,
                authority_class="default",
                rationale="auto-generated",
            )
        prior = flag.confidence_bps / 10000.0
        # treat J as likelihood that the edge is "true"
        marginal = bayes_marginal(prior, J, false_positive)
        posterior = bayes_update(prior, J, marginal)

        # Combine data‑driven Δ with epistemic posterior to obtain weight
        # Small posterior → more uncertain → higher weight
        epistemic_factor = 1.0 - posterior  # 0 (certain) → 0, 1 (uncertain) → 1
        w = delta * (1.0 + epistemic_factor)

        # ----- Physarum‑Sheaf flux gain -------------------------------------------
        alpha = J / (1.0 + w)  # larger similarity and lower weight → stronger flux

        # Discrepancy term for sheaf update (scaled version of Δ)
        d = delta * J

        edge_props[(u, v)] = {
            "delta": delta,
            "J": J,
            "weight": w,
            "alpha": alpha,
            "discrepancy": d,
            "posterior_confidence": posterior,
            "flag": flag,
        }
    return edge_props


def update_sheaf(
    sections: Dict[str, np.ndarray],
    edge_props: Dict[Tuple[str, str], Dict[str, Any]],
    eta: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    Perform one Physarum‑Sheaf update step using the hybrid edge properties.

    Parameters
    ----------
    sections : node → vector (numpy array) representing the sheaf section s_u.
    edge_props : output of `compute_edge_properties`.
    eta : learning rate / step size for the flux exchange.

    Returns
    -------
    new_sections : dict with updated vectors.
    """
    # Initialise accumulation of flux contributions
    flux_accum: Dict[str, np.ndarray] = {node: np.zeros_like(vec) for node, vec in sections.items()}

    for (u, v), prop in edge_props.items():
        s_u, s_v = sections[u], sections[v]
        diff = s_u - s_v
        # Flux magnitude proportional to alpha and diff
        q = prop["alpha"] * diff
        # Update accumulators (conservative exchange)
        flux_accum[u] -= eta * q
        flux_accum[v] += eta * q

        # Optional: incorporate discrepancy as a damping term
        damping = prop["discrepancy"] * 1e-5
        flux_accum[u] -= damping * s_u
        flux_accum[v] -= damping * s_v

    # Apply accumulated updates
    new_sections = {
        node: sections[node] + flux_accum[node] for node in sections
    }
    return new_sections


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple graph with three nodes
    nodes = ["A", "B", "C"]
    tokens_by_node = {
        "A": ["alpha", "beta", "gamma", "delta"],
        "B": ["beta", "epsilon", "zeta", "eta"],
        "C": ["theta", "iota", "kappa", "lambda", "alpha"],
    }

    edges = [("A", "B"), ("B", "C"), ("C", "A")]

    # Random epistemic flags per edge
    def random_flag() -> CertaintyFlag:
        label = random.choice(EPISTEMIC_FLAGS)
        confidence = random.randint(2000, 8000)
        return CertaintyFlag(
            label=label,
            confidence_bps=confidence,
            authority_class="test",
            rationale="randomly generated",
        )

    flags_by_edge = {e: random_flag() for e in edges}

    # Build sketches and MinHash signatures
    sketches, signatures = build_features(tokens_by_node)

    # Initialise sheaf sections as random vectors of dimension 5
    dim = 5
    sections = {node: np.random.rand(dim) for node in nodes}

    # Compute hybrid edge properties
    edge_props = compute_edge_properties(edges, sketches, signatures, flags_by_edge)

    # Perform a single update step
    new_sections = update_sheaf(sections, edge_props, eta=0.2)

    # Output a concise summary
    print("=== Edge properties ===")
    for e, p in edge_props.items():
        print(
            f"{e}: Δ={p['delta']:.2f}, J={p['J']:.2f}, weight={p['weight']:.2f}, "
            f"alpha={p['alpha']:.4f}, posterior_conf={p['posterior_confidence']:.3f}"
        )
    print("\n=== Sheaf sections (before → after) ===")
    for node in nodes:
        print(f"{node}: {sections[node]} → {new_sections[node]}")
    sys.exit(0)