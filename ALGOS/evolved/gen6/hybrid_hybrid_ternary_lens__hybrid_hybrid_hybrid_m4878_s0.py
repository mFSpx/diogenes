# DARWIN HAMMER — match 4878, survivor 0
# gen: 6
# parent_a: hybrid_ternary_lens_audit_decreasing_pruning_m15_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m1326_s2.py (gen5)
# born: 2026-05-29T23:58:39Z

"""
Hybrid Audit–Sheaf Fusion

Parents:
- hybrid_ternary_lens_audit_decreasing_pruning_m15_s0.py (Algorithm A)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_nlms_o_m1326_s2.py (Algorithm B)

Mathematical Bridge:
Algorithm A provides a set of *candidates* each with a classification and a list of
findings.  Algorithm B treats abstract objects as *sections* of a sheaf defined on a
graph, updating them with a Normalised Least‑Mean‑Squares (NLMS) rule and then
selecting a subset via a Minimum‑Cost Spanning Tree (MST).

The fusion interprets every candidate as a node of a sheaf graph.  The NLMS update
adjusts a scalar *relevance score* for each node using the number of findings as
the desired signal.  Edge costs are derived from a similarity measure between the
candidates (the “lens” similarity).  An MST provides a globally coherent
sub‑graph, and a decreasing pruning schedule (Algorithm A) finally discards the
least‑relevant nodes while preserving the MST structure.

Thus the core equations of both parents are combined:

    w_i ← w_i + μ · (d_i − w_i)                      # NLMS (scalar case)
    cost(e_{ij}) = 1 − sim(i, j)                    # edge cost from lens similarity
    prune according to p_k = p_0·γ^k                # decreasing pruning schedule
"""

import json
import math
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np

# ----------------------------------------------------------------------
# Constants & Helpers (from Algorithm A)
# ----------------------------------------------------------------------
CLASSIFICATIONS = {
    "usable_now",
    "research_only",
    "needs_conversion",
    "unsafe_for_fastpath",
    "unsupported",
}
LOCAL_PATTERNS = ["*bitnet*", "*BitNet*", "*fairyfuse*", "*FairyFuse*", "*lora*", "*LoRA*", "*adapter*"]


def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 format without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_manifest(path: Path) -> dict:
    """
    Load a vendor manifest (Algorithm A) and validate classifications.
    Returns the parsed JSON dictionary.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    for candidate in data.get("vendors", []):
        classification = candidate.get("classification")
        if classification not in CLASSIFICATIONS:
            raise SystemExit(f"invalid classification {classification!r} for {candidate.get('candidate_key')}")
    return data


# ----------------------------------------------------------------------
# NLMS Update (Algorithm B – scalar case)
# ----------------------------------------------------------------------
def nlms_update(current: float, desired: float, mu: float = 0.1) -> float:
    """
    Perform a Normalised Least‑Mean‑Squares update on a scalar section.
    The normalisation factor for a scalar input is 1, so the update reduces to:
        w ← w + μ·(d − w)
    """
    return current + mu * (desired - current)


# ----------------------------------------------------------------------
# Similarity & Edge Cost (bridge between lens audit and sheaf graph)
# ----------------------------------------------------------------------
def candidate_similarity(c1: dict, c2: dict) -> float:
    """
    Simple similarity between two candidates:
      - +1 if classifications match, else 0
      - +0.5 if both have a finding containing the word “fast_path”
      - Normalised by the total number of findings (Jaccard‑like)
    Returns a value in [0, 1].
    """
    score = 0.0
    if c1.get("classification") == c2.get("classification"):
        score += 1.0

    f1 = set(c1.get("findings", []))
    f2 = set(c2.get("findings", []))
    if "fast_path" in f1 and "fast_path" in f2:
        score += 0.5

    # Jaccard over findings
    union = f1 | f2
    inter = f1 & f2
    if union:
        score += len(inter) / len(union)
    else:
        score += 1.0  # both have no findings -> maximal similarity

    # Normalise to [0,1] (max possible score = 2.5)
    return min(score / 2.5, 1.0)


def edge_cost(similarity: float) -> float:
    """
    Convert similarity to a cost suitable for MST.
    Higher similarity → lower cost.
    """
    return 1.0 - similarity


# ----------------------------------------------------------------------
# Union‑Find for Kruskal's MST (Algorithm B)
# ----------------------------------------------------------------------
class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


def minimum_spanning_tree(nodes, edges):
    """
    Kruskal's algorithm.
    `nodes` – iterable of node identifiers.
    `edges` – list of (cost, u, v) tuples.
    Returns the set of edges belonging to the MST.
    """
    uf = UnionFind(nodes)
    mst = []
    for cost, u, v in sorted(edges, key=lambda e: e[0]):
        if uf.union(u, v):
            mst.append((u, v, cost))
        if len(mst) == len(nodes) - 1:
            break
    return mst


# ----------------------------------------------------------------------
# Decreasing Pruning Schedule (Algorithm A)
# ----------------------------------------------------------------------
def decreasing_prune(scores: dict, keep_ratio: float = 0.8, factor: float = 0.9) -> set:
    """
    Iteratively prune the lowest‑scoring candidates.
    `keep_ratio` is the fraction kept after the first pass.
    `factor` reduces the keep ratio each iteration until only the MST nodes remain.
    Returns the set of candidate keys that survive all pruning steps.
    """
    survivors = set(scores.keys())
    current_ratio = keep_ratio
    while True:
        # Determine threshold
        sorted_scores = sorted(((k, scores[k]) for k in survivors), key=lambda kv: kv[1], reverse=True)
        cutoff = int(math.ceil(len(sorted_scores) * current_ratio))
        survivors = {k for k, _ in sorted_scores[:cutoff]}
        if current_ratio <= factor:
            break
        current_ratio *= factor
    return survivors


# ----------------------------------------------------------------------
# Core Hybrid Operation
# ----------------------------------------------------------------------
def hybrid_audit_sheaf(manifest_path: Path, mu: float = 0.15) -> list[dict]:
    """
    End‑to‑end hybrid algorithm:
      1. Load candidates (Algorithm A).
      2. Initialise a scalar relevance score for each candidate (start at 1.0).
      3. Update scores with NLMS using the number of findings as the desired signal.
      4. Build a complete graph where edge cost = 1 − similarity.
      5. Compute the Minimum‑Cost Spanning Tree (Algorithm B).
      6. Apply a decreasing pruning schedule, protecting all MST nodes.
      7. Return the list of surviving candidate dictionaries.
    """
    data = load_manifest(manifest_path)
    candidates = data.get("vendors", [])
    if not candidates:
        return []

    # 1‑3. Initialise and NLMS‑update scores
    scores = {}
    for cand in candidates:
        key = cand.get("candidate_key")
        findings = cand.get("findings", [])
        desired = 1.0 + 0.2 * len(findings)          # desired relevance grows with findings
        current = 1.0                               # start from a neutral score
        updated = nlms_update(current, desired, mu)
        scores[key] = updated
        cand["relevance_score"] = updated           # store for later inspection

    # 4. Build complete graph edges
    edges = []
    keys = [c.get("candidate_key") for c in candidates]
    key_to_cand = {c.get("candidate_key"): c for c in candidates}
    for i, ki in enumerate(keys):
        for kj in keys[i + 1 :]:
            sim = candidate_similarity(key_to_cand[ki], key_to_cand[kj])
            cost = edge_cost(sim)
            edges.append((cost, ki, kj))

    # 5. Minimum‑Cost Spanning Tree
    mst_edges = minimum_spanning_tree(keys, edges)
    mst_nodes = {u for u, v, _ in mst_edges} | {v for u, v, _ in mst_edges}

    # 6. Decreasing pruning while protecting MST nodes
    survivors = decreasing_prune(scores, keep_ratio=0.8, factor=0.85)
    survivors |= mst_nodes  # ensure MST connectivity

    # 7. Return surviving candidates
    return [key_to_cand[k] for k in survivors]


# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny synthetic manifest if none is supplied
    test_path = Path("test_manifest.json")
    if not test_path.exists():
        sample = {
            "vendors": [
                {
                    "candidate_key": "model_A",
                    "classification": "usable_now",
                    "findings": ["fast_path", "low_latency"],
                },
                {
                    "candidate_key": "model_B",
                    "classification": "research_only",
                    "findings": ["needs_conversion"],
                },
                {
                    "candidate_key": "model_C",
                    "classification": "needs_conversion",
                    "findings": ["fast_path"],
                },
                {
                    "candidate_key": "model_D",
                    "classification": "unsafe_for_fastpath",
                    "findings": [],
                },
            ]
        }
        test_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")

    survivors = hybrid_audit_sheaf(test_path)
    print(f"{utc_now()} – Surviving candidates ({len(survivors)}):")
    for cand in survivors:
        print(f"  - {cand['candidate_key']} (score={cand['relevance_score']:.3f})")
    # Clean up synthetic file
    if test_path.exists():
        test_path.unlink()