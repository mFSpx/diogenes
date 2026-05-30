# DARWIN HAMMER — match 1526, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_infotaxis_min_hybrid_hybrid_distri_m20_s2.py (gen3)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m981_s0.py (gen5)
# born: 2026-05-29T23:37:05Z

"""Hybrid Algorithm combining:
- Parent A: hybrid_hybrid_infotaxis_min_hybrid_hybrid_distri_m20_s2.py (MinHash, entropy, drag‑based strike integration)
- Parent B: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m981_s0.py (tree metrics, SSIM, VRAM planning)

Mathematical bridge:
The MinHash signature of each node provides a high‑dimensional binary‑like representation.
Pairwise MinHash similarity is interpreted as a “force magnitude” in the chelydrid ambush‑strike
model; integrating this force with the drag equation yields a physical cost for selecting a node
as a cluster representative.  The distribution of these costs forms a probability mass whose
entropy modulates the pruning probability in the ternary routing model and directly scales the
VRAM allocation plan.  Thus entropy, MinHash similarity, drag‑based integration, and VRAM
planning are fused into a single pipeline."""
from __future__ import annotations

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Utilities from Parent A
# ----------------------------------------------------------------------
def entropy(probabilities: List[float], eps: float = 1e-12) -> float:
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p / total) * math.log(max(p / total, eps)) for p in probabilities if p > 0)


def _hash(seed: int, token: str) -> int:
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def signature(tokens: List[str], k: int = 128) -> List[int]:
    """MinHash signature for a set of tokens."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [2**64 - 1] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard‑like similarity between two MinHash signatures."""
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have equal length")
    if not sig_a:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def integrate_strike(
    force_series: List[float],
    dt: float,
    m_head: float,
    drag_cd: float = 0.3,
    fluid_density: float = 1000.0,
    area: float = 0.001,
    v0: float = 0.0,
) -> Dict[str, float]:
    """
    Integrate a 1‑D strike under quadratic drag.
    Returns final position, velocity, and peak velocity.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    v = v0
    x = 0.0
    peak = v0
    for force in force_series:
        # Quadratic drag force magnitude
        drag = drag_cd * fluid_density * area * v * abs(v) / 2.0
        # Net acceleration (force - drag) / mass
        acc = (force - drag) / m_head
        v += acc * dt
        x += v * dt
        if v > peak:
            peak = v
    return {"position": x, "velocity": v, "peak_velocity": peak}


# ----------------------------------------------------------------------
# Utilities from Parent B
# ----------------------------------------------------------------------
def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_metrics(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float], Dict[str, float]]:
    """Compute adjacency list, edge lengths and distance from root."""
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    edge_len: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        edge_len[(a, b)] = length(nodes[a], nodes[b])
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                edge_key = (cur, nxt) if (cur, nxt) in edge_len else (nxt, cur)
                dist[nxt] = dist[cur] + edge_len[edge_key]
                stack.append(nxt)
    return adj, edge_len, dist


def ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.std(x)
    sigma_y = np.std(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
    return numerator / denominator if denominator != 0 else 0.0


@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Hybrid core functions (the three required demonstrations)
# ----------------------------------------------------------------------
def compute_node_signatures(
    node_tokens: Dict[str, List[str]], k: int = 128
) -> Dict[str, List[int]]:
    """Compute MinHash signatures for every node."""
    return {nid: signature(toks, k) for nid, toks in node_tokens.items()}


def select_representative_nodes(
    signatures: Dict[str, List[int]],
    adjacency: Dict[str, List[str]],
    dt: float = 0.01,
    m_head: float = 0.1,
) -> List[Tuple[str, float]]:
    """
    For each node, treat MinHash similarity to its neighbours as a force series.
    Integrate the forces to obtain a cost; nodes with lowest cost are chosen as
    representatives. Returns a list of (node_id, cost) sorted ascending.
    """
    costs: List[Tuple[str, float]] = []
    for nid, sig in signatures.items():
        # Build force series: similarity * scale_factor for each neighbour
        forces = []
        for nbr in adjacency.get(nid, []):
            sim = similarity(sig, signatures[nbr])
            # Map similarity [0,1] to a force magnitude in Newtons
            forces.append(sim * 5.0)  # arbitrary scaling
        if not forces:
            # Isolated node – treat as minimal force
            forces = [0.0]
        integration = integrate_strike(
            force_series=forces,
            dt=dt,
            m_head=m_head,
            drag_cd=0.3,
            fluid_density=1000.0,
            area=0.001,
            v0=0.0,
        )
        costs.append((nid, integration["position"]))
    # Sort by increasing integrated position (lower cost)
    costs.sort(key=lambda x: x[1])
    return costs


def plan_vram_allocation(
    node_costs: List[Tuple[str, float]],
    total_mb: int = 1024,
    entropy_weight: float = 0.6,
) -> List[VramSlotPlan]:
    """
    Convert node selection costs into VRAM allocation plans.
    The probability of allocating to a node is proportional to
    exp(-entropy_weight * normalized_cost). Entropy of the cost distribution
    modulates the overall aggressiveness of allocation.
    """
    # Normalise costs to a probability distribution
    costs = np.array([c for _, c in node_costs], dtype=float)
    max_cost = costs.max() if costs.size else 1.0
    norm = costs / (max_cost + 1e-12)
    probs = np.exp(-entropy_weight * norm)
    probs /= probs.sum() if probs.sum() > 0 else 1.0

    # Compute entropy of the probability vector
    ent = entropy(probs.tolist())

    # Scale total VRAM by entropy (more uncertain => allocate more conservatively)
    effective_mb = int(total_mb * (1.0 - 0.5 * ent / math.log(len(probs) + 1e-12)))
    plans: List[VramSlotPlan] = []
    for (nid, _), p in zip(node_costs, probs):
        alloc = int(effective_mb * p)
        plan = VramSlotPlan(
            artifact_id=nid,
            artifact_kind="node_rep",
            action="allocate",
            estimated_mb=max(alloc, 1),  # at least 1 MB
            reason="hybrid_cost_entropy",
            detail={"probability": p, "entropy": ent},
        )
        plans.append(plan)
    return plans


def hybrid_process(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    node_tokens: Dict[str, List[str]],
    root: str,
) -> Tuple[List[Tuple[str, float]], List[VramSlotPlan]]:
    """
    End‑to‑end hybrid pipeline:
    1. Build tree metrics.
    2. Compute MinHash signatures per node.
    3. Select representative nodes using drag‑based integration.
    4. Allocate VRAM based on the cost distribution entropy.
    Returns the ordered representative list and the VRAM plan list.
    """
    # 1. Tree metrics (provides adjacency)
    adjacency, _, _ = tree_metrics(nodes, edges, root)

    # 2. Signatures
    sigs = compute_node_signatures(node_tokens)

    # 3. Representative selection
    rep_costs = select_representative_nodes(sigs, adjacency)

    # 4. VRAM planning
    vram_plans = plan_vram_allocation(rep_costs)

    return rep_costs, vram_plans


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple synthetic tree
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, 0.866),
        "D": (1.5, 0.866),
    }
    edges = [("A", "B"), ("A", "C"), ("B", "D")]
    root = "A"

    # Tokens per node (simulating textual content)
    node_tokens = {
        "A": ["alpha", "beta", "gamma"],
        "B": ["beta", "delta", "epsilon"],
        "C": ["gamma", "zeta", "eta"],
        "D": ["delta", "theta", "iota"],
    }

    reps, vram = hybrid_process(nodes, edges, node_tokens, root)

    print("Representative nodes (sorted by cost):")
    for nid, cost in reps:
        print(f"  {nid}: integrated position = {cost:.4f}")

    print("\nVRAM allocation plan:")
    for plan in vram:
        d = plan.as_dict()
        print(f"  Node {d['artifact_id']} -> {d['estimated_mb']} MB (prob={d['detail']['probability']:.3f})")