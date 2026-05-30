# DARWIN HAMMER — match 1161, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s2.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_hybrid_hybrid_minimu_m363_s0.py (gen3)
# born: 2026-05-29T23:33:17Z

import math
import sys
import pathlib
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
import random

# ----------------------------------------------------------------------
# Basic geometric utilities
# ----------------------------------------------------------------------
Point = Tuple[float, float]
Edge = Tuple[str, str]

def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian kernel evaluated at *theta*."""
    if width <= 0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information of a Gaussian beam.
    Returns a non‑negative scalar that grows with the steepness of the beam.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """
    Structural Similarity Index (SSIM) for 1‑D signals.
    Works for any numeric 1‑D arrays (e.g. ordinal character codes).
    """
    if x.shape != y.shape:
        raise ValueError("arrays must have identical shape")
    if x.size == 0:
        raise ValueError("arrays must not be empty")

    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator

def length(a: Point, b: Point) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

# ----------------------------------------------------------------------
# Tree cost with Fisher‑adjusted edge weights
# ----------------------------------------------------------------------
def tree_cost(nodes: Dict[str, Point],
              edges: List[Edge],
              root: str,
              path_weight: float = 0.2,
              fisher_center: float = 0.0,
              fisher_width: float = 1.0) -> float:
    """
    Compute a cost for a spanning tree.

    *material*   – sum of Euclidean lengths, each inflated by a Fisher term.
    *path_cost*  – weighted sum of distances from the root to every node,
                   also inflated by the same Fisher term.
    """
    # Build adjacency list and pre‑compute Fisher scores for both directions
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    fisher_scores: Dict[Tuple[str, str], float] = {}
    material = 0.0

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

        d = length(nodes[a], nodes[b])
        f = fisher_score(d, fisher_center, fisher_width)
        fisher_scores[(a, b)] = f
        fisher_scores[(b, a)] = f          # undirected symmetry
        material += d * (1.0 + f)

    # Depth‑first traversal to accumulate root‑to‑node distances
    dist: Dict[str, float] = {root: 0.0}
    stack = [root]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                edge_f = fisher_scores[(cur, nxt)]
                step = length(nodes[cur], nodes[nxt]) * (1.0 + edge_f)
                dist[nxt] = dist[cur] + step
                stack.append(nxt)

    path_cost = path_weight * sum(dist.values())
    return material + path_cost

# ----------------------------------------------------------------------
# Text similarity utilities (deeper integration)
# ----------------------------------------------------------------------
def text_to_numeric(text: str) -> np.ndarray:
    """
    Convert a Unicode string to a 1‑D float array of code points.
    Normalises to the range [0, 255] to keep the SSIM dynamic range sensible.
    """
    if not text:
        return np.array([], dtype=float)
    codes = np.fromiter((ord(ch) for ch in text), dtype=float)
    # Scale to typical image range; avoid division by zero for single‑character strings
    if codes.max() != codes.min():
        codes = 255.0 * (codes - codes.min()) / (codes.max() - codes.min())
    else:
        codes = np.full_like(codes, 127.5)   # midpoint of 0‑255
    return codes

# ----------------------------------------------------------------------
# Hybrid routing that couples SSIM, Fisher, and Bayesian adaptation
# ----------------------------------------------------------------------
def hybrid_routing(packet: Dict,
                   reference_text: str,
                   nodes: Dict[str, Point],
                   edges: List[Edge],
                   root: str,
                   prior_success: float = 0.5,
                   false_positive: float = 0.1) -> Dict[str, float]:
    """
    Perform routing with three intertwined mathematical layers:

    1. **Similarity** – SSIM on ordinal representations of the packet text.
    2. **Structure** – Fisher‑adjusted tree cost.
    3. **Adaptation** – Bayesian update of a success probability that
       modulates the *path_weight* used in the tree cost.

    Returns a dictionary with the raw SSIM, the updated success probability,
    and the final tree cost.
    """
    # 1️⃣ Extract textual payload
    text = str(packet.get("text_surface") or
               packet.get("raw_command") or
               packet.get("text") or "")
    x = text_to_numeric(text)
    y = text_to_numeric(reference_text)

    # 2️⃣ Similarity (guard against empty payload)
    if x.size == 0 or y.size == 0:
        ssim_value = 0.0
    else:
        ssim_value = ssim(x, y, dynamic_range=255.0)

    # 3️⃣ Fisher parameters derived from the current graph geometry
    edge_lengths = [length(nodes[a], nodes[b]) for a, b in edges]
    fisher_center = float(np.mean(edge_lengths)) if edge_lengths else 0.0
    fisher_width = float(np.std(edge_lengths)) if edge_lengths else 1.0
    fisher_width = max(fisher_width, 1e-6)   # avoid division by zero

    # 4️⃣ Bayesian adaptation: treat SSIM as evidence of a "successful" routing
    likelihood = ssim_value          # already in [0,1] for normalized inputs
    marginal = bayes_marginal(prior_success, likelihood, false_positive)
    posterior = bayes_update(prior_success, likelihood, marginal)

    # 5️⃣ Modulate path_weight: higher confidence ⇒ cheaper traversal penalty
    adaptive_path_weight = max(0.01, 0.2 * (1.0 - posterior))

    # 6️⃣ Tree cost with adapted weight
    cost = tree_cost(nodes,
                     edges,
                     root,
                     path_weight=adaptive_path_weight,
                     fisher_center=fisher_center,
                     fisher_width=fisher_width)

    return {
        "ssim": float(ssim_value),
        "posterior_success": float(posterior),
        "tree_cost": float(cost)
    }

# ----------------------------------------------------------------------
# Bayesian helpers
# ----------------------------------------------------------------------
def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """
    P(E) = P(E|H)·P(H) + P(E|¬H)·P(¬H)
    """
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("probabilities must be in [0, 1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """
    Posterior P(H|E) = P(E|H)·P(H) / P(E)
    """
    if marginal <= 0.0:
        raise ValueError("marginal probability must be > 0")
    return prior * likelihood / marginal

# ----------------------------------------------------------------------
# Demonstration / simple sanity test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple triangular graph
    nodes = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.5, 1.0)
    }
    edges = [("A", "B"), ("A", "C"), ("B", "C")]
    root = "A"

    packet = {"text_surface": "Hello"}
    reference_text = "Hello World"

    result = hybrid_routing(packet,
                            reference_text,
                            nodes,
                            edges,
                            root,
                            prior_success=0.6,
                            false_positive=0.05)

    print("Hybrid routing result:")
    for k, v in result.items():
        print(f"  {k}: {v:.6f}")