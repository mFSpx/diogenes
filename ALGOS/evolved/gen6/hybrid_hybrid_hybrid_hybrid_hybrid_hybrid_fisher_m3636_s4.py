# DARWIN HAMMER — match 3636, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_decreasing_pr_m1229_s4.py (gen5)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_minimum_cost__m29_s0.py (gen2)
# born: 2026-05-29T23:51:08Z

import math
import numpy as np
from typing import Dict, List, Tuple, Iterable

# ----------------------------------------------------------------------
# Constants & Helper tables for Lanczos approximation (used only if needed)
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_COEFFS = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
])

def _gamma(z: float) -> float:
    """
    Lanczos approximation of the Gamma function.
    For positive arguments we use the approximation directly,
    for 0 < z < 0.5 we employ the reflection formula.
    """
    if z < 0.5:
        # Reflection formula: Gamma(z) = pi / (sin(pi*z) * Gamma(1-z))
        return math.pi / (math.sin(math.pi * z) * _gamma(1 - z))

    z -= 1.0
    x = _LANCZOS_COEFFS[0]
    for i in range(1, len(_LANCZOS_COEFFS)):
        x += _LANCZOS_COEFFS[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x


def caputo_kernel(alpha: float, t: np.ndarray) -> np.ndarray:
    """
    Raw (unnormalized) Caputo kernel for a vector of time/length indices.
    The kernel is  t^{alpha-1} / Gamma(alpha)  for alpha>0.
    """
    if alpha <= 0.0:
        raise ValueError("Fractional order alpha must be positive.")
    # avoid division by zero
    t = np.where(t == 0, 1e-12, t)
    return t ** (alpha - 1) / _gamma(alpha)


def caputo_weight(length: float, alpha: float) -> float:
    """
    Edge weight after applying the Caputo kernel.
    """
    return float(length * caputo_kernel(alpha, np.array([length]))[0])


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """
    Normalised Gaussian intensity.
    """
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """
    Fisher information for a Gaussian model evaluated at theta.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def euclidean_length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """2‑D Euclidean distance."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------------------------------------------------
# Core algorithmic primitives
# ----------------------------------------------------------------------
def _build_adjacency(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
) -> Dict[str, List[str]]:
    """Create an adjacency list from node identifiers."""
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _bfs_distances(
    adj: Dict[str, List[str]],
    nodes: Dict[str, Tuple[float, float]],
    root: str,
) -> Dict[str, float]:
    """
    Breadth‑first traversal that records the cumulative Euclidean distance
    from the root to every reachable node.
    """
    dist: Dict[str, float] = {root: 0.0}
    stack: List[str] = [root]

    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in dist:
                edge_len = euclidean_length(nodes[cur], nodes[nxt])
                dist[nxt] = dist[cur] + edge_len
                stack.append(nxt)
    return dist


def hybrid_operation(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    root: str,
    alpha: float,
    center: float,
    width: float,
    path_weight: float = 0.2,
) -> float:
    """
    Integrated metric:
      * Edge material is weighted by the Caputo kernel.
      * Node‑wise Fisher information (based on graph‑geodesic distance)
        is added to the material.
      * An optional linear path penalty (path_weight) can be applied to the
        sum of distances from the root, encouraging compact trees.
    """
    if root not in nodes:
        raise KeyError(f"Root node '{root}' not present in nodes dictionary.")
    if not (0.0 <= path_weight):
        raise ValueError("path_weight must be non‑negative.")

    adj = _build_adjacency(nodes, edges)

    # ---- material (Caputo‑weighted edges) ----
    material = 0.0
    for u, v in edges:
        raw_len = euclidean_length(nodes[u], nodes[v])
        material += caputo_weight(raw_len, alpha)

    # ---- distances from root (geodesic) ----
    distances = _bfs_distances(adj, nodes, root)

    # ---- Fisher information accumulated over nodes ----
    fisher_total = sum(fisher_score(d, center, width) for d in distances.values())

    # ---- optional path penalty (linear) ----
    path_penalty = path_weight * sum(distances.values())

    return material + fisher_total + path_penalty


def adaptive_pruning(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    alpha: float,
    center: float,
    width: float,
    threshold: float = 0.5,
) -> List[Tuple[str, str]]:
    """
    Keep edges whose Caputo‑scaled length falls inside the high‑intensity region
    of the Gaussian beam.
    """
    kept: List[Tuple[str, str]] = []
    for u, v in edges:
        raw_len = euclidean_length(nodes[u], nodes[v])
        scaled_len = caputo_weight(raw_len, alpha)
        if gaussian_beam(scaled_len, center, width) > threshold:
            kept.append((u, v))
    return kept


def epistemic_certainty(
    nodes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    alpha: float,
    center: float,
    width: float,
) -> Dict[str, float]:
    """
    Certainty of a node is the Gaussian beam evaluated on the
    Caputo‑scaled sum of incident edge lengths.
    """
    certainty: Dict[str, float] = {}
    # Pre‑compute adjacency for fast neighbor lookup
    adj = _build_adjacency(nodes, edges)

    for node, neigh in adj.items():
        incident_len = sum(euclidean_length(nodes[node], nodes[n]) for n in neigh)
        scaled = caputo_weight(incident_len, alpha)
        certainty[node] = gaussian_beam(scaled, center, width)

    return certainty


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple triangular graph
    nodes_example = {
        "A": (0.0, 0.0),
        "B": (3.0, 4.0),
        "C": (6.0, 8.0),
    }
    edges_example = [("A", "B"), ("B", "C"), ("C", "A")]
    root_example = "A"

    alpha_example = 0.5
    center_example = 5.0
    width_example = 2.0

    print("Hybrid operation value:",
          hybrid_operation(
              nodes_example,
              edges_example,
              root_example,
              alpha_example,
              center_example,
              width_example,
          ))
    print("Adaptively pruned edges:",
          adaptive_pruning(
              nodes_example,
              edges_example,
              alpha_example,
              center_example,
              width_example,
          ))
    print("Epistemic certainty per node:",
          epistemic_certainty(
              nodes_example,
              edges_example,
              alpha_example,
              center_example,
              width_example,
          ))