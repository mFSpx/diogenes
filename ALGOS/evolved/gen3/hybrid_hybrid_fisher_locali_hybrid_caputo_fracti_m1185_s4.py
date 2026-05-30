# DARWIN HAMMER — match 1185, survivor 4
# gen: 3
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s0.py (gen2)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s4.py (gen1)
# born: 2026-05-29T23:33:32Z

import math
import numpy as np
from collections import deque
from typing import Any, Dict, List, Tuple


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def gamma_func(z: float) -> float:
    """Accurate Gamma using log‑Gamma to avoid overflow."""
    if z <= 0:
        raise ValueError("Gamma argument must be positive")
    return math.exp(math.lgamma(z))


def fractional_decay(alpha: float, t: float) -> float:
    """
    Fractional decay kernel κ_α(t) = t^{α‑1} / Γ(α).

    Parameters
    ----------
    alpha : float
        Order of the fractional operator (0 < α ≤ 1).
    t : float
        Positive time instant.

    Returns
    -------
    float
        Decay factor.
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")
    if t <= 0:
        raise ValueError("time t must be positive")
    return t ** (alpha - 1) / gamma_func(alpha)


# ----------------------------------------------------------------------
# Fisher‑information based on character distribution
# ----------------------------------------------------------------------
def fisher_information(text: str, eps: float = 1e-12) -> float:
    """
    Compute a scalar Fisher information for a string based on its
    character‑frequency distribution.

    The probability vector p_i is estimated from the histogram of byte
    values (0‑255). Fisher information for a discrete distribution is
    I = Σ ( (∂p_i/∂θ)^2 / p_i ), where θ is a dummy parameter.
    Here we approximate ∂p_i/∂θ by the discrete gradient of the
    histogram, which captures how rapidly the distribution changes.

    Returns a non‑negative scalar; larger values indicate a more
    “informative” (i.e. less uniform) text.
    """
    if not text:
        return eps

    # histogram of byte values
    bytes_arr = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    hist, _ = np.histogram(bytes_arr, bins=256, range=(0, 256), density=False)
    p = hist.astype(np.float64) / hist.sum() + eps  # avoid zero probabilities

    # discrete gradient (central differences)
    grad = np.gradient(p)

    fisher = np.sum((grad ** 2) / p)
    return max(fisher, eps)


# ----------------------------------------------------------------------
# SSIM for 1‑D normalized histograms
# ----------------------------------------------------------------------
def ssim_hist(x: np.ndarray, y: np.ndarray,
              dynamic_range: float = 1.0,
              k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Compute SSIM on two 1‑D probability vectors (histograms).

    The vectors are assumed to be already normalized (sum to 1).
    """
    if x.shape != y.shape:
        raise ValueError("Histogram vectors must have the same shape")

    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx ** 2 + my ** 2 + c1) * (vx + vy + c2)
    return numerator / denominator


def ssim_text(text: str, reference: str) -> float:
    """
    SSIM between two strings based on their normalized byte‑frequency histograms.
    """
    def norm_hist(s: str) -> np.ndarray:
        if not s:
            return np.zeros(256, dtype=np.float64)
        bytes_arr = np.frombuffer(s.encode("utf-8"), dtype=np.uint8)
        hist, _ = np.histogram(bytes_arr, bins=256, range=(0, 256), density=False)
        return hist.astype(np.float64) / hist.sum()

    h1 = norm_hist(text)
    h2 = norm_hist(reference)
    return ssim_hist(h1, h2)


# ----------------------------------------------------------------------
# Tree utilities
# ----------------------------------------------------------------------
def bfs_path_costs(
    nodes: List[Any],
    edges: List[Tuple[Any, Any, float]],
    root: Any,
) -> Dict[Any, float]:
    """
    Compute accumulated cost from the root to every node using BFS.
    Edge weights are assumed to be positive.
    Returns a dict mapping node → cumulative cost.
    """
    adj: Dict[Any, List[Tuple[Any, float]]] = {n: [] for n in nodes}
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    costs: Dict[Any, float] = {root: 0.0}
    visited = {root}
    q = deque([root])

    while q:
        cur = q.popleft()
        cur_cost = costs[cur]
        for nbr, w in adj[cur]:
            if nbr not in visited:
                visited.add(nbr)
                costs[nbr] = cur_cost + w
                q.append(nbr)

    return costs


def total_tree_cost(
    nodes: List[Any],
    edges: List[Tuple[Any, Any, float]],
    root: Any,
) -> float:
    """
    Sum of root‑to‑node accumulated costs for all nodes.
    """
    path_costs = bfs_path_costs(nodes, edges, root)
    return sum(path_costs.values())


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def adaptive_alpha(
    base_alpha: float,
    fisher: float,
    scale: float = 0.3,
) -> float:
    """
    Modulate the fractional order α based on Fisher information.
    α = base_alpha + scale * (fisher / (fisher + 1)).
    The result is clipped to (0, 1].
    """
    α = base_alpha + scale * (fisher / (fisher + 1.0))
    return min(max(α, 1e-6), 1.0)


def weighted_edge(
    w: float,
    decay: float,
    ssim_val: float,
    beta: float = 0.5,
) -> float:
    """
    Combine raw weight, fractional decay and structural similarity.

    The SSIM term reduces the effective weight when the packet is
    structurally similar to the reference (i.e., high SSIM → lower cost).
    """
    return w * decay * (1.0 - beta * ssim_val)


def hybrid_route_and_tree(
    packet: Dict[str, Any],
    reference_text: str,
    nodes: List[Any],
    edges: List[Tuple[Any, Any, float]],
    root: Any,
    *,
    base_alpha: float = 0.7,
    time_t: float = 5.0,
    beta: float = 0.5,
) -> Dict[str, Any]:
    """
    End‑to‑end hybrid operation.

    1. Compute Fisher information from the packet text.
    2. Compute SSIM between packet text and reference.
    3. Derive an adaptive fractional order α.
    4. Apply a decay kernel to each edge and modulate by SSIM.
    5. Compute the total tree cost (sum of root‑to‑node path costs).
    6. Choose the next hop from the root that yields the smallest
       combined decayed weight.

    Returns a dictionary with the fused cost, chosen next hop,
    and the intermediate metrics for inspection.
    """
    # ------------------------------------------------------------------
    # 1‑2. Text‑based metrics
    # ------------------------------------------------------------------
    text = str(
        packet.get("text_surface")
        or packet.get("raw_command")
        or packet.get("text")
        or ""
    )
    fisher_val = fisher_information(text)
    ssim_val = ssim_text(text, reference_text)

    # ------------------------------------------------------------------
    # 3. Adaptive fractional order
    # ------------------------------------------------------------------
    α = adaptive_alpha(base_alpha, fisher_val)

    # ------------------------------------------------------------------
    # 4. Decay and edge weighting
    # ------------------------------------------------------------------
    decay_factor = fractional_decay(α, time_t)

    decayed_edges: List[Tuple[Any, Any, float]] = []
    for u, v, w in edges:
        new_w = weighted_edge(w, decay_factor, ssim_val, beta=beta)
        decayed_edges.append((u, v, new_w))

    # ------------------------------------------------------------------
    # 5. Total tree cost (deep integration of fractional dynamics)
    # ------------------------------------------------------------------
    fused_cost = total_tree_cost(nodes, decayed_edges, root)

    # ------------------------------------------------------------------
    # 6. Routing decision (next hop from root)
    # ------------------------------------------------------------------
    # Build adjacency for root only (no need to recompute whole graph)
    root_adj: List[Tuple[Any, float]] = []
    for u, v, w in edges:
        if u == root:
            root_adj.append((v, w))
        elif v == root:
            root_adj.append((u, w))

    if not root_adj:
        next_hop = None
    else:
        # Apply same decay/ssim modulation to compare fairly
        candidates = [
            (nbr, weighted_edge(w, decay_factor, ssim_val, beta=beta))
            for nbr, w in root_adj
        ]
        next_hop = min(candidates, key=lambda kv: kv[1])[0]

    return {
        "fused_cost": fused_cost,
        "next_hop": next_hop,
        "alpha_used": α,
        "fisher_information": fisher_val,
        "ssim": ssim_val,
        "decay_factor": decay_factor,
    }