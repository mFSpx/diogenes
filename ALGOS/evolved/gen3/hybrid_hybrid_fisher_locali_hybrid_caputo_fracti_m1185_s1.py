# DARWIN HAMMER — match 1185, survivor 1
# gen: 3
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s0.py (gen2)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s4.py (gen1)
# born: 2026-05-29T23:33:32Z

"""Hybrid Fisher‑SSIM Fractional Tree Algorithm
Parents:
- hybrid_fisher_localization_hybrid_ternary_route_m40_s0.py (Fisher‑information scoring + SSIM routing)
- hybrid_caputo_fractional_minimum_cost_tree_m35_s4.py (Caputo fractional derivative + minimum‑cost tree)

Mathematical bridge:
The Fisher score of a packet’s “text surface” is used as a *fractional decay kernel* that modulates the
edge‑weight decay in a minimum‑cost tree.  The SSIM between the packet text and a reference sample
produces a similarity factor that scales the tree’s total cost.  Together they yield a unified cost
functional  

    C_fused = ( Σ_w  w·κ_α(t) ) · (1 – S_sim) + λ·F_score  

where κ_α(t)=t^{α‑1}/Γ(α) is the Caputo fractional decay, S_sim is the SSIM, and F_score is the Fisher
information score.  The algorithm can be used for routing decisions that respect both information‑theoretic
confidence (Fisher) and structural similarity (SSIM) while the underlying network topology evolves
according to a fractional‑order dynamics.
"""

import math
import random
import sys
import pathlib
import numpy as np

# ----------------------------------------------------------------------
# Core components from Parent A
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher‑information score for a scalar angle."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index (SSIM) for 1‑D signals."""
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x.size:
        raise ValueError('samples must not be empty')
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.cov(x, y, ddof=0)[0, 1]
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


# ----------------------------------------------------------------------
# Core components from Parent B
# ----------------------------------------------------------------------
_LANCZOS_G = 7
_LANCZOS_C = np.array([
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


def gamma_lanczos(z: float) -> float:
    """Lanczos approximation of the Gamma function for z > 0."""
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x


def caputo_derivative(f, alpha: float, t: float) -> float:
    """Caputo fractional derivative of order alpha of function f at time t."""
    dt = 0.01
    tau = np.arange(0, t, dt)
    f_tau = f(tau)
    integral = np.trapz(f_tau / (t - tau) ** alpha, tau)
    return integral / gamma_lanczos(1 - alpha)


def fractional_decay(alpha: float, t: float) -> float:
    """Fractional decay kernel κ_α(t) = t^{α‑1} / Γ(α)."""
    return t ** (alpha - 1) / gamma_lanczos(alpha)


def length(a: tuple, b: tuple) -> float:
    """Euclidean distance between two 2‑D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_cost(nodes, edges, root):
    """
    Simple minimum‑cost tree cost.
    edges: list of (parent, child, weight)
    Returns the sum of weights on the unique path from root to every node.
    """
    adj = {n: [] for n in nodes}
    for parent, child, w in edges:
        adj[parent].append((child, w))
        adj[child].append((parent, w))  # undirected for traversal

    visited = set()
    total = 0.0

    def dfs(node, acc):
        nonlocal total
        visited.add(node)
        total += acc
        for nbr, w in adj[node]:
            if nbr not in visited:
                dfs(nbr, w)

    dfs(root, 0.0)
    return total


# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def fisher_ssim_weight(packet: dict, reference_text: str,
                       center: float, width: float) -> float:
    """
    Compute a combined Fisher‑information / SSIM weight for a packet.
    - Fisher score uses the length of the packet text as a surrogate angle.
    - SSIM compares the UTF‑8 byte values of packet text vs reference.
    Returns a scalar in (0, ∞) where larger means higher confidence.
    """
    text = str(packet.get("text_surface") or packet.get("raw_command")
               or packet.get("text") or "")
    theta = float(len(text))  # surrogate angle proportional to length
    f_score = fisher_score(theta, center, width)

    # Prepare numeric arrays for SSIM (byte values normalized to 0‑255)
    x = np.frombuffer(text.encode('utf-8'), dtype=np.uint8).astype(np.float64)
    y = np.frombuffer(reference_text.encode('utf-8'), dtype=np.uint8).astype(np.float64)

    # Pad the shorter array to match lengths (required by ssim)
    if x.size < y.size:
        x = np.pad(x, (0, y.size - x.size), constant_values=0)
    elif y.size < x.size:
        y = np.pad(y, (0, x.size - y.size), constant_values=0)

    ssim_val = ssim(x, y)
    # Combine multiplicatively; add a small epsilon to avoid zero.
    return f_score * (ssim_val + 1e-12)


def fractional_tree_cost(nodes, edges, root,
                         alpha: float, t: float,
                         weight_factor: float) -> float:
    """
    Compute a tree cost that is modulated by a fractional decay kernel
    and an external weight factor (e.g., Fisher‑SSIM weight).
    The decay is applied to each edge weight according to the current time t.
    """
    # Apply fractional decay to each edge weight
    decayed_edges = []
    for parent, child, w in edges:
        decayed_w = w * fractional_decay(alpha, t)
        decayed_edges.append((parent, child, decayed_w))

    base_cost = tree_cost(nodes, decayed_edges, root)
    # Fuse with external factor (higher factor should increase total cost)
    return base_cost * weight_factor


def hybrid_route_and_tree(packet: dict, reference_text: str,
                          nodes, edges, root,
                          center: float = 0.0, width: float = 10.0,
                          alpha: float = 0.7, t: float = 5.0) -> dict:
    """
    End‑to‑end hybrid operation:
    1. Compute Fisher‑SSIM weight from the packet.
    2. Compute a fractional‑decayed tree cost using that weight.
    3. Return a routing decision dict containing the cost and a simple next‑hop choice.
    """
    weight = fisher_ssim_weight(packet, reference_text, center, width)

    cost = fractional_tree_cost(nodes, edges, root, alpha, t, weight)

    # Simple routing decision: choose the neighbor of the root with minimal decayed weight.
    # Build adjacency for quick lookup.
    adj = {n: [] for n in nodes}
    for p, c, w in edges:
        adj[p].append((c, w))
        adj[c].append((p, w))

    # Find candidate children of root
    candidates = adj[root]
    if not candidates:
        next_hop = None
    else:
        # Apply same fractional decay to compare fairly
        decayed = [(nbr, w * fractional_decay(alpha, t)) for nbr, w in candidates]
        next_hop = min(decayed, key=lambda kv: kv[1])[0]

    return {
        "fisher_ssim_weight": weight,
        "fractional_tree_cost": cost,
        "next_hop": next_hop,
        "routing_timestamp": t
    }


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy packet
    pkt = {
        "text_surface": "Hello world! This is a test packet.",
        "source": "sensor_A"
    }

    ref = "Hello world! This is a reference message."

    # Simple graph (tree)
    nodes = ["root", "A", "B", "C"]
    edges = [
        ("root", "A", 1.2),
        ("root", "B", 0.9),
        ("A", "C", 0.5)
    ]
    root_node = "root"

    result = hybrid_route_and_tree(
        packet=pkt,
        reference_text=ref,
        nodes=nodes,
        edges=edges,
        root=root_node,
        center=20.0,
        width=5.0,
        alpha=0.8,
        t=3.0
    )

    print("Hybrid routing result:")
    for k, v in result.items():
        print(f"  {k}: {v}")