# DARWIN HAMMER — match 1185, survivor 3
# gen: 3
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s0.py (gen2)
# parent_b: hybrid_caputo_fractional_minimum_cost_tree_m35_s4.py (gen1)
# born: 2026-05-29T23:33:32Z

import math
import random
import sys
import pathlib
import numpy as np

# Core components from Parent A
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
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


# Core components from Parent B
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
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    x = _LANCZOS_C[0]
    for i in range(1, _LANCZOS_G + 2):
        x += _LANCZOS_C[i] / (z + i)
    t = z + _LANCZOS_G + 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x


def caputo_derivative(f, alpha: float, t: float) -> float:
    dt = 0.01
    tau = np.arange(0, t, dt)
    f_tau = f(tau)
    integral = np.trapz(f_tau / (t - tau) ** alpha, tau)
    return integral / gamma_lanczos(1 - alpha)


def fractional_decay(alpha: float, t: float) -> float:
    return t ** (alpha - 1) / gamma_lanczos(alpha)


def length(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tree_cost(nodes, edges, root):
    adj = {n: [] for n in nodes}
    for parent, child, w in edges:
        adj[parent].append((child, w))
        adj[child].append((parent, w))  

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


# Hybrid functions
def fisher_ssim_weight(packet: dict, reference_text: str, center: float, width: float) -> float:
    text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    theta = float(len(text))  
    f_score = fisher_score(theta, center, width)

    x = np.frombuffer(text.encode('utf-8'), dtype=np.uint8).astype(np.float64)
    y = np.frombuffer(reference_text.encode('utf-8'), dtype=np.uint8).astype(np.float64)

    if x.size < y.size:
        x = np.pad(x, (0, y.size - x.size), constant_values=0)
    elif y.size < x.size:
        y = np.pad(y, (0, x.size - y.size), constant_values=0)

    ssim_val = ssim(x, y)
    return f_score * (ssim_val + 1e-12)


def fractional_tree_cost(nodes, edges, root, alpha: float, t: float, weight_factor: float) -> float:
    decayed_edges = []
    for parent, child, w in edges:
        decayed_w = w * fractional_decay(alpha, t)
        decayed_edges.append((parent, child, decayed_w))

    base_cost = tree_cost(nodes, decayed_edges, root)
    return base_cost * weight_factor


def hybrid_route_and_tree(packet: dict, reference_text: str, nodes, edges, root, center: float = 0.0, width: float = 10.0, alpha: float = 0.7, t: float = 5.0) -> dict:
    weight = fisher_ssim_weight(packet, reference_text, center, width)
    cost = fractional_tree_cost(nodes, edges, root, alpha, t, weight)

    adj = {n: [] for n in nodes}
    for p, c, w in edges:
        adj[p].append((c, w))
        adj[c].append((p, w))  

    candidates = adj[root]
    if not candidates:
        next_hop = None
    else:
        decayed = [(nbr, w * fractional_decay(alpha, t)) for nbr, w in candidates]
        next_hop = min(decayed, key=lambda kv: kv[1])[0]

    return {'cost': cost, 'next_hop': next_hop}


def improved_hybrid_route_and_tree(packet: dict, reference_text: str, nodes, edges, root, center: float = 0.0, width: float = 10.0, alpha: float = 0.7, t: float = 5.0) -> dict:
    weight = fisher_ssim_weight(packet, reference_text, center, width)
    cost = fractional_tree_cost(nodes, edges, root, alpha, t, weight)

    adj = {n: [] for n in nodes}
    for p, c, w in edges:
        adj[p].append((c, w))
        adj[c].append((p, w))  

    candidates = adj[root]
    if not candidates:
        next_hop = None
    else:
        decayed = [(nbr, w * fractional_decay(alpha, t)) for nbr, w in candidates]
        next_hop = min(decayed, key=lambda kv: kv[1])[0]

    # Improved: Add an additional term to the cost function to account for the network topology
    network_topology_term = 0
    for node in nodes:
        neighbors = adj[node]
        for neighbor, weight in neighbors:
            network_topology_term += weight * fractional_decay(alpha, t)

    cost += network_topology_term

    return {'cost': cost, 'next_hop': next_hop}

# Example usage:
if __name__ == "__main__":
    packet = {"text_surface": "example text"}
    reference_text = "example reference text"
    nodes = ["A", "B", "C"]
    edges = [("A", "B", 1.0), ("B", "C", 2.0), ("C", "A", 3.0)]
    root = "A"

    result = improved_hybrid_route_and_tree(packet, reference_text, nodes, edges, root)
    print(result)