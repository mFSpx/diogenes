# DARWIN HAMMER — match 35, survivor 1
# gen: 1
# parent_a: caputo_fractional.py (gen0)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:25:21Z

import numpy as np
import math
import random

__all__ = [
    "gamma_lanczos",
    "caputo_derivative",
    "fractional_decay",
    "fractional_ssm_step",
    "minimum_cost_tree",
    "hybrid_fusion"
]

# Lanczos g=7 coefficients (Numerical Recipes 3rd ed., table 6.1)
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


def gamma_lanczos(z):
    """Lanczos approximation of Gamma(z) for z > 0.

    Uses g=7 Lanczos coefficients from Numerical Recipes.  Accurate to ~15
    significant figures for real z > 0.  For z < 0.5 uses the reflection
    formula Gamma(z)*Gamma(1-z) = pi/sin(pi*z) to stay in the stable region.

    :param z: Input value
    :return: Approximated Gamma(z)
    """
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    else:
        return math.sqrt(2 * math.pi) * (z + _LANCZOS_G + 0.5)**(z + 0.5) * math.exp(-(z + _LANCZOS_G + 0.5)) * np.polyval(_LANCZOS_C, z)


def caputo_derivative(f, alpha, t, tau):
    """Caputo Fractional Derivative

    :param f: Function to differentiate
    :param alpha: Fractional order
    :param t: Time point
    :param tau: Time span
    :return: Caputo Fractional Derivative
    """
    return 1 / gamma_lanczos(1 - alpha) * np.sum(f[tau] / (t - tau)**alpha)


def fractional_decay(alpha):
    """Power-law decay kernel

    :param alpha: Fractional order
    :return: Decay kernel
    """
    return lambda t: t**(alpha - 1) / gamma_lanczos(alpha)


def fractional_ssm_step(alpha, A, B, x, h, tau):
    """Fractional SSM step

    :param alpha: Fractional order
    :param A: System matrix
    :param B: Input matrix
    :param x: Input signal
    :param h: History
    :param tau: Time span
    :return: Updated history
    """
    w = [fractional_decay(alpha)(t) / np.sum([fractional_decay(alpha)(t_i) for t_i in tau]) for t in tau]
    return np.sum([w_i * (A @ h[t_i]) + B * x[t] for t_i, w_i in enumerate(w)])


def minimum_cost_tree(nodes, edges, root, path_weight=0.2):
    """Minimum-cost tree scoring for length/path trade-offs

    :param nodes: Node coordinates
    :param edges: Edges between nodes
    :param root: Root node
    :param path_weight: Path weight
    :return: Minimum-cost tree score
    """
    adj = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    dist = {root: 0.0}
    stack = [root]
    while stack:
        a = stack.pop()
        for b in adj[a]:
            if b not in dist:
                dist[b] = dist[a] + length(nodes[a], nodes[b])
                stack.append(b)
    material = 0.0
    for a, b in edges:
        material += length(nodes[a], nodes[b])
    return material + path_weight * sum(dist.values())


def hybrid_fusion(A, B, alpha, nodes, edges, root, path_weight=0.2):
    """Hybrid fusion of Caputo Fractional Derivative and Minimum-cost Tree

    :param A: System matrix
    :param B: Input matrix
    :param alpha: Fractional order
    :param nodes: Node coordinates
    :param edges: Edges between nodes
    :param root: Root node
    :param path_weight: Path weight
    :return: Hybrid score
    """
    h = [0.0] * len(nodes)
    tau = [0.0]
    tree_score = minimum_cost_tree(nodes, edges, root, path_weight)
    ssm_score = fractional_ssm_step(alpha, A, B, [1.0], h, tau)
    return tree_score + ssm_score


def length(a, b):
    """Euclidean distance between two points

    :param a: Point a
    :param b: Point b
    :return: Euclidean distance
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


if __name__ == "__main__":
    # Smoke test
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([1.0, 1.0])
    alpha = 0.5
    nodes = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    edges = [('0', '1'), ('0', '2')]
    root = '0'
    path_weight = 0.2
    hybrid_score = hybrid_fusion(A, B, alpha, nodes, edges, root, path_weight)
    print(hybrid_score)