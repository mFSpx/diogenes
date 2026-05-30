# DARWIN HAMMER — match 35, survivor 5
# gen: 1
# parent_a: caputo_fractional.py (gen0)
# parent_b: minimum_cost_tree.py (gen0)
# born: 2026-05-29T23:25:21Z

# Import required modules
import numpy as np
import math
import random
import sys
import pathlib

# Docstring for the hybrid module
"""
This module presents a hybrid algorithm that combines the power-law memory
kernel of Caputo fractional derivatives (caputo_fractional.py) with
minimum-cost tree scoring for path trade-offs (minimum_cost_tree.py). The
bridge between these structures is found in their common use of weighted
sums and path distances. The hybrid module defines a new class FractionalTree,
which incorporates the Caputo power-law kernel into the path weights of the
minimum-cost tree scoring. This allows for the optimization of path trade-offs
under the influence of fractional memory.

Functions:
  - FractionalTree: a class representing a weighted tree with Caputo power-law
    kernel path weights
  - fractional_tree_cost: compute the minimum-cost tree score with Caputo power-law
    kernel path weights for a given tree and path weight
  - caputo_tree_distance: compute the weighted distance between two nodes in a
    tree with Caputo power-law kernel path weights
  - hybrid_ssm_step: perform a single step of the fractional SSM with Caputo power-law
    kernel path weights
"""

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

    Parameters:
      z (float): input value for Gamma(z) approximation

    Returns:
      float: Lanczos approximation of Gamma(z)
    """
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1 - z))
    else:
        return math.sqrt(2 * math.pi) * (z + _LANCZOS_G + 0.5) ** (z + 0.5) \
               * math.exp(-(z + _LANCZOS_G + 0.5)) \
               * _LANCZOS_C.sum(axis=0) / math.prod(z + i for i in range(_LANCZOS_G + 1))


def caputo_derivative(alpha, f, t, dt):
    """Compute the Caputo fractional derivative of f at time t with order alpha.

    Parameters:
      alpha (float): order of the Caputo fractional derivative
      f (function): input function
      t (float): time point at which to compute the derivative
      dt (float): time step size

    Returns:
      float: Caputo fractional derivative of f at time t
    """
    integral = 0.0
    for tau in np.arange(0, t, dt):
        integral += (f(tau + dt) - f(tau)) / ((t - tau) ** alpha)
    return 1 / gamma_lanczos(1 - alpha) * integral


def fractional_decay(alpha, t, dt):
    """Compute the power-law decay kernel at time t with order alpha.

    Parameters:
      alpha (float): order of the power-law decay kernel
      t (float): time point at which to compute the kernel
      dt (float): time step size

    Returns:
      float: power-law decay kernel at time t
    """
    return t ** (alpha - 1) / gamma_lanczos(alpha)


def caputo_tree_distance(alpha, tree, node1, node2, dt):
    """Compute the weighted distance between two nodes in a tree with Caputo power-law kernel path weights.

    Parameters:
      alpha (float): order of the Caputo power-law kernel
      tree (FractionalTree): weighted tree object
      node1 (str): first node
      node2 (str): second node
      dt (float): time step size

    Returns:
      float: weighted distance between node1 and node2
    """
    dist, _ = tree.path_weighted_dijkstra(node1, node2)
    return dist * fractional_decay(alpha, tree.tau, dt)


class FractionalTree:
    """A weighted tree with Caputo power-law kernel path weights."""

    def __init__(self, nodes, edges, root, alpha, path_weight, dt):
        """Initialize the weighted tree with Caputo power-law kernel path weights.

        Parameters:
          nodes (dict[str, tuple[float, float]]): node positions
          edges (list[tuple[str, str]]): edge connections
          root (str): root node
          alpha (float): order of the Caputo power-law kernel
          path_weight (float): path weight penalty
          dt (float): time step size
        """
        self.nodes = nodes
        self.edges = edges
        self.root = root
        self.alpha = alpha
        self.path_weight = path_weight
        self.dt = dt
        self.build_tree()

    def build_tree(self):
        """Build the weighted tree with Caputo power-law kernel path weights."""
        self.adj = {n: [] for n in self.nodes}
        self.material = 0.0
        for a, b in self.edges:
            self.adj[a].append(b)
            self.adj[b].append(a)
            self.material += self.length(self.nodes[a], self.nodes[b])
        self.tau = {n: 0.0 for n in self.nodes}
        self._dijkstra(self.root)

    def length(self, a, b):
        """Compute the Euclidean distance between two nodes.

        Parameters:
          a (tuple[float, float]): node coordinates
          b (tuple[float, float]): node coordinates

        Returns:
          float: Euclidean distance
        """
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _dijkstra(self, node):
        """Compute the weighted distances from the given node to all other nodes."""
        self.dist = {n: float('inf') for n in self.nodes}
        self.tau[n] = 0.0
        self.dist[n] = 0.0
        self.stack = [n]
        while self.stack:
            a = self.stack.pop()
            for b in self.adj[a]:
                if self.dist[b] == float('inf'):
                    self.dist[b] = self.dist[a] + self.length(self.nodes[a], self.nodes[b])
                    self.tau[b] = self.tau[a] + self.length(self.nodes[a], self.nodes[b])
                    self.stack.append(b)
        self.tau = {n: 0.0 for n in self.nodes}

    def path_weighted_dijkstra(self, node1, node2):
        """Compute the weighted distances from the given node to all other nodes.

        Parameters:
          node1 (str): starting node
          node2 (str): target node

        Returns:
          tuple[float, dict[str, tuple[float, float]]]: weighted distance
            and weighted distance map
        """
        self._dijkstra(node1)
        dist = self.dist[node2]
        return dist, self.dist


def fractional_tree_cost(tree, nodes, edges, root, path_weight, alpha):
    """Compute the minimum-cost tree score with Caputo power-law kernel path weights.

    Parameters:
      tree (FractionalTree): weighted tree object
      nodes (dict[str, tuple[float, float]]): node positions
      edges (list[tuple[str, str]]): edge connections
      root (str): root node
      path_weight (float): path weight penalty
      alpha (float): order of the Caputo power-law kernel

    Returns:
      float: minimum-cost tree score
    """
    adj = {n: [] for n in nodes}
    material = 0.0
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        material += tree.length(nodes[a], nodes[b])
    tree.build_tree()
    dist, _ = tree.path_weighted_dijkstra(root, root)
    return material + path_weight * dist * fractional_decay(alpha, tree.tau[root], 1.0)


def hybrid_ssm_step(alpha, state, input, dt):
    """Perform a single step of the fractional SSM with Caputo power-law kernel path weights.

    Parameters:
      alpha (float): order of the Caputo power-law kernel
      state (list[float]): current state
      input (list[float]): current input
      dt (float): time step size

    Returns:
      list[float]: updated state
    """
    w = [fractional_decay(alpha, tree.tau[node], dt) for node in state]
    w = np.array(w) / np.sum(w)
    state = np.sum([w[i] * (state[i - 1] + input[i - 1]) for i in range(len(state))], axis=0)
    return state


if __name__ == "__main__":
    nodes = {
        'A': (0.0, 0.0),
        'B': (1.0, 0.0),
        'C': (1.0, 1.0),
        'D': (0.0, 1.0)
    }
    edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('D', 'C')]
    root = 'A'
    path_weight = 0.2
    alpha = 0.7
    dt = 0.1
    tree = FractionalTree(nodes, edges, root, alpha, path_weight, dt)
    print(fractional_tree_cost(tree, nodes, edges, root, path_weight, alpha))
    state = [0.0, 0.0, 0.0, 0.0]
    input = [1.0, 1.0, 1.0, 1.0]
    print(hybrid_ssm_step(alpha, state, input, dt))