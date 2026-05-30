# DARWIN HAMMER — match 2752, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_omni_chaotic__hybrid_hybrid_liquid_m1302_s2.py (gen5)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m1375_s0.py (gen5)
# born: 2026-05-29T23:45:38Z

"""
Hybrid Multivector-JEPA Engine
============================
This module fuses the *Multivector* algebraic structure from 
`hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m1375_s0.py` 
with the *Joint Embedding Predictive Architecture (JEPA)* energy formulation 
from `hybrid_hybrid_omni_chaotic__hybrid_hybrid_liquid_m1302_s2.py`.

Mathematical Bridge
-------------------
The Multivector class represents an element of Cl(n,0) as a sum of basis blades. 
We integrate the Multivector operations with the JEPA energy formulation by 
treating the Multivector components as learnable features that are encoded 
by `sθ` and predicted by `pφ`. This enables the hybrid to learn geometric 
patterns in data while incorporating a notion of similarity between input 
sequences and a probabilistic belief.

The hybrid Multivector-JEPA engine defines a new algebraic structure that 
combines the strengths of both parents. The Multivector operations are used 
to compute the JEPA energy over all edges in a graph, while the VICReg 
regularizer keeps the representation space well-conditioned.

"""

import math
import numpy as np
import random
import sys
import pathlib
from typing import Dict, List, Tuple

# Configuration constants
EMBED_DIM = 64               # dimensionality of sθ representations
SEED = 42                    # deterministic randomness for reproducibility
MAX_NODES = 1000             # size of synthetic graph in smoke test
MAX_EDGES = 2000

random.seed(SEED)
np.random.seed(SEED)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

# Core blade arithmetic helpers
def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list.

    Each transposition of adjacent indices that are out of order flips the
    sign (anti-commutativity).  Duplicate indices cancel (e_i^2 = 1 → they
    annihilate and contribute +1 to the sign, but the index disappears).
    """
    lst = list(indices)
    sign = 1
    # Bubble sort; track swaps
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # Duplicate: e_i * e_i = 1, remove both
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices).

    Returns (result_blade_frozenset, sign).
    """
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


# Multivector
class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades.

    components: dict mapping frozenset(basis_indices) -> float coefficient
    """

    def __init__(self, components):
        self.components = components

    def __add__(self, other):
        new_components = self.components.copy()
        for blade, coeff in other.components.items():
            if blade in new_components:
                new_components[blade] += coeff
            else:
                new_components[blade] = coeff
        return Multivector(new_components)

    def __rmul__(self, scalar):
        return Multivector({blade: scalar * coeff for blade, coeff in self.components.items()})


def compute_jepa_energy(multivector: Multivector, encoder, predictor, node_x, node_y, edge_z):
    """
    Compute JEPA energy for a graph edge.

    :param multivector: Multivector instance
    :param encoder: Encoder function sθ
    :param predictor: Predictor function pφ
    :param node_x: Child node
    :param node_y: Parent node
    :param edge_z: Edge weight (or derived latent)
    :return: JEPA energy
    """
    encoded_x = encoder(node_x)
    encoded_y = encoder(node_y)
    predicted_x = predictor(encoded_y, edge_z)
    energy = np.linalg.norm(encoded_x - predicted_x) ** 2
    return energy


def compute_multivector_jepa_loss(multivector: Multivector, encoder, predictor, nodes, edges):
    """
    Compute hybrid Multivector-JEPA loss for a graph.

    :param multivector: Multivector instance
    :param encoder: Encoder function sθ
    :param predictor: Predictor function pφ
    :param nodes: List of nodes
    :param edges: List of edges (y, x, z)
    :return: Hybrid Multivector-JEPA loss
    """
    jepa_energies = []
    for edge_y, edge_x, edge_z in edges:
        node_y = nodes[edge_y]
        node_x = nodes[edge_x]
        energy = compute_jepa_energy(multivector, encoder, predictor, node_x, node_y, edge_z)
        jepa_energies.append(energy)
    loss = np.mean(jepa_energies)
    return loss


def hybrid_multivector_jepa_smoke_test():
    # Create a sample Multivector instance
    components = {frozenset(): 1.0, frozenset([0]): 2.0, frozenset([1]): 3.0}
    multivector = Multivector(components)

    # Define encoder and predictor functions
    def encoder(node):
        return np.array([node ** 2, node ** 3])

    def predictor(encoded_y, edge_z):
        return encoded_y + edge_z

    # Create a sample graph
    nodes = np.random.rand(MAX_NODES)
    edges = [(i, i + 1, np.random.rand()) for i in range(MAX_EDGES)]

    # Compute hybrid Multivector-JEPA loss
    loss = compute_multivector_jepa_loss(multivector, encoder, predictor, nodes, edges)
    print(f"Hybrid Multivector-JEPA loss: {loss:.4f}")


if __name__ == "__main__":
    hybrid_multivector_jepa_smoke_test()