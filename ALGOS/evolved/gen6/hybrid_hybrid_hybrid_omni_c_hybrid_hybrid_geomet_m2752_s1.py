# DARWIN HAMMER — match 2752, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_omni_chaotic__hybrid_hybrid_liquid_m1302_s2.py (gen5)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m1375_s0.py (gen5)
# born: 2026-05-29T23:45:37Z

"""
Hybrid JEPA-Geometric Engine
============================
This module fuses the *Joint Embedding Predictive Architecture (JEPA)* energy formulation 
with the geometric algebra based dynamics in the *Hybrid Geometric Product Voronoi Partition*.

Mathematical Bridge
-------------------
We found a mathematical bridge between JEPA and the geometric algebra based dynamics 
by integrating the geometric product within the JEPA's prediction energy, 
using the multivector representation to model the temporal transitions and the latent space.
This fusion enables the hybrid to learn complex patterns in sequential data 
while incorporating a notion of geometric similarity between the input sequences.

The hybrid treats every graph edge as a JEPA transition:
* ``y``  = parent node,
* ``x``  = child node,
* ``z``  = edge weight (or a derived latent).

Node attributes are embedded by ``sθ`` (the *encoder*).  The predictor
``pφ`` is a simple affine map that mixes the encoded parent with the latent.

We use a multivector representation to model the temporal transitions and the latent space, 
and the geometric product to compute the energy of the transitions.
"""

import math
import random
import sys
import pathlib
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------
EMBED_DIM = 64               # dimensionality of sθ representations
SEED = 42                    # deterministic randomness for reproducibility
MAX_NODES = 1000             # size of synthetic graph in smoke test
MAX_EDGES = 2000

random.seed(SEED)
np.random.seed(SEED)

class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades.

    components: dict mapping frozenset(basis_indices) -> float coeffi
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

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def je_pa_energy(x: np.ndarray, y: np.ndarray, z: float) -> float:
    """
    Joint Embedding Predictive Architecture energy function.

    :param x: child node
    :param y: parent node
    :param z: edge weight (or a derived latent)
    :return: energy value
    """
    return np.linalg.norm(sigmoid(x) - sigmoid(y) * z)

def geometric_product(multivector_a: Multivector, multivector_b: Multivector) -> Multivector:
    """
    Geometric product of two multivectors.

    :param multivector_a:
    :param multivector_b:
    :return: product multivector
    """
    result = Multivector({})
    for blade_a, coeff_a in multivector_a.components.items():
        for blade_b, coeff_b in multivector_b.components.items():
            result_blade, sign = _multiply_blades(blade_a, blade_b)
            result.components[result_blade] = result.components.get(result_blade, 0) + sign * coeff_a * coeff_b
    return result

def hybrid_je_pa_geometric_energy(x: np.ndarray, y: np.ndarray, z: float) -> float:
    """
    Hybrid JEPA-Geometric energy function.

    :param x: child node
    :param y: parent node
    :param z: edge weight (or a derived latent)
    :return: energy value
    """
    multivector_x = Multivector({frozenset([0]): 1.0})
    multivector_y = Multivector({frozenset([0]): 1.0})
    multivector_z = Multivector({frozenset([0]): z})
    multivector_product = geometric_product(multivector_x, multivector_y)
    multivector_product = multivector_product * multivector_z
    return je_pa_energy(x, y, z) + np.linalg.norm(multivector_product.components)

if __name__ == "__main__":
    # Smoke test
    x = np.random.rand(EMBED_DIM)
    y = np.random.rand(EMBED_DIM)
    z = np.random.rand()
    energy = hybrid_je_pa_geometric_energy(x, y, z)
    print("Hybrid JEPA-Geometric energy:", energy)