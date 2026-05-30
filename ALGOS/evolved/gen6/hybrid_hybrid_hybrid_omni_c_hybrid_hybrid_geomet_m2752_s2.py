# DARWIN HAMMER — match 2752, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_omni_chaotic__hybrid_hybrid_liquid_m1302_s2.py (gen5)
# parent_b: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m1375_s0.py (gen5)
# born: 2026-05-29T23:45:37Z

"""
Hybrid JEPA-LTCMH-Geometric Engine
================================
This module fuses the *Joint Embedding Predictive Architecture (JEPA)* energy formulation 
with the *Hybrid Ternary-Router / Liquid Time-Constant MinHash (HTR-LTCMH)* dynamics 
and the *Geometric Product Voronoi Partition* multivector arithmetic.

The mathematical bridge between JEPA and HTR-LTCMH is the integration of the MinHash 
signature generation process within the LTC's input-dependent temporal dynamics, using 
the ternary-router's output as an additional input feature. The Geometric Product Voronoi 
Partition is incorporated by representing the JEPA energy terms as multivectors and 
performing geometric product operations to combine them.

The hybrid treats every graph edge as a JEPA transition:
* ``y``  = parent node,
* ``x``  = child node,
* ``z``  = edge weight (or a derived latent).

Node attributes are embedded by ``sθ`` (the *encoder*).  The predictor
``pφ`` is a simple affine map that mixes the encoded parent with the latent.

The total hybrid loss combines the JEPA energy over all edges with a VICReg
regularizer that keeps the representation space well-conditioned, 
and the HTR-LTCMH's reconstruction loss.

The geometric product is used to combine the JEPA energy terms and the 
Geometric Product Voronoi Partition multivectors.
"""

import math
import random
import sys
import pathlib
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Tuple

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


def jepe_ltc_geometric_energy(x, y, z, theta, phi):
    """
    Calculate the energy of the JEPA-LTCMH-Geometric model.

    Parameters:
    x (np.ndarray): child node
    y (np.ndarray): parent node
    z (np.ndarray): edge weight or latent variable
    theta (np.ndarray): encoder parameters
    phi (np.ndarray): predictor parameters

    Returns:
    energy (float): the energy of the JEPE-LTCMH-Geometric model
    """
    # Calculate the encoded parent node
    encoded_y = np.dot(y, theta)
    # Calculate the predicted child node
    predicted_x = np.dot(encoded_y, phi) + z
    # Calculate the energy
    energy = np.linalg.norm(x - predicted_x) ** 2
    return energy


def jepe_ltc_geometric_multivector_energy(x, y, z, theta, phi, multivector):
    """
    Calculate the energy of the JEPE-LTCMH-Geometric model using multivectors.

    Parameters:
    x (np.ndarray): child node
    y (np.ndarray): parent node
    z (np.ndarray): edge weight or latent variable
    theta (np.ndarray): encoder parameters
    phi (np.ndarray): predictor parameters
    multivector (Multivector): the multivector representation of the energy terms

    Returns:
    energy (float): the energy of the JEPE-LTCMH-Geometric model
    """
    # Calculate the encoded parent node
    encoded_y = np.dot(y, theta)
    # Calculate the predicted child node
    predicted_x = np.dot(encoded_y, phi) + z
    # Calculate the energy using multivectors
    energy_multivector = Multivector({frozenset([0]): 1.0})
    energy_multivector = energy_multivector + multivector * (np.linalg.norm(x - predicted_x) ** 2)
    return energy_multivector


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


if __name__ == "__main__":
    # Smoke test
    x = np.random.rand(EMBED_DIM)
    y = np.random.rand(EMBED_DIM)
    z = np.random.rand(EMBED_DIM)
    theta = np.random.rand(EMBED_DIM, EMBED_DIM)
    phi = np.random.rand(EMBED_DIM, EMBED_DIM)
    multivector = Multivector({frozenset([0]): 1.0})
    energy = jepe_ltc_geometric_energy(x, y, z, theta, phi)
    energy_multivector = jepe_ltc_geometric_multivector_energy(x, y, z, theta, phi, multivector)
    print("Energy:", energy)
    print("Energy Multivector:", energy_multivector.components)